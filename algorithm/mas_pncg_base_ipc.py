"""
MAS-PNCG Solver with IPC Contact Support.

Combines:
- collision_detection_bvh_final.py: LBVH-based collision detection
- mas_pncg_solver_nocolli.py: MAS-PNCG optimization framework
- IPC barrier functions for contact handling
- Per-subdomain CCD for conservative line search (Algorithm 2 in paper)
- Woodbury updates for incremental contact changes

This is a clean, modular implementation of MAS-PNCG with IPC.
"""

import time
import taichi as ti
from algorithm.collision_detection_bvh_final import collision_detection_bvh_final_module
from algorithm.mas_preconditioner_contact import MASPreconditionerContact, BANKSIZE
from util.model_loading import model_loading
from math_utils.graphic_util import (
    point_triangle_ccd_lower_bound,
    edge_edge_ccd_lower_bound
)

# Constants
RESTART_THRESHOLD = 0.5  # Powell's restart threshold


@ti.data_oriented
class MASPNCGBaseIPC(collision_detection_bvh_final_module):
    """
    MAS-PNCG solver with IPC contact support.

    Inherits from collision_detection_bvh_final_module for optimized BVH collision detection.
    Uses MASPreconditionerContact for preconditioning with contact Hessian support.

    Key features:
    - Per-subdomain CCD for conservative line search (Algorithm 2 in paper)
    - Woodbury updates for incremental contact changes
    - 2D subspace minimization
    - Powell's restart criterion
    """

    def __init__(self, demo='cube_0'):
        """Initialize solver."""
        init_start = time.perf_counter()

        # Load model
        t0 = time.perf_counter()
        model = model_loading(demo=demo)
        t_model_loading = (time.perf_counter() - t0) * 1000

        self.demo = demo
        print(f'[MAS-PNCG-IPC] demo={demo}')

        # Store parameters
        self.dict = model.dict
        self.mu = ti.field(dtype=ti.f32, shape=())
        self.la = ti.field(dtype=ti.f32, shape=())
        self.mu[None], self.la[None] = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.ground_barrier = model.ground_barrier
        self.frame = 0
        self.SMALL_NUM = 1e-6

        # Initialize vertex fields
        t0 = time.perf_counter()
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),          # Current position
            'v': ti.types.vector(3, float),          # Velocity
            'm': float,                               # Mass
            'x_n': ti.types.vector(3, float),        # Position at frame start
            'x_hat': ti.types.vector(3, float),      # Inertial position
            'x_prev': ti.types.vector(3, float),     # Previous iteration position
            'x_init': ti.types.vector(3, float),     # Rest configuration
            'grad': ti.types.vector(3, float),       # Gradient
            'grad_prev': ti.types.vector(3, float),  # Previous gradient
            'p': ti.types.vector(3, float),          # Search direction
            'z': ti.types.vector(3, float),          # Preconditioned gradient
            'z_prev': ti.types.vector(3, float),     # Previous z (for Powell)
            'w': ti.types.vector(3, float),          # H*p
            'Hv': ti.types.vector(3, float),         # H*z
            'diagH': ti.types.vector(3, float),      # Diagonal Hessian
        })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        t_mesh_fields = (time.perf_counter() - t0) * 1000
        print(f'Mesh: {self.n_verts} verts, {self.n_cells} cells')

        # Precompute
        t0 = time.perf_counter()
        self.precompute()
        t_precompute = (time.perf_counter() - t0) * 1000

        # Initialize indices
        t0 = time.perf_counter()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        t_indices = (time.perf_counter() - t0) * 1000

        # Assign elastic type
        t0 = time.perf_counter()
        self.assign_elastic_type(model.elastic_type)
        t_elastic = (time.perf_counter() - t0) * 1000

        self.set_point_lights()

        # Boundary elements
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print(f'Boundary: {self.n_boundary_points} points, {self.n_boundary_edges} edges, '
              f'{self.n_boundary_triangles} triangles')

        # IPC parameters
        t0 = time.perf_counter()
        self.kappa = model.kappa
        self.dHat = model.dHat
        self.barrier_type = getattr(model, 'barrier_type', 'log')
        self.init_bvh()
        t_bvh = (time.perf_counter() - t0) * 1000
        print(f'IPC: dHat={self.dHat}, kappa={self.kappa}, barrier={self.barrier_type}')

        self.config = model.dict
        self.config['dHat'] = self.dHat
        self.config['kappa'] = self.kappa
        self.config['barrier_type'] = self.barrier_type

        # MAS Preconditioner with contact support
        t0 = time.perf_counter()
        print('[MAS-PNCG-IPC] Initializing MAS preconditioner with contact support...')
        self.mas_preconditioner = MASPreconditionerContact(
            self.mesh,
            metis_reordered=True,
            max_contacts=2**18
        )
        t_mas = (time.perf_counter() - t0) * 1000
        print(f'[MAS-PNCG-IPC] MAS initialized with {self.mas_preconditioner.level_num} levels')

        # Per-subdomain CCD fields
        t0 = time.perf_counter()
        self.n_subdomains = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.subdomain_alpha = ti.field(dtype=ti.f32, shape=self.n_subdomains)
        t_ccd = (time.perf_counter() - t0) * 1000

        # Buffer fields for hessian_matvec
        t0 = time.perf_counter()
        self.hv_input = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.hv_output = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        t_buffers = (time.perf_counter() - t0) * 1000

        # State variables
        self.restart_threshold = RESTART_THRESHOLD

        # Scalar fields for 2D subspace
        t0 = time.perf_counter()
        self.z_H_z = ti.field(dtype=ti.f32, shape=())
        self.z_H_p = ti.field(dtype=ti.f32, shape=())
        self.p_H_p = ti.field(dtype=ti.f32, shape=())
        self.z_g = ti.field(dtype=ti.f32, shape=())
        self.p_g = ti.field(dtype=ti.f32, shape=())
        self.g_z_prev = ti.field(dtype=ti.f32, shape=())
        self.g_z = ti.field(dtype=ti.f32, shape=())
        t_scalars = (time.perf_counter() - t0) * 1000

        init_total = (time.perf_counter() - init_start) * 1000

        print(f'\n[Init Timing Summary]')
        print(f'  Model loading:      {t_model_loading:7.2f} ms')
        print(f'  Mesh fields:        {t_mesh_fields:7.2f} ms')
        print(f'  Precompute (B,m):   {t_precompute:7.2f} ms')
        print(f'  Indices:            {t_indices:7.2f} ms')
        print(f'  Elastic type:       {t_elastic:7.2f} ms')
        print(f'  BVH init:           {t_bvh:7.2f} ms')
        print(f'  MAS preconditioner: {t_mas:7.2f} ms')
        print(f'  Per-subdomain CCD:  {t_ccd:7.2f} ms')
        print(f'  Buffers:            {t_buffers:7.2f} ms')
        print(f'  Scalar fields:      {t_scalars:7.2f} ms')
        print(f'  --------------------------------')
        print(f'  Total:              {init_total:7.2f} ms\n')

    # ========================================================================
    # Barrier Functions
    # ========================================================================

    @ti.func
    def barrier_E(self, d):
        """Log barrier energy."""
        E = 0.0
        if d < self.dHat and d > 1e-10:
            E = -self.kappa * (d - self.dHat) ** 2 * ti.log(d / self.dHat)
        return E

    @ti.func
    def barrier_g(self, d):
        """Log barrier gradient."""
        g = 0.0
        if d < self.dHat and d > 1e-10:
            t2 = d - self.dHat
            g = self.kappa * (t2 * ti.log(d / self.dHat) * (-2.0) - (t2 ** 2) / d)
        return g

    @ti.func
    def barrier_H(self, d):
        """Log barrier Hessian."""
        H = 0.0
        if d < self.dHat and d > 1e-10:
            dHat = self.dHat
            H = self.kappa * ((-2) * ti.log(d / dHat) - 4 + 4 * dHat / d + (d - dHat) ** 2 / d ** 2)
        return H

    @ti.func
    def cubic_barrier_E(self, d):
        """Cubic barrier energy."""
        E = 0.0
        if d < self.dHat:
            y = d - self.dHat
            E = -2.0 * self.kappa * (y * y * y) / (3.0 * self.dHat)
        return E

    @ti.func
    def cubic_barrier_g(self, d):
        """Cubic barrier gradient."""
        g = 0.0
        if d < self.dHat:
            y = d - self.dHat
            g = -2.0 * self.kappa * (y * y) / self.dHat
        return g

    @ti.func
    def cubic_barrier_H(self, d):
        """Cubic barrier Hessian."""
        H = 0.0
        if d < self.dHat:
            H = 4.0 * self.kappa * (1.0 - d / self.dHat)
        return H

    @ti.func
    def get_barrier_E(self, d):
        """Dispatch to appropriate barrier energy."""
        E = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            E = self.cubic_barrier_E(d)
        else:
            E = self.barrier_E(d)
        return E

    @ti.func
    def get_barrier_g(self, d):
        """Dispatch to appropriate barrier gradient."""
        g = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            g = self.cubic_barrier_g(d)
        else:
            g = self.barrier_g(d)
        return g

    @ti.func
    def get_barrier_H(self, d):
        """Dispatch to appropriate barrier Hessian."""
        H = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            H = self.cubic_barrier_H(d)
        else:
            H = self.barrier_H(d)
        return H

    # ========================================================================
    # Gradient Computation
    # ========================================================================

    @ti.kernel
    def compute_grad(self):
        """Compute gradient: inertia + elastic + contact."""
        # Inertia term
        for vert in self.mesh.verts:
            vert.grad_prev = vert.grad
            vert.grad = vert.m * (vert.x - vert.x_hat)

        # Elastic term
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu[None], self.la[None])
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)

        # Contact term
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            bg = self.get_barrier_g(dist)
            scale = self.dt ** 2

            for i in ti.static(range(4)):
                vi = ti.i32(ids[i])
                if vi >= 0 and vi < self.n_verts:
                    self.mesh.verts.grad[vi] += scale * bg * cord[i] * t

    @ti.kernel
    def add_grad_ground_barrier(self):
        """Add ground barrier gradient."""
        for vert in self.mesh.verts:
            d = vert.x[1] - self.ground
            if d < self.dHat and d > 1e-10:
                bg = self.get_barrier_g(d)
                vert.grad[1] += self.dt ** 2 * bg

    # ========================================================================
    # Preconditioner
    # ========================================================================

    def apply_preconditioner(self):
        """Apply MAS preconditioner."""
        self.mas_preconditioner.apply()

    # ========================================================================
    # Hessian-Vector Products
    # ========================================================================

    @ti.kernel
    def _copy_z_to_buffer(self):
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.z

    @ti.kernel
    def _copy_buffer_to_Hv(self):
        for vert in self.mesh.verts:
            vert.Hv = self.hv_output[vert.id]

    def compute_Hv_z(self):
        """Compute Hv = H * z."""
        self._copy_z_to_buffer()
        self.mas_preconditioner.hessian_matvec(self.hv_input, self.hv_output)
        self._copy_buffer_to_Hv()

    @ti.kernel
    def _copy_p_to_buffer(self):
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.p

    @ti.kernel
    def _copy_buffer_to_w(self):
        for vert in self.mesh.verts:
            vert.w = self.hv_output[vert.id]

    def compute_Hv_p(self):
        """Compute w = H * p."""
        self._copy_p_to_buffer()
        self.mas_preconditioner.hessian_matvec(self.hv_input, self.hv_output)
        self._copy_buffer_to_w()

    # ========================================================================
    # 2D Subspace Minimization
    # ========================================================================

    @ti.kernel
    def compute_subspace_scalars(self):
        """Compute scalar products for 2x2 system."""
        self.z_H_z[None] = 0.0
        self.z_H_p[None] = 0.0
        self.p_H_p[None] = 0.0
        self.z_g[None] = 0.0
        self.p_g[None] = 0.0

        for vert in self.mesh.verts:
            z = vert.z
            p = vert.p
            Hv = vert.Hv
            w = vert.w
            g = vert.grad

            self.z_H_z[None] += ti.f32(z.dot(Hv))
            self.z_H_p[None] += ti.f32(z.dot(w))
            self.p_H_p[None] += ti.f32(p.dot(w))
            self.z_g[None] += ti.f32(z.dot(g))
            self.p_g[None] += ti.f32(p.dot(g))

    def solve_2x2_subspace(self) -> tuple:
        """Solve 2x2 system for (mu, nu)."""
        A11 = float(self.z_H_z[None])
        A12 = -float(self.z_H_p[None])
        A22 = float(self.p_H_p[None])
        b1 = float(self.z_g[None])
        b2 = -float(self.p_g[None])

        det = A11 * A22 - A12 * A12
        eps_sing = 1e-12
        max_diag = max(abs(A11 * A22), eps_sing)

        if abs(det) < eps_sing * max_diag:
            if abs(A11) > eps_sing:
                mu = b1 / A11
            else:
                mu = 1.0
            nu = 0.0
        else:
            mu = (b1 * A22 - b2 * A12) / det
            nu = (A11 * b2 - A12 * b1) / det

        return mu, nu

    @ti.kernel
    def update_search_direction(self, mu: float, nu: float):
        """Update: p = -mu * z + nu * p"""
        for vert in self.mesh.verts:
            vert.p = -mu * vert.z + nu * vert.p

    @ti.kernel
    def compute_init_search_direction(self):
        """First iteration: p = -mu * z"""
        self.z_g[None] = 0.0
        self.z_H_z[None] = 0.0
        for vert in self.mesh.verts:
            self.z_g[None] += ti.f32(vert.z.dot(vert.grad))
            self.z_H_z[None] += ti.f32(vert.z.dot(vert.Hv))

        mu = self.z_g[None] / ti.max(self.z_H_z[None], 1e-12)

        for vert in self.mesh.verts:
            vert.p = -mu * vert.z

    # ========================================================================
    # Powell's Restart Criterion
    # ========================================================================

    @ti.kernel
    def cache_z_prev(self):
        for vert in self.mesh.verts:
            vert.z_prev = vert.z

    @ti.kernel
    def compute_powell_scalars(self):
        self.g_z_prev[None] = 0.0
        self.g_z[None] = 0.0
        for vert in self.mesh.verts:
            g = vert.grad
            z = vert.z
            z_prev = vert.z_prev
            self.g_z_prev[None] += ti.f32(g.dot(z_prev))
            self.g_z[None] += ti.f32(g.dot(z))

    def check_powell_restart(self) -> bool:
        g_z_prev = abs(float(self.g_z_prev[None]))
        g_z = float(self.g_z[None])
        if g_z < 1e-12:
            return True
        r_k = g_z_prev / g_z
        return r_k > self.restart_threshold

    # ========================================================================
    # Line Search with Per-Subdomain CCD
    # ========================================================================

    @ti.kernel
    def compute_p_inf_norm(self) -> float:
        p_max = 0.0
        for vert in self.mesh.verts:
            p_norm = vert.p.norm()
            ti.atomic_max(p_max, p_norm)
        return p_max

    @ti.kernel
    def _init_subdomain_alpha(self, alpha_init: float):
        """Initialize per-subdomain alpha to given value."""
        for d in range(self.n_subdomains):
            self.subdomain_alpha[d] = alpha_init

    @ti.kernel
    def _compute_subdomain_ccd_pt(self):
        """
        Per-subdomain CCD for point-triangle contacts.

        For each contact pair involving a subdomain, compute the CCD bound
        and update that subdomain's alpha conservatively.
        """
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            # contact_type = pair.e  # 0 = PT, 1 = EE (if stored)

            # Get vertex positions and search directions
            x0 = self.mesh.verts.x[ti.i32(ids[0])]
            x1 = self.mesh.verts.x[ti.i32(ids[1])]
            x2 = self.mesh.verts.x[ti.i32(ids[2])]
            x3 = self.mesh.verts.x[ti.i32(ids[3])]

            p0 = self.mesh.verts.p[ti.i32(ids[0])]
            p1 = self.mesh.verts.p[ti.i32(ids[1])]
            p2 = self.mesh.verts.p[ti.i32(ids[2])]
            p3 = self.mesh.verts.p[ti.i32(ids[3])]

            # Compute CCD lower bound (conservative step)
            # Use point_triangle_ccd_lower_bound from math_utils
            toi = point_triangle_ccd_lower_bound(
                x0, x1, x2, x3,
                p0, p1, p2, p3,
                self.dHat * 0.5  # Target distance
            )

            # Clamp to reasonable range
            toi = ti.max(toi, 1e-6)

            # Update each involved subdomain's alpha
            for i in ti.static(range(4)):
                vid = ti.i32(ids[i])
                if vid >= 0 and vid < self.n_verts:
                    subdomain_id = vid // BANKSIZE
                    ti.atomic_min(self.subdomain_alpha[subdomain_id], toi)

    @ti.kernel
    def _compute_subdomain_ccd_ee(self):
        """
        Per-subdomain CCD for edge-edge contacts.
        """
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a

            x0 = self.mesh.verts.x[ti.i32(ids[0])]
            x1 = self.mesh.verts.x[ti.i32(ids[1])]
            x2 = self.mesh.verts.x[ti.i32(ids[2])]
            x3 = self.mesh.verts.x[ti.i32(ids[3])]

            p0 = self.mesh.verts.p[ti.i32(ids[0])]
            p1 = self.mesh.verts.p[ti.i32(ids[1])]
            p2 = self.mesh.verts.p[ti.i32(ids[2])]
            p3 = self.mesh.verts.p[ti.i32(ids[3])]

            toi = edge_edge_ccd_lower_bound(
                x0, x1, x2, x3,
                p0, p1, p2, p3,
                self.dHat * 0.5
            )

            toi = ti.max(toi, 1e-6)

            for i in ti.static(range(4)):
                vid = ti.i32(ids[i])
                if vid >= 0 and vid < self.n_verts:
                    subdomain_id = vid // BANKSIZE
                    ti.atomic_min(self.subdomain_alpha[subdomain_id], toi)

    @ti.kernel
    def _get_global_min_alpha(self) -> float:
        """Get the minimum alpha across all subdomains."""
        alpha_min = 1.0
        for d in range(self.n_subdomains):
            ti.atomic_min(alpha_min, self.subdomain_alpha[d])
        return alpha_min

    @ti.kernel
    def _apply_subdomain_alpha(self):
        """
        Apply per-subdomain step sizes to update positions.

        x_new = x + alpha[subdomain(v)] * p
        """
        for vert in self.mesh.verts:
            subdomain_id = vert.id // BANKSIZE
            alpha = self.subdomain_alpha[subdomain_id]
            vert.x += alpha * vert.p

    def compute_per_subdomain_ccd_alpha(self, alpha_init: float = 1.0) -> float:
        """
        Compute per-subdomain CCD step sizes (Algorithm 2 in paper).

        This allows different subdomains to take different step sizes,
        enabling more aggressive steps for subdomains not involved in contacts.

        Args:
            alpha_init: Initial step size (from Newton step)

        Returns:
            Global minimum alpha (for logging)
        """
        # Initialize all subdomains to the Newton step size
        self._init_subdomain_alpha(alpha_init)

        # Compute CCD bounds for contacts
        if self.n_contacts[None] > 0:
            self._compute_subdomain_ccd_pt()
            # Note: Could also call _compute_subdomain_ccd_ee for edge-edge

        return self._get_global_min_alpha()

    def compute_ccd_alpha(self) -> float:
        """Compute safe step size using global CCD (fallback)."""
        p_max = self.compute_p_inf_norm()
        if p_max < 1e-12:
            return 1.0
        # Conservative: limit step to 0.5 * dHat
        alpha_max = 0.5 * self.dHat / p_max
        return min(alpha_max, 1.0)

    # ========================================================================
    # Position Update
    # ========================================================================

    @ti.kernel
    def update_x(self, alpha: float):
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p

    @ti.kernel
    def assign_xn_xhat(self):
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.dt * self.dt * self.gravity

    @ti.kernel
    def update_v(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def compute_z_norm(self) -> float:
        z_sq = 0.0
        for vert in self.mesh.verts:
            z_sq += vert.z.dot(vert.z)
        return ti.sqrt(z_sq)

    @ti.kernel
    def compute_energy(self) -> float:
        """Compute total energy: inertia + elastic + contact."""
        E = 0.0

        # Inertia
        for vert in self.mesh.verts:
            diff = vert.x - vert.x_hat
            E += 0.5 * vert.m * diff.dot(diff)

        # Elastic
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            para = c.W * self.dt ** 2
            E += para * self.compute_Psi(F, self.mu[None], self.la[None])

        # Contact
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            dist = pair.b
            E += self.dt ** 2 * self.get_barrier_E(dist)

        return E

    @ti.kernel
    def add_E_ground_barrier(self) -> float:
        """Add ground barrier energy."""
        E = 0.0
        for vert in self.mesh.verts:
            d = vert.x[1] - self.ground
            if d < self.dHat and d > 1e-10:
                E += self.dt ** 2 * self.get_barrier_E(d)
        return E

    # ========================================================================
    # Main Step Function
    # ========================================================================

    def step(self, verbose=False, use_woodbury=True, use_per_subdomain_ccd=True):
        """
        Main MAS-PNCG step with IPC contact.

        Args:
            verbose: Print detailed iteration info
            use_woodbury: Enable Woodbury updates for incremental contact changes
            use_per_subdomain_ccd: Enable per-subdomain CCD for line search

        Returns: number of iterations
        """
        if verbose:
            print(f'\n{"="*120}')
            print(f'Frame {self.frame}')
            print(f'{"="*120}')
            print(f'{"iter":>4} {"E":>12} {"|g|_inf":>10} {"|z|":>10} {"mu":>10} {"nu":>10} '
                  f'{"r_k":>8} {"rst":>3} {"upd":>8} {"n_cnt":>6} {"t_cnt":>7} {"t_rbd":>7} {"t_app":>7}')
            print(f'{"-"*120}')

        self.assign_xn_xhat()

        do_restart = True
        mu, nu = 0.0, 0.0
        r_k = 0.0
        first_iter = True

        for iter in range(self.iter_max):
            # Step 1: Find contacts
            t_cnt_start = time.perf_counter()
            self.find_cnts(PRINT=False)
            n_contacts = self.n_contacts[None]
            t_cnt = (time.perf_counter() - t_cnt_start) * 1000

            # Step 2: Compute gradient
            self.compute_grad()
            if self.ground_barrier == 1:
                self.add_grad_ground_barrier()

            # Step 3: Check convergence
            grad_inf = self.compute_grad_inf_norm()
            energy = self.compute_energy()
            if self.ground_barrier == 1:
                energy += self.add_E_ground_barrier()

            if grad_inf < self.epsilon:
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>8} {n_contacts:>6} {t_cnt:>6.2f}ms {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                break

            # Step 4: Update preconditioner
            t_rebuild_start = time.perf_counter()
            update_type = "cached"

            if first_iter:
                # First iteration: full rebuild + save base state for Woodbury
                self.mas_preconditioner.rebuild_with_contacts(self)
                if use_woodbury:
                    self.mas_preconditioner.save_base_state(self)
                update_type = "full"
                first_iter = False
            elif do_restart:
                # Restart: full rebuild
                self.mas_preconditioner.rebuild_with_contacts(self)
                if use_woodbury:
                    self.mas_preconditioner.save_base_state(self)
                update_type = "rebuild"
            elif use_woodbury and self.mas_preconditioner.should_use_woodbury(self):
                # Incremental update with Woodbury
                self.mas_preconditioner.woodbury_update(self)
                update_type = "woodbury"
            else:
                # Fall back to full rebuild
                self.mas_preconditioner.rebuild_with_contacts(self)
                update_type = "rebuild"

            ti.sync()
            t_rebuild = (time.perf_counter() - t_rebuild_start) * 1000

            # Step 5: Cache z_prev
            if iter > 0:
                self.cache_z_prev()

            # Step 6: Apply preconditioner
            t_apply_start = time.perf_counter()
            if use_woodbury and update_type == "woodbury":
                self.mas_preconditioner.apply_with_woodbury()
            else:
                self.apply_preconditioner()
            t_apply = (time.perf_counter() - t_apply_start) * 1000

            # Step 7: Compute Hv = H * z
            self.compute_Hv_z()

            if verbose:
                z_norm = self.compute_z_norm()

            # Step 8: Compute search direction
            if iter == 0 or do_restart:
                self.compute_init_search_direction()
                mu = float(self.z_g[None]) / max(float(self.z_H_z[None]), 1e-12)
                nu = 0.0
            else:
                self.compute_subspace_scalars()
                mu, nu = self.solve_2x2_subspace()
                self.update_search_direction(mu, nu)

            # Step 9: Compute w = H * p
            self.compute_Hv_p()

            # Log
            if verbose:
                restart_str = "Y" if do_restart else "N"
                r_k_str = f'{r_k:>8.4f}' if iter > 0 else f'{"--":>8}'
                print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {z_norm:>10.2e} {mu:>10.4f} {nu:>10.4f} '
                      f'{r_k_str} {restart_str:>3} {update_type:>8} {n_contacts:>6} {t_cnt:>6.2f}ms {t_rebuild:>6.2f}ms {t_apply:>6.2f}ms')

            # Step 10: Line search with per-subdomain CCD or global CCD
            if use_per_subdomain_ccd and n_contacts > 0:
                # Compute Newton step size first
                alpha_newton = self.compute_ccd_alpha()
                # Then apply per-subdomain CCD
                alpha_min = self.compute_per_subdomain_ccd_alpha(alpha_newton)
                # Apply per-subdomain step sizes
                self._apply_subdomain_alpha()
            else:
                # Fall back to global CCD
                alpha = self.compute_ccd_alpha()
                self.update_x(alpha)

            # Step 11: Powell's restart criterion
            if iter > 0:
                self.compute_powell_scalars()
                g_z_prev = abs(float(self.g_z_prev[None]))
                g_z = float(self.g_z[None])
                r_k = g_z_prev / g_z if g_z > 1e-12 else 1.0
                do_restart = self.check_powell_restart()
            else:
                do_restart = False
                r_k = 0.0

        self.update_v()
        self.frame += 1
        return iter + 1


def test_solver():
    """Test the MAS-PNCG-IPC solver."""
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path='.taichi_cache_mas_ipc')

    print("\n" + "="*60)
    print("Testing MAS-PNCG-IPC Solver")
    print("="*60)

    solver = MASPNCGBaseIPC(demo='cube_drop')

    print("\n--- Test 1: Full rebuild (no Woodbury) ---")
    for f in range(3):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True, use_woodbury=False)
        t1 = time.perf_counter()
        print(f"Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms")

    print("\n--- Test 2: With Woodbury updates ---")
    solver.frame = 0
    for f in range(3):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True, use_woodbury=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms")

    print("\n--- Test 3: With per-subdomain CCD ---")
    solver.frame = 0
    for f in range(3):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True, use_woodbury=True, use_per_subdomain_ccd=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms")


if __name__ == '__main__':
    test_solver()
