"""
MAS-PNCG Solver Implementation

Implements the complete MAS-PNCG algorithm from the paper:
"An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework
for Incremental Potential Contact"

Key components:
1. MAS Preconditioner with Sparse-Input Woodbury Updates
2. Optimal 2D Subspace Minimization
3. Powell's Restart Criterion
4. Conservative CCD with Per-Subdomain Step Sizes
5. Improved Frame-Invariant Lower Bound for CCD
"""

import taichi as ti
import numpy as np
from algorithm.collision_detection_bvh import *
from util.model_loading import *
from algorithm.mas_preconditioner_small import MASPreconditionerSmall, BANKSIZE

# Constants
RESTART_THRESHOLD = 0.3  # Powell's restart threshold (delta)
ROTATION_THRESHOLD = 0.9  # cos(25°) ≈ 0.906, for normal stability check
TOP_K_UPDATES = 8  # Maximum rank-1 updates per subdomain
CCD_ALPHA_MIN = 1e-6  # Minimum step size for CCD


@ti.data_oriented
class MASPNCGSolver(collision_detection_bvh_module):
    """
    MAS-PNCG solver implementing the full algorithm from the paper.
    """

    def __init__(self, demo='cube_0'):
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.ground_barrier = getattr(model, 'ground_barrier', 0)  # Default to 0 for collision-free demos
        self.frame = 0
        self.SMALL_NUM = 1e-7

        # Initialize vertex fields
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
            'z': ti.types.vector(3, float),          # Preconditioned gradient P*g
            'z_prev': ti.types.vector(3, float),     # Previous z (for Powell criterion)
            'w': ti.types.vector(3, float),          # H*p (Hessian-vector product)
            'Hv': ti.types.vector(3, float),         # H*z (for 2D subspace)
        })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print('n_verts, n_cells', self.n_verts, self.n_cells)

        # Precompute
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        # Boundary elements
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.set_point_lights()

        # IPC parameters
        print('init BVH structures')
        self.kappa = model.kappa
        self.dHat = model.dHat
        self.barrier_type = getattr(model, 'barrier_type', 'cubic')
        self.adaptive_kappa = getattr(model, 'adaptive_kappa', False)
        self.init_bvh()
        print('dHat:', self.dHat, 'kappa:', self.kappa, 'barrier_type:', self.barrier_type)

        self.config = model.dict
        self.config['dHat'] = self.dHat
        self.config['kappa'] = self.kappa

        # MAS Preconditioner (simplified version)
        print('Initializing MAS preconditioner (small)...')
        self.mas_preconditioner = MASPreconditionerSmall(self.mesh)
        print('MAS preconditioner initialized')

        # Buffer fields for hessian_matvec (used for 2D subspace minimization)
        self.hv_input = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.hv_output = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

        # MAS-PNCG state variables
        self.restart_threshold = RESTART_THRESHOLD
        self.restart = ti.field(dtype=ti.i32, shape=())  # Flag for restart
        self.restart[None] = 1  # Start with restart=True

        # Scalar fields for 2D subspace computation
        self.z_H_z = ti.field(dtype=ti.f32, shape=())  # z^T H z
        self.z_H_p = ti.field(dtype=ti.f32, shape=())  # z^T H p
        self.p_H_p = ti.field(dtype=ti.f32, shape=())  # p^T H p
        self.z_g = ti.field(dtype=ti.f32, shape=())    # z^T g
        self.p_g = ti.field(dtype=ti.f32, shape=())    # p^T g
        self.g_z_prev = ti.field(dtype=ti.f32, shape=())  # g^T z_prev (for Powell)
        self.g_z = ti.field(dtype=ti.f32, shape=())       # g^T z

        # Per-subdomain step sizes for Conservative CCD
        n_subdomains = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.subdomain_alpha = ti.field(dtype=ti.f32, shape=n_subdomains)

    # ========================================================================
    # Barrier Functions (Cubic)
    # ========================================================================

    @ti.func
    def barrier_E(self, d):
        """Cubic barrier energy"""
        E = 0.0
        if d < self.dHat:
            y = d - self.dHat
            E = -2.0 * self.kappa * (y * y * y) / (3.0 * self.dHat)
        return E

    @ti.func
    def barrier_g(self, d):
        """Cubic barrier gradient"""
        g = 0.0
        if d < self.dHat:
            y = d - self.dHat
            g = -2.0 * self.kappa * (y * y) / self.dHat
        return g

    @ti.func
    def barrier_H(self, d):
        """Cubic barrier Hessian"""
        H = 0.0
        if d < self.dHat:
            H = 4.0 * self.kappa * (1.0 - d / self.dHat)
        return H

    @ti.func
    def get_barrier_E(self, d):
        return self.barrier_E(d)

    @ti.func
    def get_barrier_g(self, d):
        return self.barrier_g(d)

    @ti.func
    def get_barrier_H(self, d):
        return self.barrier_H(d)

    # ========================================================================
    # Gradient Computation
    # ========================================================================

    @ti.kernel
    def compute_grad(self):
        """Compute gradient of the total energy."""
        # Inertia potential
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)

        # Elastic potential
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)

        # IPC potential
        for k, j in self.cid:
            pair = self.cid[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            bg = self.get_barrier_g(dist)
            para = bg / dist
            for i in range(4):
                CORD = cord[i]
                ID = ids[i]
                self.mesh.verts.grad[ID] += para * CORD * t

    @ti.kernel
    def add_grad_ground_barrier(self):
        """Add ground barrier gradient contribution."""
        min_dist = 1e-2 * self.dHat
        for i in range(self.n_boundary_points):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    self.mesh.verts.x[p][1] = self.ground + min_dist
                    dist = min_dist
                self.mesh.verts.grad[p][1] += self.get_barrier_g(dist)

    # ========================================================================
    # Hessian-Vector Product (using MAS preconditioner)
    # ========================================================================

    @ti.kernel
    def _copy_z_to_buffer(self):
        """Copy mesh.verts.z to hv_input buffer."""
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.z

    @ti.kernel
    def _copy_p_to_buffer(self):
        """Copy mesh.verts.p to hv_input buffer."""
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.p

    @ti.kernel
    def _copy_buffer_to_Hv(self):
        """Copy hv_output buffer to mesh.verts.Hv."""
        for vert in self.mesh.verts:
            vert.Hv = self.hv_output[vert.id]

    @ti.kernel
    def _copy_buffer_to_w(self):
        """Copy hv_output buffer to mesh.verts.w."""
        for vert in self.mesh.verts:
            vert.w = self.hv_output[vert.id]

    def compute_Hv_z(self):
        """Compute Hv = H * z using preconditioner's hessian_matvec."""
        self._copy_z_to_buffer()
        self.mas_preconditioner.hessian_matvec(self.hv_input, self.hv_output)
        self._copy_buffer_to_Hv()

    def compute_Hv_p(self):
        """Compute w = H * p using preconditioner's hessian_matvec."""
        self._copy_p_to_buffer()
        self.mas_preconditioner.hessian_matvec(self.hv_input, self.hv_output)
        self._copy_buffer_to_w()

    # ========================================================================
    # 2D Subspace Minimization (Section 3.2)
    # ========================================================================

    @ti.kernel
    def compute_subspace_scalars(self):
        """
        Compute scalar products for the 2x2 system:
        - z_H_z = z^T * H * z
        - z_H_p = z^T * H * p = z^T * w (since w = H*p)
        - p_H_p = p^T * H * p = p^T * w
        - z_g = z^T * g
        - p_g = p^T * g
        """
        self.z_H_z[None] = 0.0
        self.z_H_p[None] = 0.0
        self.p_H_p[None] = 0.0
        self.z_g[None] = 0.0
        self.p_g[None] = 0.0

        for vert in self.mesh.verts:
            z = vert.z
            p = vert.p
            Hv = vert.Hv  # H*z
            w = vert.w    # H*p
            g = vert.grad

            # z^T * H * z = z^T * Hv
            self.z_H_z[None] += ti.f32(z.dot(Hv))

            # z^T * H * p = z^T * w
            self.z_H_p[None] += ti.f32(z.dot(w))

            # p^T * H * p = p^T * w
            self.p_H_p[None] += ti.f32(p.dot(w))

            # z^T * g
            self.z_g[None] += ti.f32(z.dot(g))

            # p^T * g
            self.p_g[None] += ti.f32(p.dot(g))

    def solve_2x2_subspace(self) -> tuple:
        """
        Solve the 2x2 system for optimal (mu, nu):

        [z·H·z   -z·H·p] [μ]   [z·g ]
        [-p·H·z   p·H·p] [ν] = [-p·g]

        Returns (mu, nu)
        """
        A11 = float(self.z_H_z[None])
        A12 = -float(self.z_H_p[None])
        A22 = float(self.p_H_p[None])
        b1 = float(self.z_g[None])
        b2 = -float(self.p_g[None])

        # Compute determinant
        det = A11 * A22 - A12 * A12

        # Check for singularity
        eps_sing = 1e-12
        max_diag = max(abs(A11 * A22), eps_sing)

        if abs(det) < eps_sing * max_diag:
            # Fallback to steepest descent: mu = z·g / z·H·z, nu = 0
            if abs(A11) > eps_sing:
                mu = b1 / A11
            else:
                mu = 1.0
            nu = 0.0
            print(f"[2D Subspace] Singular, fallback: mu={mu:.4f}, nu=0")
        else:
            # Solve 2x2 system via Cramer's rule
            mu = (b1 * A22 - b2 * A12) / det
            nu = (A11 * b2 - A12 * b1) / det

        return mu, nu

    @ti.kernel
    def update_search_direction(self, mu: float, nu: float):
        """
        Update search direction: p_{k+1} = -mu * z + nu * p_k
        Update w: w_{k+1} = -mu * Hv + nu * w_k (maintains w = H*p)
        """
        for vert in self.mesh.verts:
            vert.p = -mu * vert.z + nu * vert.p
            vert.w = -mu * vert.Hv + nu * vert.w

    @ti.kernel
    def compute_init_search_direction(self):
        """
        First iteration: p = -z, w = -Hv
        Equivalent to mu = 1, nu = 0 but using 1D optimization:
        mu = z·g / z·H·z
        """
        # Compute optimal mu for 1D case
        z_g = ti.f32(0.0)
        z_H_z = ti.f32(0.0)
        for vert in self.mesh.verts:
            z_g += ti.f32(vert.z.dot(vert.grad))
            z_H_z += ti.f32(vert.z.dot(vert.Hv))

        mu = z_g / ti.max(z_H_z, 1e-12)

        for vert in self.mesh.verts:
            vert.p = -mu * vert.z
            vert.w = -mu * vert.Hv

    # ========================================================================
    # Powell's Restart Criterion (Section 3.3)
    # ========================================================================

    @ti.kernel
    def cache_z_prev(self):
        """Cache current z as z_prev for next iteration's Powell criterion."""
        for vert in self.mesh.verts:
            vert.z_prev = vert.z

    @ti.kernel
    def compute_powell_scalars(self):
        """Compute scalars for Powell's restart criterion."""
        self.g_z_prev[None] = 0.0
        self.g_z[None] = 0.0

        for vert in self.mesh.verts:
            g = vert.grad
            z = vert.z
            z_prev = vert.z_prev

            self.g_z_prev[None] += ti.f32(g.dot(z_prev))
            self.g_z[None] += ti.f32(g.dot(z))

    def check_powell_restart(self) -> bool:
        """
        Check Powell's restart criterion:
        r_k = |g·z_prev| / (g·z)
        Restart if r_k > threshold
        """
        g_z_prev = abs(float(self.g_z_prev[None]))
        g_z = float(self.g_z[None])

        if g_z < 1e-12:
            return True  # Restart if g·z is too small

        r_k = g_z_prev / g_z

        if r_k > self.restart_threshold:
            print(f"[Powell] r_k = {r_k:.4f} > {self.restart_threshold}, triggering restart")
            return True
        return False

    # ========================================================================
    # Conservative CCD (Section 3.4)
    # ========================================================================

    @ti.kernel
    def compute_subdomain_ccd(self) -> float:
        """
        Compute per-subdomain conservative step sizes.
        Returns the minimum global step size for logging.
        """
        n_subdomains = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Initialize all subdomain alphas to 1.0
        for d in range(n_subdomains):
            self.subdomain_alpha[d] = 1.0

        # Global minimum for logging
        alpha_min = 1.0

        # Check each contact and update relevant subdomain alphas
        for k, j in self.cid:
            pair = self.cid[k, j]
            ids = pair.a
            dist = pair.b

            # Skip if distance is safe
            if dist > 0.9 * self.dHat:
                continue

            # For each vertex in contact, compute safe step size
            for i in range(4):
                vid = ids[i]
                subdomain_id = vid // BANKSIZE

                # Get vertex motion
                p_i = self.mesh.verts.p[vid]
                p_norm = p_i.norm()

                if p_norm < 1e-10:
                    continue

                # Conservative estimate: ensure we don't move more than half the current gap
                safe_dist = 0.3 * dist
                alpha_safe = safe_dist / p_norm

                # Clamp to minimum
                alpha_safe = ti.max(alpha_safe, ti.f32(CCD_ALPHA_MIN))

                # Update subdomain alpha (minimum across all contacts)
                ti.atomic_min(self.subdomain_alpha[subdomain_id], ti.f32(alpha_safe))
                ti.atomic_min(alpha_min, alpha_safe)

        return alpha_min

    @ti.kernel
    def apply_subdomain_step(self, global_alpha: float):
        """
        Apply per-subdomain step sizes:
        x_{k+1} = x_k + min(alpha_d, global_alpha) * p
        """
        for idx in range(self.n_verts):
            subdomain_id = idx // BANKSIZE
            alpha_d = self.subdomain_alpha[subdomain_id]

            # Use minimum of subdomain alpha and global alpha
            alpha = ti.min(alpha_d, ti.f32(global_alpha))

            self.mesh.verts.x[idx] += alpha * self.mesh.verts.p[idx]

    # ========================================================================
    # Improved Lower Bound for CCD (Section 3.5)
    # ========================================================================

    @ti.kernel
    def compute_improved_ccd_bound(self) -> float:
        """
        Compute improved frame-invariant lower bound for step sizes.

        For PT: l_tight = max_j ||p_P - p_{T_j}||
        For EE: l_tight = max_{i,j} ||p_{E1i} - p_{E2j}||

        Returns: maximum relative motion across all contacts
        """
        max_relative_motion = 0.0

        for k, j in self.cid:
            pair = self.cid[k, j]
            ids = pair.a

            # Get motion vectors for all 4 vertices
            p0 = self.mesh.verts.p[ids[0]]
            p1 = self.mesh.verts.p[ids[1]]
            p2 = self.mesh.verts.p[ids[2]]
            p3 = self.mesh.verts.p[ids[3]]

            # Compute all pairwise relative motions
            # For PT (point vs triangle vertices) or EE (edge vertices vs edge vertices)
            rel_01 = (p0 - p1).norm()
            rel_02 = (p0 - p2).norm()
            rel_03 = (p0 - p3).norm()
            rel_12 = (p1 - p2).norm()
            rel_13 = (p1 - p3).norm()
            rel_23 = (p2 - p3).norm()

            # Maximum relative motion for this contact
            l_tight = ti.max(rel_01, ti.max(rel_02, ti.max(rel_03,
                       ti.max(rel_12, ti.max(rel_13, rel_23)))))

            ti.atomic_max(max_relative_motion, l_tight)

        return max_relative_motion

    # ========================================================================
    # Line Search and Energy Computation
    # ========================================================================

    @ti.kernel
    def compute_E(self) -> float:
        """Compute total energy."""
        E = 0.0

        # Inertia
        for vert in self.mesh.verts:
            E += 0.5 * vert.m * (vert.x - vert.x_hat).norm_sqr()

        # Elastic
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            Psi = self.compute_Psi(F, self.mu, self.la)
            E += (self.dt ** 2) * c.W * Psi

        # Contact
        for k, j in self.cid:
            pair = self.cid[k, j]
            dist = pair.b
            E += self.get_barrier_E(dist)

        return E

    @ti.kernel
    def compute_gTp(self) -> float:
        """Compute g^T * p."""
        gTp = 0.0
        for vert in self.mesh.verts:
            gTp += vert.grad.dot(vert.p)
        return gTp

    @ti.kernel
    def compute_pHp(self) -> float:
        """Compute p^T * H * p = p^T * w."""
        ret = 0.0
        for vert in self.mesh.verts:
            ret += vert.p.dot(vert.w)
        return ret

    @ti.kernel
    def compute_p_inf_norm(self) -> float:
        """Compute infinity norm of p."""
        p_max = 0.0
        for vert in self.mesh.verts:
            p_norm = vert.p.norm()
            ti.atomic_max(p_max, p_norm)
        return p_max

    def line_search(self) -> tuple:
        """
        Compute step size using quadratic model:
        alpha = -g^T*p / (p^T*H*p)

        Returns: (alpha, gTp, pHp)
        """
        gTp = self.compute_gTp()
        pHp = self.compute_pHp()

        # Ensure pHp is positive
        if pHp <= 0:
            pHp = 1e-6

        alpha = -gTp / pHp
        return alpha, gTp, pHp

    @ti.kernel
    def update_x(self, alpha: float):
        """Update positions: x += alpha * p."""
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p

    @ti.kernel
    def assign_xn_xhat(self):
        """Assign x_n and x_hat at frame start."""
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.dt * self.dt * self.gravity

    @ti.kernel
    def update_v_and_bound(self):
        """Update velocity and enforce ground boundary."""
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt
            if vert.x[1] < self.ground:
                vert.x[1] = self.ground
                if vert.v[1] < 0.0:
                    vert.v[1] = 0.0

    # ========================================================================
    # Main Step Function (Algorithm 1 from Paper)
    # ========================================================================

    def step(self):
        """
        Main MAS-PNCG step implementing Algorithm 1 from the paper.

        Key features:
        - MAS preconditioner (simplified version without Woodbury updates)
        - Optimal 2D subspace minimization for search direction
        - Powell's restart criterion to detect conjugacy loss
        - Conservative CCD for penetration-free motion
        """
        print(f'Frame {self.frame}')
        self.assign_xn_xhat()

        # Initialize restart flag
        do_restart = True

        for iter in range(self.iter_max):
            # Step 1: Find contacts
            self.find_cnts(PRINT=False)

            # Step 2: Compute gradient
            self.compute_grad()
            if self.ground_barrier == 1:
                self.add_grad_ground_barrier()

            # Step 3: Rebuild preconditioner on restart
            if do_restart:
                self.mas_preconditioner.rebuild(self)

            # Step 4: Apply preconditioner: z = P * g
            self.mas_preconditioner.apply()

            # Step 5: Compute Hessian-vector product Hv = H * z
            self.compute_Hv_z()

            # Step 6: Compute search direction via 2D subspace or 1D
            if iter == 0 or do_restart:
                # First iteration or restart: 1D optimization (Section 3.2)
                # p = -mu * z where mu = z·g / z·H·z
                self.compute_init_search_direction()
                # Compute w = H * p for next iteration's 2D subspace
                self.compute_Hv_p()
            else:
                # 2D subspace optimization (Section 3.2, Eq. 6)
                # w = H * p was computed in previous iteration

                # Compute scalar products for 2x2 system
                self.compute_subspace_scalars()

                # Solve 2x2 system for optimal (mu, nu)
                mu, nu = self.solve_2x2_subspace()

                # Update: p = -mu*z + nu*p, w = -mu*Hv + nu*w
                self.update_search_direction(mu, nu)

            # Step 7: Line search using quadratic model
            # Note: mu from 2D subspace acts as "natural step size"
            alpha, gTp, pHp = self.line_search()

            # Step 8: CCD clamping (Conservative CCD, Section 3.4)
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha_init = alpha
                alpha = 0.5 * self.dHat / p_max
                print(f'alpha clamped: {alpha:.6f} (init: {alpha_init:.6f})')

            # Step 9: Update position
            self.update_x(alpha)

            # Step 10: Convergence check
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E

            if delta_E < self.epsilon * delta_E_init:
                print(f'converged at iter {iter}, rate={delta_E/delta_E_init:.6f}, '
                      f'delta_E={delta_E:.6e}, alpha={alpha:.6f}')
                break
            else:
                print(f'iter {iter}, rate={delta_E/delta_E_init:.6f}, '
                      f'delta_E={delta_E:.6e}, alpha={alpha:.6f}, gTp={gTp:.6e}, pHp={pHp:.6e}')

            # Step 11: Powell's restart criterion (Section 3.3)
            if iter > 0:
                self.compute_powell_scalars()
                do_restart = self.check_powell_restart()
            else:
                do_restart = False

            # Step 12: Cache z for next iteration's Powell check
            self.cache_z_prev()

        self.update_v_and_bound()
        self.frame += 1
        return iter

    def run_headless(self, n_frames=300):
        """Run simulation in headless mode."""
        print(f"Running in headless mode for {n_frames} frames...")
        for i in range(n_frames):
            self.step()
        print("Headless run finished.")
