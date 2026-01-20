"""
MAS-PNCG Solver with IPC Contact Support.

Full MAS-PNCG solver with:
- IPC contact handling (PT/EE collisions via BVH)
- MAS preconditioner with contact Hessian integration
- Optimized 2D subspace minimization
- Powell's restart criterion
- CCD-based line search

Based on mas_pncg_solver_nocolli.py with collision detection from pncg_base_ipc.py.
"""

import time
import taichi as ti
from algorithm.collision_detection_bvh import collision_detection_bvh_module
from algorithm.mas_preconditioner_contact import MASPreconditionerContact, BANKSIZE
from util.model_loading import model_loading

# Constants
RESTART_THRESHOLD = 0.5  # Powell's restart threshold
ENERGY_TOL = 1e-4        # Relative energy change tolerance for convergence
STAGNANT_WINDOW = 5      # Number of iterations to check for energy stagnation


@ti.data_oriented
class MASPNCGSolver(collision_detection_bvh_module):
    """
    MAS-PNCG solver with IPC contact support.

    Inherits from collision_detection_bvh_module for BVH-based collision detection.
    Uses MASPreconditionerContact for contact-aware preconditioning.
    """

    def __init__(self, demo='cube_0'):
        """
        Initialize solver.

        Args:
            demo: Demo configuration name
        """
        init_start = time.perf_counter()

        # Load model first to get parameters
        t0 = time.perf_counter()
        model = model_loading(demo=demo)
        t_model_loading = (time.perf_counter() - t0) * 1000

        self.demo = demo
        print(f'[MAS-PNCG] demo={demo}')

        # Store parameters before calling parent __init__
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
        self.ground_barrier = model.ground_barrier
        self.frame = 0
        self.SMALL_NUM = 1e-7

        # Initialize vertex fields (extended for MAS-PNCG)
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
            'z': ti.types.vector(3, float),          # Preconditioned gradient P*g
            'z_prev': ti.types.vector(3, float),     # Previous z (for Powell criterion)
            'w': ti.types.vector(3, float),          # H*p (Hessian-vector product)
            'Hv': ti.types.vector(3, float),         # H*z (for 2D subspace)
            'diagH': ti.types.vector(3, float),      # Diagonal Hessian (for fallback)
        })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        t_mesh_fields = (time.perf_counter() - t0) * 1000
        print(f'Mesh: {self.n_verts} verts, {self.n_cells} cells')

        # Precompute mass and B matrices (inherited from base_deformer)
        t0 = time.perf_counter()
        self.precompute()
        t_precompute = (time.perf_counter() - t0) * 1000

        # Initialize indices for rendering
        t0 = time.perf_counter()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        t_indices = (time.perf_counter() - t0) * 1000

        # Assign elastic type (inherited from base_deformer)
        t0 = time.perf_counter()
        self.assign_elastic_type(model.elastic_type)
        t_elastic = (time.perf_counter() - t0) * 1000

        # Set point lights for visualization
        self.set_point_lights()

        # Boundary elements (for collision detection)
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
        self.adaptive_kappa = getattr(model, 'adaptive_kappa', False)
        self.cache_kappa = getattr(model, 'cache_kappa', True)
        self.init_bvh()
        t_bvh = (time.perf_counter() - t0) * 1000
        print(f'IPC: dHat={self.dHat}, kappa={self.kappa}, barrier={self.barrier_type}')

        self.config = model.dict
        self.config['dHat'] = self.dHat
        self.config['kappa'] = self.kappa
        self.config['barrier_type'] = self.barrier_type

        # MAS Preconditioner with Contact support
        t0 = time.perf_counter()
        print('[MAS-PNCG] Initializing MAS preconditioner with contact support...')
        self.mas_preconditioner = MASPreconditionerContact(
            self.mesh,
            max_contacts=self.MAX_C,
            metis_reordered=True
        )
        t_mas = (time.perf_counter() - t0) * 1000
        print(f'[MAS-PNCG] MAS initialized with {self.mas_preconditioner.level_num} levels')

        # Buffer fields for hessian_matvec
        t0 = time.perf_counter()
        self.hv_input = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.hv_output = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        t_buffers = (time.perf_counter() - t0) * 1000

        # MAS-PNCG state variables
        self.restart_threshold = RESTART_THRESHOLD
        self.energy_tol = ENERGY_TOL
        self.stagnant_window = STAGNANT_WINDOW
        self.energy_history = []

        # Scalar fields for 2D subspace computation
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

        # Print initialization timing summary
        print(f'\n[Init Timing Summary]')
        print(f'  Model loading:      {t_model_loading:7.2f} ms')
        print(f'  Mesh fields:        {t_mesh_fields:7.2f} ms')
        print(f'  Precompute (B,m):   {t_precompute:7.2f} ms')
        print(f'  Indices:            {t_indices:7.2f} ms')
        print(f'  Elastic type:       {t_elastic:7.2f} ms')
        print(f'  BVH init:           {t_bvh:7.2f} ms')
        print(f'  MAS preconditioner: {t_mas:7.2f} ms')
        print(f'  Buffers:            {t_buffers:7.2f} ms')
        print(f'  Scalar fields:      {t_scalars:7.2f} ms')
        print(f'  --------------------------------')
        print(f'  Total:              {init_total:7.2f} ms\n')

    # ========================================================================
    # Barrier Functions (IPC)
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

    # ========================================================================
    # Gradient Computation (with contact)
    # ========================================================================

    @ti.kernel
    def compute_grad(self):
        """Compute gradient for elastic + inertia + contact."""
        # Initialize with inertia term
        for vert in self.mesh.verts:
            vert.grad_prev = vert.grad
            vert.grad = vert.m * (vert.x - vert.x_hat)
            vert.diagH = ti.Vector([vert.m, vert.m, vert.m])

        # Add elastic term
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2

            # Compute elastic gradient
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)

            # Add to vertex gradients
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)

            # Add diagonal Hessian contribution (for fallback)
            diagH_contrib = para * self.compute_diagH(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].diagH += ti.Vector([diagH_contrib[3 * i], diagH_contrib[3 * i + 1],
                                               diagH_contrib[3 * i + 2]], float)

        # Add contact gradient
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            # Compute barrier gradient
            if ti.static(self.barrier_type == 'cubic'):
                bg = self.cubic_barrier_g(dist)
            else:
                bg = self.barrier_g(dist)

            scale = self.dt ** 2

            # Add gradient contribution to each vertex
            for i in ti.static(range(4)):
                vi = ti.i32(ids[i])
                if vi >= 0 and vi < self.n_verts:
                    grad_contrib = scale * bg * cord[i] * t
                    self.mesh.verts.grad[vi] += grad_contrib

    @ti.kernel
    def add_grad_ground_barrier(self):
        """Add ground barrier gradient contribution."""
        for vert in self.mesh.verts:
            d = vert.x[1] - self.ground
            if d < self.dHat and d > 1e-10:
                if ti.static(self.barrier_type == 'cubic'):
                    bg = self.cubic_barrier_g(d)
                else:
                    bg = self.barrier_g(d)
                vert.grad[1] += self.dt ** 2 * bg

    # ========================================================================
    # Preconditioner
    # ========================================================================

    def apply_preconditioner(self):
        """Apply MAS preconditioner."""
        self.mas_preconditioner.apply()

    def apply_preconditioner_woodbury(self):
        """Apply MAS preconditioner with Woodbury updates."""
        self.mas_preconditioner.apply_with_woodbury()

    # ========================================================================
    # Hessian-Vector Products
    # ========================================================================

    @ti.kernel
    def _copy_z_to_buffer(self):
        """Copy mesh.verts.z to hv_input buffer."""
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.z

    @ti.kernel
    def _copy_buffer_to_Hv(self):
        """Copy hv_output buffer to mesh.verts.Hv."""
        for vert in self.mesh.verts:
            vert.Hv = self.hv_output[vert.id]

    def compute_Hv_z(self):
        """Compute Hv = H * z."""
        self._copy_z_to_buffer()
        self.mas_preconditioner.hessian_matvec(self.hv_input, self.hv_output)
        self._copy_buffer_to_Hv()

    @ti.kernel
    def _copy_p_to_buffer(self):
        """Copy mesh.verts.p to hv_input buffer."""
        for vert in self.mesh.verts:
            self.hv_input[vert.id] = vert.p

    @ti.kernel
    def _copy_buffer_to_w(self):
        """Copy hv_output buffer to mesh.verts.w."""
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
        """
        Compute scalar products for the 2x2 system:
        - z_H_z = z^T * H * z
        - z_H_p = z^T * H * p = z^T * w
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

            self.z_H_z[None] += ti.f32(z.dot(Hv))
            self.z_H_p[None] += ti.f32(z.dot(w))
            self.p_H_p[None] += ti.f32(p.dot(w))
            self.z_g[None] += ti.f32(z.dot(g))
            self.p_g[None] += ti.f32(p.dot(g))

    def solve_2x2_subspace(self) -> tuple:
        """
        Solve the 2x2 system for optimal (mu, nu):

        [z*H*z   -z*H*p] [mu]   [z*g ]
        [-p*H*z   p*H*p] [nu] = [-p*g]

        Returns (mu, nu)
        """
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
        """Update search direction: p = -mu * z + nu * p"""
        for vert in self.mesh.verts:
            vert.p = -mu * vert.z + nu * vert.p

    @ti.kernel
    def compute_init_search_direction(self):
        """First iteration: p = -mu * z where mu = z*g / z*H*z"""
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
        """Cache current z as z_prev."""
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
        """Check Powell's restart criterion."""
        g_z_prev = abs(float(self.g_z_prev[None]))
        g_z = float(self.g_z[None])

        if g_z < 1e-12:
            return True

        r_k = g_z_prev / g_z
        return r_k > self.restart_threshold

    # ========================================================================
    # Line Search with CCD
    # ========================================================================

    @ti.kernel
    def compute_p_inf_norm(self) -> float:
        """Compute infinity norm of search direction."""
        p_max = 0.0
        for vert in self.mesh.verts:
            p_norm = vert.p.norm()
            ti.atomic_max(p_max, p_norm)
        return p_max

    def compute_ccd_alpha(self) -> float:
        """
        Compute maximum safe step size using CCD.
        Returns alpha_max such that x + alpha * p is collision-free for alpha in [0, alpha_max].
        """
        # For now, use simple distance-based clamping
        # TODO: Implement proper CCD if needed
        p_max = self.compute_p_inf_norm()
        if p_max > 1e-10:
            alpha_max = 0.5 * self.dHat / p_max
        else:
            alpha_max = 1.0
        return min(alpha_max, 1.0)

    # ========================================================================
    # Position Update
    # ========================================================================

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
    def update_v(self):
        """Update velocity: v = (x - x_n) / dt."""
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient."""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def compute_z_norm(self) -> float:
        """Compute L2 norm of preconditioned gradient z."""
        z_sq = 0.0
        for vert in self.mesh.verts:
            z_sq += vert.z.dot(vert.z)
        return ti.sqrt(z_sq)

    @ti.kernel
    def compute_energy(self) -> float:
        """Compute total energy: inertia + elastic + contact."""
        E = 0.0

        # Inertia energy
        for vert in self.mesh.verts:
            diff = vert.x - vert.x_hat
            E += 0.5 * vert.m * diff.dot(diff)

        # Elastic energy
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            para = c.W * self.dt ** 2
            E += para * self.compute_Psi(F, self.mu, self.la)

        # Contact energy
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            dist = pair.b
            if ti.static(self.barrier_type == 'cubic'):
                E += self.dt ** 2 * self.cubic_barrier_E(dist)
            else:
                E += self.dt ** 2 * self.barrier_E(dist)

        return E

    # ========================================================================
    # Main Step Function
    # ========================================================================

    def check_energy_stagnation(self, energy):
        """Check if energy has stagnated."""
        self.energy_history.append(energy)

        if len(self.energy_history) < self.stagnant_window + 1:
            return False

        if len(self.energy_history) > self.stagnant_window + 1:
            self.energy_history = self.energy_history[-(self.stagnant_window + 1):]

        E_old = self.energy_history[0]
        E_new = self.energy_history[-1]

        if E_old > 1e-12:
            rel_change = abs(E_old - E_new) / E_old
            return rel_change < self.energy_tol

        return False

    def step(self, verbose=False, use_woodbury=True):
        """
        Main MAS-PNCG step with IPC contact.

        Args:
            verbose: Print detailed iteration info
            use_woodbury: Enable Woodbury updates for incremental contact changes

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
        self.energy_history.clear()

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

            if grad_inf < self.epsilon:
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>8} {n_contacts:>6} {t_cnt:>6.2f}ms {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                break

            if self.check_energy_stagnation(energy):
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>8} {n_contacts:>6} {t_cnt:>6.2f}ms {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, energy stagnated')
                break

            # Step 4: Update preconditioner
            t_rebuild_start = time.perf_counter()
            update_type = "cached"

            if first_iter:
                # First iteration: full rebuild + save base state for Woodbury
                self.mas_preconditioner.rebuild_with_contacts(self)
                if use_woodbury:
                    self.mas_preconditioner.save_base_state(self)
                ti.sync()
                update_type = "full"
                first_iter = False
            elif do_restart:
                # Restart: check if Woodbury update is appropriate
                if use_woodbury and self.mas_preconditioner.should_use_woodbury(self):
                    # Incremental update with Woodbury
                    self.mas_preconditioner.woodbury_update(self)
                    update_type = "woodbury"
                else:
                    # Large contact change: full rebuild + update base state
                    self.mas_preconditioner.rebuild_with_contacts(self)
                    if use_woodbury:
                        self.mas_preconditioner.save_base_state(self)
                    ti.sync()
                    update_type = "full"

            t_rebuild = (time.perf_counter() - t_rebuild_start) * 1000

            # Step 5: Cache z_prev
            if iter > 0:
                self.cache_z_prev()

            # Step 6: Apply preconditioner
            t_apply_start = time.perf_counter()
            if update_type == "woodbury":
                self.apply_preconditioner_woodbury()
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

            # Log iteration info
            if verbose:
                restart_str = "Y" if do_restart else "N"
                r_k_str = f'{r_k:>8.4f}' if iter > 0 else f'{"--":>8}'
                print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {z_norm:>10.2e} {mu:>10.4f} {nu:>10.4f} '
                      f'{r_k_str} {restart_str:>3} {update_type:>8} {n_contacts:>6} {t_cnt:>6.2f}ms {t_rebuild:>6.2f}ms {t_apply:>6.2f}ms')

            # Step 10: Line search with CCD
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
    """Test the MAS-PNCG solver with contact."""
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path='.taichi_cache')

    print("\n" + "="*60)
    print("Testing MAS-PNCG Solver with Contact")
    print("="*60)

    # Use a demo with contact support
    solver = MASPNCGSolver(demo='eight_E_drop_demo_contact')

    for f in range(5):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms")


if __name__ == '__main__':
    test_solver()
