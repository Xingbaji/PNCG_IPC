"""
MAS-PNCG Solver (No Collision Version)

Simplified MAS-PNCG solver for collision-free scenarios:
- Elastic + Inertia energy only (no IPC contact)
- Optimized 2D subspace minimization
- Powell's restart criterion

This version is optimized for free-fall validation tests where collision
detection is disabled.
"""

import time
import taichi as ti
from algorithm.base_deformer import base_deformer
from algorithm.mas_preconditioner_small import MASPreconditionerSmall, BANKSIZE
from util.model_loading import model_loading

# Constants
RESTART_THRESHOLD = 0.5  # Powell's restart threshold (increased from 0.3 to reduce restart frequency)
ENERGY_TOL = 1e-4        # Relative energy change tolerance for convergence
STAGNANT_WINDOW = 5      # Number of iterations to check for energy stagnation


@ti.data_oriented
class MASPNCGSolverNoCollision(base_deformer):
    """
    MAS-PNCG solver for collision-free scenarios.

    Inherits from base_deformer for common functionality.
    Optimized for elastic + inertia simulations without contact handling.
    """

    def __init__(self, demo='cube_freefall_10'):
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
        print(f'[MAS-PNCG NoCollision] demo={demo}')

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
        self.frame = 0

        # Initialize vertex fields (extended from base_deformer)
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
        self.config = model.dict

        # MAS Preconditioner
        # Note: model_loading applies METIS reordering, so metis_reordered=True
        t0 = time.perf_counter()
        print('[MAS-PNCG] Initializing MAS preconditioner...')

        # Get actual partition count from METIS result (may differ from ceil(n_verts/BANKSIZE)
        # due to component-aware partitioning)
        metis_n_parts = None
        metis_sorted_to_partition = None
        if hasattr(model, 'metis_result') and model.metis_result is not None:
            if hasattr(model.metis_result, 'n_parts'):
                metis_n_parts = model.metis_result.n_parts
            if hasattr(model.metis_result, 'sorted_to_partition'):
                metis_sorted_to_partition = model.metis_result.sorted_to_partition

        self.mas_preconditioner = MASPreconditionerSmall(
            self.mesh,
            metis_reordered=True,
            metis_n_parts=metis_n_parts
        )

        # Pass METIS partition mapping for correct going_next computation
        # This is needed when partition sizes < BANKSIZE (common with component-aware METIS)
        if metis_sorted_to_partition is not None:
            self.mas_preconditioner.sorted_to_partition = metis_sorted_to_partition

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
        self.energy_history = []  # For tracking energy stagnation

        # Scalar fields for 2D subspace computation
        t0 = time.perf_counter()
        self.z_H_z = ti.field(dtype=ti.f32, shape=())  # z^T H z
        self.z_H_p = ti.field(dtype=ti.f32, shape=())  # z^T H p
        self.p_H_p = ti.field(dtype=ti.f32, shape=())  # p^T H p
        self.z_g = ti.field(dtype=ti.f32, shape=())    # z^T g
        self.p_g = ti.field(dtype=ti.f32, shape=())    # p^T g
        self.g_z_prev = ti.field(dtype=ti.f32, shape=())  # g^T z_prev (for Powell)
        self.g_z = ti.field(dtype=ti.f32, shape=())       # g^T z
        t_scalars = (time.perf_counter() - t0) * 1000

        init_total = (time.perf_counter() - init_start) * 1000

        # Print initialization timing summary
        print(f'\n[Init Timing Summary]')
        print(f'  Model loading:      {t_model_loading:7.2f} ms')
        print(f'  Mesh fields:        {t_mesh_fields:7.2f} ms')
        print(f'  Precompute (B,m):   {t_precompute:7.2f} ms')
        print(f'  Indices:            {t_indices:7.2f} ms')
        print(f'  Elastic type:       {t_elastic:7.2f} ms')
        print(f'  MAS preconditioner: {t_mas:7.2f} ms')
        print(f'  Buffers:            {t_buffers:7.2f} ms')
        print(f'  Scalar fields:      {t_scalars:7.2f} ms')
        print(f'  --------------------------------')
        print(f'  Total:              {init_total:7.2f} ms\n')

    # ========================================================================
    # Gradient Computation
    # ========================================================================

    @ti.kernel
    def compute_grad(self):
        """Compute gradient for elastic + inertia."""
        # Initialize with inertia term
        # Note: ti.mesh_local disabled due to CUDA scalarize bug in taichi 1.7.4
        # ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            vert.grad_prev = vert.grad
            vert.grad = vert.m * (vert.x - vert.x_hat)

        # Add elastic term
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2

            # Compute elastic gradient (returns 12x1 vector)
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)

            # Add to vertex gradients
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)

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
        """Compute w = H * p (exact Hessian-vector product for current x)."""
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

        # Compute determinant
        det = A11 * A22 - A12 * A12

        # Check for singularity
        eps_sing = 1e-12
        max_diag = max(abs(A11 * A22), eps_sing)

        if abs(det) < eps_sing * max_diag:
            print("Fallback to steepest descent")
            if abs(A11) > eps_sing:
                mu = b1 / A11
            else:
                mu = 1.0
            nu = 0.0
        else:
            # Solve 2x2 system via Cramer's rule
            mu = (b1 * A22 - b2 * A12) / det
            nu = (A11 * b2 - A12 * b1) / det

        return mu, nu

    @ti.kernel
    def update_search_direction(self, mu: float, nu: float):
        """
        Update search direction: p = -mu * z + nu * p
        Note: w = H * p will be computed separately via compute_Hv_p()
        """
        for vert in self.mesh.verts:
            vert.p = -mu * vert.z + nu * vert.p

    @ti.kernel
    def compute_init_search_direction(self):
        """
        First iteration: p = -mu * z where mu = z*g / z*H*z
        Note: w = H * p will be computed separately via compute_Hv_p()
        Also stores z_g and z_H_z to scalar fields for logging.
        """
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
        if r_k > self.restart_threshold:
            return True
        return False

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
        """Compute total energy: inertia + elastic."""
        E = 0.0

        # Inertia energy: 0.5 * (x - x_hat)^T M (x - x_hat)
        for vert in self.mesh.verts:
            diff = vert.x - vert.x_hat
            E += 0.5 * vert.m * diff.dot(diff)

        # Elastic energy (ARAP or SNH depending on elastic_type)
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            para = c.W * self.dt ** 2
            E += para * self.compute_Psi(F, self.mu, self.la)

        return E

    # ========================================================================
    # Main Step Function
    # ========================================================================

    def check_energy_stagnation(self, energy):
        """
        Check if energy has stagnated (relative change below tolerance).

        Returns: True if converged due to energy stagnation
        """
        self.energy_history.append(energy)

        if len(self.energy_history) < self.stagnant_window + 1:
            return False

        # Keep only recent history
        if len(self.energy_history) > self.stagnant_window + 1:
            self.energy_history = self.energy_history[-(self.stagnant_window + 1):]

        # Check relative energy change over window
        E_old = self.energy_history[0]
        E_new = self.energy_history[-1]

        if E_old > 1e-12:
            rel_change = abs(E_old - E_new) / E_old
            return rel_change < self.energy_tol

        return False

    def step(self, verbose=False):
        """
        Main MAS-PNCG step for collision-free scenarios.

        Convergence criteria:
        1. Gradient norm: |g|_inf < epsilon
        2. Energy stagnation: relative energy change < energy_tol over stagnant_window iters

        Returns: number of iterations
        """
        if verbose:
            print(f'\n{"="*100}')
            print(f'Frame {self.frame}')
            print(f'{"="*100}')
            print(f'{"iter":>4} {"E":>12} {"|g|_inf":>10} {"|z|":>10} {"mu":>10} {"nu":>10} '
                  f'{"r_k":>8} {"rst":>3} {"t_rbd":>7} {"t_app":>7} {"t_Hz":>7} {"t_Hp":>7}')
            print(f'{"-"*100}')

        self.assign_xn_xhat()
        self.energy_history.clear()  # Reset energy history for this frame

        do_restart = True
        mu, nu = 0.0, 0.0  # Initialize for logging
        r_k = 0.0  # Powell criterion value

        for iter in range(self.iter_max):
            # Step 1: Compute gradient
            self.compute_grad()

            # Step 2: Check convergence (at iteration start, after gradient computation)
            grad_inf = self.compute_grad_inf_norm()

            # Compute energy (always needed for convergence check)
            energy = self.compute_energy()

            # Check gradient convergence
            if grad_inf < self.epsilon:
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>7} {"--":>7} {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                break

            # Check energy stagnation convergence
            if self.check_energy_stagnation(energy):
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>7} {"--":>7} {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, energy stagnated (rel_change < {self.energy_tol:.0e})')
                break

            # Step 3: Rebuild preconditioner on restart
            t_rebuild_start = time.perf_counter()
            if do_restart:
                self.mas_preconditioner.rebuild(self)
                ti.sync()  # Ensure GPU work completes before timing
            t_rebuild = (time.perf_counter() - t_rebuild_start) * 1000

            # Step 3.5: Cache z_prev BEFORE applying preconditioner (for Powell criterion)
            # z_prev stores z from previous iteration, needed to compute g^T z_prev
            if iter > 0:
                self.cache_z_prev()

            # Step 4: Apply preconditioner: z = P^{-1} g
            t_apply_start = time.perf_counter()
            self.apply_preconditioner()
            t_apply = (time.perf_counter() - t_apply_start) * 1000

            # Step 5: Compute Hv = H * z (for 2D subspace)
            t_hz_start = time.perf_counter()
            self.compute_Hv_z()
            t_hz = (time.perf_counter() - t_hz_start) * 1000

            # Compute z norm for logging
            if verbose:
                z_norm = self.compute_z_norm()

            # Step 6: Compute search direction via 2D subspace minimization
            # p = -mu * z + nu * p
            if iter == 0 or do_restart:
                self.compute_init_search_direction()
                # For restart, mu is computed inside, nu = 0
                mu = float(self.z_g[None]) / max(float(self.z_H_z[None]), 1e-12)
                nu = 0.0
            else:
                self.compute_subspace_scalars()
                mu, nu = self.solve_2x2_subspace()
                self.update_search_direction(mu, nu)

            # Step 6.5: Compute w = H * p (exact, using current Hessian)
            # This ensures w accurately reflects H_{k+1} * p_{k+1} for 2D subspace in next iter
            t_hp_start = time.perf_counter()
            self.compute_Hv_p()
            t_hp = (time.perf_counter() - t_hp_start) * 1000

            # Log iteration info
            if verbose:
                restart_str = "Y" if do_restart else "N"
                r_k_str = f'{r_k:>8.4f}' if iter > 0 else f'{"--":>8}'
                print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {z_norm:>10.2e} {mu:>10.4f} {nu:>10.4f} '
                      f'{r_k_str} {restart_str:>3} {t_rebuild:>6.2f}ms {t_apply:>6.2f}ms {t_hz:>6.2f}ms {t_hp:>6.2f}ms')

            # Step 7: Update position (alpha=1, optimal step from 2D subspace)
            self.update_x(1.0)

            # Step 8: Powell's restart criterion
            if iter > 0:
                self.compute_powell_scalars()
                # Compute r_k for logging
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
    """Test the collision-free MAS-PNCG solver."""
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    print("\n" + "="*60)
    print("Testing MAS-PNCG Solver (No Collision)")
    print("="*60)

    solver = MASPNCGSolverNoCollision(demo='eight_E_freefall')

    for f in range(5):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms")


if __name__ == '__main__':
    test_solver()
