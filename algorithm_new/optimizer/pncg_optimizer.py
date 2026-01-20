"""
PNCGOptimizer: Preconditioned Nonlinear Conjugate Gradient optimizer.

This module implements the PNCG algorithm with:
- Modular gradient/Hessian contribution system
- Pluggable preconditioner
- 2D subspace minimization
- Powell's restart criterion
"""

import taichi as ti
import time
from typing import List, Optional, Any, TYPE_CHECKING
from ..core.precision import PrecisionType, get_precision_config, PrecisionMixin

if TYPE_CHECKING:
    from ..core.protocols import GradientContributor, HessianContributor, Preconditioner


# Constants
RESTART_THRESHOLD = 0.5
ENERGY_TOL = 1e-4
STAGNANT_WINDOW = 5


@ti.data_oriented
class PNCGOptimizer(PrecisionMixin):
    """
    Preconditioned Nonlinear Conjugate Gradient optimizer.

    This class implements the core PNCG algorithm with:
    - Modular gradient/Hessian contribution system
    - Pluggable preconditioner
    - 2D subspace minimization
    - Powell's restart criterion
    """

    def __init__(
        self,
        mesh: Any,
        mesh_system: Any,
        dt: float = 0.04,
        epsilon: float = 1e-5,
        iter_max: int = 50,
        restart_threshold: float = RESTART_THRESHOLD,
        energy_tol: float = ENERGY_TOL,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize the PNCG optimizer.

        Args:
            mesh: MeshTaichi mesh object
            mesh_system: MeshSystem instance for elastic/inertia computation
            dt: Time step
            epsilon: Convergence tolerance
            iter_max: Maximum iterations per step
            restart_threshold: Powell's restart threshold
            energy_tol: Energy stagnation tolerance
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)

        self.mesh = mesh
        self.mesh_system = mesh_system
        self.dt = dt
        self.epsilon = epsilon
        self.iter_max = iter_max
        self.restart_threshold = restart_threshold
        self.energy_tol = energy_tol
        self.stagnant_window = STAGNANT_WINDOW

        self.n_verts = len(mesh.verts)

        # Module registrations
        self._gradient_contributors: List['GradientContributor'] = []
        self._hessian_contributors: List['HessianContributor'] = []
        self._preconditioner: Optional['Preconditioner'] = None

        # Energy history for stagnation check
        self.energy_history = []

        # Scalar fields for 2D subspace computation
        float_type = self.cfg.float_type
        self.z_H_z = ti.field(dtype=float_type, shape=())
        self.z_H_p = ti.field(dtype=float_type, shape=())
        self.p_H_p = ti.field(dtype=float_type, shape=())
        self.z_g = ti.field(dtype=float_type, shape=())
        self.p_g = ti.field(dtype=float_type, shape=())
        self.g_z_prev = ti.field(dtype=float_type, shape=())
        self.g_z = ti.field(dtype=float_type, shape=())

        # Buffer fields for hessian_matvec
        self.hv_input = ti.Vector.field(3, dtype=float_type, shape=self.n_verts)
        self.hv_output = ti.Vector.field(3, dtype=float_type, shape=self.n_verts)

        # Register mesh_system as default contributor
        self.register_gradient_contributor(mesh_system)
        self.register_hessian_contributor(mesh_system)

        print(f'[PNCGOptimizer] Initialized: eps={epsilon}, iter_max={iter_max}, '
              f'restart_threshold={restart_threshold}')

    # ========================================================================
    # Module Registration
    # ========================================================================

    def register_gradient_contributor(self, contributor: 'GradientContributor'):
        """Register a module that contributes to gradient."""
        self._gradient_contributors.append(contributor)

    def register_hessian_contributor(self, contributor: 'HessianContributor'):
        """Register a module that contributes to Hessian."""
        self._hessian_contributors.append(contributor)

    def set_preconditioner(self, preconditioner: 'Preconditioner'):
        """Set the preconditioner to use."""
        self._preconditioner = preconditioner

    # ========================================================================
    # Gradient Computation
    # ========================================================================

    def compute_gradient(self):
        """Compute gradient from all contributors."""
        # Clear gradient and diagH
        self.mesh_system.clear_fields()

        # Add contributions from all registered modules
        for contributor in self._gradient_contributors:
            contributor.add_gradient(self.mesh, self.dt, self.mesh_system.mu, self.mesh_system.la)
            contributor.add_diagonal_hessian(self.mesh, self.dt, self.mesh_system.mu, self.mesh_system.la)

    # ========================================================================
    # Preconditioner
    # ========================================================================

    def apply_preconditioner(self):
        """Apply preconditioner: z = M^{-1} g."""
        if self._preconditioner is not None:
            self._preconditioner.apply()
        else:
            self._diagonal_precondition()

    @ti.kernel
    def _diagonal_precondition(self):
        """Fallback diagonal preconditioning: z = g / diagH."""
        for vert in self.mesh.verts:
            diagH = vert.diagH
            for i in ti.static(range(3)):
                if diagH[i] > 1e-12:
                    vert.z[i] = vert.grad[i] / diagH[i]
                else:
                    vert.z[i] = vert.grad[i]

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
        if self._preconditioner is not None and hasattr(self._preconditioner, 'hessian_matvec'):
            self._copy_z_to_buffer()
            self._preconditioner.hessian_matvec(self.hv_input, self.hv_output)
            self._copy_buffer_to_Hv()
        else:
            # Fallback: use diagonal approximation
            self._compute_Hv_z_diagonal()

    @ti.kernel
    def _compute_Hv_z_diagonal(self):
        """Fallback Hv = diagH * z."""
        for vert in self.mesh.verts:
            vert.Hv = vert.diagH * vert.z

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
        if self._preconditioner is not None and hasattr(self._preconditioner, 'hessian_matvec'):
            self._copy_p_to_buffer()
            self._preconditioner.hessian_matvec(self.hv_input, self.hv_output)
            self._copy_buffer_to_w()
        else:
            self._compute_Hv_p_diagonal()

    @ti.kernel
    def _compute_Hv_p_diagonal(self):
        """Fallback w = diagH * p."""
        for vert in self.mesh.verts:
            vert.w = vert.diagH * vert.p

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

            self.z_H_z[None] += z.dot(Hv)
            self.z_H_p[None] += z.dot(w)
            self.p_H_p[None] += p.dot(w)
            self.z_g[None] += z.dot(g)
            self.p_g[None] += p.dot(g)

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
            # Fallback to steepest descent
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
        """Update search direction: p = -mu * z + nu * p."""
        for vert in self.mesh.verts:
            vert.p = -mu * vert.z + nu * vert.p

    @ti.kernel
    def compute_init_search_direction(self):
        """
        First iteration: p = -mu * z where mu = z*g / z*H*z.
        """
        self.z_g[None] = 0.0
        self.z_H_z[None] = 0.0
        for vert in self.mesh.verts:
            self.z_g[None] += vert.z.dot(vert.grad)
            self.z_H_z[None] += vert.z.dot(vert.Hv)

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

            self.g_z_prev[None] += g.dot(z_prev)
            self.g_z[None] += g.dot(z)

    def check_powell_restart(self) -> tuple:
        """
        Check Powell's restart criterion.

        Returns: (do_restart, r_k)
        """
        g_z_prev = abs(float(self.g_z_prev[None]))
        g_z = float(self.g_z[None])

        if g_z < 1e-12:
            return True, 1.0

        r_k = g_z_prev / g_z
        if r_k > self.restart_threshold:
            return True, r_k
        return False, r_k

    # ========================================================================
    # Energy and Convergence
    # ========================================================================

    def compute_energy(self) -> float:
        """Compute total energy."""
        return self.mesh_system.compute_energy(self.mesh, self.dt,
                                               self.mesh_system.mu, self.mesh_system.la)

    def check_energy_stagnation(self, energy: float) -> bool:
        """Check if energy has stagnated."""
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

    @ti.kernel
    def compute_z_norm(self) -> float:
        """Compute L2 norm of preconditioned gradient z."""
        z_sq = 0.0
        for vert in self.mesh.verts:
            z_sq += vert.z.dot(vert.z)
        return ti.sqrt(z_sq)

    # ========================================================================
    # Main Step Function
    # ========================================================================

    def step(self, verbose: bool = False) -> int:
        """
        Perform one optimization step.

        Returns: number of iterations
        """
        if verbose:
            print(f'\n{"="*100}')
            print(f'{"iter":>4} {"E":>12} {"|g|_inf":>10} {"|z|":>10} {"mu":>10} {"nu":>10} '
                  f'{"r_k":>8} {"rst":>3} {"t_rbd":>7} {"t_app":>7} {"t_Hz":>7} {"t_Hp":>7}')
            print(f'{"-"*100}')

        # Frame setup
        self.mesh_system.assign_xn_xhat()
        self.energy_history.clear()

        do_restart = True
        mu, nu = 0.0, 0.0
        r_k = 0.0

        for iter in range(self.iter_max):
            # Step 1: Compute gradient
            self.compute_gradient()

            # Step 2: Check convergence
            grad_inf = self.mesh_system.compute_grad_inf_norm()
            energy = self.compute_energy()

            if grad_inf < self.epsilon:
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>7} {"--":>7} {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                break

            if self.check_energy_stagnation(energy):
                if verbose:
                    print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {"--":>10} {"--":>10} {"--":>10} '
                          f'{"--":>8} {"--":>3} {"--":>7} {"--":>7} {"--":>7} {"--":>7}')
                    print(f'  => Converged at iter {iter}, energy stagnated')
                break

            # Step 3: Rebuild preconditioner on restart
            t_rebuild_start = time.perf_counter()
            if do_restart and self._preconditioner is not None:
                self._preconditioner.rebuild(self)
                ti.sync()
            t_rebuild = (time.perf_counter() - t_rebuild_start) * 1000

            # Cache z_prev for Powell criterion
            if iter > 0:
                self.cache_z_prev()

            # Step 4: Apply preconditioner
            t_apply_start = time.perf_counter()
            self.apply_preconditioner()
            t_apply = (time.perf_counter() - t_apply_start) * 1000

            # Step 5: Compute Hv = H * z
            t_hz_start = time.perf_counter()
            self.compute_Hv_z()
            t_hz = (time.perf_counter() - t_hz_start) * 1000

            # Compute z norm for logging
            z_norm = self.compute_z_norm() if verbose else 0.0

            # Step 6: Compute search direction
            if iter == 0 or do_restart:
                self.compute_init_search_direction()
                mu = float(self.z_g[None]) / max(float(self.z_H_z[None]), 1e-12)
                nu = 0.0
            else:
                self.compute_subspace_scalars()
                mu, nu = self.solve_2x2_subspace()
                self.update_search_direction(mu, nu)

            # Step 6.5: Compute w = H * p
            t_hp_start = time.perf_counter()
            self.compute_Hv_p()
            t_hp = (time.perf_counter() - t_hp_start) * 1000

            # Log iteration info
            if verbose:
                restart_str = "Y" if do_restart else "N"
                r_k_str = f'{r_k:>8.4f}' if iter > 0 else f'{"--":>8}'
                print(f'{iter:>4} {energy:>12.4e} {grad_inf:>10.2e} {z_norm:>10.2e} {mu:>10.4f} {nu:>10.4f} '
                      f'{r_k_str} {restart_str:>3} {t_rebuild:>6.2f}ms {t_apply:>6.2f}ms {t_hz:>6.2f}ms {t_hp:>6.2f}ms')

            # Step 7: Update position (alpha=1 for 2D subspace)
            self.mesh_system.update_x(1.0)

            # Step 8: Powell's restart criterion
            if iter > 0:
                self.compute_powell_scalars()
                do_restart, r_k = self.check_powell_restart()
            else:
                do_restart = False
                r_k = 0.0

        # Update velocity
        self.mesh_system.update_v()
        return iter + 1
