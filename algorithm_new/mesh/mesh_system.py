"""
MeshSystem: Mesh management and elastic energy computation.

This module handles:
- Mesh loading and initialization
- Vertex/cell field management
- Elastic energy, gradient, and Hessian computation
- Implements GradientContributor and HessianContributor protocols
"""

import taichi as ti
import numpy as np
from typing import Optional, Callable, Any
from ..core.precision import PrecisionType, PrecisionConfig, get_precision_config, PrecisionMixin


@ti.data_oriented
class MeshSystem(PrecisionMixin):
    """
    Mesh system for handling geometry and elastic energy.

    This class manages:
    - Mesh topology and vertex positions
    - Rest pose storage (B matrices)
    - Mass computation
    - Elastic energy/gradient/Hessian

    Implements GradientContributor and HessianContributor protocols.
    """

    def __init__(
        self,
        mesh: Any,
        density: float,
        mu: float,
        la: float,
        elastic_type: str = 'ARAP_filter',
        precision: PrecisionType = 'f32',
        dt: float = 0.04,
        gravity: float = -9.8,
    ):
        """
        Initialize the mesh system.

        Args:
            mesh: MeshTaichi mesh object (already loaded)
            density: Material density (kg/m^3)
            mu: Lame's first parameter
            la: Lame's second parameter
            elastic_type: Elastic constitutive model
            precision: Float precision ('f32' or 'f64')
            dt: Time step
            gravity: Gravity acceleration (negative for downward)
        """
        self.init_precision(precision)

        self.mesh = mesh
        self.density = density
        self.mu = mu
        self.la = la
        self.dt = dt
        self.gravity = gravity
        self.elastic_type = elastic_type

        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        # Place vertex fields
        self._place_vertex_fields()

        # Place cell fields
        self._place_cell_fields()

        # Initialize positions
        self.mesh.verts.x.from_numpy(mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)

        # Assign elastic energy functions
        self._assign_elastic_functions(elastic_type)

        # Precompute B matrices and masses
        self._precompute()

        print(f'[MeshSystem] Initialized: {self.n_verts} verts, {self.n_cells} cells, '
              f'precision={precision}, elastic={elastic_type}')

    def _place_vertex_fields(self):
        """Place standard vertex fields on mesh."""
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),          # Current position
            'v': ti.types.vector(3, float),          # Velocity
            'm': float,                               # Mass
            'x_n': ti.types.vector(3, float),        # Position at frame start
            'x_hat': ti.types.vector(3, float),      # Inertial target
            'x_prev': ti.types.vector(3, float),     # Previous iteration position
            'x_init': ti.types.vector(3, float),     # Rest configuration
            'grad': ti.types.vector(3, float),       # Gradient
            'grad_prev': ti.types.vector(3, float),  # Previous gradient
            'p': ti.types.vector(3, float),          # Search direction
            'diagH': ti.types.vector(3, float),      # Diagonal Hessian
            'z': ti.types.vector(3, float),          # Preconditioned gradient
            'z_prev': ti.types.vector(3, float),     # Previous z
            'Hv': ti.types.vector(3, float),         # H*z
            'w': ti.types.vector(3, float),          # H*p
        })

    def _place_cell_fields(self):
        """Place cell fields on mesh."""
        self.mesh.cells.place({
            'B': ti.math.mat3,  # Reference frame inverse
            'W': float,          # Reference volume
        })

    def _assign_elastic_functions(self, elastic_type: str):
        """Assign elastic energy functions based on type."""
        from math_utils.elastic_util import (
            compute_Psi_ARAP, compute_dPsidx_ARAP,
            compute_diag_d2Psidx2_ARAP, compute_pHp_ARAP,
            compute_diag_d2Psidx2_ARAP_filter, compute_pHp_ARAP_filter,
            compute_Psi_SNH, compute_dPsidx_SNH,
            compute_diag_d2Psidx2_SNH, compute_pHp_SNH,
            compute_Psi_FCR, compute_dPsidx_FCR,
            compute_diag_d2Psidx2_FCR, compute_pHp_FCR,
        )

        elastic_funcs = {
            'ARAP': (
                compute_Psi_ARAP,
                compute_dPsidx_ARAP,
                compute_diag_d2Psidx2_ARAP,
                compute_pHp_ARAP,
            ),
            'ARAP_filter': (
                compute_Psi_ARAP,
                compute_dPsidx_ARAP,
                compute_diag_d2Psidx2_ARAP_filter,
                compute_pHp_ARAP_filter,
            ),
            'SNH': (
                compute_Psi_SNH,
                compute_dPsidx_SNH,
                compute_diag_d2Psidx2_SNH,
                compute_pHp_SNH,
            ),
            'FCR': (
                compute_Psi_FCR,
                compute_dPsidx_FCR,
                compute_diag_d2Psidx2_FCR,
                compute_pHp_FCR,
            ),
        }

        if elastic_type not in elastic_funcs:
            available = list(elastic_funcs.keys())
            raise ValueError(f"Unknown elastic type: {elastic_type}. Available: {available}")

        (self.compute_Psi, self.compute_dPsidx,
         self.compute_diag_d2Psidx2, self.compute_pHp) = elastic_funcs[elastic_type]

    @ti.kernel
    def _precompute(self):
        """Precompute B matrices and vertex masses."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x
                                  for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4.0

    # ========================================================================
    # GradientContributor Protocol Implementation
    # ========================================================================

    @ti.kernel
    def _clear_grad(self):
        """Clear gradient fields."""
        for vert in self.mesh.verts:
            vert.grad_prev = vert.grad
            vert.grad = ti.Vector.zero(float, 3)

    @ti.kernel
    def _clear_diagH(self):
        """Clear diagonal Hessian fields."""
        for vert in self.mesh.verts:
            vert.diagH = ti.Vector.zero(float, 3)

    @ti.kernel
    def _add_gradient_inertia(self):
        """Add inertia gradient: m * (x - x_hat)."""
        for vert in self.mesh.verts:
            vert.grad += vert.m * (vert.x - vert.x_hat)

    @ti.kernel
    def _add_gradient_elastic(self, para_scale: float):
        """Add elastic gradient contribution."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x
                                  for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            para = c.W * para_scale

            dPsidx = para * self.compute_dPsidx(F, c.B, self.mu, self.la)

            for i in range(4):
                grad_i = ti.Vector([dPsidx[3*i], dPsidx[3*i+1], dPsidx[3*i+2]], float)
                c.verts[i].grad += grad_i

    def add_gradient(self, mesh: Any, dt: float, mu: float, la: float) -> None:
        """Add gradient contribution (GradientContributor protocol)."""
        para_scale = dt * dt
        self._add_gradient_inertia()
        self._add_gradient_elastic(para_scale)

    @ti.kernel
    def _add_diagonal_hessian_inertia(self):
        """Add inertia diagonal Hessian: m * I."""
        for vert in self.mesh.verts:
            vert.diagH += vert.m * ti.Vector.one(float, 3)

    @ti.kernel
    def _add_diagonal_hessian_elastic(self, para_scale: float):
        """Add elastic diagonal Hessian contribution."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x
                                  for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            para = c.W * para_scale

            diagH = para * self.compute_diag_d2Psidx2(F, c.B, self.mu, self.la)

            for i in range(4):
                diag_i = ti.Vector([diagH[3*i], diagH[3*i+1], diagH[3*i+2]])
                c.verts[i].diagH += ti.max(diag_i, 0.0)

    def add_diagonal_hessian(self, mesh: Any, dt: float, mu: float, la: float) -> None:
        """Add diagonal Hessian contribution (GradientContributor protocol)."""
        para_scale = dt * dt
        self._add_diagonal_hessian_inertia()
        self._add_diagonal_hessian_elastic(para_scale)

    # ========================================================================
    # HessianContributor Protocol Implementation
    # ========================================================================

    @ti.kernel
    def _compute_pHp_inertia(self) -> float:
        """Compute inertia p^T H p = p^T M p."""
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.p.norm_sqr() * vert.m
        return result

    @ti.kernel
    def _compute_pHp_elastic(self, para_scale: float) -> float:
        """Compute elastic p^T H p."""
        result = 0.0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x
                                  for i in ti.static(range(1, 4))])
            F = Ds @ c.B

            # Gather p values
            p = ti.Vector.zero(float, 12)
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    p[3*i + j] = c.verts[i].p[j]

            pHp = self.compute_pHp(F, c.B, p, self.mu, self.la)
            result += c.W * para_scale * ti.max(pHp, 0.0)

        return result

    def add_pHp(self, mesh: Any, dt: float, mu: float, la: float) -> float:
        """Compute p^T H p (HessianContributor protocol)."""
        para_scale = dt * dt
        pHp_inertia = self._compute_pHp_inertia()
        pHp_elastic = self._compute_pHp_elastic(para_scale)
        return pHp_inertia + pHp_elastic

    # ========================================================================
    # EnergyComputer Protocol Implementation
    # ========================================================================

    @ti.kernel
    def _compute_energy_inertia(self) -> float:
        """Compute inertia energy: 0.5 * (x - x_hat)^T M (x - x_hat)."""
        E = 0.0
        for vert in self.mesh.verts:
            diff = vert.x - vert.x_hat
            E += 0.5 * vert.m * diff.dot(diff)
        return E

    @ti.kernel
    def _compute_energy_elastic(self, para_scale: float) -> float:
        """Compute elastic energy."""
        E = 0.0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x
                                  for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            E += c.W * para_scale * self.compute_Psi(F, self.mu, self.la)
        return E

    def compute_energy(self, mesh: Any = None, dt: float = None,
                       mu: float = None, la: float = None) -> float:
        """Compute total energy (EnergyComputer protocol)."""
        dt = dt if dt is not None else self.dt
        para_scale = dt * dt
        E_inertia = self._compute_energy_inertia()
        E_elastic = self._compute_energy_elastic(para_scale)
        return E_inertia + E_elastic

    # ========================================================================
    # State Management
    # ========================================================================

    @ti.kernel
    def assign_xn_xhat(self):
        """Assign x_n and x_hat at frame start."""
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.dt * self.dt * self.gravity

    @ti.kernel
    def update_x(self, alpha: float):
        """Update positions: x += alpha * p."""
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p

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

    def clear_fields(self):
        """Clear gradient and diagH fields."""
        self._clear_grad()
        self._clear_diagH()
