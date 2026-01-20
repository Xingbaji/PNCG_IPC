"""
Diagonal preconditioner implementation.

Simple and fast preconditioner using the diagonal of the Hessian.
"""

import taichi as ti
from typing import Any
from .registry import PreconditionerRegistry
from ..core.precision import PrecisionType, get_precision_config, PrecisionMixin


@PreconditionerRegistry.register('diagonal')
@ti.data_oriented
class DiagonalPreconditioner(PrecisionMixin):
    """
    Simple diagonal preconditioner.

    Uses the diagonal of the Hessian: z = g / diagH.
    Fast but may have poor convergence for ill-conditioned problems.
    """

    def __init__(
        self,
        mesh: Any,
        precision: PrecisionType = 'f32',
        **kwargs
    ):
        """
        Initialize diagonal preconditioner.

        Args:
            mesh: MeshTaichi mesh object
            precision: Float precision ('f32' or 'f64')
            **kwargs: Additional arguments (ignored)
        """
        self.init_precision(precision)
        self.mesh = mesh
        self.n_verts = len(mesh.verts)

        print(f'[DiagonalPreconditioner] Initialized, n_verts={self.n_verts}')

    def rebuild(self, solver: Any) -> None:
        """
        Rebuild preconditioner.

        For diagonal preconditioner, this is a no-op since diagH
        is computed during gradient computation.

        Args:
            solver: Solver instance (unused)
        """
        pass  # No rebuild needed

    @ti.kernel
    def apply(self):
        """Apply preconditioner: z = g / diagH."""
        for vert in self.mesh.verts:
            diagH = vert.diagH
            for i in ti.static(range(3)):
                if diagH[i] > 1e-12:
                    vert.z[i] = vert.grad[i] / diagH[i]
                else:
                    vert.z[i] = vert.grad[i]

    def hessian_matvec(self, v: Any, result: Any) -> None:
        """
        Compute Hessian-vector product: result = H * v.

        For diagonal preconditioner, uses diagH as approximation.

        Args:
            v: Input vector field
            result: Output vector field
        """
        self._hessian_matvec_kernel(v, result)

    @ti.kernel
    def _hessian_matvec_kernel(self, v: ti.template(), result: ti.template()):
        """Kernel for Hessian-vector product."""
        for vert in self.mesh.verts:
            result[vert.id] = vert.diagH * v[vert.id]
