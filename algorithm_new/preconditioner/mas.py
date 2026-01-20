"""
MAS (Multilevel Additive Schwarz) preconditioner wrapper.

This module provides a wrapper around the existing MAS preconditioner
implementations with f32/f64 precision selection.
"""

import taichi as ti
from typing import Any, Optional
from .registry import PreconditionerRegistry
from ..core.precision import PrecisionType, get_precision_config, PrecisionMixin


@PreconditionerRegistry.register('mas')
@ti.data_oriented
class MASPreconditioner(PrecisionMixin):
    """
    MAS (Multilevel Additive Schwarz) preconditioner.

    Wraps the existing MAS preconditioner implementations with
    automatic precision selection.
    """

    def __init__(
        self,
        mesh: Any,
        precision: PrecisionType = 'f32',
        metis_reordered: bool = True,
        metis_n_parts: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize MAS preconditioner.

        Args:
            mesh: MeshTaichi mesh object
            precision: Float precision ('f32' or 'f64')
            metis_reordered: Whether mesh is METIS reordered
            metis_n_parts: Number of METIS partitions (optional)
            **kwargs: Additional arguments for MAS preconditioner
        """
        self.init_precision(precision)
        self.mesh = mesh
        self.n_verts = len(mesh.verts)

        # Import appropriate MAS implementation based on precision
        if precision == 'f64':
            from algorithm.mas_preconditioner_small.core_f64 import MASPreconditionerSmallF64
            precond_class = MASPreconditionerSmallF64
            print(f'[MASPreconditioner] Using float64 implementation')
        else:
            from algorithm.mas_preconditioner_small.core import MASPreconditionerSmall
            precond_class = MASPreconditionerSmall
            print(f'[MASPreconditioner] Using float32 implementation')

        # Create the underlying implementation
        self._impl = precond_class(
            mesh,
            metis_reordered=metis_reordered,
            metis_n_parts=metis_n_parts,
            **kwargs
        )

        print(f'[MASPreconditioner] Initialized with {self._impl.level_num} levels, '
              f'n_verts={self.n_verts}')

    @property
    def level_num(self) -> int:
        """Number of hierarchy levels."""
        return self._impl.level_num

    def rebuild(self, solver: Any) -> None:
        """
        Rebuild preconditioner with current solver state.

        Args:
            solver: Solver or optimizer instance
        """
        self._impl.rebuild(solver)

    def apply(self) -> None:
        """Apply preconditioner: z = M^{-1} g."""
        self._impl.apply()

    def hessian_matvec(self, v: Any, result: Any) -> None:
        """
        Compute Hessian-vector product: result = H * v.

        Args:
            v: Input vector field
            result: Output vector field
        """
        self._impl.hessian_matvec(v, result)

    # Expose internal implementation for advanced usage
    @property
    def impl(self):
        """Access the underlying MAS implementation."""
        return self._impl
