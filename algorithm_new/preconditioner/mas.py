"""
MAS (Multilevel Additive Schwarz) preconditioner wrapper.

This module provides a wrapper around the existing MAS preconditioner
implementations with f32/f64 precision selection and contact/Woodbury support.

Features:
- f32/f64 precision selection
- Contact Hessian integration (MASPreconditionerContact)
- Woodbury incremental updates for efficient contact handling
"""

import taichi as ti
from typing import Any, Optional, Literal
from .registry import PreconditionerRegistry
from ..core.precision import PrecisionType, get_precision_config, PrecisionMixin


# Default constants
DEFAULT_BANKSIZE = 16


@PreconditionerRegistry.register('mas')
@ti.data_oriented
class MASPreconditioner(PrecisionMixin):
    """
    MAS (Multilevel Additive Schwarz) preconditioner.

    Wraps the existing MAS preconditioner implementations with
    automatic precision selection and optional contact support.

    Features:
    - f32/f64 precision selection
    - Contact Hessian integration (via MASPreconditionerContact)
    - Woodbury incremental updates for efficient contact handling
    - Per-subdomain CCD integration support

    Usage without contacts:
        precond = MASPreconditioner(mesh)
        precond.rebuild(solver)
        precond.apply()

    Usage with contacts:
        precond = MASPreconditioner(mesh, with_contacts=True)
        precond.rebuild_with_contacts(solver)
        precond.apply()

    Usage with Woodbury updates:
        precond = MASPreconditioner(mesh, with_contacts=True)
        precond.rebuild_with_contacts(solver)
        precond.save_base_state(solver)
        # ... iteration ...
        if precond.should_use_woodbury(solver):
            precond.woodbury_update(solver)
            precond.apply_with_woodbury()
        else:
            precond.rebuild_with_contacts(solver)
            precond.apply()
    """

    def __init__(
        self,
        mesh: Any,
        precision: PrecisionType = 'f32',
        metis_reordered: bool = True,
        metis_n_parts: Optional[int] = None,
        with_contacts: bool = False,
        max_contacts: int = 2**18,
        banksize: int = DEFAULT_BANKSIZE,
        **kwargs
    ):
        """
        Initialize MAS preconditioner.

        Args:
            mesh: MeshTaichi mesh object
            precision: Float precision ('f32' or 'f64')
            metis_reordered: Whether mesh is METIS reordered
            metis_n_parts: Number of METIS partitions (optional)
            with_contacts: Enable contact Hessian support
            max_contacts: Maximum number of contact pairs (if with_contacts=True)
            banksize: Nodes per subdomain (typically 16)
            **kwargs: Additional arguments for MAS preconditioner
        """
        self.init_precision(precision)
        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.with_contacts = with_contacts
        self.banksize = banksize
        self.n_subdomains = (self.n_verts + banksize - 1) // banksize

        # Select implementation based on contact support and precision
        if with_contacts:
            # Use contact-aware MAS preconditioner
            from algorithm.mas_preconditioner_contact import MASPreconditionerContact
            self._impl = MASPreconditionerContact(
                mesh,
                max_contacts=max_contacts,
                metis_reordered=metis_reordered,
                metis_n_parts=metis_n_parts,
                **kwargs
            )
            print(f'[MASPreconditioner] Using contact-aware implementation')
        else:
            # Use base MAS preconditioner (no contacts)
            if precision == 'f64':
                from algorithm.mas_preconditioner_small.core_f64 import MASPreconditionerSmallF64
                precond_class = MASPreconditionerSmallF64
                print(f'[MASPreconditioner] Using float64 implementation')
            else:
                from algorithm.mas_preconditioner_small.core import MASPreconditionerSmall
                precond_class = MASPreconditionerSmall
                print(f'[MASPreconditioner] Using float32 implementation')

            self._impl = precond_class(
                mesh,
                metis_reordered=metis_reordered,
                metis_n_parts=metis_n_parts,
                **kwargs
            )

        # Woodbury support (lazy initialization)
        self._woodbury_initialized = False

        print(f'[MASPreconditioner] Initialized with {self._impl.level_num} levels, '
              f'n_verts={self.n_verts}, contacts={with_contacts}')

    @property
    def level_num(self) -> int:
        """Number of hierarchy levels."""
        return self._impl.level_num

    # ========================================================================
    # Basic Preconditioner Interface
    # ========================================================================

    def rebuild(self, solver: Any) -> None:
        """
        Rebuild preconditioner with current solver state (no contacts).

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
        if self.with_contacts and hasattr(self._impl, 'hessian_matvec_with_contacts'):
            self._impl.hessian_matvec_with_contacts(v, result)
        else:
            self._impl.hessian_matvec(v, result)

    # ========================================================================
    # Contact Support
    # ========================================================================

    def rebuild_with_contacts(self, solver: Any) -> None:
        """
        Rebuild preconditioner including contact Hessians.

        This is the primary rebuild method when contacts are present.
        Call this at each Newton iteration when contacts may have changed.

        Args:
            solver: IPC solver with contact_pairs, n_contacts, dHat, kappa
        """
        if not self.with_contacts:
            raise RuntimeError("MASPreconditioner not initialized with with_contacts=True")

        self._impl.rebuild_with_contacts(solver)

    def rebuild_with_contacts_spd(
        self,
        solver: Any,
        mu: float = 0.0,
        friction_eps: float = 1e-4,
    ) -> None:
        """
        Rebuild preconditioner using SPD contact Hessian formulation.

        Uses the guaranteed-PSD formulation from PPF-Contact-Solver:
        - Barrier Hessian: H = curvature * (e⊗e^T) / ||e||²
        - Friction Hessian: H = λ * P (projection matrix)

        Args:
            solver: IPC solver with contact data
            mu: Friction coefficient (0 to disable)
            friction_eps: Minimum displacement for friction regularization
        """
        if not self.with_contacts:
            raise RuntimeError("MASPreconditioner not initialized with with_contacts=True")

        if hasattr(self._impl, 'rebuild_with_contacts_spd'):
            self._impl.rebuild_with_contacts_spd(solver, mu, friction_eps)
        else:
            # Fallback to standard rebuild
            self._impl.rebuild_with_contacts(solver)

    # ========================================================================
    # Woodbury Update Support
    # ========================================================================

    def _ensure_woodbury_initialized(self):
        """Ensure Woodbury structures are initialized."""
        if not self._woodbury_initialized:
            if not self.with_contacts:
                raise RuntimeError("Woodbury requires with_contacts=True")
            if hasattr(self._impl, 'init_woodbury'):
                self._impl.init_woodbury()
            self._woodbury_initialized = True

    def save_base_state(self, solver: Any) -> None:
        """
        Save current contact state as base for Woodbury updates.

        Call this after a full rebuild to establish the baseline.
        Subsequent iterations can use woodbury_update() for efficiency.

        Args:
            solver: IPC solver containing contact information
        """
        self._ensure_woodbury_initialized()
        if hasattr(self._impl, 'save_base_state'):
            self._impl.save_base_state(solver)

    def woodbury_update(self, solver: Any) -> None:
        """
        Compute Woodbury updates from contact changes.

        Call this instead of full rebuild when contacts change incrementally.
        Use should_use_woodbury() to determine if Woodbury is appropriate.

        Args:
            solver: IPC solver containing current contact information
        """
        self._ensure_woodbury_initialized()
        if hasattr(self._impl, 'woodbury_update'):
            self._impl.woodbury_update(solver)

    def apply_with_woodbury(self) -> None:
        """
        Apply preconditioner with Woodbury corrections.

        Use this instead of apply() when Woodbury updates have been computed.
        """
        self._ensure_woodbury_initialized()
        if hasattr(self._impl, 'apply_with_woodbury'):
            self._impl.apply_with_woodbury()
        else:
            # Fallback to standard apply
            self._impl.apply()

    def should_use_woodbury(self, solver: Any) -> bool:
        """
        Determine if Woodbury update is appropriate.

        Returns True if:
        1. Woodbury is initialized
        2. Base contacts exist (not first iteration)
        3. Contact change is incremental (< 50% change in count)

        Args:
            solver: IPC solver containing contact information

        Returns:
            bool: True if Woodbury update should be used
        """
        if not self._woodbury_initialized:
            return False

        if hasattr(self._impl, 'should_use_woodbury'):
            return self._impl.should_use_woodbury(solver)

        return False

    # ========================================================================
    # Statistics and Debugging
    # ========================================================================

    def get_contact_stats(self) -> dict:
        """Get statistics about contact storage."""
        if hasattr(self._impl, 'get_contact_stats'):
            return self._impl.get_contact_stats()
        return {'n_contacts': 0, 'has_contact_support': self.with_contacts}

    def get_woodbury_stats(self) -> dict:
        """Get statistics about Woodbury updates."""
        if hasattr(self._impl, 'get_woodbury_stats'):
            return self._impl.get_woodbury_stats()
        return {'initialized': self._woodbury_initialized}

    def get_cross_block_stats(self) -> dict:
        """Get statistics about cross-block coupling storage."""
        if hasattr(self._impl, 'get_cross_block_stats'):
            return self._impl.get_cross_block_stats()
        return {}

    # Expose internal implementation for advanced usage
    @property
    def impl(self):
        """Access the underlying MAS implementation."""
        return self._impl


# ============================================================================
# BANKSIZE=8 MAS Preconditioner with Contact
# ============================================================================

@PreconditionerRegistry.register('mas8_contact')
@ti.data_oriented
class MASPreconditioner8(PrecisionMixin):
    """
    MAS preconditioner with BANKSIZE=8 and contact support.

    Optimizations over BANKSIZE=16:
    - One-way Gauss-Jordan elimination for matrix inverse (~3x faster)
    - Compact block storage (36 sym blocks vs 136)
    - Two-pass symmetric matvec with reduced branching

    Usage:
        precond = MASPreconditioner8(mesh, precision='f64')
        precond.rebuild(solver, dt)
        precond.apply()
    """

    def __init__(
        self,
        mesh: Any,
        precision: PrecisionType = 'f32',
        metis_reordered: bool = True,
        max_contacts: int = 2**18,
        **kwargs
    ):
        """
        Initialize MAS-8 preconditioner.

        Args:
            mesh: MeshTaichi mesh object
            precision: Float precision ('f32' or 'f64')
            metis_reordered: Whether mesh is METIS reordered
            max_contacts: Maximum number of contact pairs
            **kwargs: Additional arguments
        """
        self.init_precision(precision)
        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.banksize = 8

        # Use the new algorithm_new implementation
        from .mas_8_contact import MASPreconditioner8Contact
        self._impl = MASPreconditioner8Contact(
            mesh=mesh,
            precision=precision,
            metis_reordered=metis_reordered,
            max_contacts=max_contacts,
            **kwargs
        )

        print(f'[MASPreconditioner8] Initialized with {self._impl.level_num} levels, '
              f'n_verts={self.n_verts}')

    @property
    def level_num(self) -> int:
        """Number of hierarchy levels."""
        return self._impl.level_num

    def rebuild(self, solver: Any, dt: float = None) -> None:
        """Rebuild preconditioner with current solver state."""
        self._impl.rebuild(solver, dt)

    def rebuild_with_contacts(self, solver: Any, dt: float = None) -> None:
        """Rebuild preconditioner including contact Hessians."""
        self._impl.rebuild_with_contacts(solver, dt)

    def rebuild_with_contacts_spd(
        self,
        solver: Any,
        dt: float = None,
        mu: float = 0.0,
        friction_eps: float = 1e-4,
    ) -> None:
        """Rebuild using SPD contact Hessian formulation."""
        self._impl.rebuild_with_contacts_spd(solver, dt, mu, friction_eps)

    def apply(self) -> None:
        """Apply preconditioner: z = M^{-1} g."""
        self._impl.apply()

    def hessian_matvec(self, v: Any, result: Any) -> None:
        """Compute Hessian-vector product: result = H * v."""
        self._impl.hessian_matvec(v, result)

    @property
    def impl(self):
        """Access the underlying implementation."""
        return self._impl


# ============================================================================
# Factory Functions
# ============================================================================

def create_mas_preconditioner(
    mesh: Any,
    precision: PrecisionType = 'f32',
    with_contacts: bool = False,
    max_contacts: int = 2**18,
    **kwargs,
) -> MASPreconditioner:
    """
    Factory function to create MAS preconditioner.

    Args:
        mesh: MeshTaichi mesh object
        precision: Float precision ('f32' or 'f64')
        with_contacts: Enable contact support
        max_contacts: Maximum contacts (if with_contacts=True)
        **kwargs: Additional arguments

    Returns:
        MASPreconditioner instance
    """
    return MASPreconditioner(
        mesh=mesh,
        precision=precision,
        with_contacts=with_contacts,
        max_contacts=max_contacts,
        **kwargs,
    )
