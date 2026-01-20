"""
MAS Preconditioner Contact - MAS preconditioner with IPC contact Hessian support.

This module extends MASPreconditionerSmall to integrate contact Hessian
into the block matrix assembly. Follows the pattern from Stiff-GIPC reference.

Key features:
- Same-block contacts: Direct assembly to level-0 blocks
- Cross-block contacts: Triplet storage for exact matvec + coarse propagation
- SPD preservation: Contact Hessian structure is naturally PSD
- Woodbury updates: Efficient incremental preconditioner updates

SPD Contact Hessian (PPF-Contact-Solver style):
- Barrier Hessian: H = curvature * (e⊗e^T) / ||e||² (PSD by construction)
- Friction Hessian: H = λ * P where P = I - n⊗n^T (PSD projection matrix)
- No eigenvalue decomposition needed, efficient GPU implementation

Woodbury Update (MAS-PNCG paper Section 3.1):
- Efficient incremental updates when contacts change slightly
- Formula: (A + U*S*U^T)^{-1} = A^{-1} - A^{-1}*U*(S^{-1} + U^T*A^{-1}*U)^{-1}*U^T*A^{-1}
- Only updates level 0 blocks; coarse levels use frozen cached inverses

Usage:
======
    from algorithm.mas_preconditioner_contact import MASPreconditionerContact

    # Create preconditioner (mesh must be METIS pre-reordered)
    precond = MASPreconditionerContact(mesh, max_contacts=MAX_C)

    # Standard usage (full rebuild):
    precond.rebuild_with_contacts(solver)
    precond.apply()

    # With Woodbury updates (incremental):
    precond.rebuild_with_contacts(solver)
    precond.save_base_state(solver)     # Save baseline
    # ... later, when contacts change slightly ...
    if precond.should_use_woodbury(solver):
        precond.woodbury_update(solver)
        precond.apply_with_woodbury()
    else:
        precond.rebuild_with_contacts(solver)
        precond.save_base_state(solver)
        precond.apply()
"""

from .core import MASPreconditionerContact
from .woodbury import WoodburySupport

# Export SPD contact assembly functions
from .contact_assembly import (
    # SPD barrier functions
    barrier_curvature_cubic,
    barrier_curvature_log,
    compute_spd_contact_hessian_3x3,
    compute_spd_edge_hessian,
    compute_spd_edge_gradient,
    # Friction functions
    compute_friction_projection_matrix,
    compute_friction_lambda,
    compute_friction_gradient,
    compute_friction_hessian,
    # Combined contact + friction
    compute_spd_contact_friction_hessian,
    compute_spd_contact_friction_gradient,
    # Extension utilities
    extend_spd_contact_hessian_12x12,
    extend_spd_contact_gradient_12,
)

# Re-export METIS utilities from mas_preconditioner_small
from algorithm.mas_preconditioner_small import (
    MetisReorderResult,
    compute_metis_reorder,
    extract_cells_from_mesh,
    check_pymetis_available,
    reorder_mesh_data_metis,
    merge_models,
    BANKSIZE,
)

__all__ = [
    # Main class
    'MASPreconditionerContact',
    # Woodbury support
    'WoodburySupport',
    # SPD barrier functions
    'barrier_curvature_cubic',
    'barrier_curvature_log',
    'compute_spd_contact_hessian_3x3',
    'compute_spd_edge_hessian',
    'compute_spd_edge_gradient',
    # Friction functions
    'compute_friction_projection_matrix',
    'compute_friction_lambda',
    'compute_friction_gradient',
    'compute_friction_hessian',
    # Combined contact + friction
    'compute_spd_contact_friction_hessian',
    'compute_spd_contact_friction_gradient',
    # Extension utilities
    'extend_spd_contact_hessian_12x12',
    'extend_spd_contact_gradient_12',
    # METIS utilities
    'MetisReorderResult',
    'compute_metis_reorder',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'reorder_mesh_data_metis',
    'merge_models',
    'BANKSIZE',
]
