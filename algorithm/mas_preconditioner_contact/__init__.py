"""
MAS Preconditioner Contact - MAS preconditioner with IPC contact Hessian support.

This module extends MASPreconditionerSmall to integrate contact Hessian
into the block matrix assembly. Follows the pattern from Stiff-GIPC reference.

Key features:
- Same-block contacts: Direct assembly to level-0 blocks
- Cross-block contacts: Triplet storage for exact matvec + coarse propagation
- SPD preservation: Contact Hessian structure is naturally PSD

SPD Contact Hessian (PPF-Contact-Solver style):
- Barrier Hessian: H = curvature * (e⊗e^T) / ||e||² (PSD by construction)
- Friction Hessian: H = λ * P where P = I - n⊗n^T (PSD projection matrix)
- No eigenvalue decomposition needed, efficient GPU implementation

Usage:
======
    from algorithm.mas_preconditioner_contact import MASPreconditionerContact

    # Create preconditioner (mesh must be METIS pre-reordered)
    precond = MASPreconditionerContact(mesh, max_contacts=MAX_C)

    # In solver step (without friction):
    precond.rebuild_with_contacts(solver)
    precond.apply()

    # With friction (SPD formulation):
    precond.rebuild_with_contacts_spd(solver, mu=0.5, friction_eps=1e-4)
    precond.apply()
"""

from .core import MASPreconditionerContact

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
