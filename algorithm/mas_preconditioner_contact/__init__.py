"""
MAS Preconditioner Contact - MAS preconditioner with IPC contact Hessian support.

This module extends MASPreconditionerSmall to integrate contact Hessian
into the block matrix assembly. Follows the pattern from Stiff-GIPC reference.

Key features:
- Same-block contacts: Direct assembly to level-0 blocks
- Cross-block contacts: Triplet storage for exact matvec + coarse propagation
- SPD preservation: Contact Hessian structure is naturally PSD

Usage:
======
    from algorithm.mas_preconditioner_contact import MASPreconditionerContact

    # Create preconditioner (mesh must be METIS pre-reordered)
    precond = MASPreconditionerContact(mesh, max_contacts=MAX_C)

    # In solver step:
    precond.rebuild_with_contacts(solver)  # Rebuild with contacts
    precond.apply()  # Apply preconditioner
"""

from .core import MASPreconditionerContact

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
    'MASPreconditionerContact',
    'MetisReorderResult',
    'compute_metis_reorder',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'reorder_mesh_data_metis',
    'merge_models',
    'BANKSIZE',
]
