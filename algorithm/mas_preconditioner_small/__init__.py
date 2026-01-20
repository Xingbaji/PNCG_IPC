"""
MAS Preconditioner Small - Simplified version for fast compilation.

This is a minimal implementation that only includes:
- ARAP elastic Hessian (with eigenvalue filter)
- IC(0) block inversion
- Banded local solve
- METIS reordering (computed once at initialization)

Removed features:
- Multiple elastic types (only ARAP filter)
- Multiple inversion methods (only IC)
- IPC contact handling
- Woodbury update
- Multiple solve methods (only banded)

Usage:
    # One-time METIS computation at simulation start
    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall, compute_metis_reorder, extract_cells_from_mesh
    )

    cells = extract_cells_from_mesh(mesh)
    metis_result = compute_metis_reorder(n_verts, cells)

    # Create preconditioner with METIS reordering
    precond = MASPreconditionerSmall(mesh, metis_result=metis_result)
"""

from .core import MASPreconditionerSmall
from .metis_reorder import (
    MetisReorderResult,
    compute_metis_reorder,
    compute_optimized_cell_data,
    extract_cells_from_mesh,
    check_pymetis_available,
    BANKSIZE,
)

__all__ = [
    'MASPreconditionerSmall',
    'MetisReorderResult',
    'compute_metis_reorder',
    'compute_optimized_cell_data',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'BANKSIZE',
]
