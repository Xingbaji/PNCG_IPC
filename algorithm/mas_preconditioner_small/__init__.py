"""
MAS Preconditioner Small - Simplified version for fast compilation.

This is a minimal implementation that only includes:
- ARAP elastic Hessian (with eigenvalue filter)
- IC(0) block inversion
- Banded local solve
- METIS pre-reordering support

Removed features:
- Multiple elastic types (only ARAP filter)
- Multiple inversion methods (only IC)
- IPC contact handling
- Woodbury update
- Multiple solve methods (only banded)

IMPORTANT: This implementation REQUIRES mesh data to be pre-reordered with METIS
BEFORE creating the MeshTaichi mesh. This provides the fastest performance with
direct indexing and banded O(5) solve.

Usage:
======
    import meshtaichi_patcher as Patcher
    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall, reorder_mesh_data_metis
    )

    # Load raw mesh data
    raw_data = Patcher.load_mesh_rawdata("model.node")
    positions, cells = raw_data[0], raw_data[3]

    # METIS reorder (one-time preprocessing - REQUIRED!)
    reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(positions, cells)

    # Create MeshTaichi mesh with reordered data
    mesh = Patcher.load_mesh({0: reordered_verts, 3: reordered_cells}, relations=["CV"])

    # Create preconditioner
    precond = MASPreconditionerSmall(mesh)

Notes:
- METIS reordering ensures vertices in same block are mesh neighbors
- This enables banded solve O(5) instead of full solve O(16)
- No runtime mapping lookup needed (fastest apply time ~0.29ms)
"""

from .core import MASPreconditionerSmall
from .core_f64 import MASPreconditionerSmallF64
from .metis_reorder import (
    MetisReorderResult,
    compute_metis_reorder,
    extract_cells_from_mesh,
    check_pymetis_available,
    reorder_mesh_data_metis,
    merge_models,
    BANKSIZE,
)

__all__ = [
    'MASPreconditionerSmall',
    'MASPreconditionerSmallF64',
    'MetisReorderResult',
    'compute_metis_reorder',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'reorder_mesh_data_metis',
    'merge_models',
    'BANKSIZE',
]
