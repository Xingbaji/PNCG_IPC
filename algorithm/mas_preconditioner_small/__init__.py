"""
MAS Preconditioner Small - Simplified version for fast compilation.

This is a minimal implementation that only includes:
- ARAP elastic Hessian (with eigenvalue filter)
- IC(0) block inversion
- Banded local solve
- METIS reordering support (two modes)

Removed features:
- Multiple elastic types (only ARAP filter)
- Multiple inversion methods (only IC)
- IPC contact handling
- Woodbury update
- Multiple solve methods (only banded)

METIS Modes Performance Comparison:
====================================
| Mode              | Assemble | Apply   | Total   | Best For            |
|-------------------|----------|---------|---------|---------------------|
| No METIS          | 0.50ms   | 0.56ms  | 1.06ms  | Baseline            |
| Runtime Mapping   | 0.95ms   | 0.33ms  | 1.28ms  | Dynamic meshes      |
| Pre-Reordered     | 0.97ms   | 0.29ms  | 1.25ms  | Static meshes (rec) |

Usage (Recommended - Pre-Reordered Mode):
=========================================
Best for static mesh simulations. Reorder mesh data BEFORE creating MeshTaichi mesh.

    import meshtaichi_patcher as Patcher
    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall, reorder_mesh_data_metis
    )

    # Load raw mesh data
    raw_data = Patcher.load_mesh_rawdata("model.node")
    positions, cells = raw_data[0], raw_data[3]

    # METIS reorder (one-time preprocessing)
    reordered_verts, reordered_cells, perm = reorder_mesh_data_metis(positions, cells)

    # Create MeshTaichi mesh with reordered data
    mesh = Patcher.load_mesh({0: reordered_verts, 3: reordered_cells}, relations=["CV"])

    # Create preconditioner (metis_reordered=True enables banded solve)
    precond = MASPreconditionerSmall(mesh, metis_reordered=True)

Usage (Runtime Mapping Mode):
=============================
For dynamic topology or when mesh cannot be pre-reordered.

    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall, compute_metis_reorder, extract_cells_from_mesh
    )

    cells = extract_cells_from_mesh(mesh)
    metis_result = compute_metis_reorder(n_verts, cells)
    precond = MASPreconditionerSmall(mesh, metis_result=metis_result)

Notes:
- Pre-Reordered mode has fastest Apply (no mapping lookup, banded O(5) solve)
- Runtime Mapping adds ~0.4ms to Assemble but enables METIS without mesh modification
- Both METIS modes use banded solve; No-METIS uses full solve O(16)
"""

from .core import MASPreconditionerSmall
from .metis_reorder import (
    MetisReorderResult,
    compute_metis_reorder,
    compute_optimized_cell_data,
    extract_cells_from_mesh,
    check_pymetis_available,
    reorder_mesh_data_metis,
    merge_models,
    BANKSIZE,
)

__all__ = [
    'MASPreconditionerSmall',
    'MetisReorderResult',
    'compute_metis_reorder',
    'compute_optimized_cell_data',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'reorder_mesh_data_metis',
    'merge_models',
    'BANKSIZE',
]
