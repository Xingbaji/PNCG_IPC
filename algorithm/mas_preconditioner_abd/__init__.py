"""
MAS Preconditioner ABD - MAS preconditioner with ABD system integration.

This module extends the MAS preconditioner to support hybrid FEM-ABD simulation.
ABD (Affine Body Dynamics) bodies are integrated into the multi-level block structure
with their 12x12 Hessian blocks.

Key features:
- METIS pre-reordering support for FEM vertices
- ABD body blocks appended after FEM blocks
- Shape energy and inertia Hessian for ABD
- Unified multi-level hierarchy for FEM + ABD

Usage:
======
    from algorithm.mas_preconditioner_abd import (
        MASPreconditionerABD, reorder_mesh_data_metis
    )
    from algorithm.abd_system import ABDSystem

    # Load raw mesh data
    raw_data = Patcher.load_mesh_rawdata("model.node")
    positions, cells = raw_data[0], raw_data[3]

    # METIS reorder for FEM
    reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(positions, cells)

    # Create MeshTaichi mesh with reordered data
    mesh = Patcher.load_mesh({0: reordered_verts, 3: reordered_cells}, relations=["CV"])

    # Create ABD system
    abd_system = ABDSystem(max_bodies=16)
    abd_system.add_body(...)

    # Create preconditioner with ABD integration
    precond = MASPreconditionerABD(mesh, abd_system)

Notes:
- FEM vertices use 3x3 blocks in 16-vertex partitions
- ABD bodies use 12x12 blocks (one per body)
- ABD blocks are stored after FEM blocks in the hierarchy
"""

from .core import MASPreconditionerABD, ABD_DOF, BANKSIZE as MAS_BANKSIZE

# Re-export from mas_preconditioner_small
from ..mas_preconditioner_small import (
    MetisReorderResult,
    compute_metis_reorder,
    extract_cells_from_mesh,
    check_pymetis_available,
    reorder_mesh_data_metis,
    merge_models,
    BANKSIZE,
)

__all__ = [
    'MASPreconditionerABD',
    'ABD_DOF',
    'MAS_BANKSIZE',
    'MetisReorderResult',
    'compute_metis_reorder',
    'extract_cells_from_mesh',
    'check_pymetis_available',
    'reorder_mesh_data_metis',
    'merge_models',
    'BANKSIZE',
]
