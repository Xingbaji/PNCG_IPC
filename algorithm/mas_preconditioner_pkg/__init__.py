"""
MAS Preconditioner Module - Modular Implementation

This package provides a modular implementation of the Multilevel Additive Schwarz (MAS)
preconditioner for the PNCG-IPC solver.

The implementation is split into several modules for better organization:
- core.py: Main MASPreconditioner class
- constants.py: Core constants (BANKSIZE, MAX_LEVELS, etc.)
- topology.py: Mesh topology and neighbor building
- assembly.py: Matrix assembly (elastic + contact Hessian)
- inversion.py: Block matrix inversion algorithms
- schwarz.py: Schwarz local solvers
- hierarchy.py: Multi-level restriction/prolongation
- woodbury.py: Sparse-Input Woodbury updates
- metis_integration.py: METIS-based reordering (includes GPU-accelerated pipeline)
- simple_api.py: Simplified API without meshtaichi
- spmv.py: SRBK SpMV implementation
- warp_utils.py: Bit manipulation and warp reduction utilities

Usage:
    from algorithm.mas_preconditioner_pkg import MASPreconditioner

    # Or import specific components:
    from algorithm.mas_preconditioner_pkg.spmv import SRBKSpMV
    from algorithm.mas_preconditioner_pkg.constants import BANKSIZE

    # METIS reordering functions:
    from algorithm.mas_preconditioner_pkg.metis_integration import (
        metis_reorder_mesh,
        check_pymetis_available,
        MetisReorderGPU
    )

Note: The monolithic mas_preconditioner.py has been deprecated and moved to tmp/.
Note: The standalone metis_reorder.py has been integrated into metis_integration.py.
"""

# Main class export
from .core import MASPreconditioner

# Constants
from .constants import (
    BANKSIZE,
    MAX_LEVELS,
    SYM_BLOCK_COUNT,
    MAX_NEIGHBORS_PER_VERTEX,
    BLOCK_DOF,
    WARP_REDUCTION_ENABLED,
    NODE_BANDWIDTH
)

# SpMV implementation
from .spmv import SRBKSpMV

# Warp utilities
from .warp_utils import WarpReductionHelper

# METIS integration
from .metis_integration import (
    metis_reorder_mesh,
    check_pymetis_available,
    MetisReorderGPU,
    METISMixin,
    apply_metis_reordering_to_mas,
    # CPU fallback functions
    build_adjacency_from_cells_cpu,
    compute_sort_index_cpu,
    compute_inverse_mapping_cpu,
    build_partition_mappings_cpu,
    # File I/O
    save_partition_file,
    load_partition_file,
)

# Module version
__version__ = "1.0.1"

# All public exports
__all__ = [
    # Main class
    "MASPreconditioner",

    # Constants
    "BANKSIZE",
    "MAX_LEVELS",
    "SYM_BLOCK_COUNT",
    "MAX_NEIGHBORS_PER_VERTEX",
    "BLOCK_DOF",
    "WARP_REDUCTION_ENABLED",
    "NODE_BANDWIDTH",

    # Utilities
    "SRBKSpMV",
    "WarpReductionHelper",

    # METIS integration
    "metis_reorder_mesh",
    "check_pymetis_available",
    "MetisReorderGPU",
    "METISMixin",
    "apply_metis_reordering_to_mas",
    "build_adjacency_from_cells_cpu",
    "compute_sort_index_cpu",
    "compute_inverse_mapping_cpu",
    "build_partition_mappings_cpu",
    "save_partition_file",
    "load_partition_file",
]
