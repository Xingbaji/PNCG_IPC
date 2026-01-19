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
- metis_integration.py: METIS-based reordering
- simple_api.py: Simplified API without meshtaichi
- spmv.py: SRBK SpMV implementation
- warp_utils.py: Bit manipulation and warp reduction utilities

Usage:
    from algorithm.mas_preconditioner import MASPreconditioner

    # Or import specific components:
    from algorithm.mas_preconditioner.spmv import SRBKSpMV
    from algorithm.mas_preconditioner.constants import BANKSIZE
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

# Module version
__version__ = "1.0.0"

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
]
