"""
Constants for the MAS (Multilevel Additive Schwarz) Preconditioner.

These constants are derived from the CUDA reference implementation and
the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate
Gradient Framework for Incremental Potential Contact".
"""

# ==============================================================================
# Core MAS Parameters
# ==============================================================================

# Nodes per subdomain (warp subdivision)
# This matches the GPU warp size subdivision for optimal parallelism
BANKSIZE = 16

# Maximum hierarchy depth
# Typically 4-6 levels are sufficient for most meshes
MAX_LEVELS = 6

# Symmetric block count for BANKSIZE x BANKSIZE upper triangular storage
# Formula: n * (n + 1) / 2 = 16 * 17 / 2 = 136
SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2  # = 136

# Maximum neighbors per vertex in adjacency list
# Conservative upper bound for tetrahedral meshes
MAX_NEIGHBORS_PER_VERTEX = 64

# Total DOFs per block (BANKSIZE vertices * 3 DOFs per vertex)
BLOCK_DOF = BANKSIZE * 3  # = 48

# ==============================================================================
# Warp-Level Optimization Constants
# ==============================================================================

# Enable warp-level reduction optimization for restriction operations
# Note: ti.simt.block.SharedArray is NOT supported in Taichi 1.7.4
# We use field-based warp reduction instead
WARP_REDUCTION_ENABLED = True

# Node bandwidth for IC(0) banded matrix inversion
# Only accesses node pairs within this bandwidth
NODE_BANDWIDTH = 2

# ==============================================================================
# Woodbury Update Constants
# ==============================================================================

# Number of top-k rank-1 updates per subdomain
TOP_K_UPDATES = 8

# Minimum step size for CCD (conservative continuous collision detection)
CCD_ALPHA_MIN = 1e-6

# ==============================================================================
# MAS-PNCG Solver Constants
# ==============================================================================

# Powell's restart threshold (delta in Algorithm 1)
# When |g_k^T z_{k-1}| / |g_k^T z_k| > delta, restart with steepest descent
RESTART_THRESHOLD = 0.3

# ==============================================================================
# Numerical Constants
# ==============================================================================

# Small epsilon for numerical stability in divisions and comparisons
EPS = 1e-10

# Regularization parameter for ensuring positive definiteness
DEFAULT_REGULARIZATION = 1e-6

# ==============================================================================
# Cell-to-Warp Mapping Constants
# ==============================================================================

# Maximum cells per warp (conservative upper bound)
# For tetrahedral meshes: avg ~4 cells per vertex -> ~64 cells per warp
MAX_CELLS_PER_WARP = 128
