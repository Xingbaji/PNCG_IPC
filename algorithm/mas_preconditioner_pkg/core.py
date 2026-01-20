"""
Core Module: Main MASPreconditioner class definition.

This module combines all the mixin classes to create the complete
Multilevel Additive Schwarz (MAS) Preconditioner for PNCG-IPC solver.

The MAS preconditioner formula:
    P = M_{(0)}^{-1} + Σ_{l=1}^{L} C_{(l)}^T M_{(l)}^{-1} C_{(l)}

Where:
    - M_{(0)}^{-1} = Level-0 local inverses (Schwarz blocks)
    - C_{(l)} = Coarsening/restriction operators (binary aggregation)
    - M_{(l)}^{-1} = Coarse-level preconditioners
"""

import taichi as ti
import numpy as np

from .constants import (
    BANKSIZE, MAX_LEVELS, SYM_BLOCK_COUNT, MAX_NEIGHBORS_PER_VERTEX,
    BLOCK_DOF, WARP_REDUCTION_ENABLED
)
from .topology import TopologyMixin
from .assembly import AssemblyMixin
from .inversion import InversionMixin
from .schwarz import SchwarzMixin
from .hierarchy import HierarchyMixin
from .woodbury import WoodburyMixin
from .metis_integration import METISMixin
from .simple_api import SimpleAPIMixin
from .hessian_matvec import HessianMatvecMixin


@ti.data_oriented
class MASPreconditioner(
    TopologyMixin,
    AssemblyMixin,
    InversionMixin,
    SchwarzMixin,
    HierarchyMixin,
    WoodburyMixin,
    METISMixin,
    SimpleAPIMixin,
    HessianMatvecMixin
):
    """
    Multilevel Additive Schwarz preconditioner for PNCG optimization.

    The preconditioner uses connectivity-aware hierarchical coarsening
    to build a multilevel structure that captures both local and global
    coupling in the Hessian matrix.

    This class combines functionality from multiple mixin classes:
    - TopologyMixin: Mesh topology and neighbor building
    - AssemblyMixin: Matrix assembly (elastic + contact Hessian)
    - InversionMixin: Block matrix inversion algorithms
    - SchwarzMixin: Schwarz local solvers
    - HierarchyMixin: Multi-level restriction/prolongation
    - WoodburyMixin: Sparse-Input Woodbury updates
    - METISMixin: METIS-based reordering
    - SimpleAPIMixin: Simplified API without meshtaichi
    - HessianMatvecMixin: Hessian matrix-vector multiplication
    """

    def __init__(self, n_verts: int, n_cells: int, mesh, use_metis: bool = True,
                 cells_np: np.ndarray = None):
        """
        Initialize MAS preconditioner.

        Args:
            n_verts: Number of vertices in the mesh
            n_cells: Number of cells in the mesh
            mesh: MeshTaichi mesh object for topology access
            use_metis: Enable METIS-based node reordering (default: True)
            cells_np: Optional cell array of shape (n_cells, 4). If provided,
                      used directly for METIS instead of extracting from mesh.
        """
        self.n_verts = n_verts
        self.n_cells = n_cells
        self.mesh = mesh
        self.level_num = self._compute_num_levels(n_verts)
        self.use_metis_reorder = False  # Will be set to True if METIS succeeds

        print(f"[MAS] Initializing with {n_verts} vertices, {self.level_num} levels")

        # Allocate all data structures
        self._allocate_neighbor_structures()
        self._allocate_hierarchy_structures()
        self._allocate_matrix_structures()
        self._allocate_preconditioning_buffers()

        # Build static neighbor list from mesh topology (only if mesh is provided)
        if mesh is not None:
            self._build_neighbor_list_from_mesh()

            # Initialize METIS reordering by default
            if use_metis:
                if cells_np is not None:
                    # Use provided cells directly
                    vertices_np = self.mesh.get_position_as_numpy()
                    self.init_metis_reordering(cells_np, vertices_np)
                else:
                    # Extract cells from mesh
                    self._init_metis_from_mesh()

        # State tracking
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

        # Cache backend detection
        self._cuda_backend = None

    # ========================================================================
    # Initialization Helpers
    # ========================================================================

    def _is_cuda_backend(self) -> bool:
        """Check if Taichi is running on CUDA backend."""
        if self._cuda_backend is None:
            try:
                arch = ti.lang.impl.current_cfg().arch
                self._cuda_backend = (arch == ti.cuda)
            except Exception:
                self._cuda_backend = False
        return self._cuda_backend

    def _compute_num_levels(self, n_verts: int) -> int:
        """Compute number of hierarchy levels based on vertex count."""
        # Each level reduces by factor of ~BANKSIZE
        # We want at least BANKSIZE vertices at coarsest level
        levels = 1
        size = n_verts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return min(levels, MAX_LEVELS)

    def _compute_total_hierarchy_size(self) -> int:
        """Estimate total nodes across all hierarchy levels."""
        # Conservative estimate: geometric series
        total = self.n_verts
        size = self.n_verts
        for _ in range(self.level_num - 1):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return int(total * 1.5)  # Add buffer

    # ========================================================================
    # Memory Allocation
    # ========================================================================

    def _allocate_neighbor_structures(self):
        """Allocate structures for mesh connectivity."""
        max_neighbor_entries = self.n_verts * MAX_NEIGHBORS_PER_VERTEX

        # Neighbor list (CSR-like format)
        self.neighbor_list = ti.field(dtype=ti.i32, shape=max_neighbor_entries)
        self.neighbor_start = ti.field(dtype=ti.i32, shape=self.n_verts + 1)
        self.neighbor_num = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Connectivity bitmasks (32-bit per vertex)
        self.fine_connect_mask = ti.field(dtype=ti.u32, shape=self.n_verts)

        # Track actual neighbor count
        self.total_neighbors = 0

    def _allocate_hierarchy_structures(self):
        """Allocate multi-level hierarchy data structures."""
        self.total_nodes_all_levels = self._compute_total_hierarchy_size()
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level sizes: [level] -> (num_nodes, offset_in_global_array)
        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS + 1)

        # Coarse space tables: maps vertex to coarse ID at each level
        self.coarse_space_tables = ti.field(dtype=ti.i32,
                                            shape=(MAX_LEVELS, self.n_verts))

        # Going next: for each node, its parent in next level
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)

        # Aggregation table: stores path through hierarchy for prolongation
        self.aggregation_table = ti.Vector.field(MAX_LEVELS - 1, dtype=ti.i32,
                                                  shape=self.n_verts)

        # Connectivity mask for coarse levels
        self.next_connect_mask = ti.field(dtype=ti.u32, shape=self.n_verts)

        # Prefix sum arrays for hierarchy construction
        self.prefix_original = ti.field(dtype=ti.i32, shape=n_warps + 1)
        self.prefix_sum_original = ti.field(dtype=ti.i32, shape=n_warps + 1)

        # Elected mask for cluster detection
        self.elected_mask = ti.field(dtype=ti.u32, shape=n_warps)

        # Cluster ID assignment
        self.cluster_id = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Coarse space table alias for METIS (same as coarse_space_tables[0])
        self.coarse_space_table_0 = ti.field(dtype=ti.i32, shape=self.n_verts)

    def _allocate_matrix_structures(self):
        """Allocate block matrix storage for all levels."""
        n_blocks_l0 = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Total blocks = blocks needed for all nodes in the global array
        # Since block_id = global_vertex_id // BANKSIZE, we need:
        # total_blocks = ceil(total_nodes_all_levels / BANKSIZE)
        self.total_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE
        self.total_blocks = int(self.total_blocks * 1.5)  # Buffer

        # Block matrices: symmetric storage (upper triangle)
        # Each block is BANKSIZE x BANKSIZE vertices = 136 entries of 3x3 matrices
        self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                               shape=(self.total_blocks, SYM_BLOCK_COUNT))

        # Inverted block matrices (single precision)
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                   shape=(self.total_blocks, SYM_BLOCK_COUNT))

        # Full 48x48 block matrices for Gauss-Jordan inversion
        # Store as dense matrix for each block (used during inversion)
        self.full_block_matrix = ti.field(dtype=ti.f32,
                                          shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f32,
                                            shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))

        # Block info: which level and local ID
        self.block_level = ti.field(dtype=ti.i32, shape=self.total_blocks)
        self.n_blocks_per_level = ti.field(dtype=ti.i32, shape=MAX_LEVELS)

        # Elastic type tracking per cell
        # 0=ARAP, 1=SNH, 2=FCR, 3=ARAP_SPD, 4=NH_SPD, 5=STVK_SPD
        self.elastic_type = 0  # Default to ARAP

        # Mapping from string elastic type to integer for assembly
        self.ELASTIC_TYPE_MAP = {
            'ARAP': 0, 'ARAP_filter': 0,
            'SNH': 1,
            'FCR': 2, 'FCR_filter': 2,
            'ARAP_SPD': 3,
            'NH_SPD': 4,
            'STVK_SPD': 5,
            'NH': 1,  # Map NH to SNH for assembly (similar structure)
        }

        # SharedArray optimization: cell-to-warp mapping for reduction
        self._allocate_cell_warp_mapping()

        # Cross-block triplet storage for exact hessian_matvec
        self._allocate_cross_block_storage()

    def _allocate_cell_warp_mapping(self):
        """
        Allocate structures for cell-to-warp mapping used in SharedArray optimization.

        This enables grouping cells by their primary warp (the warp containing most vertices)
        to reduce atomic operations through shared memory accumulation.
        """
        # Primary warp ID for each cell (warp containing vertex 0)
        self.cell_primary_warp = ti.field(dtype=ti.i32, shape=self.n_cells)

        # Count of cells per warp for load balancing
        n_warps_l0 = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.cells_per_warp = ti.field(dtype=ti.i32, shape=n_warps_l0)

        # Cell list per warp (CSR format for warp-grouped cell processing)
        # Estimate: average ~4 cells per vertex, so ~64 cells per warp
        max_cells_per_warp = 128  # Conservative upper bound
        self.warp_cell_list = ti.field(dtype=ti.i32,
                                        shape=(n_warps_l0, max_cells_per_warp))
        self.warp_cell_count = ti.field(dtype=ti.i32, shape=n_warps_l0)
        self.max_cells_per_warp = max_cells_per_warp

        # Flag for whether mapping is built
        self.cell_warp_mapping_built = False

    def _allocate_preconditioning_buffers(self):
        """Allocate buffers for restrict/solve/prolong operations."""
        # Multi-level residual (restricted gradient at each level)
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32,
                                              shape=self.total_nodes_all_levels)

        # Multi-level solution (z at each level before prolongation)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32,
                                              shape=self.total_nodes_all_levels)

        # P1 Optimization: Warp-level reduction buffers
        # For fully-connected warps (prefix==1), we use parallel reduction
        # For multi-component warps, we use elected-node accumulation
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Warp sum buffer: stores the sum of residuals for each warp's elected nodes
        # Shape: [n_warps, BANKSIZE, 3] - each lane can be an elected node
        self.warp_sum_buffer = ti.field(dtype=ti.f32, shape=(n_warps, BANKSIZE, 3))

        # Warp prefix cache: stores prefix_original[warp_id] for fast access
        # prefix==1 means fully connected warp (can use fast reduction)
        self.warp_prefix_cache = ti.field(dtype=ti.i32, shape=n_warps)

    # ========================================================================
    # Public Interface
    # ========================================================================

    def rebuild(self, solver, use_full_hessian: bool = True,
                method: str = 'ic', adaptive_regularization: float = 0.05):
        """
        Full rebuild of preconditioner.
        Called on first iteration or when restart is needed.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling.
            method: Inversion method ('ic', 'cholesky', 'gauss_jordan', etc.)
            adaptive_regularization: Per-block regularization (default: 0.05)
        """
        if not self.hierarchy_built:
            self.build_hierarchy()

        # Set elastic type from solver if available
        if hasattr(solver, 'elastic_type'):
            elastic_str = solver.elastic_type
            if isinstance(elastic_str, str):
                self.elastic_type = self.ELASTIC_TYPE_MAP.get(elastic_str, 0)
                print(f"[MAS] Elastic type: {elastic_str} -> {self.elastic_type}")
            else:
                self.elastic_type = elastic_str

        self.assemble_block_matrices(solver, use_full_hessian)
        self.invert_block_matrices(method=method, adaptive_regularization=adaptive_regularization)

    def get_stats(self) -> dict:
        """Return statistics about the preconditioner."""
        return {
            'n_verts': self.n_verts,
            'n_levels': self.actual_levels if hasattr(self, 'actual_levels') else 0,
            'total_neighbors': self.total_neighbors,
            'hierarchy_built': self.hierarchy_built,
            'matrices_inverted': self.matrices_inverted,
        }

    # ========================================================================
    # Bit Manipulation Functions (used by mixins)
    # ========================================================================

    @ti.func
    def _popcount(self, x: ti.u32) -> ti.i32:
        """Count number of set bits in a 32-bit integer."""
        count = 0
        for _ in range(32):
            count += ti.i32(x & 1)
            x >>= 1
        return count

    @ti.func
    def _find_first_set(self, x: ti.u32) -> ti.i32:
        """Find index of first (lowest) set bit, or -1 if none."""
        result = -1
        for i in range(32):
            if ((x >> ti.u32(i)) & 1) and result == -1:
                result = i
        return result

    @ti.func
    def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
        """Return mask with bits set for lanes < lane_id."""
        result = ti.u32(0)
        if lane_id > 0:
            result = (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)
        return result

    @ti.func
    def _sym_index(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """
        Compute symmetric storage index for (i, j) where i <= j.
        Uses upper triangular storage: row-major within upper triangle.
        """
        # Ensure i <= j
        min_idx = ti.min(i, j)
        max_idx = ti.max(i, j)
        # Formula: sum of (BANKSIZE - k) for k in [0, min_idx) + (max_idx - min_idx)
        # = BANKSIZE * min_idx - min_idx * (min_idx + 1) / 2 + (max_idx - min_idx)
        # = BANKSIZE * min_idx - min_idx * (min_idx - 1) / 2 - min_idx + max_idx
        return BANKSIZE * min_idx - min_idx * (min_idx + 1) // 2 + max_idx
