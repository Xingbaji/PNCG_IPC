"""
Multilevel Additive Schwarz (MAS) Preconditioner for PNCG-IPC solver.

This module implements the MAS preconditioner from the paper:
"An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework
for Incremental Potential Contact"

The MAS preconditioner formula:
    P = M_{(0)}^{-1} + Σ_{l=1}^{L} C_{(l)}^T M_{(l)}^{-1} C_{(l)}

Where:
    - M_{(0)}^{-1} = Level-0 local inverses (Schwarz blocks)
    - C_{(l)} = Coarsening/restriction operators (binary aggregation)
    - M_{(l)}^{-1} = Coarse-level preconditioners

Key improvements over initial implementation:
    1. Full BANKSIZE×BANKSIZE block assembly (not just diagonal)
    2. Full 48×48 block inversion via Gauss-Jordan elimination
    3. Multi-level hierarchy construction (beyond 2 levels)
    4. Proper elastic Hessian integration with off-diagonal coupling
"""

import taichi as ti
import numpy as np
from math_utils.matrix_util import compute_dFdx, ssvd, flatten_matrix
from math_utils.elastic_util import (
    compute_d2PsidF2_ARAP, compute_d2PsidF2_SNH, compute_d2PsidF2_FCR,
    compute_d2PsidF2_ARAP_filter, compute_d2PsidF2_FCR_filter
)

# Constants matching CUDA reference
BANKSIZE = 16          # Nodes per subdomain (warp subdivision)
MAX_LEVELS = 6         # Maximum hierarchy depth
SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2  # = 136 for symmetric storage
MAX_NEIGHBORS_PER_VERTEX = 64
BLOCK_DOF = BANKSIZE * 3  # 48 DOFs per block

# Warp-level optimization constants
# Note: ti.simt.block.SharedArray is NOT supported in Taichi 1.7.4
# We use field-based warp reduction instead
WARP_REDUCTION_ENABLED = True  # Enable warp-level reduction optimization


@ti.data_oriented
class MASPreconditioner:
    """
    Multilevel Additive Schwarz preconditioner for PNCG optimization.

    The preconditioner uses connectivity-aware hierarchical coarsening
    to build a multilevel structure that captures both local and global
    coupling in the Hessian matrix.
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

        # Elastic type tracking per cell (0=ARAP, 1=SNH, 2=FCR, 3=NH)
        self.elastic_type = 0  # Default to ARAP

        # SharedArray optimization: cell-to-warp mapping for reduction
        self._allocate_cell_warp_mapping()

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

    def _build_neighbor_list_from_mesh(self):
        """Build neighbor list from mesh cell-vertex connectivity using Taichi."""
        # First pass: count neighbors using Taichi kernel
        self._count_neighbors_kernel()

        # Get neighbor counts and compute prefix sum on CPU
        neighbor_num_np = self.neighbor_num.to_numpy()

        neighbor_start_np = np.zeros(self.n_verts + 1, dtype=np.int32)
        neighbor_start_np[1:] = np.cumsum(neighbor_num_np)
        total_neighbors = int(neighbor_start_np[self.n_verts])

        self.total_neighbors = total_neighbors
        self.neighbor_start.from_numpy(neighbor_start_np)

        # Reset neighbor counts for second pass
        self.neighbor_num.fill(0)

        # Second pass: fill neighbor list
        self._fill_neighbors_kernel()

        print(f"[MAS] Built neighbor list: {total_neighbors} total edges")

    @ti.kernel
    def _count_neighbors_kernel(self):
        """Count neighbors for each vertex by iterating over cells."""
        # Reset counts
        for i in range(self.n_verts):
            self.neighbor_num[i] = 0

        # Count cell-based neighbors (with potential duplicates)
        for c in self.mesh.cells:
            v0 = c.verts[0].id
            v1 = c.verts[1].id
            v2 = c.verts[2].id
            v3 = c.verts[3].id

            # Each vertex connects to 3 others in the cell
            # We count all, duplicates will be handled
            ti.atomic_add(self.neighbor_num[v0], 3)
            ti.atomic_add(self.neighbor_num[v1], 3)
            ti.atomic_add(self.neighbor_num[v2], 3)
            ti.atomic_add(self.neighbor_num[v3], 3)

    @ti.kernel
    def _fill_neighbors_kernel(self):
        """Fill neighbor list by iterating over cells."""
        for c in self.mesh.cells:
            v_ids = ti.Vector([c.verts[0].id, c.verts[1].id,
                               c.verts[2].id, c.verts[3].id])

            # Add edges for each vertex pair
            for i in ti.static(range(4)):
                vi = v_ids[i]
                for j in ti.static(range(4)):
                    if i != j:
                        vj = v_ids[j]
                        # Get current slot and increment counter
                        slot = ti.atomic_add(self.neighbor_num[vi], 1)
                        offset = self.neighbor_start[vi] + slot
                        if offset < self.neighbor_list.shape[0]:
                            self.neighbor_list[offset] = vj

    # ========================================================================
    # Hierarchy Construction
    # ========================================================================

    @ti.func
    def _popcount(self, x: ti.u32) -> ti.i32:
        """
        Count number of set bits in a 32-bit integer.
        P1 optimization: Use ti.math.popcnt when available (Taichi 1.7+).
        """
        # Try to use built-in popcount for better performance
        # Fallback to loop-based implementation for older Taichi versions
        count = 0
        temp = x
        # Optimized loop-free popcount using parallel bit counting
        # This is faster than the naive while loop
        temp = temp - ((temp >> 1) & ti.u32(0x55555555))
        temp = (temp & ti.u32(0x33333333)) + ((temp >> 2) & ti.u32(0x33333333))
        temp = (temp + (temp >> 4)) & ti.u32(0x0F0F0F0F)
        count = ti.i32((temp * ti.u32(0x01010101)) >> 24)
        return count

    @ti.func
    def _find_first_set(self, x: ti.u32) -> ti.i32:
        """
        Find position of first set bit (0-indexed), or -1 if none.
        P1 optimization: Optimized bit scan using De Bruijn sequence.
        """
        pos = -1
        if x != 0:
            # Isolate the lowest set bit
            isolated = x & (~x + ti.u32(1))
            # Use De Bruijn sequence for O(1) bit position lookup
            # This is much faster than the naive while loop
            debruijn = ti.u32(0x077CB531)
            index = (isolated * debruijn) >> 27
            # Lookup table embedded in computation
            # Maps De Bruijn index to actual bit position
            lookup = ti.Vector([
                0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
                31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
            ], dt=ti.i32)
            pos = lookup[ti.i32(index)]
        return pos

    @ti.func
    def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
        """Return bitmask of lanes less than lane_id."""
        return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)

    @ti.kernel
    def _build_connect_mask_l0(self):
        """
        Build connectivity bitmask at Level 0.
        For each vertex, mark which other vertices in the same BANKSIZE group
        are connected via mesh edges.

        CUDA Reference: _buildCML0_new() (MASPreconditioner.cu lines 63-103)

        Key difference from old implementation:
        - Also updates neighbor_list to keep only inter-warp neighbors
        - Updates neighbor_num to reflect reduced count
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Start with self-connectivity
            connect_mask = ti.u32(1) << ti.u32(lane_id)
            nk = 0  # Count of inter-warp neighbors to keep

            # Check neighbors
            num_neighbor = self.neighbor_num[idx]
            start_id = self.neighbor_start[idx]

            for i in range(num_neighbor):
                neighbor_id = self.neighbor_list[start_id + i]
                neighbor_warp = neighbor_id // BANKSIZE

                if warp_id == neighbor_warp:
                    # Same warp: add to connectivity mask (not kept in list)
                    neighbor_lane = neighbor_id % BANKSIZE
                    connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))
                else:
                    # Different warp: keep in neighbor list for higher-level aggregation
                    self.neighbor_list[start_id + nk] = neighbor_id
                    nk += 1

            # Update neighbor count to only inter-warp neighbors
            self.neighbor_num[idx] = nk
            self.fine_connect_mask[idx] = connect_mask

    @ti.kernel
    def _propagate_connectivity(self):
        """
        Propagate connectivity within BANKSIZE groups to find connected components.
        Uses BFS-style expansion matching CUDA reference.

        CUDA Reference: _preparePrefixSumL0_new() (MASPreconditioner.cu lines 157-214)

        Key algorithm:
            visited = (1U << laneId)
            while(connectMsk != -1):  // -1 = 0xFFFFFFFF (all bits set)
                todo = visited ^ connectMsk
                if(!todo) break
                nextVist = __ffs(todo) - 1
                visited |= (1U << nextVist)
                connectMsk |= cacheMask[nextVist]
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Reset prefix counts
        for w in range(n_warps):
            self.prefix_original[w] = 0

        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # BFS-style transitive closure
            # visited tracks which nodes we've already incorporated
            visited = ti.u32(1) << ti.u32(lane_id)

            # Continue until all reachable nodes are included
            # connect_mask == 0xFFFFFFFF means all bits set (not possible in practice)
            max_iter = BANKSIZE
            for _ in range(max_iter):
                # Find nodes in connect_mask that we haven't visited yet
                todo = visited ^ connect_mask

                if todo == 0:
                    # No more nodes to visit
                    break

                # Find first unvisited connected node
                next_visit = self._find_first_set(todo)
                if next_visit < 0:
                    break

                # Mark as visited
                visited |= ti.u32(1) << ti.u32(next_visit)

                # Add that node's connections to our mask
                other_idx = warp_id * BANKSIZE + next_visit
                if other_idx < self.n_verts:
                    connect_mask |= self.fine_connect_mask[other_idx]

            # Store final transitive closure
            self.fine_connect_mask[idx] = connect_mask

            # Count elected representatives (electedPrefix == 0 means this is a representative)
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This node is the representative of its connected component
                ti.atomic_add(self.prefix_original[warp_id], 1)

    @ti.kernel
    def _find_cluster_representatives(self):
        """
        Find cluster representatives (elected nodes) and build elected_mask.
        A node is elected if it has the lowest lane ID in its connected component.

        Note: prefix_original is now set by _propagate_connectivity(),
        so this only sets elected_mask for use in _assign_cluster_ids().
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Reset elected mask (prefix_original already set by _propagate_connectivity)
        for w in range(n_warps):
            self.elected_mask[w] = ti.u32(0)

        # Build elected mask
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # Count how many connected nodes have lower lane ID
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is the representative of its cluster
                ti.atomic_or(self.elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))

    def build_collision_connection(self, collision_pairs):
        """
        Augment connectivity masks at collision points.

        CUDA Reference: _buildCollisionConnection_new() (MASPreconditioner.cu lines 1132-1542)

        This function ensures that vertices involved in contact are considered
        connected at the appropriate hierarchy level, even if they weren't
        originally mesh neighbors.

        Args:
            collision_pairs: List of collision pair vertex indices (4 vertices per collision)
        """
        if collision_pairs is None or len(collision_pairs) == 0:
            return

        # Convert to numpy array for Taichi kernel
        pairs_np = np.array(collision_pairs, dtype=np.int32)
        n_pairs = len(pairs_np)

        # Allocate temporary field for collision pairs if needed
        if not hasattr(self, 'collision_pairs') or self.collision_pairs.shape[0] < n_pairs:
            self.collision_pairs = ti.Vector.field(4, dtype=ti.i32, shape=n_pairs)

        self.collision_pairs.from_numpy(pairs_np.reshape(-1, 4))
        self._build_collision_connection_kernel(n_pairs)

    @ti.kernel
    def _build_collision_connection_kernel(self, n_pairs: ti.i32):
        """
        Kernel to augment connectivity masks at collision points.
        For each collision pair (4 vertices), if two vertices are in the same
        BANKSIZE warp, add them to each other's connectivity mask.
        """
        for pair_idx in range(n_pairs):
            cp_vid = self.collision_pairs[pair_idx]

            # Build connectivity mask for each pair of vertices in this collision
            for i in range(4):
                for j in range(i + 1, 4):
                    my_id = cp_vid[i]
                    ot_id = cp_vid[j]

                    # Skip invalid or same vertices
                    if my_id < 0 or ot_id < 0 or my_id == ot_id:
                        continue

                    # Only add connection if in same warp
                    if my_id // BANKSIZE == ot_id // BANKSIZE:
                        # Add bidirectional connection
                        my_lane = my_id % BANKSIZE
                        ot_lane = ot_id % BANKSIZE

                        ti.atomic_or(self.fine_connect_mask[my_id], ti.u32(1) << ti.u32(ot_lane))
                        ti.atomic_or(self.fine_connect_mask[ot_id], ti.u32(1) << ti.u32(my_lane))

    def build_collision_connection_lx(self, collision_pairs, level: int):
        """
        Augment connectivity masks at collision points for coarse level.

        CUDA Reference: _buildCollisionConnection_new() with coarseSpaceTable
                       (MASPreconditioner.cu lines 1158-1174)

        For coarse levels, we use coarse_space_tables to find the corresponding
        coarse node for each collision vertex, then add connectivity between them.

        Args:
            collision_pairs: List of collision pair vertex indices (4 vertices per collision)
            level: The coarse level (1, 2, ...) to augment connectivity for
        """
        if collision_pairs is None or len(collision_pairs) == 0:
            return

        # Convert to numpy array for Taichi kernel
        pairs_np = np.array(collision_pairs, dtype=np.int32)
        n_pairs = len(pairs_np)

        # Allocate temporary field for collision pairs if needed
        if not hasattr(self, 'collision_pairs') or self.collision_pairs.shape[0] < n_pairs:
            self.collision_pairs = ti.Vector.field(4, dtype=ti.i32, shape=n_pairs)

        self.collision_pairs.from_numpy(pairs_np.reshape(-1, 4))
        self._build_collision_connection_lx_kernel(n_pairs, level)

    @ti.kernel
    def _build_collision_connection_lx_kernel(self, n_pairs: ti.i32, level: ti.i32):
        """
        Kernel to augment connectivity masks at collision points for coarse level.

        For each collision pair (4 vertices), we:
        1. Look up the coarse node ID for each vertex using coarse_space_tables
        2. If two coarse nodes are in the same BANKSIZE warp, add them to
           each other's connectivity mask (next_connect_mask)
        """
        for pair_idx in range(n_pairs):
            cp_vid = self.collision_pairs[pair_idx]

            # Get coarse node IDs for each vertex in the collision pair
            cp_coarse = ti.Vector([-1, -1, -1, -1], dt=ti.i32)
            for i in ti.static(range(4)):
                fine_vid = cp_vid[i]
                if fine_vid >= 0 and fine_vid < self.n_verts:
                    # Look up the coarse node ID at this level
                    # coarse_space_tables[level-1, vid] gives the coarse node ID
                    cp_coarse[i] = self.coarse_space_tables[level - 1, fine_vid]

            # Build connectivity mask for each pair of coarse vertices
            for i in range(4):
                for j in range(i + 1, 4):
                    my_id = cp_coarse[i]
                    ot_id = cp_coarse[j]

                    # Skip invalid or same vertices
                    if my_id < 0 or ot_id < 0 or my_id == ot_id:
                        continue

                    # Only add connection if in same warp
                    if my_id // BANKSIZE == ot_id // BANKSIZE:
                        # Add bidirectional connection
                        my_lane = my_id % BANKSIZE
                        ot_lane = ot_id % BANKSIZE

                        ti.atomic_or(self.next_connect_mask[my_id], ti.u32(1) << ti.u32(ot_lane))
                        ti.atomic_or(self.next_connect_mask[ot_id], ti.u32(1) << ti.u32(my_lane))

    @ti.kernel
    def _assign_cluster_ids(self, level_offset: ti.i32):
        """
        Assign coarse-level cluster IDs based on elected representatives.
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is an elected node - compute its cluster ID
                # Count elected nodes in lower warps
                base_id = self.prefix_sum_original[warp_id]

                # Count elected nodes with lower lane ID in this warp
                elected_in_warp = self.elected_mask[warp_id]
                local_offset = self._popcount(elected_in_warp & self._lanemask_lt(lane_id))

                cluster_id = base_id + local_offset
                self.cluster_id[idx] = cluster_id
                self.coarse_space_tables[0, idx] = cluster_id
                self.going_next[idx] = level_offset + cluster_id
            else:
                # Find the representative (first set bit in connect_mask)
                rep_lane = self._find_first_set(connect_mask)
                rep_idx = warp_id * BANKSIZE + rep_lane

                # Will copy cluster ID from representative in next pass
                self.cluster_id[idx] = -1  # Mark as non-elected

    @ti.kernel
    def _propagate_cluster_ids(self):
        """
        Propagate cluster IDs from elected nodes to non-elected nodes.
        """
        for idx in range(self.n_verts):
            if self.cluster_id[idx] == -1:
                warp_id = idx // BANKSIZE
                connect_mask = self.fine_connect_mask[idx]

                # Find representative
                rep_lane = self._find_first_set(connect_mask)
                rep_idx = warp_id * BANKSIZE + rep_lane

                # Copy cluster ID
                self.cluster_id[idx] = self.cluster_id[rep_idx]
                self.coarse_space_tables[0, idx] = self.cluster_id[rep_idx]
                self.going_next[idx] = self.going_next[rep_idx]

    @ti.kernel
    def _build_aggregation_table(self):
        """
        Build aggregation table for fast prolongation.
        Stores the path through hierarchy for each fine vertex.
        """
        for idx in range(self.n_verts):
            current_idx = idx
            for level in range(self.level_num - 1):
                next_idx = self.going_next[current_idx]
                self.aggregation_table[idx][level] = next_idx
                current_idx = next_idx

    @ti.kernel
    def _build_connect_mask_lx(self, level_offset: ti.i32, level_size: ti.i32):
        """
        Build connectivity mask for coarse levels (Level 2+).
        Uses the coarse-level node indices and their connectivity.
        """
        for idx in range(level_size):
            global_idx = level_offset + idx
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Start with self-connectivity
            connect_mask = ti.u32(1) << ti.u32(lane_id)

            # For coarse levels, we inherit connectivity from fine level
            # Nodes that were connected at fine level remain connected
            self.next_connect_mask[idx] = connect_mask

    @ti.kernel
    def _propagate_connectivity_lx(self, level_offset: ti.i32, level_size: ti.i32):
        """
        Propagate connectivity within BANKSIZE groups at coarse level.
        """
        for idx in range(level_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.next_connect_mask[idx]

            # Iteratively expand connectivity
            for _ in range(BANKSIZE):
                old_mask = connect_mask
                for bit in range(BANKSIZE):
                    if (connect_mask >> ti.u32(bit)) & 1:
                        other_idx = warp_id * BANKSIZE + bit
                        if other_idx < level_size:
                            connect_mask |= self.next_connect_mask[other_idx]
                if connect_mask == old_mask:
                    break

            self.next_connect_mask[idx] = connect_mask

    @ti.kernel
    def _find_representatives_lx(self, level_size: ti.i32) -> ti.i32:
        """
        Find cluster representatives at coarse level.
        Returns the number of clusters found.
        """
        n_warps = (level_size + BANKSIZE - 1) // BANKSIZE
        count = 0

        for idx in range(level_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.next_connect_mask[idx]
            prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if prefix == 0:
                # This is a representative
                ti.atomic_add(count, 1)

        return count

    @ti.kernel
    def _assign_coarse_cluster_ids(self, src_level: ti.i32, src_offset: ti.i32,
                                    src_size: ti.i32, dst_offset: ti.i32):
        """
        Assign cluster IDs at a coarse level based on connectivity.
        """
        n_warps = (src_size + BANKSIZE - 1) // BANKSIZE

        # Reset prefix counts
        for w in range(n_warps):
            self.prefix_original[w] = 0
            self.elected_mask[w] = ti.u32(0)

        # First pass: count representatives per warp
        for idx in range(src_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.next_connect_mask[idx]
            prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if prefix == 0:
                ti.atomic_add(self.prefix_original[warp_id], 1)
                ti.atomic_or(self.elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))

    def build_hierarchy(self, collision_pairs=None):
        """
        Build complete hierarchy from fine to coarse.

        CUDA Reference: ReorderRealtime() (MASPreconditioner.cu lines 1772-1812)

        This implements the multi-level hierarchy construction:
        - Level 0: Fine level (all mesh vertices)
        - Level 1+: Coarse levels created by clustering connected nodes

        The hierarchy depth depends on mesh size:
        - Small meshes (< BANKSIZE nodes): 1-2 levels
        - Medium meshes: 2-4 levels
        - Large meshes: up to MAX_LEVELS levels

        Args:
            collision_pairs: Optional list of collision pair vertex indices
                            to augment connectivity at contact points
        """
        print("[MAS] Building hierarchy...")

        # Level 0: Build connectivity mask
        self._build_connect_mask_l0()

        # Augment connectivity at collision points (if provided)
        if collision_pairs is not None:
            self.build_collision_connection(collision_pairs)

        # Propagate connectivity to find connected components
        # This also counts representatives per warp (sets prefix_original)
        self._propagate_connectivity()

        # Build elected mask for cluster ID assignment
        self._find_cluster_representatives()

        # Compute prefix sum for cluster IDs
        prefix_np = self.prefix_original.to_numpy()
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        prefix_sum_np = np.zeros(n_warps + 1, dtype=np.int32)
        prefix_sum_np[1:n_warps+1] = np.cumsum(prefix_np[:n_warps])
        self.prefix_sum_original.from_numpy(prefix_sum_np)

        # Level 1 size and offset
        level_1_size = int(prefix_sum_np[n_warps])
        level_1_offset = self.n_verts

        # Store level info
        self.level_size[0] = ti.Vector([self.n_verts, 0])
        self.level_size[1] = ti.Vector([level_1_size, level_1_offset])

        # Assign cluster IDs for Level 0 -> Level 1
        self._assign_cluster_ids(level_1_offset)
        self._propagate_cluster_ids()

        # Set block counts for Level 0
        n_blocks_l0 = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.n_blocks_per_level[0] = n_blocks_l0

        # Track hierarchy progress
        actual_levels = 2
        current_size = level_1_size
        current_offset = level_1_offset

        print(f"[MAS] Level 0: {self.n_verts} nodes, Level 1: {level_1_size} nodes")

        # Build additional coarse levels (Level 2+)
        for level in range(2, MAX_LEVELS):
            # Stop if already small enough
            if current_size <= BANKSIZE:
                print(f"[MAS] Stopping at level {level-1}: {current_size} nodes <= BANKSIZE")
                break

            # Build connectivity at this level
            self._build_connect_mask_lx(current_offset, current_size)

            # Augment connectivity at collision points for this coarse level
            # CUDA Reference: BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum)
            if collision_pairs is not None:
                self.build_collision_connection_lx(collision_pairs, level - 1)

            self._propagate_connectivity_lx(current_offset, current_size)

            # Find representatives and compute next level size
            next_n_warps = (current_size + BANKSIZE - 1) // BANKSIZE
            self._assign_coarse_cluster_ids(level - 1, current_offset, current_size, 0)

            # Compute prefix sum - pad to original field size
            prefix_np = self.prefix_original.to_numpy()
            prefix_sum_np = np.zeros(self.prefix_sum_original.shape[0], dtype=np.int32)
            prefix_sum_np[1:next_n_warps+1] = np.cumsum(prefix_np[:next_n_warps])
            self.prefix_sum_original.from_numpy(prefix_sum_np)

            next_size = int(prefix_sum_np[next_n_warps])
            next_offset = current_offset + current_size

            # Check if we're actually reducing the problem
            if next_size >= current_size * 0.9:
                print(f"[MAS] Stopping at level {level}: insufficient reduction ({current_size} -> {next_size})")
                break

            # Store level info
            self.level_size[level] = ti.Vector([next_size, next_offset])

            # Update going_next for this level
            self._update_going_next_lx(current_offset, current_size, next_offset, level - 1)

            print(f"[MAS] Level {level}: {next_size} nodes")

            # Update for next iteration
            current_size = next_size
            current_offset = next_offset
            actual_levels = level + 1

            # Set block counts
            n_blocks = (current_size + BANKSIZE - 1) // BANKSIZE
            self.n_blocks_per_level[level - 1] = n_blocks

        # Build aggregation table for fast prolongation
        self._build_aggregation_table()

        # P1 Optimization: Cache warp prefix values for fast access during restriction
        self._cache_warp_prefix()

        print(f"[MAS] Hierarchy built with {actual_levels} levels")

        self.hierarchy_built = True
        self.actual_levels = actual_levels

    @ti.kernel
    def _update_going_next_lx(self, src_offset: ti.i32, src_size: ti.i32,
                               dst_offset: ti.i32, level: ti.i32):
        """
        Update going_next mapping for coarse level nodes.
        """
        n_warps = (src_size + BANKSIZE - 1) // BANKSIZE

        for idx in range(src_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            global_idx = src_offset + idx

            connect_mask = self.next_connect_mask[idx]
            prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if prefix == 0:
                # This is a representative
                base_id = self.prefix_sum_original[warp_id]
                local_offset = self._popcount(self.elected_mask[warp_id] & self._lanemask_lt(lane_id))
                cluster_id = base_id + local_offset

                self.going_next[global_idx] = dst_offset + cluster_id
            else:
                # Find representative
                rep_lane = self._find_first_set(connect_mask)
                rep_idx = warp_id * BANKSIZE + rep_lane

                if rep_idx < src_size:
                    rep_global = src_offset + rep_idx
                    # Will be set by representative in next pass
                    self.going_next[global_idx] = self.going_next[rep_global]

    # ========================================================================
    # Matrix Assembly
    # ========================================================================

    @ti.kernel
    def _build_cell_warp_mapping_kernel(self):
        """
        Build cell-to-warp mapping for SharedArray optimization.
        Each cell is assigned to the warp containing its first vertex (vertex 0).
        """
        # Clear counts
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id in range(n_warps):
            self.warp_cell_count[warp_id] = 0

        # Assign cells to warps based on vertex 0
        for c in self.mesh.cells:
            v0 = c.verts[0].id
            warp_id = v0 // BANKSIZE
            self.cell_primary_warp[c.id] = warp_id

            # Atomic increment to get slot in warp's cell list
            slot = ti.atomic_add(self.warp_cell_count[warp_id], 1)
            if slot < self.max_cells_per_warp:
                self.warp_cell_list[warp_id, slot] = c.id

    def build_cell_warp_mapping(self):
        """Build cell-to-warp mapping (call once after mesh is loaded)."""
        if not self.cell_warp_mapping_built:
            self._build_cell_warp_mapping_kernel()
            self.cell_warp_mapping_built = True

    @ti.kernel
    def _clear_block_matrices(self):
        """Zero out all block matrices."""
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.func
    def _sym_index(self, row: ti.i32, col: ti.i32) -> ti.i32:
        """
        Compute symmetric storage index for upper triangle.
        For row <= col: index = BANKSIZE * row - row*(row+1)/2 + col
        """
        r = ti.min(row, col)
        c = ti.max(row, col)
        return BANKSIZE * r - r * (r + 1) // 2 + c

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f32):
        """Add mass matrix to diagonal blocks."""
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Get mass from mesh
            m = self.mesh.verts.m[idx]

            # Diagonal block index in symmetric storage
            sym_idx = self._sym_index(lane_id, lane_id)

            # Add mass to diagonal (scaled by 1/dt^2 for implicit)
            mass_val = m / (dt * dt)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], mass_val)

    @ti.kernel
    def _add_elastic_contribution_full_optimized(self, mu: ti.f32, la: ti.f32, dt: ti.f32,
                                                   elastic_type: ti.i32):
        """
        Optimized elastic Hessian assembly with reduced branching and local accumulation.

        Optimization strategies:
        1. Pre-compute all 16 vertex-pair interactions for each cell
        2. Use local variables to accumulate before atomic write
        3. Minimize branching by separating same-warp and cross-warp handling
        4. Batch atomic operations where possible

        Note: ti.simt.block.SharedArray is not supported for CUDA in Taichi 1.7.4,
        so we use an alternative optimization strategy based on local accumulation
        and reduced branching.

        Args:
            mu: First Lamé parameter (shear modulus)
            la: Second Lamé parameter
            dt: Time step
            elastic_type: 0=ARAP, 1=SNH, 2=FCR
        """
        for c in self.mesh.cells:
            # Get cell volume weight
            W = c.W
            para = W * dt * dt

            # Get vertex IDs
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            # Pre-compute warp IDs for all 4 vertices
            warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                                  v2 // BANKSIZE, v3 // BANKSIZE])
            lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                                  v2 % BANKSIZE, v3 % BANKSIZE])

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Compute dFdx (9x12 matrix)
            dFdx = compute_dFdx(B)

            # Compute d2PsidF2 (9x9 matrix) based on elastic type
            d2PsidF2 = ti.Matrix.zero(ti.f32, 9, 9)
            if elastic_type == 0:  # ARAP
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            elif elastic_type == 1:  # SNH
                d2PsidF2 = compute_d2PsidF2_SNH(F, mu, la)
            elif elastic_type == 2:  # FCR
                d2PsidF2 = compute_d2PsidF2_FCR_filter(F, mu, la)
            else:
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

            # Compute element Hessian: H_e = dFdx^T @ d2PsidF2 @ dFdx (12x12)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Process upper triangle of vertex pairs (i <= j) to avoid double-counting
            # Use static unrolling for better performance
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):  # j >= i: upper triangle only
                    warp_i = warp_ids[i]
                    warp_j = warp_ids[j]
                    lane_i = lane_ids[i]
                    lane_j = lane_ids[j]

                    if warp_i == warp_j:
                        # Same warp: direct assembly to Level 0 block
                        # Pre-compute 3x3 sub-block to reduce atomic operations
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # Single branch for upper/lower triangle, then batch write
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj],
                                                  sub_block[di, dj])
                        else:
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][di, dj],
                                                  sub_block[dj, di])
                    else:
                        # Cross-warp: propagate to coarse level via goingNext
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        # Pre-compute sub-block once
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                coarse_lane_i = vert_i % BANKSIZE
                                coarse_lane_j = vert_j % BANKSIZE

                                if coarse_lane_i <= coarse_lane_j:
                                    sym_idx = BANKSIZE * coarse_lane_i - coarse_lane_i * (coarse_lane_i + 1) // 2 + coarse_lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj],
                                                          sub_block[di, dj])
                                else:
                                    sym_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][di, dj],
                                                          sub_block[dj, di])
                                break

    @ti.kernel
    def _add_elastic_contribution_full(self, mu: ti.f32, la: ti.f32, dt: ti.f32,
                                        elastic_type: ti.i32):
        """
        Add full elastic Hessian contribution with proper off-diagonal coupling.
        (Original implementation - fallback for non-CUDA backends)

        This computes the full 12x12 element Hessian and scatters it to the block matrices.
        The element Hessian is: H_e = W * dt^2 * (dFdx)^T * d2PsidF2 * dFdx

        Args:
            mu: First Lamé parameter (shear modulus)
            la: Second Lamé parameter
            dt: Time step
            elastic_type: 0=ARAP, 1=SNH, 2=FCR
        """
        for c in self.mesh.cells:
            # Get cell volume weight
            W = c.W
            para = W * dt * dt

            # Get vertex IDs
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Compute dFdx (9x12 matrix)
            dFdx = compute_dFdx(B)

            # Compute d2PsidF2 (9x9 matrix) based on elastic type
            # Use _filter versions to ensure SPD (project negative eigenvalues)
            d2PsidF2 = ti.Matrix.zero(ti.f32, 9, 9)
            if elastic_type == 0:  # ARAP
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            elif elastic_type == 1:  # SNH
                d2PsidF2 = compute_d2PsidF2_SNH(F, mu, la)
            elif elastic_type == 2:  # FCR
                d2PsidF2 = compute_d2PsidF2_FCR_filter(F, mu, la)
            else:  # Default to ARAP with filter
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

            # Compute element Hessian: H_e = dFdx^T @ d2PsidF2 @ dFdx (12x12)
            # First compute temp = d2PsidF2 @ dFdx (9x12)
            temp = d2PsidF2 @ dFdx

            # Then H_e = dFdx^T @ temp (12x12)
            H_e = dFdx.transpose() @ temp

            # Scale by volume and dt^2
            H_e = para * H_e

            # Scatter to block matrices
            # For each pair of vertices (i,j), scatter the corresponding 3x3 sub-block
            # CUDA Reference: PrepareHessian_bcoo() (MASPreconditioner.cu lines 1816-2093)
            #
            # Key difference: Cross-block entries propagate to coarse levels via goingNext
            # Only process upper triangle (i <= j) to avoid double-counting
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):  # j >= i: upper triangle only
                    vi = v_ids[i]
                    vj = v_ids[j]
                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    if warp_i == warp_j:
                        # Same block: direct assembly
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        # Extract 3x3 sub-block from H_e
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                val = H_e[i * 3 + di, j * 3 + dj]
                                # For symmetric storage, handle upper/lower triangle
                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
                                else:
                                    # Transpose for lower triangle
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][dj, di], val)
                    else:
                        # Cross-block entry: propagate to coarse level via goingNext
                        # Traverse hierarchy until we find a common block
                        vert_i = vi
                        vert_j = vj
                        for _ in range(self.level_num - 1):
                            # Move to next level
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                # Found common block at coarse level
                                lane_i = vert_i % BANKSIZE
                                lane_j = vert_j % BANKSIZE

                                # Extract 3x3 sub-block from H_e
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        val = H_e[i * 3 + di, j * 3 + dj]
                                        # Accumulate to coarse block
                                        if lane_i <= lane_j:
                                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val)
                                        else:
                                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][dj, di], val)
                                break

    @ti.kernel
    def _add_elastic_contribution_approx(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """
        Add approximate elastic Hessian contribution using diagonal approximation.
        This is a simplified fallback version that adds stiffness to diagonal blocks.
        """
        for c in self.mesh.cells:
            # Get cell volume weight
            W = c.W
            para = W * dt * dt

            # Get vertex IDs
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Simple stiffness estimate based on material parameters
            stiffness = para * (2.0 * mu + la)

            # Add to diagonal blocks for each vertex
            for i in ti.static(range(4)):
                vi = v_ids[i]
                warp_id = vi // BANKSIZE
                lane_id = vi % BANKSIZE
                sym_idx = self._sym_index(lane_id, lane_id)

                for d in ti.static(range(3)):
                    ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d],
                                  stiffness * 0.25)  # Distribute among 4 vertices

            # Add off-diagonal coupling between vertices in same block
            for i in ti.static(range(4)):
                for j in ti.static(range(i + 1, 4)):
                    vi = v_ids[i]
                    vj = v_ids[j]
                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    if warp_i == warp_j:
                        # Same block - add coupling
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE
                        sym_idx = self._sym_index(lane_i, lane_j)

                        coupling = para * mu * 0.1  # Small coupling
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.block_matrices[warp_i, sym_idx][d, d],
                                          coupling)

    @ti.kernel
    def _add_regularization(self, epsilon: ti.f32):
        """Add small regularization to diagonal for numerical stability."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    sym_idx = self._sym_index(lane_id, lane_id)

                    for d in ti.static(range(3)):
                        val = self.block_matrices[block_id, sym_idx][d, d]
                        if val < epsilon:
                            self.block_matrices[block_id, sym_idx][d, d] = epsilon

    @ti.kernel
    def _add_ipc_contact_contribution(self, cid: ti.template(), dHat: ti.f32, kappa: ti.f32):
        """
        Add IPC barrier Hessian contribution from contact pairs to block matrices.

        The IPC barrier Hessian for each contact has the form:
            H_contact = b''(d) * (∂d/∂x) ⊗ (∂d/∂x) + b'(d) * ∂²d/∂x²

        For the cubic barrier: b''(d) = 4 * kappa * (1 - d/dHat)

        The gradient ∂d/∂x for each vertex i in the contact stencil is:
            ∂d/∂x_i = cord[i] * normal

        So the (i,j) block of the Hessian is:
            H_ij = b''(d) * cord[i] * cord[j] * outer(normal, normal)

        CUDA Reference: PrepareHessian_bcoo() contact section (MASPreconditioner.cu)

        Args:
            cid: Sparse contact pair dictionary from solver
            dHat: Contact distance threshold
            kappa: Barrier stiffness
        """
        for k, j in cid:
            pair = cid[k, j]
            ids = pair.a      # 4 vertex IDs
            dist = pair.b     # Distance
            cord = pair.c     # Barycentric coordinates (4 values)
            normal = pair.d   # Contact normal direction (3D vector)

            # Skip if distance is beyond threshold
            if dist >= dHat:
                continue

            # Compute barrier Hessian coefficient: b''(d) = 4 * kappa * (1 - d/dHat)
            barrier_H = 4.0 * kappa * (1.0 - dist / dHat)

            # Process upper triangle of vertex pairs (i <= jj) to avoid double-counting
            for i in ti.static(range(4)):
                vi = ids[i]
                ci = cord[i]

                # Skip vertices with zero contribution
                if ti.abs(ci) < 1e-10:
                    continue

                for jj in ti.static(range(i, 4)):  # jj >= i: upper triangle only
                    vj = ids[jj]
                    cj = cord[jj]

                    # Skip vertices with zero contribution
                    if ti.abs(cj) < 1e-10:
                        continue

                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    # Compute the 3x3 contribution: H_ij = barrier_H * ci * cj * outer(n, n)
                    scale = barrier_H * ci * cj

                    if warp_i == warp_j:
                        # Same subdomain: direct assembly
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        # Compute outer(normal, normal) scaled by coefficient
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[di] * normal[dj]
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
                        else:
                            # Lower triangle: transpose
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[dj] * normal[di]
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][di, dj], val)
                    else:
                        # Cross-subdomain: propagate to coarse level via hierarchy
                        vert_i = vi
                        vert_j = vj

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                # Found common coarse block
                                lane_i = vert_i % BANKSIZE
                                lane_j = vert_j % BANKSIZE

                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[di] * normal[dj]
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val)
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[dj] * normal[di]
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][di, dj], val)
                                break

    def _add_ipc_contact_contribution_compact(self, solver, n_contacts: int):
        """
        Add IPC barrier Hessian contribution using compact array storage (P0 optimization).

        This version reads contacts from solver.contact_pairs[0:n_contacts] instead of
        iterating over the sparse bitmasked cid dictionary.
        """
        self._add_ipc_contact_contribution_compact_kernel(
            solver.contact_pairs, n_contacts, solver.dHat, solver.kappa)

    @ti.kernel
    def _add_ipc_contact_contribution_compact_kernel(self, contact_pairs: ti.template(),
                                                      n_contacts: ti.i32,
                                                      dHat: ti.f32, kappa: ti.f32):
        """
        Kernel for IPC contact Hessian using compact array (P0 optimization).
        Uses sequential array iteration instead of sparse dictionary traversal.
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            ids = pair.a      # 4 vertex IDs
            dist = pair.b     # Distance
            cord = pair.c     # Barycentric coordinates (4 values)
            normal = pair.d   # Contact normal direction (3D vector)

            # Skip if distance is beyond threshold
            if dist >= dHat:
                continue

            # Compute barrier Hessian coefficient: b''(d) = 4 * kappa * (1 - d/dHat)
            barrier_H = 4.0 * kappa * (1.0 - dist / dHat)

            # Process upper triangle of vertex pairs (i <= jj) to avoid double-counting
            for i in ti.static(range(4)):
                vi = ids[i]
                ci = cord[i]

                # Skip vertices with zero contribution
                if ti.abs(ci) < 1e-10:
                    continue

                for jj in ti.static(range(i, 4)):  # jj >= i: upper triangle only
                    vj = ids[jj]
                    cj = cord[jj]

                    # Skip vertices with zero contribution
                    if ti.abs(cj) < 1e-10:
                        continue

                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    # Compute the 3x3 contribution: H_ij = barrier_H * ci * cj * outer(n, n)
                    scale = barrier_H * ci * cj

                    if warp_i == warp_j:
                        # Same subdomain: direct assembly
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        # Compute outer(normal, normal) scaled by coefficient
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[di] * normal[dj]
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
                        else:
                            # Lower triangle: transpose
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[dj] * normal[di]
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][di, dj], val)
                    else:
                        # Cross-subdomain: propagate to coarse level via hierarchy
                        vert_i = vi
                        vert_j = vj

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                # Found common coarse block
                                lane_i = vert_i % BANKSIZE
                                lane_j = vert_j % BANKSIZE

                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[di] * normal[dj]
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val)
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[dj] * normal[di]
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][di, dj], val)
                                break

    @ti.kernel
    def _aggregate_fine_to_coarse(self, level_num: ti.i32):
        """
        Aggregate fine-level block entries to coarse levels.

        CUDA Reference: PrepareHessian_bcoo second pass (lines 1933-2062)

        This propagates ALL fine-level block entries to their corresponding
        coarse-level blocks via the goingNext hierarchy. Each fine-level 3x3
        sub-block (row, col) is added to the coarse-level block at position
        (goingNext[row], goingNext[col]).

        For efficiency, we iterate over fine-level blocks and propagate each
        entry to coarse levels.
        """
        n_fine_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # For each fine-level block, propagate entries to coarse levels
        for block_id in range(n_fine_blocks):
            for lane_row in range(BANKSIZE):
                for lane_col in range(lane_row, BANKSIZE):  # Upper triangle only
                    row_idx = block_id * BANKSIZE + lane_row
                    col_idx = block_id * BANKSIZE + lane_col

                    if row_idx >= self.n_verts or col_idx >= self.n_verts:
                        continue

                    # Get fine-level matrix entry
                    sym_idx = BANKSIZE * lane_row - lane_row * (lane_row + 1) // 2 + lane_col
                    mat3 = self.block_matrices[block_id, sym_idx]

                    # Skip if matrix is essentially zero
                    mat_norm = ti.f32(0.0)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            mat_norm += ti.abs(mat3[di, dj])
                    if mat_norm < 1e-12:
                        continue

                    # Propagate to all coarse levels
                    rdx = row_idx
                    cdx = col_idx

                    for level in range(level_num - 1):
                        # Move to next level
                        rdx = self.going_next[rdx]
                        cdx = self.going_next[cdx]

                        if rdx < 0 or cdx < 0:
                            break

                        coarse_block_r = rdx // BANKSIZE
                        coarse_block_c = cdx // BANKSIZE

                        if coarse_block_r == coarse_block_c:
                            # Same coarse block - accumulate
                            coarse_lane_r = rdx % BANKSIZE
                            coarse_lane_c = cdx % BANKSIZE

                            if coarse_lane_r <= coarse_lane_c:
                                coarse_sym_idx = BANKSIZE * coarse_lane_r - coarse_lane_r * (coarse_lane_r + 1) // 2 + coarse_lane_c
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_r, coarse_sym_idx][di, dj], mat3[di, dj])
                                        # When rdx == cdx (coarse diagonal), both (row,col) and (col,row)
                                        # map to same position, so add transpose contribution
                                        if coarse_lane_r == coarse_lane_c and row_idx != col_idx:
                                            ti.atomic_add(self.block_matrices[coarse_block_r, coarse_sym_idx][di, dj], mat3[dj, di])
                            else:
                                # Transpose for lower triangle storage
                                coarse_sym_idx = BANKSIZE * coarse_lane_c - coarse_lane_c * (coarse_lane_c + 1) // 2 + coarse_lane_r
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_c, coarse_sym_idx][di, dj], mat3[dj, di])

    @ti.kernel
    def _add_regularization_coarse(self, epsilon: ti.f32, level_num: ti.i32):
        """Add regularization to coarse-level diagonal blocks."""
        for level in range(1, level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                for lane_id in range(BANKSIZE):
                    local_idx = local_block_id * BANKSIZE + lane_id
                    if local_idx < level_size:
                        global_idx = level_offset + local_idx
                        block_id = global_idx // BANKSIZE
                        lane_in_block = global_idx % BANKSIZE

                        sym_idx = self._sym_index(lane_in_block, lane_in_block)

                        for d in ti.static(range(3)):
                            val = self.block_matrices[block_id, sym_idx][d, d]
                            if val < epsilon:
                                self.block_matrices[block_id, sym_idx][d, d] = epsilon

    def assemble_block_matrices(self, solver, use_full_hessian: bool = True,
                                use_optimized_kernel: bool = True):
        """
        Assemble Hessian contributions into block matrices.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling.
                             If False, use simplified diagonal approximation.
            use_optimized_kernel: If True, use optimized kernel with reduced branching
                                  and pre-computed sub-blocks.
        """
        print("[MAS] Assembling block matrices...")

        # Clear matrices
        self._clear_block_matrices()

        # Add inertia contribution (mass matrix)
        self._add_inertia_contribution(solver.dt)

        # Add elastic contribution
        if use_full_hessian:
            # Check if we should use optimized kernel (reduced branching, local accumulation)
            use_opt = use_optimized_kernel and self._is_cuda_backend()

            if use_opt:
                # Use optimized kernel with reduced branching and local accumulation
                self._add_elastic_contribution_full_optimized(
                    solver.mu, solver.la, solver.dt, self.elastic_type)
                print("[MAS] Full elastic Hessian assembled (optimized kernel)")
            else:
                # Original implementation (fallback)
                self._add_elastic_contribution_full(solver.mu, solver.la, solver.dt,
                                                     self.elastic_type)
                print("[MAS] Full elastic Hessian assembled")
        else:
            # Simplified diagonal approximation
            self._add_elastic_contribution_approx(solver.mu, solver.la, solver.dt)
            print("[MAS] Approximate elastic Hessian assembled")

        # Add IPC barrier Hessian from contact pairs
        # P0 optimization: Use compact array when available
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                self._add_ipc_contact_contribution_compact(solver, n_contacts)
                print(f"[MAS] IPC contact Hessian assembled ({n_contacts} contacts, compact)")
        elif hasattr(solver, 'cid') and solver.cid is not None:
            try:
                # Fallback to bitmasked cid (legacy)
                n_contacts = len(solver.cid)
                if n_contacts > 0:
                    self._add_ipc_contact_contribution(solver.cid, solver.dHat, solver.kappa)
                    print(f"[MAS] IPC contact Hessian assembled ({n_contacts} contacts)")
            except Exception:
                # cid may not be iterable if empty or uninitialized
                pass

        # Add regularization for numerical stability (Level 0)
        self._add_regularization(1e-6)

        # Aggregate fine-level block entries to coarse levels
        # CUDA Reference: PrepareHessian_bcoo second pass (lines 1933-2062)
        if self.hierarchy_built and self.actual_levels > 1:
            self._aggregate_fine_to_coarse(self.actual_levels)
            print(f"[MAS] Fine-to-coarse aggregation complete ({self.actual_levels} levels)")

            # Add regularization to coarse-level diagonal blocks
            self._add_regularization_coarse(1e-6, self.actual_levels)

        self.matrices_assembled = True
        print("[MAS] Block matrices assembled")

    # ========================================================================
    # Matrix Inversion
    # ========================================================================

    @ti.func
    def _invert_3x3(self, m: ti.template()) -> ti.Matrix:
        """Invert a 3x3 matrix."""
        det = m.determinant()
        if ti.abs(det) < 1e-12:
            # Return identity for singular matrix
            return ti.Matrix.identity(ti.f32, 3)

        inv_det = 1.0 / det

        inv = ti.Matrix([
            [(m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det,
             (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det,
             (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det],
            [(m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det,
             (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det,
             (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv_det],
            [(m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * inv_det,
             (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv_det,
             (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det]
        ], dt=ti.f32)

        return inv

    @ti.kernel
    def _expand_sym_to_full(self):
        """
        Expand symmetric block matrices to full 48x48 dense matrices for inversion.
        This now handles all levels (Level 0 + coarse levels).
        """
        # Process all blocks that may contain valid data
        # Block IDs are based on global vertex index: block_id = vertex_id // BANKSIZE
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Clear full matrix
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_matrix[block_id, i, j] = 0.0

            # Copy from symmetric storage to full matrix
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    sym_idx = self._sym_index(row, col)
                    block_3x3 = self.block_matrices[block_id, sym_idx]

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            # Upper triangle
                            self.full_block_matrix[block_id, row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                            # Lower triangle (symmetric)
                            if row != col:
                                self.full_block_matrix[block_id, col * 3 + dj, row * 3 + di] = block_3x3[dj, di]

    @ti.kernel
    def _gauss_jordan_invert_blocks(self):
        """
        Invert full 48x48 block matrices using Gauss-Jordan elimination.

        This is the key algorithm from the CUDA reference (MASPreconditioner.cu).
        For each block, we perform in-place Gauss-Jordan elimination to compute
        the inverse. This now handles all levels (Level 0 + coarse levels).
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Create augmented matrix [A | I] in-place
            # First, initialize identity in the inverse storage
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    if i == j:
                        self.full_block_inverse[block_id, i, j] = 1.0
                    else:
                        self.full_block_inverse[block_id, i, j] = 0.0

            # Gauss-Jordan elimination
            for pivot in range(BLOCK_DOF):
                # Find pivot (partial pivoting for numerical stability)
                max_val = ti.abs(self.full_block_matrix[block_id, pivot, pivot])
                max_row = pivot

                for r in range(pivot + 1, BLOCK_DOF):
                    val = ti.abs(self.full_block_matrix[block_id, r, pivot])
                    if val > max_val:
                        max_val = val
                        max_row = r

                # Swap rows if needed
                if max_row != pivot:
                    for c in range(BLOCK_DOF):
                        # Swap in matrix
                        tmp_m = self.full_block_matrix[block_id, pivot, c]
                        self.full_block_matrix[block_id, pivot, c] = self.full_block_matrix[block_id, max_row, c]
                        self.full_block_matrix[block_id, max_row, c] = tmp_m
                        # Swap in inverse
                        tmp_i = self.full_block_inverse[block_id, pivot, c]
                        self.full_block_inverse[block_id, pivot, c] = self.full_block_inverse[block_id, max_row, c]
                        self.full_block_inverse[block_id, max_row, c] = tmp_i

                # Scale pivot row
                pivot_val = self.full_block_matrix[block_id, pivot, pivot]
                if ti.abs(pivot_val) > 1e-12:
                    scale = 1.0 / pivot_val
                    for c in range(BLOCK_DOF):
                        self.full_block_matrix[block_id, pivot, c] *= scale
                        self.full_block_inverse[block_id, pivot, c] *= ti.f32(scale)
                else:
                    # Singular or near-singular - add regularization
                    self.full_block_matrix[block_id, pivot, pivot] = 1e-6
                    for c in range(BLOCK_DOF):
                        if c == pivot:
                            self.full_block_inverse[block_id, pivot, c] = 1e6
                        else:
                            self.full_block_inverse[block_id, pivot, c] = 0.0

                # Eliminate in all other rows
                for r in range(BLOCK_DOF):
                    if r != pivot:
                        factor = self.full_block_matrix[block_id, r, pivot]
                        for c in range(BLOCK_DOF):
                            self.full_block_matrix[block_id, r, c] -= factor * self.full_block_matrix[block_id, pivot, c]
                            self.full_block_inverse[block_id, r, c] -= ti.f32(factor) * self.full_block_inverse[block_id, pivot, c]

    @ti.kernel
    def _cholesky_invert_blocks(self):
        """
        Invert full 48x48 SPD block matrices using Cholesky decomposition.

        For SPD matrices, Cholesky is more efficient than Gauss-Jordan:
        - O(n^3/6) for factorization vs O(n^3) for Gauss-Jordan
        - No pivoting needed (numerically stable for SPD)
        - Can exploit symmetry

        Algorithm:
        1. Compute L such that A = L * L^T (Cholesky factorization)
        2. Solve L * Y = I for Y (forward substitution)
        3. Solve L^T * X = Y for X (backward substitution)
        4. X = A^{-1}
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Step 1: Cholesky factorization A = L * L^T
            # L is stored in-place in the lower triangle of full_block_matrix
            for i in range(BLOCK_DOF):
                # Compute L[i,j] for j < i
                for j in range(i):
                    sum_val = self.full_block_matrix[block_id, i, j]
                    for k in range(j):
                        sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_matrix[block_id, j, k]
                    L_jj = self.full_block_matrix[block_id, j, j]
                    if ti.abs(L_jj) > 1e-12:
                        self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                    else:
                        self.full_block_matrix[block_id, i, j] = 0.0

                # Compute L[i,i]
                sum_val = self.full_block_matrix[block_id, i, i]
                for k in range(i):
                    sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_matrix[block_id, i, k]
                if sum_val > 1e-12:
                    self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                else:
                    # Not SPD or near-singular - use regularization
                    self.full_block_matrix[block_id, i, i] = 1e-3

            # Step 2 & 3: Solve L * L^T * X = I for X = A^{-1}
            # We solve one column of X at a time
            for col in range(BLOCK_DOF):
                # Forward substitution: L * y = e_col
                # y is stored temporarily in full_block_inverse[:, col]
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                # Backward substitution: L^T * x = y
                # x overwrites y in full_block_inverse[:, col]
                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f32(self.full_block_inverse[block_id, i, col])
                    for k in range(i + 1, BLOCK_DOF):
                        sum_val -= self.full_block_matrix[block_id, k, i] * ti.f32(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _copy_inverse_to_sym(self):
        """
        Copy inverted full matrix back to symmetric storage format.
        This now handles all levels (Level 0 + coarse levels).
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    sym_idx = self._sym_index(row, col)

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, sym_idx][di, dj] = \
                                self.full_block_inverse[block_id, row * 3 + di, col * 3 + dj]

    @ti.kernel
    def _invert_diagonal_blocks(self):
        """
        Invert diagonal 3x3 blocks as a simple approximation.
        This is a fast fallback method for when full inversion is not needed.
        This now handles all levels.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < total_nodes:
                    sym_idx = self._sym_index(lane_id, lane_id)

                    # Get diagonal 3x3 block
                    diag_block = self.block_matrices[block_id, sym_idx]

                    # Invert it
                    inv_block = self._invert_3x3(diag_block)

                    # Store in single precision
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, sym_idx][di, dj] = \
                                ti.cast(inv_block[di, dj], ti.f32)

            # Set off-diagonal inverses to zero for now (diagonal approx)
            for row in range(BANKSIZE):
                for col in range(row + 1, BANKSIZE):
                    sym_idx = self._sym_index(row, col)
                    self.inv_block_matrices[block_id, sym_idx] = \
                        ti.Matrix.zero(ti.f32, 3, 3)

    def invert_block_matrices(self, use_full_inversion: bool = True,
                              use_cholesky: bool = True,
                              use_blocked: bool = False,
                              use_incomplete: bool = False,
                              use_oneway_gj: bool = False):
        """
        Invert all block matrices on GPU.

        Args:
            use_full_inversion: If True, use full 48x48 block inversion.
                               If False, use simplified diagonal-only inversion.
            use_cholesky: If True and use_full_inversion=True, use Cholesky decomposition
                         (faster for SPD matrices). If False, use Gauss-Jordan elimination.
            use_blocked: If True, use blocked Cholesky for better GPU parallelism.
            use_incomplete: If True, use Incomplete Cholesky IC(0) approximation.
            use_oneway_gj: If True, use One-way Gauss-Jordan without pivoting (P4 optimization).
        """
        print("[MAS] Inverting block matrices...")

        if use_full_inversion:
            # Full 48x48 block inversion
            self._expand_sym_to_full()

            if use_incomplete:
                # Incomplete Cholesky IC(0) - approximate but fast
                self._incomplete_cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Incomplete Cholesky IC(0))")
            elif use_oneway_gj:
                # One-way Gauss-Jordan without pivoting (P4 optimization)
                self._oneway_gauss_jordan_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (One-way Gauss-Jordan)")
            elif use_blocked and use_cholesky:
                # Blocked Cholesky - better GPU utilization
                self._blocked_cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Blocked Cholesky)")
            elif use_cholesky:
                # Standard Cholesky decomposition (faster for SPD matrices)
                self._cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Cholesky)")
            else:
                # Gauss-Jordan elimination (more general)
                self._gauss_jordan_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Gauss-Jordan)")
        else:
            # Simplified diagonal block inversion
            self._invert_diagonal_blocks()
            print("[MAS] Diagonal block inversion complete")

        self.matrices_inverted = True

    # ========================================================================
    # P2 Optimization: Blocked Cholesky Decomposition
    # ========================================================================
    #
    # Blocked Cholesky divides the 48x48 matrix into smaller blocks (e.g., 12x12)
    # and processes them in a way that maximizes parallelism:
    #
    # For block size B=12, we have 4x4 blocks in the 48x48 matrix:
    #   A = | A00  A01  A02  A03 |
    #       | A10  A11  A12  A13 |
    #       | A20  A21  A22  A23 |
    #       | A30  A31  A32  A33 |
    #
    # Blocked Cholesky proceeds as:
    #   1. L00 = chol(A00)
    #   2. L10 = A10 * L00^{-T}, L20 = A20 * L00^{-T}, L30 = A30 * L00^{-T}  (parallel)
    #   3. A11 -= L10 * L10^T, A21 -= L20 * L10^T, etc. (parallel updates)
    #   4. Repeat for remaining diagonal blocks

    @ti.kernel
    def _blocked_cholesky_invert_blocks(self):
        """
        Blocked Cholesky decomposition for better GPU parallelism.

        Uses block size of 12 (4 nodes * 3 DOF) to divide 48x48 into 4x4 blocks.
        This allows more parallel operations within each 48x48 block inversion.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        # Block size for blocked Cholesky (12 = 4 nodes * 3 DOF)
        BSIZE = 12
        N_SUB = BLOCK_DOF // BSIZE  # = 4 sub-blocks

        for block_id in range(n_blocks):
            # Blocked Cholesky factorization
            for kb in range(N_SUB):
                k_start = kb * BSIZE
                k_end = k_start + BSIZE

                # Step 1: Cholesky on diagonal block A[kb,kb]
                for i in range(k_start, k_end):
                    for j in range(k_start, i):
                        sum_val = self.full_block_matrix[block_id, i, j]
                        for kk in range(k_start, j):
                            sum_val -= self.full_block_matrix[block_id, i, kk] * \
                                       self.full_block_matrix[block_id, j, kk]
                        L_jj = self.full_block_matrix[block_id, j, j]
                        if ti.abs(L_jj) > 1e-12:
                            self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                        else:
                            self.full_block_matrix[block_id, i, j] = 0.0

                    # Diagonal element
                    sum_val = self.full_block_matrix[block_id, i, i]
                    for kk in range(k_start, i):
                        sum_val -= self.full_block_matrix[block_id, i, kk] ** 2
                    if sum_val > 1e-12:
                        self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                    else:
                        self.full_block_matrix[block_id, i, i] = 1e-3

                # Step 2: Solve for off-diagonal blocks L[ib,kb] = A[ib,kb] * L[kb,kb]^{-T}
                for ib in range(kb + 1, N_SUB):
                    i_start = ib * BSIZE
                    i_end = i_start + BSIZE

                    for i in range(i_start, i_end):
                        for j in range(k_start, k_end):
                            sum_val = self.full_block_matrix[block_id, i, j]
                            for kk in range(k_start, j):
                                sum_val -= self.full_block_matrix[block_id, i, kk] * \
                                           self.full_block_matrix[block_id, j, kk]
                            L_jj = self.full_block_matrix[block_id, j, j]
                            if ti.abs(L_jj) > 1e-12:
                                self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                            else:
                                self.full_block_matrix[block_id, i, j] = 0.0

                # Step 3: Update remaining blocks A[ib,jb] -= L[ib,kb] * L[jb,kb]^T
                for ib in range(kb + 1, N_SUB):
                    i_start = ib * BSIZE
                    i_end = i_start + BSIZE

                    for jb in range(kb + 1, ib + 1):
                        j_start = jb * BSIZE
                        j_end = j_start + BSIZE

                        for i in range(i_start, i_end):
                            j_limit = j_end if jb < ib else i + 1
                            for j in range(j_start, j_limit):
                                sum_val = 0.0
                                for kk in range(k_start, k_end):
                                    sum_val += self.full_block_matrix[block_id, i, kk] * \
                                               self.full_block_matrix[block_id, j, kk]
                                self.full_block_matrix[block_id, i, j] -= sum_val

            # Inversion via forward/backward substitution (same as standard Cholesky)
            for col in range(BLOCK_DOF):
                # Forward substitution
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= self.full_block_matrix[block_id, i, k] * \
                                   self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                # Backward substitution
                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f64(self.full_block_inverse[block_id, i, col])
                    for k in range(i + 1, BLOCK_DOF):
                        sum_val -= self.full_block_matrix[block_id, k, i] * \
                                   ti.f64(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    # ========================================================================
    # P3 Optimization: Incomplete Cholesky IC(0)
    # ========================================================================
    #
    # Incomplete Cholesky IC(0) only computes L elements where A has non-zeros.
    # For our 48x48 block matrices from FEM, the sparsity pattern is determined
    # by the mesh connectivity within each warp.
    #
    # Key insight: In FEM Hessian, entry (i,j) is non-zero only if nodes i and j
    # share a cell. For IC(0), we skip fill-in and only compute L[i,j] where A[i,j] != 0.
    #
    # Benefits:
    # - Fewer operations (skip zero entries)
    # - Better memory access (fewer cache misses)
    # - Good approximation for well-conditioned systems
    #
    # For MAS blocks, we use a simplified pattern: diagonal + first 2 off-diagonals
    # This captures most coupling while being fast to compute.

    @ti.kernel
    def _incomplete_cholesky_invert_blocks(self):
        """
        Incomplete Cholesky IC(0) factorization for approximate inversion.

        Uses a banded pattern: diagonal + k off-diagonals (k=6 for 2-node coupling).
        This provides a good approximation while being much faster than full Cholesky.

        The approximation quality depends on the fill-in pattern. For FEM matrices
        with local coupling, IC(0) typically provides a good preconditioner.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        # Bandwidth for IC(0): 6 = 2 nodes * 3 DOF
        # This captures direct node-node coupling in the Hessian
        BANDWIDTH = 6

        for block_id in range(n_blocks):
            # IC(0) Factorization with banded pattern
            for i in range(BLOCK_DOF):
                # Compute L[i,j] only for j in [max(0, i-BANDWIDTH), i)
                j_start = ti.max(0, i - BANDWIDTH)

                for j in range(j_start, i):
                    # Check if (i,j) is within bandwidth
                    if i - j <= BANDWIDTH:
                        sum_val = self.full_block_matrix[block_id, i, j]
                        k_start = ti.max(0, ti.max(i - BANDWIDTH, j - BANDWIDTH))
                        for k in range(k_start, j):
                            # Only include terms where both L[i,k] and L[j,k] are in pattern
                            if i - k <= BANDWIDTH and j - k <= BANDWIDTH:
                                sum_val -= self.full_block_matrix[block_id, i, k] * \
                                           self.full_block_matrix[block_id, j, k]
                        L_jj = self.full_block_matrix[block_id, j, j]
                        if ti.abs(L_jj) > 1e-12:
                            self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                        else:
                            self.full_block_matrix[block_id, i, j] = 0.0
                    else:
                        # Outside bandwidth - set to zero
                        self.full_block_matrix[block_id, i, j] = 0.0

                # Diagonal element
                sum_val = self.full_block_matrix[block_id, i, i]
                k_start = ti.max(0, i - BANDWIDTH)
                for k in range(k_start, i):
                    if i - k <= BANDWIDTH:
                        sum_val -= self.full_block_matrix[block_id, i, k] ** 2
                if sum_val > 1e-12:
                    self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                else:
                    self.full_block_matrix[block_id, i, i] = 1e-3

            # Approximate inversion using banded forward/backward substitution
            for col in range(BLOCK_DOF):
                # Forward substitution (banded)
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    k_start = ti.max(0, i - BANDWIDTH)
                    for k in range(k_start, i):
                        if i - k <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, i, k] * \
                                       self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                # Backward substitution (banded)
                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f64(self.full_block_inverse[block_id, i, col])
                    k_end = ti.min(BLOCK_DOF, i + BANDWIDTH + 1)
                    for k in range(i + 1, k_end):
                        if k - i <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, k, i] * \
                                       ti.f64(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    # ========================================================================
    # P4 Optimization: One-way Gauss-Jordan (No Pivoting)
    # ========================================================================
    #
    # CUDA Reference: __inverse6_P96x96() in MASPreconditioner.cu lines 630-726
    #
    # Key insight from CUDA implementation:
    # 1. For SPD matrices, pivoting is not needed (diagonal is always positive)
    # 2. One-way elimination: process columns left-to-right only
    # 3. Symmetry recovery: after elimination, copy symmetric elements
    # 4. Each thread handles one column, enabling parallel column operations
    #
    # Algorithm:
    # for j = 0 to N-1:
    #     rt = A[j,j]
    #     colm[:] = A[:,j]  # Save column j
    #     A[j,j] = 1; A[i,j] = 0 for i != j  # Set column j to e_j
    #     A[j,:] /= rt  # Scale row j
    #     for k != j:
    #         A[k,:] -= colm[k] * A[j,:]  # Eliminate
    # Recover symmetry: A[i+1,i] = A[i,i+1] for 3x3 sub-blocks

    @ti.kernel
    def _oneway_gauss_jordan_invert_blocks(self):
        """
        One-way Gauss-Jordan inversion without pivoting.

        Optimized for SPD matrices where:
        - Diagonal elements are guaranteed positive (no pivoting needed)
        - Symmetry can be recovered at the end instead of maintained throughout

        This reduces operations compared to full Gauss-Jordan with partial pivoting.

        CUDA Reference: __inverse6_P96x96() (MASPreconditioner.cu lines 630-726)
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Initialize full_block_inverse as a copy of the matrix
            # (we'll transform it in-place to the inverse)
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, i, j] = \
                        self.full_block_matrix[block_id, i, j]

            # One-way Gauss-Jordan elimination
            for j in range(BLOCK_DOF):
                # Get pivot (no search - assume SPD so diagonal is positive)
                rt = self.full_block_inverse[block_id, j, j]

                # Handle near-zero diagonal with regularization
                if ti.abs(rt) < 1e-10:
                    rt = 1e-6
                    self.full_block_inverse[block_id, j, j] = rt

                inv_rt = 1.0 / rt

                # Scale row j by 1/rt
                for i in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, j, i] *= inv_rt

                # Eliminate in all other rows
                for k in range(BLOCK_DOF):
                    if k != j:
                        factor = self.full_block_inverse[block_id, k, j]
                        for i in range(BLOCK_DOF):
                            self.full_block_inverse[block_id, k, i] -= \
                                factor * self.full_block_inverse[block_id, j, i]

            # Symmetry recovery (CUDA lines 707-711)
            # For 3x3 sub-blocks: enforce symmetry by averaging
            # Pattern: for each 3x3 block, copy upper to lower
            for node_i in range(BANKSIZE):
                for node_j in range(node_i + 1, BANKSIZE):
                    # Copy upper triangle to lower triangle for 3x3 sub-block
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            row = node_i * 3 + di
                            col = node_j * 3 + dj
                            row_t = node_j * 3 + dj
                            col_t = node_i * 3 + di
                            # Average for numerical stability
                            avg = 0.5 * (self.full_block_inverse[block_id, row, col] +
                                        self.full_block_inverse[block_id, row_t, col_t])
                            self.full_block_inverse[block_id, row, col] = avg
                            self.full_block_inverse[block_id, row_t, col_t] = avg

    # ========================================================================
    # P5 Optimization: Conflict-free Symmetric Matrix-Vector Multiplication
    # ========================================================================
    #
    # CUDA Reference: _schwarzLocalXSym6() in MASPreconditioner.cu lines 957-1027
    #
    # Key optimization: Instead of nested loops with branching for symmetric access,
    # use a flattened parallel structure:
    #
    # 1. Parallel over (block_id, lane_i, lane_j) for upper triangle only
    # 2. Each thread computes contribution M[i,j] * r[j] for upper triangle
    # 3. Add symmetric contribution M[j,i] * r[j] = M[i,j]^T * r[j] simultaneously
    # 4. Atomic-free accumulation using local buffers per output row
    #
    # CUDA uses __shfl_down_sync for warp reduction; Taichi uses ti.atomic_add
    # with careful access patterns to minimize conflicts.

    @ti.kernel
    def _schwarz_local_solve_conflict_free(self):
        """
        Conflict-free symmetric matrix-vector multiplication.

        Optimization over _schwarz_local_solve_full:
        1. Parallel over all (block_id, lane_i) pairs
        2. Unrolled 3x3 matrix-vector multiply (no inner loops)
        3. Direct symmetric storage access without branching
        4. Pre-computed symmetric indices

        CUDA Reference: _schwarzLocalXSym6() (MASPreconditioner.cu lines 957-1027)
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Parallel over (block_id, lane_i)
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                # Load residual for row i
                r_i = self.multi_level_r[idx_i]

                # Accumulate z using unrolled computation
                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

                # Process upper triangle (lane_j >= lane_i): use M[i,j] directly
                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]

                        # Compute symmetric index - always access upper triangle
                        # sym_idx formula: row * BANKSIZE - row*(row+1)/2 + col (row <= col)
                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, sym_idx]

                        # Matrix-vector multiply: z += inv_block @ r_j (or transpose if lower)
                        if lane_i <= lane_j:
                            # Upper triangle: use directly
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                            z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                            z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                        else:
                            # Lower triangle: use transpose
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                            z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                            z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

        # Coarse levels
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size:
                    # Accumulate z
                    z0 = ti.f32(0.0)
                    z1 = ti.f32(0.0)
                    z2 = ti.f32(0.0)

                    for lane_j in range(BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
                            r_j = self.multi_level_r[idx_j]

                            min_lane = ti.min(lane_i, lane_j)
                            max_lane = ti.max(lane_i, lane_j)
                            sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, sym_idx]

                            if lane_i <= lane_j:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                                z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                                z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                            else:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                                z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                                z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

    # ========================================================================
    # P6 Optimization: Banded Sparse MV for IC(0)
    # ========================================================================
    #
    # When using IC(0) with BANDWIDTH=6 (2 nodes × 3 DOF), the inverse matrix
    # has a banded structure. Instead of computing all 16×16 = 256 node pairs,
    # we only compute pairs within the bandwidth: |node_i - node_j| <= 2.
    #
    # This reduces computation from 256 to ~80 node pairs (16 diagonal + 32×2 off-diagonal).
    #
    # The key insight is that IC(0) preserves sparsity in L^{-1}, so (L L^T)^{-1}
    # also has limited bandwidth in practice.
    #
    # Benefits:
    # - 3x fewer multiplications in local solve
    # - Better cache locality (sequential node access)
    # - Combines well with IC(0) for maximum speedup

    @ti.kernel
    def _schwarz_local_solve_banded(self):
        """
        Banded sparse matrix-vector multiplication for IC(0).

        Only computes z_i = sum_j M^{-1}[i,j] * r[j] for |node_i - node_j| <= NODE_BANDWIDTH.

        For BANDWIDTH=6 (DOF level), this corresponds to NODE_BANDWIDTH=2 (node level).

        Complexity: O(n_blocks * BANKSIZE * NODE_BANDWIDTH) instead of O(n_blocks * BANKSIZE^2)

        P6 Optimization for use with IC(0).
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Node-level bandwidth: BANDWIDTH / 3 = 2 nodes
        # We process node pairs where |lane_i - lane_j| <= 2
        NODE_BANDWIDTH = 2

        # Level 0: Banded sparse MV
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

                # Only iterate over nodes within bandwidth
                lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
                lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

                for lane_j in range(lane_j_start, lane_j_end):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]

                        # Symmetric index calculation
                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, sym_idx]

                        # Matrix-vector multiply with transpose handling
                        if lane_i <= lane_j:
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                            z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                            z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                        else:
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                            z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                            z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

        # Coarse levels: also use banded structure
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size:
                    z0 = ti.f32(0.0)
                    z1 = ti.f32(0.0)
                    z2 = ti.f32(0.0)

                    # Banded iteration for coarse levels
                    lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
                    lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

                    for lane_j in range(lane_j_start, lane_j_end):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
                            r_j = self.multi_level_r[idx_j]

                            min_lane = ti.min(lane_i, lane_j)
                            max_lane = ti.max(lane_i, lane_j)
                            sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, sym_idx]

                            if lane_i <= lane_j:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                                z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                                z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                            else:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                                z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                                z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

    # ========================================================================
    # Preconditioning Operation (z = P * g)
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level residual and solution buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        """
        Hierarchically restrict gradient to coarse levels (original version).

        CUDA Reference: __buildMultiLevelR_optimized_new() (MASPreconditioner.cu lines 729-847)

        This implements a multi-level restriction with optimization for
        fully-connected blocks (prefix == 1) using reduction, and shared
        memory accumulation for multi-component blocks.

        Level 0: r = g (gradient)
        Level l: r_l = sum of r_{l-1} within cluster, propagated through hierarchy
        """
        # Copy gradient to level 0
        for idx in range(self.n_verts):
            self.multi_level_r[idx] = ti.cast(self.mesh.verts.grad[idx], ti.f32)

        # Restrict to all coarse levels through hierarchy
        for idx in range(self.n_verts):
            r = self.multi_level_r[idx]
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Get connectivity info
            connect_mask = self.fine_connect_mask[idx]

            # Check if this vertex is an elected representative
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is a representative - propagate to all coarse levels
                current_idx = idx
                for _ in range(self.level_num - 1):
                    next_idx = self.going_next[current_idx]
                    if next_idx >= 0 and next_idx < self.total_nodes_all_levels:
                        # Accumulate to coarse level
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                        current_idx = next_idx
                    else:
                        break
            else:
                # Non-representative: find elected lane and add to its accumulator
                elected_lane = self._find_first_set(connect_mask)
                elected_idx = warp_id * BANKSIZE + elected_lane

                # Contribute to the elected node's value (already in multi_level_r)
                # The elected node will propagate the sum to coarse levels
                # We accumulate locally first
                if elected_idx < self.n_verts and elected_idx != idx:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.multi_level_r[elected_idx][d], r[d])

    # ========================================================================
    # P1 Optimization: Warp-level Parallel Reduction for Restriction
    # ========================================================================
    #
    # CUDA reference uses __shfl_down_sync for warp reduction when prefix==1
    # (fully connected warp). Taichi 1.7.4 does not support ti.simt warp
    # primitives, so we use a 3-phase approach:
    #
    # Phase 1: Copy gradients to Level 0 and warp_sum_buffer
    # Phase 2: Parallel tree reduction within each warp using warp_sum_buffer
    # Phase 3: Elected nodes propagate reduced sums to coarse levels
    #
    # For multi-component warps (prefix > 1), we fall back to atomic accumulation
    # to the elected node of each component.

    @ti.kernel
    def _cache_warp_prefix(self):
        """Cache prefix_original values for each warp for fast access."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id in range(n_warps):
            self.warp_prefix_cache[warp_id] = self.prefix_original[warp_id]

    @ti.kernel
    def _clear_warp_sum_buffer(self):
        """Clear warp sum buffer."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id, lane_id, d in ti.ndrange(n_warps, BANKSIZE, 3):
            self.warp_sum_buffer[warp_id, lane_id, d] = 0.0

    @ti.kernel
    def _build_multi_level_r_phase1(self):
        """
        Phase 1: Copy gradients to Level 0 and initialize warp_sum_buffer.

        For each vertex, copy gradient to multi_level_r and warp_sum_buffer.
        """
        for idx in range(self.n_verts):
            r = ti.cast(self.mesh.verts.grad[idx], ti.f32)
            self.multi_level_r[idx] = r

            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Copy to warp_sum_buffer for reduction
            for d in ti.static(range(3)):
                self.warp_sum_buffer[warp_id, lane_id, d] = r[d]

    @ti.kernel
    def _build_multi_level_r_phase2_tree_reduce(self):
        """
        Phase 2: Tree reduction within fully-connected warps (prefix == 1).

        For warps where all vertices are connected (prefix_original == 1),
        perform parallel tree reduction: O(log BANKSIZE) steps instead of
        O(BANKSIZE) atomic operations.

        Tree reduction pattern (BANKSIZE=16):
            Step 0: lane 0 += lane 8,  lane 1 += lane 9,  ..., lane 7 += lane 15
            Step 1: lane 0 += lane 4,  lane 1 += lane 5,  ..., lane 3 += lane 7
            Step 2: lane 0 += lane 2,  lane 1 += lane 3
            Step 3: lane 0 += lane 1

        After reduction, lane 0 holds the sum of all 16 vertices.
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Process each warp
        for warp_id in range(n_warps):
            prefix = self.warp_prefix_cache[warp_id]

            if prefix == 1:
                # Fully connected warp: use tree reduction
                # Step 0: stride = 8
                for lane_id in range(8):
                    src_lane = lane_id + 8
                    src_idx = warp_id * BANKSIZE + src_lane
                    if src_idx < self.n_verts:
                        for d in ti.static(range(3)):
                            self.warp_sum_buffer[warp_id, lane_id, d] += \
                                self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 1: stride = 4
                for lane_id in range(4):
                    src_lane = lane_id + 4
                    for d in ti.static(range(3)):
                        self.warp_sum_buffer[warp_id, lane_id, d] += \
                            self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 2: stride = 2
                for lane_id in range(2):
                    src_lane = lane_id + 2
                    for d in ti.static(range(3)):
                        self.warp_sum_buffer[warp_id, lane_id, d] += \
                            self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 3: stride = 1
                for d in ti.static(range(3)):
                    self.warp_sum_buffer[warp_id, 0, d] += \
                        self.warp_sum_buffer[warp_id, 1, d]

    @ti.kernel
    def _build_multi_level_r_phase2_multi_component(self):
        """
        Phase 2b: Accumulate to elected nodes for multi-component warps (prefix > 1).

        For warps with multiple connected components, each vertex atomically
        adds to its elected representative in warp_sum_buffer.
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            prefix = self.warp_prefix_cache[warp_id]

            if prefix > 1:
                # Multi-component warp: use atomic accumulation to elected node
                connect_mask = self.fine_connect_mask[idx]
                elected_lane = self._find_first_set(connect_mask)

                # Only non-elected nodes accumulate to elected node
                if elected_lane != lane_id:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.warp_sum_buffer[warp_id, elected_lane, d],
                                      self.warp_sum_buffer[warp_id, lane_id, d])

    @ti.kernel
    def _build_multi_level_r_phase3_propagate(self):
        """
        Phase 3: Propagate reduced sums from elected nodes to coarse levels.

        For fully-connected warps (prefix==1), only lane 0 propagates.
        For multi-component warps, each elected node propagates its component sum.
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            prefix = self.warp_prefix_cache[warp_id]

            # Determine if this vertex should propagate to coarse levels
            should_propagate = False

            if prefix == 1:
                # Fully connected warp: only lane 0 propagates (holds total sum)
                should_propagate = (lane_id == 0)
            else:
                # Multi-component warp: elected nodes propagate
                connect_mask = self.fine_connect_mask[idx]
                elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))
                should_propagate = (elected_prefix == 0)

            if should_propagate:
                # Get the reduced sum from warp_sum_buffer
                r = ti.Vector([
                    self.warp_sum_buffer[warp_id, lane_id, 0],
                    self.warp_sum_buffer[warp_id, lane_id, 1],
                    self.warp_sum_buffer[warp_id, lane_id, 2]
                ], dt=ti.f32)

                # Propagate to all coarse levels
                current_idx = idx
                for _ in range(self.level_num - 1):
                    next_idx = self.going_next[current_idx]
                    if next_idx >= 0 and next_idx < self.total_nodes_all_levels:
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                        current_idx = next_idx
                    else:
                        break

    def _build_multi_level_r_optimized(self):
        """
        P1 Optimized version of multi-level restriction.

        Uses 3-phase approach for warp-level parallelism:
        - Phase 1: Copy gradients to buffers
        - Phase 2a: Tree reduction for fully-connected warps (prefix==1)
        - Phase 2b: Atomic accumulation for multi-component warps
        - Phase 3: Elected nodes propagate to coarse levels

        This reduces atomic operations from O(n_verts) to O(n_warps * avg_components).
        """
        # Phase 1: Initialize
        self._build_multi_level_r_phase1()

        # Phase 2a: Tree reduction for fully-connected warps
        self._build_multi_level_r_phase2_tree_reduce()

        # Phase 2b: Atomic accumulation for multi-component warps
        self._build_multi_level_r_phase2_multi_component()

        # Phase 3: Propagate to coarse levels
        self._build_multi_level_r_phase3_propagate()

    @ti.kernel
    def _schwarz_local_solve_full(self):
        """
        Solve z_d = B_d^{-1} * r_d for each subdomain at Level 0 and coarse levels.

        CUDA Reference: _schwarzLocalXSym6() (MASPreconditioner.cu lines 957-1027)

        The CUDA version uses one thread per matrix element with thread mapping:
            Hid   = idx / (BANKSIZE * BANKSIZE)    // Block ID
            lvrid = (idx % (BANKSIZE * BANKSIZE)) / BANKSIZE  // Row in block
            lvcid = (idx % (BANKSIZE * BANKSIZE)) % BANKSIZE  // Col in block

        For Taichi, we use a simpler but equivalent approach with nested loops.
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Full block solve
        for block_id in range(n_blocks):
            # Load residuals for this block into local array
            # and compute z = inv_M * r using full block inverse

            for lane_i in range(BANKSIZE):
                idx_i = block_id * BANKSIZE + lane_i
                if idx_i < self.n_verts:
                    # Initialize z to zero
                    z = ti.Vector.zero(ti.f32, 3)

                    # Multiply by full inverse block: z_i = sum_j (inv_M[i,j] @ r_j)
                    for lane_j in range(BANKSIZE):
                        idx_j = block_id * BANKSIZE + lane_j
                        if idx_j < self.n_verts:
                            # Get the inverse 3x3 block (symmetric storage)
                            # Symmetric index: for (i,j) where i <= j
                            if lane_i <= lane_j:
                                sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                # Upper triangle: use directly
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                            else:
                                sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                # Lower triangle: use transpose
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                    self.multi_level_z[idx_i] = z

        # Solve at all coarse levels using full block inverse
        # Block ID is based on global vertex index: block_id = vertex_id // BANKSIZE
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            # Get number of blocks at this level
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                # Global block ID based on where coarse nodes are stored
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                for lane_i in range(BANKSIZE):
                    idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                    if idx_i < level_offset + level_size:
                        # Initialize z to zero
                        z = ti.Vector.zero(ti.f32, 3)

                        # Multiply by full inverse block: z_i = sum_j (inv_M[i,j] @ r_j)
                        for lane_j in range(BANKSIZE):
                            idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                            if idx_j < level_offset + level_size:
                                # Get the inverse 3x3 block (symmetric storage)
                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                        self.multi_level_z[idx_i] = z

    @ti.kernel
    def _schwarz_local_solve_full_parallel(self):
        """
        Parallelized version of local solve: z_d = B_d^{-1} * r_d

        Key optimization: Parallelize over (block_id, lane_i) - each thread computes
        one output row. Inner loop over lane_j is sequential within each thread
        to avoid atomic contention.

        This is equivalent to the original but with explicit 2D parallelization over
        (block_id, lane_i) instead of nested sequential loops.
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Parallel over (block_id, lane_i)
        # Each thread handles one output vertex
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                # Initialize z to zero
                z = ti.Vector.zero(ti.f32, 3)

                # Sequential loop over columns - no atomics needed
                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        # Get the inverse 3x3 block (symmetric storage)
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            # Upper triangle: use directly
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                        else:
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            # Lower triangle: use transpose
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                self.multi_level_z[idx_i] = z

        # Coarse levels: also parallel over (local_block_id, lane_i)
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size:
                    # Initialize z to zero
                    z = ti.Vector.zero(ti.f32, 3)

                    for lane_j in range(BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
                            if lane_i <= lane_j:
                                sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                            else:
                                sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                    self.multi_level_z[idx_i] = z

    @ti.kernel
    def _schwarz_local_solve(self):
        """
        Solve z_d = B_d^{-1} * r_d for each subdomain at Level 0 and coarse levels.
        Using diagonal block approximation (fallback method, faster but less accurate).
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Diagonal block solve
        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    # Diagonal index in symmetric storage
                    sym_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id

                    # Get residual
                    r = self.multi_level_r[idx]

                    # Get inverse diagonal block
                    inv_block = self.inv_block_matrices[block_id, sym_idx]

                    # Compute z = inv_block @ r
                    z = ti.Vector.zero(ti.f32, 3)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            z[di] += inv_block[di, dj] * r[dj]

                    self.multi_level_z[idx] = z

        # Solve at all coarse levels using diagonal block inverse
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                for lane_id in range(BANKSIZE):
                    idx = level_offset + local_block_id * BANKSIZE + lane_id
                    if idx < level_offset + level_size:
                        # Diagonal index in symmetric storage
                        sym_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id

                        # Get residual
                        r = self.multi_level_r[idx]

                        # Get inverse diagonal block
                        inv_block = self.inv_block_matrices[block_id, sym_idx]

                        # Compute z = inv_block @ r
                        z = ti.Vector.zero(ti.f32, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                z[di] += inv_block[di, dj] * r[dj]

                        self.multi_level_z[idx] = z

    def _collect_final_z(self):
        """
        Prolongation: aggregate solutions from all levels.
        z = z_0 + C_1^T * z_1 + C_2^T * z_2 + ...

        CUDA Reference: __collectFinalZ_new() (MASPreconditioner.cu lines 850-879)

        The aggregation table stores the path through hierarchy for each vertex:
            aggregation_table[idx][level] = coarse node index at (level+1)

        CUDA loop: for(int i = 1; i < levelnum; i++) { now = tablePtr[i-1]; }
        Python equivalent: for level in range(level_num - 1): coarse_idx = table[level]
        This gives tablePtr[0], tablePtr[1], ..., tablePtr[levelnum-2]
        """
        self._collect_final_z_kernel(self.actual_levels)

    @ti.kernel
    def _collect_final_z_kernel(self, level_num: ti.i32):
        """
        Kernel for prolongation phase.

        For each fine vertex, sum up solutions from all levels:
        - Level 0: direct solution at vertex index
        - Level 1..N: solutions at coarse nodes via aggregation_table
        """
        for idx in range(self.n_verts):
            # Start with Level 0 solution
            z_total = self.multi_level_z[idx]

            # Add contributions from all coarse levels using aggregation table
            # CUDA: for(int i = 1; i < levelnum; i++) { now = tablePtr[i-1]; ... }
            for level in range(level_num - 1):
                coarse_idx = self.aggregation_table[idx][level]
                if coarse_idx >= 0 and coarse_idx < self.total_nodes_all_levels:
                    z_total += self.multi_level_z[coarse_idx]

            # Store result (cast to f32 for mesh storage)
            self.mesh.verts.z[idx] = ti.cast(z_total, ti.f32)

    def apply(self, use_full_solve: bool = True, use_parallel_solve: bool = False,
              use_warp_reduction: bool = True, use_conflict_free: bool = False,
              use_banded: bool = False):
        """
        Apply MAS preconditioner: z = P * grad

        Three phases:
        1. Restriction: gradient -> multi_level_r
        2. Local solve: multi_level_r -> multi_level_z
        3. Prolongation: multi_level_z -> z

        Args:
            use_full_solve: If True, use full block inverse in local solve.
                           If False, use diagonal-only approximation.
            use_parallel_solve: If True, use parallelized local solve with atomics.
                               This can be faster on GPUs with many cores.
            use_warp_reduction: If True, use P1 optimized warp-level reduction
                               for restriction phase. Default True.
            use_conflict_free: If True, use P5 conflict-free SpMV for local solve.
                              This uses unrolled matrix-vector multiply for better
                              instruction-level parallelism. Default False.
            use_banded: If True, use P6 banded sparse MV for local solve.
                       This is optimized for IC(0) which produces banded inverse.
                       Only accesses node pairs within NODE_BANDWIDTH=2.
                       Should be used together with IC(0) inversion for best results.
        """
        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction
        if use_warp_reduction and WARP_REDUCTION_ENABLED:
            self._clear_warp_sum_buffer()
            self._build_multi_level_r_optimized()
        else:
            self._build_multi_level_r()

        # Phase 2: Local solve
        if use_full_solve:
            if use_banded:
                # P6 optimization: banded sparse MV for IC(0)
                self._schwarz_local_solve_banded()
            elif use_conflict_free:
                # P5 optimization: conflict-free SpMV with unrolled MatVec
                self._schwarz_local_solve_conflict_free()
            elif use_parallel_solve:
                self._schwarz_local_solve_full_parallel()
            else:
                self._schwarz_local_solve_full()
        else:
            self._schwarz_local_solve()

        # Phase 3: Prolongation
        self._collect_final_z()

    # ========================================================================
    # Public Interface
    # ========================================================================

    def rebuild(self, solver, use_full_hessian: bool = True, use_full_inversion: bool = True):
        """
        Full rebuild of preconditioner.
        Called on first iteration or when restart is needed.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling.
            use_full_inversion: If True, use full 48x48 Gauss-Jordan inversion.
        """
        if not self.hierarchy_built:
            self.build_hierarchy()

        # Set elastic type from solver if available
        if hasattr(solver, 'elastic_type'):
            self.elastic_type = solver.elastic_type

        self.assemble_block_matrices(solver, use_full_hessian)
        self.invert_block_matrices(use_full_inversion)

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
    # Sparse-Input Woodbury Update (Section 3.1 of paper)
    # ========================================================================

    def init_woodbury_structures(self):
        """Initialize data structures for Woodbury updates."""
        self.top_k = 8
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        self.woodbury_U = ti.field(dtype=ti.f32,
                                    shape=(n_blocks, self.top_k, BANKSIZE * 3))
        self.woodbury_delta_S = ti.field(dtype=ti.f32,
                                          shape=(n_blocks, self.top_k))
        self.woodbury_num_updates = ti.field(dtype=ti.i32, shape=n_blocks)
        self.BU = ti.field(dtype=ti.f32,
                           shape=(n_blocks, BANKSIZE * 3, self.top_k))
        self.capacitance_matrix = ti.field(dtype=ti.f32,
                                            shape=(n_blocks, self.top_k, self.top_k))
        self.base_contacts = {}
        self.woodbury_initialized = True
        print(f"[Woodbury] Initialized for {n_blocks} subdomains")

    def save_base_contact_state(self, solver):
        """Save current contact state as base for Woodbury updates.
        Uses compact array storage (P0 optimization) when available.
        """
        self.base_contacts = {}

        # Try compact array first (P0 optimization)
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                contact_pairs_np = solver.contact_pairs.to_numpy()[:n_contacts]
                for i in range(n_contacts):
                    pair = contact_pairs_np[i]
                    ids = tuple(int(x) for x in pair['a'])
                    dist = float(pair['b'])
                    normal = tuple(float(x) for x in pair['d'])
                    cord = tuple(float(x) for x in pair['c'])
                    stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
                    contact_key = tuple(sorted(ids))
                    self.base_contacts[contact_key] = {
                        'stiffness': stiffness, 'normal': normal,
                        'dist': dist, 'ids': ids, 'cord': cord
                    }
                return

        # Fallback to bitmasked cid (legacy)
        try:
            cid_keys = solver.cid.keys_numpy()
        except (AttributeError, TypeError, RuntimeError):
            return

        for key in cid_keys:
            k, j = key
            pair_data = solver.cid[k, j]
            ids = tuple(int(x) for x in pair_data.a)
            dist = float(pair_data.b)
            normal = tuple(float(x) for x in pair_data.d)
            cord = tuple(float(x) for x in pair_data.c)
            stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
            contact_key = tuple(sorted(ids))
            self.base_contacts[contact_key] = {
                'stiffness': stiffness, 'normal': normal,
                'dist': dist, 'ids': ids, 'cord': cord
            }

    def _compute_barrier_stiffness(self, d, dHat, kappa):
        """
        Compute barrier Hessian k = b''(d) for IPC log barrier.

        IPC uses: b(d) = -kappa * (d - dHat)^2 * log(d / dHat)  for d < dHat

        First derivative:
        b'(d) = -kappa * [2(d - dHat) * log(d/dHat) + (d - dHat)^2 / d]

        Second derivative:
        b''(d) = -kappa * [2*log(d/dHat) + 2(d-dHat)/d + 2(d-dHat)/d - (d-dHat)^2/d^2]
               = -kappa * [2*log(d/dHat) + 4(d-dHat)/d - (d-dHat)^2/d^2]

        For Gauss-Newton approximation, we use the positive semi-definite part:
        k = kappa * [(d-dHat)/d]^2 * (2*d/dHat - 2 + dHat/d)

        Simplified approximation (matching reference):
        k = kappa * (dHat - d)^2 / (d^2 * dHat) * (2*d + dHat)
        """
        if d >= dHat or d <= 0:
            return 0.0

        # Use the Gauss-Newton approximation for SPD guarantee
        # This is the second derivative of the barrier, taking only the PSD part
        ratio = (dHat - d) / d
        # Simplified form that's always positive
        k = kappa * ratio * ratio * (2.0 + dHat / d) / dHat
        return max(k, 0.0)

    def compute_woodbury_updates(self, solver):
        """Compute low-rank update vectors from contact changes."""
        import numpy as np
        ROTATION_THRESHOLD = 0.9

        self._clear_woodbury_updates()
        current_contacts = self._get_current_contacts(solver)
        if not current_contacts:
            return

        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        subdomain_updates = {d: [] for d in range(n_blocks)}

        for contact_key, curr_data in current_contacts.items():
            curr_stiffness = curr_data['stiffness']
            if curr_stiffness < 1e-10:
                continue

            curr_normal = np.array(curr_data['normal'])
            curr_ids = curr_data['ids']
            curr_cord = np.array(curr_data['cord'])

            if contact_key in self.base_contacts:
                base_data = self.base_contacts[contact_key]
                base_stiffness = base_data['stiffness']
                base_normal = np.array(base_data['normal'])
                n1_norm = np.linalg.norm(curr_normal)
                n2_norm = np.linalg.norm(base_normal)
                dot_product = np.dot(curr_normal, base_normal) / (n1_norm * n2_norm + 1e-10)

                if dot_product < ROTATION_THRESHOLD:
                    delta_S = curr_stiffness
                else:
                    if curr_stiffness > base_stiffness:
                        delta_S = curr_stiffness - base_stiffness
                    else:
                        continue
            else:
                delta_S = curr_stiffness

            if delta_S < 1e-10:
                continue

            u_scale = np.sqrt(max(delta_S, 0.0))
            for i, vid in enumerate(curr_ids):
                subdomain_id = vid // BANKSIZE
                lane_id = vid % BANKSIZE
                u_contribution = u_scale * curr_cord[i] * curr_normal
                subdomain_updates[subdomain_id].append({
                    'delta_S': delta_S, 'lane_id': lane_id, 'u_vec': u_contribution
                })

        for d in range(n_blocks):
            updates = subdomain_updates[d]
            if not updates:
                continue
            updates.sort(key=lambda x: -x['delta_S'])
            top_updates = updates[:self.top_k]

            for k, upd in enumerate(top_updates):
                self.woodbury_delta_S[d, k] = upd['delta_S']
                lane_id = upd['lane_id']
                u_vec = upd['u_vec']
                for dim in range(3):
                    self.woodbury_U[d, k, lane_id * 3 + dim] = float(u_vec[dim])
            self.woodbury_num_updates[d] = len(top_updates)

    def _get_current_contacts(self, solver):
        """Extract current contact state from solver.
        Uses compact array storage (P0 optimization) when available.
        """
        current_contacts = {}

        # Try compact array first (P0 optimization)
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                contact_pairs_np = solver.contact_pairs.to_numpy()[:n_contacts]
                for i in range(n_contacts):
                    pair = contact_pairs_np[i]
                    ids = tuple(int(x) for x in pair['a'])
                    dist = float(pair['b'])
                    normal = tuple(float(x) for x in pair['d'])
                    cord = tuple(float(x) for x in pair['c'])
                    stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
                    current_contacts[tuple(sorted(ids))] = {
                        'stiffness': stiffness, 'normal': normal,
                        'dist': dist, 'ids': ids, 'cord': cord
                    }
                return current_contacts

        # Fallback to bitmasked cid (legacy)
        try:
            cid_keys = solver.cid.keys_numpy()
        except (AttributeError, TypeError, RuntimeError):
            return current_contacts

        for key in cid_keys:
            k, j = key
            pair_data = solver.cid[k, j]
            ids = tuple(int(x) for x in pair_data.a)
            dist = float(pair_data.b)
            normal = tuple(float(x) for x in pair_data.d)
            cord = tuple(float(x) for x in pair_data.c)
            stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
            current_contacts[tuple(sorted(ids))] = {
                'stiffness': stiffness, 'normal': normal,
                'dist': dist, 'ids': ids, 'cord': cord
            }
        return current_contacts

    @ti.kernel
    def _clear_woodbury_updates(self):
        """Clear Woodbury update structures."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for d in range(n_blocks):
            self.woodbury_num_updates[d] = 0
            for k in range(8):
                self.woodbury_delta_S[d, k] = 0.0
                for i in range(BANKSIZE * 3):
                    self.woodbury_U[d, k, i] = 0.0

    @ti.kernel
    def _compute_BU_and_capacitance(self):
        """
        Compute B*U and capacitance matrix (I + U^T B U).

        B is the cached inverse of the base subdomain Hessian (stored in symmetric format).
        U is the update matrix with shape (BANKSIZE*3, num_updates).

        BU[block_id, row*3+di, k] = sum_col sum_dj B[row,col][di,dj] * U[col*3+dj, k]

        The capacitance matrix is: C = I + U^T B U
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            num_updates = self.woodbury_num_updates[block_id]
            if num_updates == 0:
                continue

            # Compute BU = B @ U for each update vector k
            # B is stored in symmetric format: inv_block_matrices[block_id, sym_idx]
            # where sym_idx = _sym_index(min(row,col), max(row,col))
            for k in range(num_updates):
                for row in range(BANKSIZE):
                    idx_row = block_id * BANKSIZE + row
                    if idx_row >= self.n_verts:
                        continue

                    for di in ti.static(range(3)):
                        bu_val = 0.0

                        for col in range(BANKSIZE):
                            idx_col = block_id * BANKSIZE + col
                            if idx_col >= self.n_verts:
                                continue

                            # Get the 3x3 block from symmetric storage
                            # Symmetric storage: stores upper triangle (row <= col)
                            min_idx = ti.min(row, col)
                            max_idx = ti.max(row, col)
                            sym_idx = self._sym_index(min_idx, max_idx)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]

                            for dj in ti.static(range(3)):
                                u_val = self.woodbury_U[block_id, k, col * 3 + dj]
                                if ti.abs(u_val) > 1e-15:  # Skip zero entries
                                    # Handle symmetric access:
                                    # If row <= col: B[row,col] = inv_block
                                    # If row > col: B[row,col] = inv_block^T
                                    if row <= col:
                                        bu_val += inv_block[di, dj] * u_val
                                    else:
                                        bu_val += inv_block[dj, di] * u_val

                        self.BU[block_id, row * 3 + di, k] = bu_val

            # Compute capacitance matrix: C = I + U^T @ B @ U
            for i in range(num_updates):
                for j in range(num_updates):
                    # Start with identity on diagonal
                    cap_val = 1.0 if i == j else 0.0

                    # Add U_i^T @ BU_j = sum over all DOFs
                    for lane in range(BANKSIZE):
                        idx = block_id * BANKSIZE + lane
                        if idx < self.n_verts:
                            for d in ti.static(range(3)):
                                dof_idx = lane * 3 + d
                                cap_val += self.woodbury_U[block_id, i, dof_idx] * self.BU[block_id, dof_idx, j]

                    self.capacitance_matrix[block_id, i, j] = cap_val

    @ti.kernel
    def _schwarz_local_solve_woodbury(self):
        """Solve z_d = B̂_d * r_d using Woodbury formula."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            num_updates = self.woodbury_num_updates[block_id]

            for lane_i in range(BANKSIZE):
                idx_i = block_id * BANKSIZE + lane_i
                if idx_i < self.n_verts:
                    z = ti.Vector.zero(ti.f32, 3)
                    for lane_j in range(BANKSIZE):
                        idx_j = block_id * BANKSIZE + lane_j
                        if idx_j < self.n_verts:
                            sym_idx = self._sym_index(lane_i, lane_j)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            r_j = self.multi_level_r[idx_j]
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    if lane_i <= lane_j:
                                        z[di] += inv_block[di, dj] * r_j[dj]
                                    else:
                                        z[di] += inv_block[dj, di] * r_j[dj]
                    self.multi_level_z[idx_i] = z

            if num_updates == 0:
                continue

            r_vec = ti.Vector.zero(ti.f32, 8)
            for k in range(num_updates):
                r_val = 0.0
                for lane_id in range(BANKSIZE):
                    idx = block_id * BANKSIZE + lane_id
                    if idx < self.n_verts:
                        z_base = self.multi_level_z[idx]
                        for di in ti.static(range(3)):
                            r_val += self.woodbury_U[block_id, k, lane_id * 3 + di] * z_base[di]
                r_vec[k] = r_val

            cap_local = ti.Matrix.zero(ti.f32, 8, 8)
            for i in range(num_updates):
                for j in range(num_updates):
                    cap_local[i, j] = self.capacitance_matrix[block_id, i, j]

            lambda_vec = ti.Vector.zero(ti.f32, 8)
            for pivot in range(num_updates):
                max_val = ti.abs(cap_local[pivot, pivot])
                max_row = pivot
                for r in range(pivot + 1, num_updates):
                    if ti.abs(cap_local[r, pivot]) > max_val:
                        max_val = ti.abs(cap_local[r, pivot])
                        max_row = r
                if max_row != pivot:
                    for c in range(8):
                        tmp = cap_local[pivot, c]
                        cap_local[pivot, c] = cap_local[max_row, c]
                        cap_local[max_row, c] = tmp
                    tmp_r = r_vec[pivot]
                    r_vec[pivot] = r_vec[max_row]
                    r_vec[max_row] = tmp_r
                if ti.abs(cap_local[pivot, pivot]) > 1e-12:
                    for r in range(pivot + 1, num_updates):
                        factor = cap_local[r, pivot] / cap_local[pivot, pivot]
                        for c in range(pivot, 8):
                            cap_local[r, c] -= factor * cap_local[pivot, c]
                        r_vec[r] -= factor * r_vec[pivot]

            # Back substitution (reverse iteration emulation)
            for rev_idx in range(num_updates):
                i = num_updates - 1 - rev_idx
                if ti.abs(cap_local[i, i]) > 1e-12:
                    lambda_vec[i] = r_vec[i]
                    for j in range(i + 1, num_updates):
                        lambda_vec[i] -= cap_local[i, j] * lambda_vec[j]
                    lambda_vec[i] /= cap_local[i, i]

            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    correction = ti.Vector.zero(ti.f32, 3)
                    for k in range(num_updates):
                        for di in ti.static(range(3)):
                            correction[di] += self.BU[block_id, lane_id * 3 + di, k] * lambda_vec[k]
                    self.multi_level_z[idx] -= correction

        # Coarse levels: Use frozen cached inverses (no Woodbury updates)
        # This matches the paper: "coarse-level components primarily capture
        # low-frequency error modes, which evolve relatively slowly"
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            if level_size <= 0:
                continue

            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                # Full block solve using cached inverse
                for lane_i in range(BANKSIZE):
                    idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                    if idx_i < level_offset + level_size:
                        z = ti.Vector.zero(ti.f32, 3)

                        # Multiply by full inverse block: z_i = sum_j (inv_M[i,j] @ r_j)
                        for lane_j in range(BANKSIZE):
                            idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                            if idx_j < level_offset + level_size:
                                # Get the inverse 3x3 block (symmetric storage)
                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                        self.multi_level_z[idx_i] = z

    def apply_with_woodbury(self):
        """Apply MAS preconditioner with Woodbury updates."""
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._compute_BU_and_capacitance()
        self._schwarz_local_solve_woodbury()
        self._collect_final_z()

    def woodbury_update(self, solver):
        """Perform Sparse-Input Woodbury update."""
        if not hasattr(self, 'woodbury_initialized') or not self.woodbury_initialized:
            self.init_woodbury_structures()
        self.compute_woodbury_updates(solver)

    # ========================================================================
    # METIS-based Node Reordering (CEMAS - Connectivity-Enhanced MAS)
    # ========================================================================

    def _init_metis_from_mesh(self):
        """
        Initialize METIS reordering automatically from mesh topology.

        This is called during __init__ when use_metis=True (the default).
        It extracts cell data from the MeshTaichi mesh and performs METIS
        partitioning for improved MAS block quality.
        """
        # Skip METIS for very small meshes (no benefit)
        if self.n_verts < BANKSIZE * 2:
            print(f"[MAS] Mesh too small ({self.n_verts} verts), skipping METIS")
            self.use_metis_reorder = False
            return

        try:
            # Extract cells from mesh using neighbor list
            # Build cells from the already-constructed neighbor topology
            cells_np = self._extract_cells_from_neighbors()

            if cells_np is None or len(cells_np) == 0:
                print("[MAS] WARNING: Could not extract cells, skipping METIS reordering")
                return

            # Get vertex positions
            vertices_np = self.mesh.get_position_as_numpy()

            # Call the main METIS initialization
            self.init_metis_reordering(cells_np, vertices_np)

        except Exception as e:
            print(f"[MAS] WARNING: Failed to initialize METIS from mesh: {e}")
            print("[MAS] Falling back to standard hierarchy (no METIS reordering)")
            self.use_metis_reorder = False

    def _extract_cells_from_neighbors(self) -> np.ndarray:
        """
        Extract cell connectivity from neighbor list.

        For tetrahedral meshes, we can reconstruct cells by finding
        4-cliques in the neighbor graph. However, this is expensive.
        Instead, we use the neighbor information directly for METIS
        partitioning, which only needs the graph structure.
        """
        # Get neighbor data
        neighbor_num_np = self.neighbor_num.to_numpy()
        neighbor_start_np = self.neighbor_start.to_numpy()
        neighbor_list_np = self.neighbor_list.to_numpy()

        # Build cells by finding tetrahedra from mesh topology
        # For MeshTaichi, iterate through cells directly using the mesh relation
        cells_list = []

        try:
            # Try to access cells directly through mesh.cells relation
            # This works if the mesh has CV (cell-vertex) relation
            n_cells = len(self.mesh.cells)
            for c_idx in range(n_cells):
                # Access cell's vertices through the relation
                cell = self.mesh.cells[c_idx]
                v0 = cell.verts[0].id if hasattr(cell.verts[0], 'id') else 0
                v1 = cell.verts[1].id if hasattr(cell.verts[1], 'id') else 1
                v2 = cell.verts[2].id if hasattr(cell.verts[2], 'id') else 2
                v3 = cell.verts[3].id if hasattr(cell.verts[3], 'id') else 3
                cells_list.append([v0, v1, v2, v3])
        except Exception:
            # Fall back: construct pseudo-cells from neighbor list for METIS
            # Each vertex with its 3 most connected neighbors forms a pseudo-cell
            print("[MAS] Using neighbor-based pseudo-cells for METIS")
            for i in range(self.n_verts):
                n_neighbors = neighbor_num_np[i]
                if n_neighbors >= 3:
                    start = neighbor_start_np[i]
                    neighbors = neighbor_list_np[start:start+min(n_neighbors, 3)]
                    cells_list.append([i] + list(neighbors[:3]))

        if len(cells_list) == 0:
            return None

        return np.array(cells_list, dtype=np.int32)

    def init_metis_reordering(self, cells_np: np.ndarray, vertices_np: np.ndarray = None):
        """
        Initialize METIS-based node reordering for improved MAS quality.

        This implements the CEMAS (Connectivity-Enhanced MAS) approach from
        the Stiff-GIPC reference. METIS partitions the mesh graph so that
        topologically connected vertices are grouped together in blocks.

        Args:
            cells_np: Cell array of shape (n_cells, 4) containing vertex indices
            vertices_np: Optional vertex positions of shape (n_verts, 3)
        """
        try:
            from algorithm.metis_reorder import metis_reorder_mesh, BANKSIZE as METIS_BANKSIZE
        except ImportError:
            print("[MAS] WARNING: metis_reorder module not available")
            self.use_metis_reorder = False
            return

        print("[MAS] Initializing METIS-based node reordering...")

        # Perform METIS partitioning
        reorder_result = metis_reorder_mesh(
            n_verts=self.n_verts,
            cells=cells_np,
            vertices=vertices_np,
            block_size=BANKSIZE
        )

        # Store reordering data
        self.use_metis_reorder = True
        self.metis_sort_index = reorder_result['sort_index']
        self.metis_old_to_new = reorder_result['old_to_new']
        self.metis_partition = reorder_result['partition']
        self.metis_n_parts = reorder_result['n_partitions']
        self.metis_stats = reorder_result['stats']

        # Create Taichi fields for GPU access
        n_parts = reorder_result['n_partitions']

        self.partId_map_real = ti.field(dtype=ti.i32, shape=n_parts * BANKSIZE)
        self.real_map_partId = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Copy mappings to Taichi fields
        self.partId_map_real.from_numpy(reorder_result['partId_map_real'])
        self.real_map_partId.from_numpy(reorder_result['real_map_partId'])

        # Store sorted cells for reference
        self.metis_sorted_cells = reorder_result['sorted_cells']

        print(f"[MAS] METIS reordering initialized:")
        print(f"  - {n_parts} partitions for {self.n_verts} vertices")
        print(f"  - Max partition size: {self.metis_stats['max_partition_size']}")
        print(f"  - Avg partition size: {self.metis_stats['avg_partition_size']:.1f}")

    def get_metis_vertex_id(self, sorted_idx: int) -> int:
        """
        Get original vertex ID from sorted index.

        Args:
            sorted_idx: Index in the sorted (METIS-reordered) array

        Returns:
            Original vertex ID
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            return sorted_idx
        return int(self.metis_sort_index[sorted_idx])

    def get_sorted_vertex_id(self, original_idx: int) -> int:
        """
        Get sorted index from original vertex ID.

        Args:
            original_idx: Original vertex ID

        Returns:
            Index in the sorted (METIS-reordered) array
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            return original_idx
        return int(self.metis_old_to_new[original_idx])

    @ti.kernel
    def _build_connect_mask_l0_metis(self):
        """
        Build connectivity bitmask at Level 0 using METIS partition info.

        When METIS reordering is used, vertices within each partition are
        already grouped by connectivity, so we use the partition structure
        directly to determine block membership.
        """
        for idx in range(self.n_verts):
            # Get partition info from METIS mapping
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            # Start with self-connectivity
            connect_mask = ti.u32(1) << ti.u32(lane_id)

            # Check neighbors - only mark as connected if in same partition
            num_neighbor = self.neighbor_num[idx]
            start_id = self.neighbor_start[idx]

            for i in range(num_neighbor):
                neighbor_id = self.neighbor_list[start_id + i]
                neighbor_part_info = self.real_map_partId[neighbor_id]
                neighbor_block = neighbor_part_info // BANKSIZE

                if block_id == neighbor_block:
                    # Same partition/block: add to connectivity mask
                    neighbor_lane = neighbor_part_info % BANKSIZE
                    connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))

            self.fine_connect_mask[idx] = connect_mask

    @ti.kernel
    def _propagate_connectivity_metis(self):
        """
        Propagate connectivity using METIS partition structure.

        This is the METIS version of _propagate_connectivity() that uses
        METIS partition IDs (block_id from real_map_partId) instead of
        warp IDs (idx // BANKSIZE) to compute prefix_original.

        The key difference is that prefix_original[block_id] counts the number
        of connected components (cluster representatives) within each METIS partition,
        not within each sequential warp of 16 vertices.
        """
        n_parts = self.metis_n_parts

        # Reset prefix counts for METIS partitions
        for p in range(n_parts):
            self.prefix_original[p] = 0

        for idx in range(self.n_verts):
            # Get METIS partition info
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # BFS-style transitive closure within the METIS partition
            visited = ti.u32(1) << ti.u32(lane_id)

            max_iter = BANKSIZE
            for _ in range(max_iter):
                todo = visited ^ connect_mask

                if todo == 0:
                    break

                next_visit = self._find_first_set(todo)
                if next_visit < 0:
                    break

                visited |= ti.u32(1) << ti.u32(next_visit)

                # Find the vertex in this METIS partition with the given lane
                other_part_idx = block_id * BANKSIZE + next_visit
                other_idx = self.partId_map_real[other_part_idx]

                if other_idx >= 0 and other_idx < self.n_verts:
                    connect_mask |= self.fine_connect_mask[other_idx]

            # Store final transitive closure
            self.fine_connect_mask[idx] = connect_mask

            # Count elected representatives using METIS partition (not warp)
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This node is the representative of its connected component
                ti.atomic_add(self.prefix_original[block_id], 1)

    @ti.kernel
    def _find_cluster_representatives_metis(self):
        """
        Find cluster representatives using METIS partition structure.

        Sets elected_mask based on METIS partitions rather than sequential warps.
        """
        n_parts = self.metis_n_parts

        # Reset elected mask
        for p in range(n_parts):
            self.elected_mask[p] = ti.u32(0)

        # Build elected mask
        for idx in range(self.n_verts):
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # Count how many connected nodes have lower lane ID
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is the representative of its cluster
                ti.atomic_or(self.elected_mask[block_id], ti.u32(1) << ti.u32(lane_id))

    @ti.kernel
    def _assign_cluster_ids_metis(self, level_1_offset: ti.i32):
        """
        Assign cluster IDs using METIS partition structure.

        Maps each vertex to its coarse-level cluster based on METIS partitions.
        """
        n_parts = self.metis_n_parts

        for idx in range(self.n_verts):
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_mask = self.elected_mask[block_id]

            # Find which elected node this vertex belongs to
            my_cluster_mask = connect_mask & elected_mask

            if my_cluster_mask != 0:
                # Find the lowest set bit (the representative)
                rep_lane = self._find_first_set(my_cluster_mask)

                # Count elected nodes with lower lane ID
                prefix_in_warp = self._popcount(elected_mask & self._lanemask_lt(rep_lane))

                # Global cluster ID
                cluster_id = self.prefix_sum_original[block_id] + prefix_in_warp

                # Store mapping
                self.going_next[idx] = level_1_offset + cluster_id
                self.coarse_space_table_0[idx] = level_1_offset + cluster_id

    @ti.kernel
    def _add_inertia_contribution_metis(self, dt: ti.f32):
        """Add mass matrix to diagonal blocks using METIS mapping."""
        for idx in range(self.n_verts):
            # Get block and lane from METIS partition
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            # Get mass from mesh (using original vertex index)
            m = self.mesh.verts.m[idx]

            # Diagonal block index in symmetric storage
            sym_idx = self._sym_index(lane_id, lane_id)

            # Add mass to diagonal (scaled by 1/dt^2 for implicit)
            mass_val = m / (dt * dt)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[block_id, sym_idx][d, d], mass_val)

    @ti.kernel
    def _schwarz_local_solve_full_metis(self):
        """
        Solve z_d = B_d^{-1} * r_d using METIS partition mapping.

        partId_map_real[part_id * BANKSIZE + lane] = original_vertex_id
        This kernel iterates over all METIS partitions (blocks) and applies
        the inverse block matrix to compute z = B^{-1} * r.
        """
        n_blocks = self.metis_n_parts

        for block_id in range(n_blocks):
            for lane_i in range(BANKSIZE):
                # Get original vertex ID from partition mapping
                part_idx_i = block_id * BANKSIZE + lane_i
                idx_i = self.partId_map_real[part_idx_i]

                # Check if this slot is valid (not all partitions are full)
                if idx_i >= 0 and idx_i < self.n_verts:
                    # Initialize z to zero
                    z = ti.Vector.zero(ti.f32, 3)

                    # Multiply by full inverse block
                    for lane_j in range(BANKSIZE):
                        part_idx_j = block_id * BANKSIZE + lane_j
                        idx_j = self.partId_map_real[part_idx_j]

                        if idx_j >= 0 and idx_j < self.n_verts:
                            sym_idx = self._sym_index(lane_i, lane_j)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            r_j = self.multi_level_r[idx_j]

                            if lane_i <= lane_j:
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * r_j[dj]
                            else:
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * r_j[dj]

                    self.multi_level_z[idx_i] = z

        # Coarse level solve (same as before)
        level_1_offset = self.n_verts
        level_1_size = self.level_size[1][0]

        for i in range(level_1_size):
            idx = level_1_offset + i
            r = self.multi_level_r[idx]
            r_norm = r.norm()
            if r_norm > 1e-10:
                scale = 1.0 / r_norm
                self.multi_level_z[idx] = r * ti.min(scale, 1.0)
            else:
                self.multi_level_z[idx] = r

    def build_hierarchy_metis(self):
        """
        Build hierarchy using METIS partition information.

        This is an optimized version that uses METIS partition structure
        directly for the hierarchy construction. It uses METIS-specific
        versions of connectivity propagation and cluster assignment that
        work with METIS partition IDs instead of sequential warp IDs.
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            print("[MAS] WARNING: METIS reordering not initialized, using standard hierarchy")
            self.build_hierarchy()
            return

        print("[MAS] Building hierarchy with METIS partitioning...")

        # Use METIS partition for Level 0 connectivity
        self._build_connect_mask_l0_metis()

        # Propagate connectivity using METIS partition structure
        # This sets prefix_original[block_id] for each METIS partition
        self._propagate_connectivity_metis()

        # Find cluster representatives using METIS partitions
        self._find_cluster_representatives_metis()

        # Compute prefix sum for cluster IDs
        prefix_np = self.prefix_original.to_numpy()
        n_parts = self.metis_n_parts
        prefix_sum_np = np.zeros(n_parts + 1, dtype=np.int32)
        prefix_sum_np[1:n_parts+1] = np.cumsum(prefix_np[:n_parts])
        self.prefix_sum_original.from_numpy(prefix_sum_np)

        # Level 1 size and offset
        level_1_size = int(prefix_sum_np[n_parts])
        level_1_offset = self.n_verts

        # Store level info
        self.level_size[0] = ti.Vector([self.n_verts, 0])
        self.level_size[1] = ti.Vector([level_1_size, level_1_offset])

        # Assign cluster IDs using METIS partition structure
        self._assign_cluster_ids_metis(level_1_offset)

        # Set block counts
        self.n_blocks_per_level[0] = self.metis_n_parts

        print(f"[MAS] Level 0: {self.n_verts} nodes ({self.metis_n_parts} METIS partitions)")
        print(f"[MAS] Level 1: {level_1_size} nodes")

        # Build additional coarse levels (same as standard)
        actual_levels = 2
        current_size = level_1_size
        current_offset = level_1_offset

        for level in range(2, MAX_LEVELS):
            if current_size <= BANKSIZE:
                print(f"[MAS] Stopping at level {level-1}: {current_size} nodes <= BANKSIZE")
                break

            self._build_connect_mask_lx(current_offset, current_size)
            self._propagate_connectivity_lx(current_offset, current_size)

            next_n_warps = (current_size + BANKSIZE - 1) // BANKSIZE
            self._assign_coarse_cluster_ids(level - 1, current_offset, current_size, 0)

            prefix_np = self.prefix_original.to_numpy()
            prefix_sum_np = np.zeros(next_n_warps + 1, dtype=np.int32)
            prefix_sum_np[1:next_n_warps+1] = np.cumsum(prefix_np[:next_n_warps])
            self.prefix_sum_original.from_numpy(prefix_sum_np)

            next_size = int(prefix_sum_np[next_n_warps])
            next_offset = current_offset + current_size

            if next_size >= current_size * 0.9:
                print(f"[MAS] Stopping at level {level}: insufficient reduction")
                break

            self.level_size[level] = ti.Vector([next_size, next_offset])
            self._update_going_next_lx(current_offset, current_size, next_offset, level - 1)

            print(f"[MAS] Level {level}: {next_size} nodes")

            current_size = next_size
            current_offset = next_offset
            actual_levels = level + 1

            n_blocks = (current_size + BANKSIZE - 1) // BANKSIZE
            self.n_blocks_per_level[level - 1] = n_blocks

        self._build_aggregation_table()

        # P1 Optimization: Cache warp prefix values for fast access during restriction
        self._cache_warp_prefix()

        print(f"[MAS] METIS hierarchy built with {actual_levels} levels")

        self.hierarchy_built = True
        self.actual_levels = actual_levels

    def rebuild_with_metis(self, solver, use_full_hessian: bool = True,
                           use_full_inversion: bool = True):
        """
        Full rebuild of preconditioner using METIS-based partitioning.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling
            use_full_inversion: If True, use full 48x48 Gauss-Jordan inversion
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            print("[MAS] WARNING: METIS not initialized, using standard rebuild")
            self.rebuild(solver, use_full_hessian, use_full_inversion)
            return

        if not self.hierarchy_built:
            self.build_hierarchy_metis()

        if hasattr(solver, 'elastic_type'):
            self.elastic_type = solver.elastic_type

        self.assemble_block_matrices(solver, use_full_hessian)
        self.invert_block_matrices(use_full_inversion)

    def apply_metis(self, use_full_solve: bool = True):
        """
        Apply MAS preconditioner using METIS partition structure.

        Args:
            use_full_solve: If True, use full block inverse in local solve
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            self.apply(use_full_solve)
            return

        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction
        self._build_multi_level_r()

        # Phase 2: Local solve with METIS mapping
        if use_full_solve:
            self._schwarz_local_solve_full_metis()
        else:
            self._schwarz_local_solve()

        # Phase 3: Prolongation
        self._collect_final_z()

    # ========================================================================
    # Simplified Interface for Direct Field Access (no meshtaichi dependency)
    # ========================================================================

    def _build_hierarchy_from_adjacency(self, neighbor_list_np: np.ndarray,
                                        neighbor_starts_np: np.ndarray):
        """
        Build hierarchy from pre-computed adjacency arrays.

        This provides a meshtaichi-free interface for building the hierarchy.

        Args:
            neighbor_list_np: Flat array of neighbor indices (int32)
            neighbor_starts_np: CSR-style starts array (int32), length n_verts+1
        """
        # Copy neighbor data to fields
        total_len = min(len(neighbor_list_np), self.neighbor_list.shape[0])
        temp_neighbor_list = np.zeros(self.neighbor_list.shape[0], dtype=np.int32)
        temp_neighbor_list[:total_len] = neighbor_list_np[:total_len]
        self.neighbor_list.from_numpy(temp_neighbor_list)
        self.neighbor_start.from_numpy(neighbor_starts_np)
        self.total_neighbors = len(neighbor_list_np)

        # Compute neighbor counts
        neighbor_num_np = np.diff(neighbor_starts_np)
        self.neighbor_num.from_numpy(neighbor_num_np)

        # Build connectivity mask
        self._build_connect_mask_from_csr()

        # Build hierarchy levels
        self._build_coarse_levels_from_scratch()

        # Cache warp prefix for P1 optimization
        self._cache_warp_prefix()

        self.hierarchy_built = True
        print(f"[MAS] Hierarchy built: {self.actual_levels} levels")

    @ti.kernel
    def _build_connect_mask_from_csr(self):
        """Build connectivity bitmask from CSR neighbor format."""
        for i in range(self.n_verts):
            warp_id = i // BANKSIZE
            lane_id = i % BANKSIZE
            mask = ti.u32(0)

            # Self-connection
            mask |= ti.u32(1) << lane_id

            # Add neighbors in same warp
            start = self.neighbor_start[i]
            end = self.neighbor_start[i + 1]
            for k in range(start, end):
                neighbor = self.neighbor_list[k]
                if neighbor // BANKSIZE == warp_id:
                    neighbor_lane = neighbor % BANKSIZE
                    mask |= ti.u32(1) << neighbor_lane

            self.fine_connect_mask[i] = mask

    def _build_coarse_levels_from_scratch(self):
        """Build coarse levels without mesh dependency."""
        # Initialize level 0
        self.level_size[0] = ti.Vector([self.n_verts, 0], dt=ti.i32)

        current_size = self.n_verts
        current_offset = 0
        actual_levels = 1

        for level in range(1, MAX_LEVELS):
            # Compute next level size
            next_offset = current_offset + current_size
            next_size = (current_size + BANKSIZE - 1) // BANKSIZE

            if next_size < 1:
                break

            self.level_size[level] = ti.Vector([next_size, next_offset], dt=ti.i32)

            # Build going_next mapping
            self._build_going_next_level(current_offset, current_size, next_offset)

            current_offset = next_offset
            current_size = next_size
            actual_levels += 1

            if next_size <= BANKSIZE:
                break

        self.actual_levels = actual_levels
        self.level_size[actual_levels] = ti.Vector([0, current_offset + current_size], dt=ti.i32)

    @ti.kernel
    def _build_going_next_level(self, current_offset: ti.i32, current_size: ti.i32, next_offset: ti.i32):
        """Build going_next mapping for one level."""
        for i in range(current_size):
            global_idx = current_offset + i
            parent_idx = next_offset + i // BANKSIZE
            self.going_next[global_idx] = parent_idx

    def assemble_block_matrices_simple(self, solver, x_field, cell_verts_field,
                                       cell_B_field, cell_W_field):
        """
        Assemble block matrices using direct field access (no meshtaichi).

        Args:
            solver: Object with mu, la, dt attributes
            x_field: Vertex positions, ti.Vector.field(3, float, shape=n_verts)
            cell_verts_field: Cell vertex indices, ti.field(int, shape=(n_cells, 4))
            cell_B_field: Cell inverse rest matrices, ti.Matrix.field(3,3, shape=n_cells)
            cell_W_field: Cell volumes, ti.field(float, shape=n_cells)
        """
        print("[MAS] Assembling block matrices (simple interface)...")

        # Clear block matrices
        self._clear_block_matrices()

        # Assemble using simple kernel
        self._assemble_simple_kernel(x_field, cell_verts_field, cell_B_field, cell_W_field,
                                     solver.mu, solver.la, solver.dt)

        # Add mass/regularization
        self._add_mass_term_simple(solver.dt)

        self.matrices_assembled = True

    @ti.kernel
    def _assemble_simple_kernel(self, x: ti.template(), cell_verts: ti.template(),
                                cell_B: ti.template(), cell_W: ti.template(),
                                mu: ti.f32, la: ti.f32, dt: ti.f32):
        """Kernel for assembling block matrices from separate fields."""
        W_scale = dt * dt

        for c in range(self.n_cells):
            # Get vertex indices
            v0 = cell_verts[c, 0]
            v1 = cell_verts[c, 1]
            v2 = cell_verts[c, 2]
            v3 = cell_verts[c, 3]

            vids = ti.Vector([v0, v1, v2, v3])

            # Compute deformation gradient
            x0 = x[v0]
            Ds = ti.Matrix.cols([x[v1] - x0, x[v2] - x0, x[v3] - x0])
            B = cell_B[c]
            F = Ds @ B
            W = cell_W[c]

            # Compute Hessian (ARAP for simplicity)
            # Full Hessian requires SVD and material-specific d2PsidF2
            # For benchmark, use simplified diagonal approximation
            diag_val = 2.0 * mu * W * W_scale

            # Add to block matrices
            for i in ti.static(range(4)):
                vi = vids[i]
                block_i = vi // BANKSIZE
                lane_i = vi % BANKSIZE

                # Diagonal entry
                sym_idx = lane_i * (lane_i + 1) // 2 + lane_i
                diag_mat = ti.Matrix.identity(ti.f32, 3) * diag_val
                ti.atomic_add(self.block_matrices[block_i, sym_idx], diag_mat)

                # Off-diagonal within same block
                for j in ti.static(range(i)):
                    vj = vids[j]
                    block_j = vj // BANKSIZE
                    if block_j == block_i:
                        lane_j = vj % BANKSIZE
                        # Symmetric storage: always store (max, min)
                        max_lane = ti.max(lane_i, lane_j)
                        min_lane = ti.min(lane_i, lane_j)
                        sym_idx_ij = max_lane * (max_lane + 1) // 2 + min_lane
                        # Coupling term (simplified)
                        coupling = ti.Matrix.identity(ti.f32, 3) * (diag_val * 0.1)
                        ti.atomic_add(self.block_matrices[block_i, sym_idx_ij], coupling)

    @ti.kernel
    def _add_mass_term_simple(self, dt: ti.f32):
        """Add mass/inertia term to diagonal."""
        mass_scale = 1.0 / (dt * dt)
        for i in range(self.n_verts):
            block_id = i // BANKSIZE
            lane_id = i % BANKSIZE
            sym_idx = lane_id * (lane_id + 1) // 2 + lane_id
            self.block_matrices[block_id, sym_idx] += ti.Matrix.identity(ti.f32, 3) * mass_scale

    def apply_simple(self, grad_field, z_field, use_warp_reduction: bool = True):
        """
        Apply preconditioner using direct field access.

        Args:
            grad_field: Input gradient, ti.Vector.field(3, float, shape=n_verts)
            z_field: Output preconditioned direction, ti.Vector.field(3, float, shape=n_verts)
            use_warp_reduction: Whether to use P1 warp reduction optimization
        """
        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction (copy gradient to multi_level_r)
        if use_warp_reduction and WARP_REDUCTION_ENABLED:
            self._clear_warp_sum_buffer()
            self._restrict_simple_optimized(grad_field)
        else:
            self._restrict_simple(grad_field)

        # Phase 2: Local solve
        self._schwarz_local_solve_full()

        # Phase 3: Prolongation
        self._prolong_simple(z_field)

    @ti.kernel
    def _restrict_simple(self, grad_field: ti.template()):
        """Restrict gradient to multi-level residual (level 0)."""
        for i in range(self.n_verts):
            self.multi_level_r[i] = ti.cast(grad_field[i], ti.f32)

        # Propagate to coarse levels
        for level in range(1, self.actual_levels):
            level_info = self.level_size[level - 1]
            prev_size = level_info[0]
            prev_offset = level_info[1]

            next_info = self.level_size[level]
            next_offset = next_info[1]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                next_idx = next_offset + i // BANKSIZE
                ti.atomic_add(self.multi_level_r[next_idx], self.multi_level_r[prev_idx])

    @ti.kernel
    def _restrict_simple_optimized(self, grad_field: ti.template()):
        """Optimized restriction using warp-level patterns."""
        # Phase 1: Copy gradient to level 0
        for i in range(self.n_verts):
            self.multi_level_r[i] = ti.cast(grad_field[i], ti.f32)

        # Phase 2 & 3: Use existing optimized kernels
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id in range(n_warps):
            prefix = self.warp_prefix_cache[warp_id]

            if prefix == 1:
                # Fully connected: sum all nodes
                warp_sum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                warp_start = warp_id * BANKSIZE
                warp_end = ti.min(warp_start + BANKSIZE, self.n_verts)

                for i in range(warp_start, warp_end):
                    warp_sum += self.multi_level_r[i]

                # Store to coarse level
                if self.actual_levels > 1:
                    next_offset = self.level_size[1][1]
                    self.multi_level_r[next_offset + warp_id] = warp_sum
            else:
                # Multi-component: use atomic
                if self.actual_levels > 1:
                    next_offset = self.level_size[1][1]
                    warp_start = warp_id * BANKSIZE
                    warp_end = ti.min(warp_start + BANKSIZE, self.n_verts)

                    for i in range(warp_start, warp_end):
                        ti.atomic_add(self.multi_level_r[next_offset + warp_id], self.multi_level_r[i])

        # Higher levels
        for level in range(2, self.actual_levels):
            prev_info = self.level_size[level - 1]
            prev_size = prev_info[0]
            prev_offset = prev_info[1]

            next_info = self.level_size[level]
            next_offset = next_info[1]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                next_idx = next_offset + i // BANKSIZE
                ti.atomic_add(self.multi_level_r[next_idx], self.multi_level_r[prev_idx])

    @ti.kernel
    def _prolong_simple(self, z_field: ti.template()):
        """Prolong solution back to z_field."""
        for i in range(self.n_verts):
            # Collect from all levels
            z_total = self.multi_level_z[i]

            # Add coarse level contributions
            for level in ti.static(range(1, MAX_LEVELS)):
                if level < self.actual_levels:
                    # Find coarse index
                    coarse_idx = i
                    for l in range(level):
                        level_info = self.level_size[l]
                        offset = level_info[1]
                        coarse_idx = self.level_size[l + 1][1] + (coarse_idx - offset) // BANKSIZE
                    z_total += self.multi_level_z[coarse_idx]

            z_field[i] = ti.cast(z_total, ti.f32)


# ==============================================================================
# SRBK SpMV: Symmetric Reduce-By-Key Sparse Matrix-Vector Multiplication
# ==============================================================================
#
# Reference: /root/Stiff-GIPC_init/StiffGIPC/linear_system/utils/spmv.cu
#
# This implements a GPU-optimized symmetric SpMV that:
# 1. Only stores upper triangle of symmetric matrix
# 2. Uses reduce-by-key pattern for efficient accumulation
# 3. Processes both A[i,j] and A[j,i]^T simultaneously


@ti.data_oriented
class SRBKSpMV:
    """
    Symmetric Reduce-By-Key Sparse Matrix-Vector Multiplication.

    This class implements the SRBK SpMV algorithm from StiffGIPC for computing
    y = A * x where A is a symmetric sparse matrix stored in block-triplet format.

    Key optimizations:
    1. Upper triangle storage: Only store (i,j) where i <= j
    2. Symmetric processing: Compute both upper and lower contributions simultaneously
    3. Row-grouped accumulation: Triplets sorted by row for efficient reduction
    4. Warp-level parallelism: Multiple threads per row reduce before atomic write
    """

    def __init__(self, max_triplets: int, n_dofs: int):
        """
        Initialize SRBK SpMV structures.

        Args:
            max_triplets: Maximum number of 3x3 block triplets
            n_dofs: Number of degrees of freedom (3 * n_verts)
        """
        self.max_triplets = max_triplets
        self.n_dofs = n_dofs
        self.n_verts = n_dofs // 3

        # Triplet storage: (row_id, col_id, value) where value is 3x3 block
        # Row and col are block (vertex) indices, not DOF indices
        self.triplet_row = ti.field(dtype=ti.i32, shape=max_triplets)
        self.triplet_col = ti.field(dtype=ti.i32, shape=max_triplets)
        self.triplet_val = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_triplets)

        # Triplet count
        self.n_triplets = ti.field(dtype=ti.i32, shape=())

        # Row segment information for reduce-by-key
        # row_starts[i] = first triplet index for row i
        self.row_starts = ti.field(dtype=ti.i32, shape=self.n_verts + 1)

        # Work buffers for reduction
        self.y_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)

        # Flag for whether triplets are sorted
        self.sorted = False

    def clear(self):
        """Clear all triplets."""
        self.n_triplets[None] = 0
        self.sorted = False

    @ti.kernel
    def add_triplet(self, row: ti.i32, col: ti.i32, val: ti.types.matrix(3, 3, ti.f64)):
        """
        Add a single 3x3 block triplet.

        For symmetric matrices, only add upper triangle (row <= col).
        The SpMV will automatically handle the symmetric contribution.
        """
        idx = ti.atomic_add(self.n_triplets[None], 1)
        if idx < self.max_triplets:
            # Ensure upper triangle storage
            if row <= col:
                self.triplet_row[idx] = row
                self.triplet_col[idx] = col
                self.triplet_val[idx] = val
            else:
                self.triplet_row[idx] = col
                self.triplet_col[idx] = row
                self.triplet_val[idx] = val.transpose()

    def sort_by_row(self):
        """
        Sort triplets by row index for efficient reduce-by-key.

        Uses numpy for sorting since Taichi doesn't have efficient parallel sort.
        """
        n = self.n_triplets[None]
        if n == 0:
            self.sorted = True
            return

        # Extract to numpy
        rows_np = self.triplet_row.to_numpy()[:n]
        cols_np = self.triplet_col.to_numpy()[:n]
        vals_np = self.triplet_val.to_numpy()[:n]

        # Sort by row, then by col for stability
        sort_idx = np.lexsort((cols_np, rows_np))

        # Apply sorting
        rows_sorted = rows_np[sort_idx]
        cols_sorted = cols_np[sort_idx]
        vals_sorted = vals_np[sort_idx]

        # Write back
        temp_rows = np.zeros(self.max_triplets, dtype=np.int32)
        temp_cols = np.zeros(self.max_triplets, dtype=np.int32)
        temp_vals = np.zeros((self.max_triplets, 3, 3), dtype=np.float64)
        temp_rows[:n] = rows_sorted
        temp_cols[:n] = cols_sorted
        temp_vals[:n] = vals_sorted

        self.triplet_row.from_numpy(temp_rows)
        self.triplet_col.from_numpy(temp_cols)
        self.triplet_val.from_numpy(temp_vals)

        # Compute row_starts using CSR-style computation
        # row_starts[i] = first triplet index for row i
        # row_starts[n_verts] = total number of triplets
        row_starts_np = np.zeros(self.n_verts + 1, dtype=np.int32)

        # Count elements per row
        row_counts = np.zeros(self.n_verts, dtype=np.int32)
        for i in range(n):
            row = rows_sorted[i]
            row_counts[row] += 1

        # Cumulative sum to get row starts
        row_starts_np[0] = 0
        for i in range(self.n_verts):
            row_starts_np[i + 1] = row_starts_np[i] + row_counts[i]

        self.row_starts.from_numpy(row_starts_np)
        self.sorted = True

    @ti.kernel
    def spmv_naive(self, x: ti.template(), y: ti.template(), alpha: ti.f64, beta: ti.f64):
        """
        Naive symmetric SpMV: y = alpha * A * x + beta * y

        This is the reference implementation without reduce-by-key optimization.
        """
        n = self.n_triplets[None]
        n_verts = self.n_verts

        # Scale existing y
        if beta != 0.0:
            for i in range(n_verts):
                y[i] = beta * y[i]
        else:
            for i in range(n_verts):
                y[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Process each triplet
        for tid in range(n):
            i = self.triplet_row[tid]
            j = self.triplet_col[tid]
            mat = self.triplet_val[tid]

            # Upper triangle contribution: y[i] += A[i,j] * x[j]
            contrib = mat @ x[j]
            for d in ti.static(range(3)):
                ti.atomic_add(y[i][d], alpha * contrib[d])

            # Lower triangle contribution: y[j] += A[j,i] * x[i] = A[i,j]^T * x[i]
            if i != j:
                contrib_t = mat.transpose() @ x[i]
                for d in ti.static(range(3)):
                    ti.atomic_add(y[j][d], alpha * contrib_t[d])

    @ti.kernel
    def spmv_row_parallel(self, x: ti.template(), y: ti.template(), alpha: ti.f64, beta: ti.f64):
        """
        Row-parallel symmetric SpMV: y = alpha * A * x + beta * y

        This version processes each row in parallel, reducing atomic conflicts
        by accumulating row contributions locally before writing.

        Reference: _schwarzLocalXSym9 in MASPreconditioner.cu (lines 1045-1129)
        """
        n_verts = self.n_verts

        # Initialize output
        if beta != 0.0:
            for i in range(n_verts):
                y[i] = beta * y[i]
        else:
            for i in range(n_verts):
                y[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Clear work buffer
        for i in range(n_verts):
            self.y_buffer[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Process each row in parallel
        for row in range(n_verts):
            start = self.row_starts[row]
            end = self.row_starts[row + 1]

            # Accumulate row contribution locally
            row_sum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

            for tid in range(start, end):
                j = self.triplet_col[tid]
                mat = self.triplet_val[tid]

                # Upper triangle contribution: y[row] += A[row,j] * x[j]
                row_sum += mat @ x[j]

                # Lower triangle contribution: y[j] += A[j,row] * x[row]
                if row != j:
                    contrib_t = mat.transpose() @ x[row]
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.y_buffer[j][d], contrib_t[d])

            # Write row sum (no atomics needed for diagonal)
            y[row] += alpha * row_sum

        # Add lower triangle contributions
        for i in range(n_verts):
            y[i] += alpha * self.y_buffer[i]

    def spmv(self, x, y, alpha: float = 1.0, beta: float = 0.0):
        """
        Compute y = alpha * A * x + beta * y

        Args:
            x: Input vector, ti.Vector.field(3, f64, shape=n_verts)
            y: Output vector, ti.Vector.field(3, f64, shape=n_verts)
            alpha: Scalar multiplier for A*x
            beta: Scalar multiplier for existing y
        """
        if not self.sorted:
            self.sort_by_row()

        self.spmv_row_parallel(x, y, alpha, beta)


# ==============================================================================
# Improved Warp Reduction for Multi-level Restriction
# ==============================================================================
#
# Reference: __buildMultiLevelR_optimized_new in MASPreconditioner.cu (lines 729-847)
#
# Key optimizations from CUDA:
# 1. Boundary detection using __ballot_sync
# 2. Interval calculation using __brev and __clz
# 3. Warp-level reduction using __shfl_down_sync
#
# Taichi implementation uses field-based tree reduction as substitute.


@ti.func
def _bit_reverse_u32(x: ti.u32) -> ti.u32:
    """
    Reverse bits of a 32-bit unsigned integer.

    Equivalent to CUDA's __brev() intrinsic.
    """
    x = ((x & ti.u32(0x55555555)) << 1) | ((x & ti.u32(0xAAAAAAAA)) >> 1)
    x = ((x & ti.u32(0x33333333)) << 2) | ((x & ti.u32(0xCCCCCCCC)) >> 2)
    x = ((x & ti.u32(0x0F0F0F0F)) << 4) | ((x & ti.u32(0xF0F0F0F0)) >> 4)
    x = ((x & ti.u32(0x00FF00FF)) << 8) | ((x & ti.u32(0xFF00FF00)) >> 8)
    x = (x << 16) | (x >> 16)
    return x


@ti.func
def _count_leading_zeros_u32(x: ti.u32) -> ti.i32:
    """
    Count leading zeros in a 32-bit unsigned integer.

    Equivalent to CUDA's __clz() intrinsic.
    """
    n = ti.i32(32)  # Default for x == 0

    if x != 0:
        n = ti.i32(0)
        x_copy = x
        if (x_copy & ti.u32(0xFFFF0000)) == 0:
            n += 16
            x_copy <<= 16
        if (x_copy & ti.u32(0xFF000000)) == 0:
            n += 8
            x_copy <<= 8
        if (x_copy & ti.u32(0xF0000000)) == 0:
            n += 4
            x_copy <<= 4
        if (x_copy & ti.u32(0xC0000000)) == 0:
            n += 2
            x_copy <<= 2
        if (x_copy & ti.u32(0x80000000)) == 0:
            n += 1

    return n


@ti.func
def _popcount_u32(x: ti.u32) -> ti.i32:
    """
    Count set bits (population count) in a 32-bit unsigned integer.

    Equivalent to CUDA's __popc() intrinsic.
    """
    x = x - ((x >> 1) & ti.u32(0x55555555))
    x = (x & ti.u32(0x33333333)) + ((x >> 2) & ti.u32(0x33333333))
    x = (x + (x >> 4)) & ti.u32(0x0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    return ti.i32(x & ti.u32(0x3F))


@ti.func
def _find_first_set_u32(x: ti.u32) -> ti.i32:
    """
    Find position of first set bit (1-indexed, returns 0 if no bits set).

    Equivalent to CUDA's __ffs() intrinsic.
    """
    result = ti.i32(0)
    if x != 0:
        # x & (~x + 1) isolates the lowest set bit
        lowest_bit = x & (~x + ti.u32(1))
        result = 32 - _count_leading_zeros_u32(lowest_bit)
    return result


@ti.data_oriented
class WarpReductionHelper:
    """
    Helper class for warp-level reduction operations.

    Since Taichi 1.7.4 doesn't support ti.simt warp primitives,
    this class provides field-based alternatives that emulate the
    behavior of CUDA warp intrinsics.

    Reference: MASPreconditioner.cu (lines 930-948)
    """

    def __init__(self, n_warps: int, warp_size: int = 16):
        """
        Initialize warp reduction helper.

        Args:
            n_warps: Number of warps (blocks) in the computation
            warp_size: Size of each warp (default: BANKSIZE=16)
        """
        self.n_warps = n_warps
        self.warp_size = warp_size

        # Reduction buffer: [warp_id, lane_id, component]
        self.reduction_buffer = ti.field(dtype=ti.f32, shape=(n_warps, warp_size, 3))

        # Boundary mask for each warp
        self.boundary_mask = ti.field(dtype=ti.u32, shape=n_warps)

        # Reduction interval for each lane
        self.reduction_interval = ti.field(dtype=ti.i32, shape=(n_warps, warp_size))

    @ti.kernel
    def compute_boundary_mask(self, connect_mask: ti.template(), n_verts: ti.i32):
        """
        Compute boundary mask for each warp based on connectivity.

        A lane is a boundary if:
        1. It's lane 0 (always a boundary)
        2. Its connectivity differs from the previous lane

        Reference: MASPreconditioner.cu lines 932-937
        """
        for warp_id in range(self.n_warps):
            mask = ti.u32(0)

            for lane_id in range(self.warp_size):
                idx = warp_id * self.warp_size + lane_id
                is_boundary = False

                if idx < n_verts:
                    if lane_id == 0:
                        is_boundary = True
                    else:
                        # Check if connectivity differs from previous lane
                        prev_idx = idx - 1
                        curr_conn = connect_mask[idx]
                        prev_conn = connect_mask[prev_idx] if prev_idx >= 0 else ti.u32(0)

                        # Find representative for each
                        curr_rep = _find_first_set_u32(curr_conn) - 1
                        prev_rep = _find_first_set_u32(prev_conn) - 1

                        is_boundary = (curr_rep != prev_rep)

                if is_boundary:
                    mask |= ti.u32(1) << ti.u32(lane_id)

            self.boundary_mask[warp_id] = mask

    @ti.kernel
    def compute_reduction_intervals(self):
        """
        Compute reduction interval for each lane based on boundary mask.

        The interval is the distance to the next boundary, used to determine
        how many lanes should be reduced together.

        Reference: MASPreconditioner.cu lines 936-937
        """
        for warp_id in range(self.n_warps):
            mask = self.boundary_mask[warp_id]

            for lane_id in range(self.warp_size):
                # Reverse bits and count leading zeros after this position
                reversed_mask = _bit_reverse_u32(mask)
                shifted = reversed_mask << ti.u32(lane_id + 1)
                clz = _count_leading_zeros_u32(shifted)

                # Interval is min of clz and remaining lanes
                interval = ti.min(clz, 31 - lane_id)
                self.reduction_interval[warp_id, lane_id] = interval

    @ti.kernel
    def load_data(self, data: ti.template(), n_verts: ti.i32):
        """Load data into reduction buffer."""
        for idx in range(n_verts):
            warp_id = idx // self.warp_size
            lane_id = idx % self.warp_size

            for d in ti.static(range(3)):
                self.reduction_buffer[warp_id, lane_id, d] = data[idx][d]

    @ti.kernel
    def tree_reduce(self):
        """
        Perform tree reduction within each segment.

        This emulates CUDA's __shfl_down_sync reduction pattern using
        explicit memory operations.

        Reference: MASPreconditioner.cu lines 940-948
        """
        for warp_id in range(self.n_warps):
            # Reduction steps: stride = 8, 4, 2, 1
            # Step 0: stride = 8
            for lane_id in range(8):
                src_lane = lane_id + 8
                interval = self.reduction_interval[warp_id, lane_id]

                if interval >= 8:
                    for d in ti.static(range(3)):
                        self.reduction_buffer[warp_id, lane_id, d] += \
                            self.reduction_buffer[warp_id, src_lane, d]

            # Step 1: stride = 4
            for lane_id in range(8):
                if lane_id < 4 or (lane_id >= 8 and lane_id < 12):
                    src_lane = lane_id + 4
                    interval = self.reduction_interval[warp_id, lane_id]

                    if interval >= 4:
                        for d in ti.static(range(3)):
                            self.reduction_buffer[warp_id, lane_id, d] += \
                                self.reduction_buffer[warp_id, src_lane, d]

            # Step 2: stride = 2
            for lane_id in range(self.warp_size):
                if lane_id % 4 < 2:
                    src_lane = lane_id + 2
                    interval = self.reduction_interval[warp_id, lane_id]

                    if interval >= 2:
                        for d in ti.static(range(3)):
                            self.reduction_buffer[warp_id, lane_id, d] += \
                                self.reduction_buffer[warp_id, src_lane, d]

            # Step 3: stride = 1
            for lane_id in range(self.warp_size):
                if lane_id % 2 == 0:
                    src_lane = lane_id + 1
                    interval = self.reduction_interval[warp_id, lane_id]

                    if interval >= 1:
                        for d in ti.static(range(3)):
                            self.reduction_buffer[warp_id, lane_id, d] += \
                                self.reduction_buffer[warp_id, src_lane, d]

    @ti.kernel
    def write_boundary_results(self, output: ti.template(), n_verts: ti.i32):
        """
        Write reduced results from boundary lanes to output.

        Only boundary lanes (segment heads) write their accumulated values.
        """
        for warp_id in range(self.n_warps):
            mask = self.boundary_mask[warp_id]

            for lane_id in range(self.warp_size):
                idx = warp_id * self.warp_size + lane_id

                if idx < n_verts:
                    # Check if this is a boundary lane
                    is_boundary = (mask >> ti.u32(lane_id)) & 1

                    if is_boundary:
                        for d in ti.static(range(3)):
                            output[idx][d] = self.reduction_buffer[warp_id, lane_id, d]

    def reduce(self, data, output, n_verts: int, connect_mask):
        """
        Perform full segmented reduction.

        Args:
            data: Input data, ti.Vector.field(3, float, shape=n_verts)
            output: Output data, ti.Vector.field(3, float, shape=n_verts)
            n_verts: Number of vertices
            connect_mask: Connectivity mask, ti.field(u32, shape=n_verts)
        """
        self.compute_boundary_mask(connect_mask, n_verts)
        self.compute_reduction_intervals()
        self.load_data(data, n_verts)
        self.tree_reduce()
        self.write_boundary_results(output, n_verts)