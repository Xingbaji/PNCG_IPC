"""
METIS Integration Module: METIS-based node reordering for improved MAS quality.

This module implements the CEMAS (Connectivity-Enhanced MAS) approach from
Stiff-GIPC that uses METIS graph partitioning to group topologically connected
vertices together in blocks for better preconditioner quality.

Reference: MASPreconditioner.cu METIS integration
"""

import taichi as ti
import numpy as np
from .constants import BANKSIZE, MAX_LEVELS


class METISMixin:
    """
    Mixin class providing METIS-based reordering and hierarchy construction.

    METIS partitions the mesh graph so that topologically connected vertices
    are grouped together, improving the quality of subdomain Hessian blocks.

    Required attributes from main class:
        - n_verts: int
        - mesh: meshtaichi mesh
        - neighbor_num, neighbor_start, neighbor_list: neighbor fields
        - fine_connect_mask: ti.field(u32)
        - prefix_original, prefix_sum_original: ti.field(i32)
        - elected_mask: ti.field(u32)
        - going_next: ti.field(i32)
        - coarse_space_table_0: ti.field(i32)
        - level_size: ti.Vector.field(2, i32)
        - n_blocks_per_level: ti.field(i32)
        - multi_level_r, multi_level_z: ti.Vector.field(3, f32)
        - block_matrices, inv_block_matrices: matrix fields
    """

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

        Returns:
            np.ndarray: Cell array of shape (n_cells, 4) or None if extraction fails
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

    # ========================================================================
    # METIS-specific Connectivity and Hierarchy Kernels
    # ========================================================================

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

    # ========================================================================
    # METIS Hierarchy Building
    # ========================================================================

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

    # ========================================================================
    # METIS Apply and Rebuild Methods
    # ========================================================================

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
