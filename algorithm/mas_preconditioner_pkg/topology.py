"""
Mesh topology and neighbor building for MAS Preconditioner.

This module handles:
- Neighbor list construction from mesh cell-vertex connectivity
- Connectivity mask building at Level 0 and coarse levels
- Connected component detection and cluster assignment
- Collision connection augmentation

Reference: MASPreconditioner.cu (CUDA reference implementation)
"""

import taichi as ti
import numpy as np

from .constants import BANKSIZE, MAX_LEVELS, MAX_NEIGHBORS_PER_VERTEX


# ==============================================================================
# Standalone Taichi Functions for topology operations
# ==============================================================================

@ti.func
def popcount(x: ti.u32) -> ti.i32:
    """
    Count number of set bits in a 32-bit integer.

    Uses optimized parallel bit counting (SWAR algorithm).
    """
    temp = x
    temp = temp - ((temp >> 1) & ti.u32(0x55555555))
    temp = (temp & ti.u32(0x33333333)) + ((temp >> 2) & ti.u32(0x33333333))
    temp = (temp + (temp >> 4)) & ti.u32(0x0F0F0F0F)
    count = ti.i32((temp * ti.u32(0x01010101)) >> 24)
    return count


@ti.func
def find_first_set(x: ti.u32) -> ti.i32:
    """
    Find position of first set bit (0-indexed), or -1 if none.

    Uses De Bruijn sequence for O(1) bit position lookup.
    """
    pos = -1
    if x != 0:
        # Isolate the lowest set bit
        isolated = x & (~x + ti.u32(1))
        # Use De Bruijn sequence for O(1) bit position lookup
        debruijn = ti.u32(0x077CB531)
        index = (isolated * debruijn) >> 27
        # Lookup table embedded in computation
        lookup = ti.Vector([
            0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
            31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
        ], dt=ti.i32)
        pos = lookup[ti.i32(index)]
    return pos


@ti.func
def lanemask_lt(lane_id: ti.i32) -> ti.u32:
    """Return bitmask of lanes less than lane_id."""
    return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)


# ==============================================================================
# TopologyBuilder Mixin Class
# ==============================================================================

class TopologyMixin:
    """
    Mixin class providing topology-related methods for MASPreconditioner.

    This class is designed to be mixed into the main MASPreconditioner class
    to provide neighbor list construction and connectivity mask building.
    """

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
    # Connectivity Mask Building
    # ========================================================================

    @ti.func
    def _popcount(self, x: ti.u32) -> ti.i32:
        """Count number of set bits in a 32-bit integer."""
        return popcount(x)

    @ti.func
    def _find_first_set(self, x: ti.u32) -> ti.i32:
        """Find position of first set bit (0-indexed), or -1 if none."""
        return find_first_set(x)

    @ti.func
    def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
        """Return bitmask of lanes less than lane_id."""
        return lanemask_lt(lane_id)

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
            visited = ti.u32(1) << ti.u32(lane_id)

            # Continue until all reachable nodes are included
            max_iter = BANKSIZE
            for _ in range(max_iter):
                # Find nodes in connect_mask that we haven't visited yet
                todo = visited ^ connect_mask

                if todo == 0:
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

            # Count elected representatives
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This node is the representative of its connected component
                ti.atomic_add(self.prefix_original[warp_id], 1)

    @ti.kernel
    def _find_cluster_representatives(self):
        """
        Find cluster representatives (elected nodes) and build elected_mask.

        A node is elected if it has the lowest lane ID in its connected component.
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Reset elected mask
        for w in range(n_warps):
            self.elected_mask[w] = ti.u32(0)

        # Build elected mask
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is the representative of its cluster
                ti.atomic_or(self.elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))

    # ========================================================================
    # Collision Connection Building
    # ========================================================================

    def build_collision_connection(self, collision_pairs):
        """
        Augment connectivity masks at collision points.

        CUDA Reference: _buildCollisionConnection_new() (MASPreconditioner.cu lines 1132-1542)

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
        """
        for pair_idx in range(n_pairs):
            cp_vid = self.collision_pairs[pair_idx]

            # Get coarse node IDs for each vertex in the collision pair
            cp_coarse = ti.Vector([-1, -1, -1, -1], dt=ti.i32)
            for i in ti.static(range(4)):
                fine_vid = cp_vid[i]
                if fine_vid >= 0 and fine_vid < self.n_verts:
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
                        my_lane = my_id % BANKSIZE
                        ot_lane = ot_id % BANKSIZE

                        ti.atomic_or(self.next_connect_mask[my_id], ti.u32(1) << ti.u32(ot_lane))
                        ti.atomic_or(self.next_connect_mask[ot_id], ti.u32(1) << ti.u32(my_lane))

    # ========================================================================
    # Cluster Assignment
    # ========================================================================

    @ti.kernel
    def _assign_cluster_ids(self, level_offset: ti.i32):
        """
        Assign coarse-level cluster IDs based on elected representatives.
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is an elected node - compute its cluster ID
                base_id = self.prefix_sum_original[warp_id]
                elected_in_warp = self.elected_mask[warp_id]
                local_offset = self._popcount(elected_in_warp & self._lanemask_lt(lane_id))

                cluster_id = base_id + local_offset
                self.cluster_id[idx] = cluster_id
                self.coarse_space_tables[0, idx] = cluster_id
                self.going_next[idx] = level_offset + cluster_id
            else:
                # Mark as non-elected
                self.cluster_id[idx] = -1

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
        """
        for idx in range(self.n_verts):
            current_idx = idx
            for level in range(self.level_num - 1):
                next_idx = self.going_next[current_idx]
                self.aggregation_table[idx][level] = next_idx
                current_idx = next_idx

    # ========================================================================
    # Coarse Level Connectivity
    # ========================================================================

    @ti.kernel
    def _build_connect_mask_lx(self, level_offset: ti.i32, level_size: ti.i32):
        """
        Build connectivity mask for coarse levels (Level 2+).
        """
        for idx in range(level_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Start with self-connectivity
            connect_mask = ti.u32(1) << ti.u32(lane_id)
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
        count = 0

        for idx in range(level_size):
            lane_id = idx % BANKSIZE
            connect_mask = self.next_connect_mask[idx]
            prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if prefix == 0:
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

        # Count representatives per warp
        for idx in range(src_size):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.next_connect_mask[idx]
            prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if prefix == 0:
                ti.atomic_add(self.prefix_original[warp_id], 1)
                ti.atomic_or(self.elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))

    @ti.kernel
    def _update_going_next_lx(self, src_offset: ti.i32, src_size: ti.i32,
                               dst_offset: ti.i32, level: ti.i32):
        """
        Update going_next mapping for coarse level nodes.
        """
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
                    self.going_next[global_idx] = self.going_next[rep_global]

    # ========================================================================
    # Main Hierarchy Building Method
    # ========================================================================

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
