"""
Final Optimized LBVH - Best performing implementation.

Combines the best optimizations:
1. Packed AABB storage (vec6): [lx, ly, lz, ux, uy, uz] for cache efficiency
2. Packed node topology (vec4<u32>): [parent, left, right, element]
3. Fused kernels and hierarchical scene bounds reduction
4. Uses Taichi's parallel_sort for stable sorting

Based on algorithm/lbvh_final.py with precision support added.
"""

import taichi as ti
from taichi.algorithms import parallel_sort

from ...core.precision import PrecisionType, PrecisionMixin


@ti.data_oriented
class LBVH_Final(PrecisionMixin):
    """
    Final optimized LBVH base class with packed storage.

    Key optimizations:
    1. Packed AABB: vec6 [lx, ly, lz, ux, uy, uz] for cache efficiency
    2. Packed node topology: vec4<u32> [parent, left, right, element]
    3. Fused kernels: Reduce kernel launch overhead
    4. Hierarchical reduction: Block-level scene bounds computation
    5. Taichi parallel_sort for stable sorting
    """

    def __init__(self, max_primitives: int, precision: PrecisionType = 'f32'):
        self.init_precision(precision)
        self.max_primitives = max_primitives
        self.num_nodes = 2 * max_primitives - 1

        float_type = self.cfg.float_type

        # Packed AABB: [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
        self.aabb = ti.Vector.field(6, dtype=float_type, shape=self.num_nodes)

        # Packed node topology: [parent, left, right, element]
        self.node_data = ti.Vector.field(4, dtype=ti.u32, shape=self.num_nodes)

        # Temp buffer for AABB reordering
        self.temp_aabb = ti.Vector.field(6, dtype=float_type, shape=max_primitives)

        # Morton codes and sorted indices
        self.morton_codes = ti.field(dtype=ti.u64, shape=max_primitives)
        self.sorted_indices = ti.field(dtype=ti.u32, shape=max_primitives)

        # Flags for bottom-up traversal synchronization
        self.flags = ti.field(dtype=ti.u32, shape=max_primitives)

        # Scene bounds (hierarchical reduction)
        self.scene_bounds = ti.Vector.field(6, dtype=float_type, shape=())
        self.BLOCK_SIZE = 256
        n_blocks = (max_primitives + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        self.block_bounds = ti.Vector.field(6, dtype=float_type, shape=max(n_blocks, 1))

        # State tracking
        self.num_primitives = ti.field(dtype=ti.i32, shape=())
        self.INVALID = 0xFFFFFFFF
        self.tree_built = False

    # ==================== Accessors ====================

    @ti.func
    def get_parent(self, idx: ti.i32) -> ti.u32:
        return self.node_data[idx][0]

    @ti.func
    def get_left(self, idx: ti.i32) -> ti.u32:
        return self.node_data[idx][1]

    @ti.func
    def get_right(self, idx: ti.i32) -> ti.u32:
        return self.node_data[idx][2]

    @ti.func
    def get_element(self, idx: ti.i32) -> ti.u32:
        return self.node_data[idx][3]

    @ti.func
    def set_node(self, idx: ti.i32, parent: ti.u32, left: ti.u32,
                 right: ti.u32, element: ti.u32):
        self.node_data[idx] = ti.Vector([parent, left, right, element], dt=ti.u32)

    @ti.func
    def is_leaf(self, idx: ti.i32, n_leaves: ti.i32) -> bool:
        return idx >= n_leaves - 1

    # ==================== Morton Code ====================

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        """Expand 10-bit integer to 30 bits by inserting 2 zeros between each bit."""
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    @ti.func
    def morton_code_3d(self, x: ti.template(), y: ti.template(), z: ti.template()) -> ti.u32:
        """Compute 30-bit Morton code from normalized [0,1] coordinates."""
        resolution = 1024.0
        xi = ti.cast(ti.min(ti.max(x * resolution, 0.0), 1023.0), ti.u32)
        yi = ti.cast(ti.min(ti.max(y * resolution, 0.0), 1023.0), ti.u32)
        zi = ti.cast(ti.min(ti.max(z * resolution, 0.0), 1023.0), ti.u32)
        return (self.expand_bits(xi) << 2) | (self.expand_bits(yi) << 1) | self.expand_bits(zi)

    # ==================== AABB Operations ====================

    @ti.func
    def aabb_merge(self, idx1: ti.i32, idx2: ti.i32, dst: ti.i32):
        """Merge two AABBs into destination."""
        a1 = self.aabb[idx1]
        a2 = self.aabb[idx2]
        lower = ti.min(ti.Vector([a1[0], a1[1], a1[2]]),
                       ti.Vector([a2[0], a2[1], a2[2]]))
        upper = ti.max(ti.Vector([a1[3], a1[4], a1[5]]),
                       ti.Vector([a2[3], a2[4], a2[5]]))
        self.aabb[dst] = ti.Vector([lower[0], lower[1], lower[2],
                                    upper[0], upper[1], upper[2]])

    # ==================== Tree Construction ====================

    @ti.func
    def common_upper_bits(self, lhs: ti.u64, rhs: ti.u64) -> ti.i32:
        """Count common upper bits (leading zeros in XOR)."""
        xor_val = lhs ^ rhs
        count = 0
        if xor_val == 0:
            count = 64
        else:
            if (xor_val >> 32) == 0:
                count += 32
                xor_val <<= 32
            if (xor_val >> 48) == 0:
                count += 16
                xor_val <<= 16
            if (xor_val >> 56) == 0:
                count += 8
                xor_val <<= 8
            if (xor_val >> 60) == 0:
                count += 4
                xor_val <<= 4
            if (xor_val >> 62) == 0:
                count += 2
                xor_val <<= 2
            if (xor_val >> 63) == 0:
                count += 1
        return count

    @ti.func
    def determine_range(self, idx: ti.i32, num_leaves: ti.i32) -> ti.math.ivec2:
        """Determine the range of leaves covered by internal node idx."""
        first = 0
        last = num_leaves - 1

        if idx != 0:
            self_code = self.morton_codes[idx]
            L_delta = self.common_upper_bits(self_code, self.morton_codes[idx - 1])
            R_delta = self.common_upper_bits(self_code, self.morton_codes[idx + 1])

            d = 1 if R_delta > L_delta else -1
            delta_min = ti.min(L_delta, R_delta)

            l_max = 2
            i_tmp = idx + d * l_max
            delta = -1
            if 0 <= i_tmp < num_leaves:
                delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])

            while delta > delta_min:
                l_max <<= 1
                i_tmp = idx + d * l_max
                delta = -1
                if 0 <= i_tmp < num_leaves:
                    delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])

            l = 0
            t = l_max >> 1
            while t > 0:
                i_tmp = idx + (l + t) * d
                delta = -1
                if 0 <= i_tmp < num_leaves:
                    delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])
                if delta > delta_min:
                    l += t
                t >>= 1

            jdx = idx + l * d
            first = ti.min(idx, jdx)
            last = ti.max(idx, jdx)

        return ti.math.ivec2(first, last)

    @ti.func
    def find_split(self, first: ti.i32, last: ti.i32) -> ti.i32:
        """Find split position for range [first, last]."""
        first_code = self.morton_codes[first]
        last_code = self.morton_codes[last]
        split = (first + last) >> 1

        if first_code != last_code:
            delta_node = self.common_upper_bits(first_code, last_code)
            split = first
            stride = last - first
            while stride > 1:
                stride = (stride + 1) >> 1
                middle = split + stride
                if middle < last:
                    delta = self.common_upper_bits(first_code, self.morton_codes[middle])
                    if delta > delta_node:
                        split = middle

        return split


@ti.data_oriented
class LBVH_Triangles_Final(LBVH_Final):
    """Final optimized LBVH for triangle primitives."""

    def __init__(self, max_triangles: int, precision: PrecisionType = 'f32'):
        super().__init__(max_triangles, precision)

    @ti.kernel
    def _compute_leaf_aabbs_and_bounds(self, vertices: ti.template(),
                                        triangles: ti.template(), n: ti.i32):
        """Fused: Compute leaf AABBs and block-level scene bounds."""
        self.num_primitives[None] = n
        n_blocks = (n + 255) // 256

        # Initialize block bounds
        for bid in range(n_blocks):
            self.block_bounds[bid] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])

        ti.sync()

        # Compute leaf AABBs
        for i in range(n):
            t0, t1, t2 = triangles[i, 0], triangles[i, 1], triangles[i, 2]
            v0, v1, v2 = vertices[t0], vertices[t1], vertices[t2]

            lower = ti.min(ti.min(v0, v1), v2)
            upper = ti.max(ti.max(v0, v1), v2)

            leaf_idx = i + n - 1
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

            # Update block bounds atomically
            bid = i // 256
            ti.atomic_min(self.block_bounds[bid][0], lower[0])
            ti.atomic_min(self.block_bounds[bid][1], lower[1])
            ti.atomic_min(self.block_bounds[bid][2], lower[2])
            ti.atomic_max(self.block_bounds[bid][3], upper[0])
            ti.atomic_max(self.block_bounds[bid][4], upper[1])
            ti.atomic_max(self.block_bounds[bid][5], upper[2])

    @ti.kernel
    def _reduce_scene_bounds(self, n: ti.i32):
        """Reduce block bounds to scene bounds."""
        n_blocks = (n + 255) // 256
        self.scene_bounds[None] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])
        ti.sync()

        for bid in range(n_blocks):
            bb = self.block_bounds[bid]
            ti.atomic_min(self.scene_bounds[None][0], bb[0])
            ti.atomic_min(self.scene_bounds[None][1], bb[1])
            ti.atomic_min(self.scene_bounds[None][2], bb[2])
            ti.atomic_max(self.scene_bounds[None][3], bb[3])
            ti.atomic_max(self.scene_bounds[None][4], bb[4])
            ti.atomic_max(self.scene_bounds[None][5], bb[5])

    @ti.kernel
    def _compute_morton_codes(self, n: ti.i32):
        """Compute Morton codes from leaf AABB centers."""
        bounds = self.scene_bounds[None]
        scene_lower = ti.Vector([bounds[0], bounds[1], bounds[2]])
        scene_size = ti.max(ti.Vector([bounds[3] - bounds[0],
                                       bounds[4] - bounds[1],
                                       bounds[5] - bounds[2]]),
                           ti.Vector([1e-10, 1e-10, 1e-10]))
        inv_size = 1.0 / scene_size

        for i in range(n):
            leaf_idx = i + n - 1
            a = self.aabb[leaf_idx]
            center = ti.Vector([(a[0] + a[3]) * 0.5,
                               (a[1] + a[4]) * 0.5,
                               (a[2] + a[5]) * 0.5])
            normalized = (center - scene_lower) * inv_size
            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            # Pack Morton code (upper 32 bits) + index (lower 32 bits)
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = ti.u32(i)

    def _sort_morton_codes(self, n: int):
        """Sort Morton codes using Taichi's parallel_sort."""
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def _reorder_leaf_aabbs(self, n: ti.i32):
        """Reorder leaf AABBs to match sorted order."""
        for i in range(n):
            orig_idx = n - 1 + ti.cast(self.sorted_indices[i], ti.i32)
            self.temp_aabb[i] = self.aabb[orig_idx]

        ti.sync()

        for i in range(n):
            leaf_idx = n - 1 + i
            self.aabb[leaf_idx] = self.temp_aabb[i]

    @ti.kernel
    def _init_nodes(self, n: ti.i32):
        """Initialize all nodes."""
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n):
            if i < n - 1:
                self.set_node(i, INVALID, INVALID, INVALID, INVALID)
            leaf_idx = i + n - 1
            self.set_node(leaf_idx, INVALID, INVALID, INVALID, self.sorted_indices[i])

    @ti.kernel
    def _build_tree_topology(self, n: ti.i32):
        """Build tree topology - set left/right children and parent pointers."""
        for i in range(n - 1):
            range_ij = self.determine_range(i, n)
            first, last = range_ij[0], range_ij[1]
            gamma = self.find_split(first, last)

            left_child = gamma
            right_child = gamma + 1

            if ti.min(first, last) == gamma:
                left_child += n - 1
            if ti.max(first, last) == gamma + 1:
                right_child += n - 1

            # Set children for this internal node
            self.node_data[i][1] = ti.u32(left_child)
            self.node_data[i][2] = ti.u32(right_child)

            # Set parent for children
            self.node_data[left_child][0] = ti.u32(i)
            self.node_data[right_child][0] = ti.u32(i)

    @ti.kernel
    def _compute_internal_aabbs(self, n: ti.i32):
        """Bottom-up AABB computation with atomic synchronization."""
        INVALID = ti.u32(0xFFFFFFFF)

        # Reset flags
        for i in range(n - 1):
            self.flags[i] = INVALID

        ti.sync()

        # Bottom-up traversal from leaves
        for i in range(n):
            leaf_idx = i + n - 1
            parent_u32 = self.get_parent(leaf_idx)

            while parent_u32 != INVALID:
                parent = ti.cast(parent_u32, ti.i32)
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    break  # First to arrive, wait for sibling

                # Second to arrive, merge children AABBs
                left = ti.cast(self.get_left(parent), ti.i32)
                right = ti.cast(self.get_right(parent), ti.i32)
                self.aabb_merge(left, right, parent)
                ti.simt.block.mem_sync()

                parent_u32 = self.get_parent(parent)

    @ti.kernel
    def _refit_leaf_aabbs(self, vertices: ti.template(),
                          triangles: ti.template(), n: ti.i32):
        """Update leaf AABBs from current vertex positions."""
        for i in range(n):
            leaf_idx = i + n - 1
            tri_idx = ti.cast(self.get_element(leaf_idx), ti.i32)

            t0, t1, t2 = triangles[tri_idx, 0], triangles[tri_idx, 1], triangles[tri_idx, 2]
            v0, v1, v2 = vertices[t0], vertices[t1], vertices[t2]

            lower = ti.min(ti.min(v0, v1), v2)
            upper = ti.max(ti.max(v0, v1), v2)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    def build(self, vertices, triangles, n_triangles: int):
        """Build BVH from scratch."""
        if n_triangles < 1:
            return

        # 1. Compute leaf AABBs and block bounds (fused)
        self._compute_leaf_aabbs_and_bounds(vertices, triangles, n_triangles)

        # 2. Reduce to scene bounds
        self._reduce_scene_bounds(n_triangles)

        # 3. Compute Morton codes
        self._compute_morton_codes(n_triangles)

        # 4. Sort
        self._sort_morton_codes(n_triangles)

        # 5. Reorder leaf AABBs
        self._reorder_leaf_aabbs(n_triangles)

        # 6. Initialize nodes
        self._init_nodes(n_triangles)

        # 7. Build tree topology
        self._build_tree_topology(n_triangles)

        # 8. Compute internal AABBs
        self._compute_internal_aabbs(n_triangles)

        self.tree_built = True

    def refit(self, vertices, triangles, n_triangles: int):
        """Refit BVH (update AABBs without rebuilding structure)."""
        if not self.tree_built or self.num_primitives[None] != n_triangles:
            self.build(vertices, triangles, n_triangles)
            return

        self._refit_leaf_aabbs(vertices, triangles, n_triangles)
        self._compute_internal_aabbs(n_triangles)


@ti.data_oriented
class LBVH_Edges_Final(LBVH_Final):
    """Final optimized LBVH for edge primitives."""

    def __init__(self, max_edges: int, precision: PrecisionType = 'f32'):
        super().__init__(max_edges, precision)

    @ti.kernel
    def _compute_leaf_aabbs_and_bounds(self, vertices: ti.template(),
                                        edges: ti.template(), n: ti.i32):
        """Fused: Compute leaf AABBs and block-level scene bounds."""
        self.num_primitives[None] = n
        n_blocks = (n + 255) // 256

        for bid in range(n_blocks):
            self.block_bounds[bid] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])

        ti.sync()

        for i in range(n):
            e0, e1 = edges[i, 0], edges[i, 1]
            v0, v1 = vertices[e0], vertices[e1]

            lower = ti.min(v0, v1)
            upper = ti.max(v0, v1)

            leaf_idx = i + n - 1
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

            bid = i // 256
            ti.atomic_min(self.block_bounds[bid][0], lower[0])
            ti.atomic_min(self.block_bounds[bid][1], lower[1])
            ti.atomic_min(self.block_bounds[bid][2], lower[2])
            ti.atomic_max(self.block_bounds[bid][3], upper[0])
            ti.atomic_max(self.block_bounds[bid][4], upper[1])
            ti.atomic_max(self.block_bounds[bid][5], upper[2])

    @ti.kernel
    def _reduce_scene_bounds(self, n: ti.i32):
        """Reduce block bounds to scene bounds."""
        n_blocks = (n + 255) // 256
        self.scene_bounds[None] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])
        ti.sync()

        for bid in range(n_blocks):
            bb = self.block_bounds[bid]
            ti.atomic_min(self.scene_bounds[None][0], bb[0])
            ti.atomic_min(self.scene_bounds[None][1], bb[1])
            ti.atomic_min(self.scene_bounds[None][2], bb[2])
            ti.atomic_max(self.scene_bounds[None][3], bb[3])
            ti.atomic_max(self.scene_bounds[None][4], bb[4])
            ti.atomic_max(self.scene_bounds[None][5], bb[5])

    @ti.kernel
    def _compute_morton_codes(self, n: ti.i32):
        """Compute Morton codes from leaf AABB centers."""
        bounds = self.scene_bounds[None]
        scene_lower = ti.Vector([bounds[0], bounds[1], bounds[2]])
        scene_size = ti.max(ti.Vector([bounds[3] - bounds[0],
                                       bounds[4] - bounds[1],
                                       bounds[5] - bounds[2]]),
                           ti.Vector([1e-10, 1e-10, 1e-10]))
        inv_size = 1.0 / scene_size

        for i in range(n):
            leaf_idx = i + n - 1
            a = self.aabb[leaf_idx]
            center = ti.Vector([(a[0] + a[3]) * 0.5,
                               (a[1] + a[4]) * 0.5,
                               (a[2] + a[5]) * 0.5])
            normalized = (center - scene_lower) * inv_size
            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = ti.u32(i)

    def _sort_morton_codes(self, n: int):
        """Sort Morton codes using Taichi's parallel_sort."""
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def _reorder_leaf_aabbs(self, n: ti.i32):
        """Reorder leaf AABBs to match sorted order."""
        for i in range(n):
            orig_idx = n - 1 + ti.cast(self.sorted_indices[i], ti.i32)
            self.temp_aabb[i] = self.aabb[orig_idx]

        ti.sync()

        for i in range(n):
            leaf_idx = n - 1 + i
            self.aabb[leaf_idx] = self.temp_aabb[i]

    @ti.kernel
    def _init_nodes(self, n: ti.i32):
        """Initialize all nodes."""
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n):
            if i < n - 1:
                self.set_node(i, INVALID, INVALID, INVALID, INVALID)
            leaf_idx = i + n - 1
            self.set_node(leaf_idx, INVALID, INVALID, INVALID, self.sorted_indices[i])

    @ti.kernel
    def _build_tree_topology(self, n: ti.i32):
        """Build tree topology."""
        for i in range(n - 1):
            range_ij = self.determine_range(i, n)
            first, last = range_ij[0], range_ij[1]
            gamma = self.find_split(first, last)

            left_child = gamma
            right_child = gamma + 1

            if ti.min(first, last) == gamma:
                left_child += n - 1
            if ti.max(first, last) == gamma + 1:
                right_child += n - 1

            # Set children
            self.node_data[i][1] = ti.u32(left_child)
            self.node_data[i][2] = ti.u32(right_child)

            # Set parent for children
            self.node_data[left_child][0] = ti.u32(i)
            self.node_data[right_child][0] = ti.u32(i)

    @ti.kernel
    def _compute_internal_aabbs(self, n: ti.i32):
        """Bottom-up AABB computation."""
        INVALID = ti.u32(0xFFFFFFFF)

        for i in range(n - 1):
            self.flags[i] = INVALID

        ti.sync()

        for i in range(n):
            leaf_idx = i + n - 1
            parent_u32 = self.get_parent(leaf_idx)

            while parent_u32 != INVALID:
                parent = ti.cast(parent_u32, ti.i32)
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    break

                left = ti.cast(self.get_left(parent), ti.i32)
                right = ti.cast(self.get_right(parent), ti.i32)
                self.aabb_merge(left, right, parent)
                ti.simt.block.mem_sync()

                parent_u32 = self.get_parent(parent)

    @ti.kernel
    def _refit_leaf_aabbs(self, vertices: ti.template(),
                          edges: ti.template(), n: ti.i32):
        """Update leaf AABBs from current vertex positions."""
        for i in range(n):
            leaf_idx = i + n - 1
            edge_idx = ti.cast(self.get_element(leaf_idx), ti.i32)

            e0, e1 = edges[edge_idx, 0], edges[edge_idx, 1]
            v0, v1 = vertices[e0], vertices[e1]

            lower = ti.min(v0, v1)
            upper = ti.max(v0, v1)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    def build(self, vertices, edges, n_edges: int):
        """Build BVH from scratch."""
        if n_edges < 1:
            return

        self._compute_leaf_aabbs_and_bounds(vertices, edges, n_edges)
        self._reduce_scene_bounds(n_edges)
        self._compute_morton_codes(n_edges)
        self._sort_morton_codes(n_edges)
        self._reorder_leaf_aabbs(n_edges)
        self._init_nodes(n_edges)
        self._build_tree_topology(n_edges)
        self._compute_internal_aabbs(n_edges)
        self.tree_built = True

    def refit(self, vertices, edges, n_edges: int):
        """Refit BVH."""
        if not self.tree_built or self.num_primitives[None] != n_edges:
            self.build(vertices, edges, n_edges)
            return

        self._refit_leaf_aabbs(vertices, edges, n_edges)
        self._compute_internal_aabbs(n_edges)
