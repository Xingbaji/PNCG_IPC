"""
Optimized Linear Bounding Volume Hierarchy (LBVH) implementation.

Key optimizations over the original lbvh.py:
1. Packed node structure - reduces cache misses by combining AABB and topology
2. Warp-aligned block reduction - 32 elements per block matches GPU warp size
3. Binary search CLZ with early exit
4. Fused AABB computation - avoids redundant memory loads
5. Optimized Morton code computation with bit manipulation
6. Single-pass leaf initialization

Based on the LBVH algorithm from Stiff-GIPC with Taichi optimizations.
"""

import taichi as ti
import numpy as np
from taichi.algorithms import parallel_sort


@ti.data_oriented
class LBVH_Optimized:
    """
    Optimized Linear BVH implementation using Morton codes for fast construction.

    The tree structure uses 2N-1 nodes where N is the number of primitives:
    - Nodes 0 to N-2 are internal nodes
    - Nodes N-1 to 2N-2 are leaf nodes

    Optimizations:
    - Packed AABB storage (lower/upper in contiguous memory)
    - Warp-aligned reductions (32 elements per block)
    - Fused operations to reduce memory traffic
    """

    def __init__(self, max_primitives: int):
        """
        Initialize BVH data structures.

        Args:
            max_primitives: Maximum number of primitives (triangles or edges)
        """
        self.max_primitives = max_primitives
        self.num_nodes = 2 * max_primitives - 1

        # Packed AABB structure: [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
        # This improves cache locality when accessing both bounds
        self.aabb = ti.Vector.field(6, dtype=ti.f32, shape=self.num_nodes)

        # Packed node topology: [parent, left, right, element]
        # Single struct reduces memory accesses during traversal
        self.node_data = ti.Vector.field(4, dtype=ti.u32, shape=self.num_nodes)

        # Temporary buffer for AABB reorder (single field, packed)
        self.temp_aabb = ti.Vector.field(6, dtype=ti.f32, shape=max_primitives)

        # Morton codes and indices for sorting
        self.morton_codes = ti.field(dtype=ti.u64, shape=max_primitives)
        self.sorted_indices = ti.field(dtype=ti.u32, shape=max_primitives)

        # Flags for bottom-up AABB computation (atomic synchronization)
        self.flags = ti.field(dtype=ti.u32, shape=max_primitives)

        # Scene bounding box (packed as single vector)
        self.scene_bounds = ti.Vector.field(6, dtype=ti.f32, shape=())

        # Current number of primitives
        self.num_primitives = ti.field(dtype=ti.i32, shape=())

        # Invalid index marker (Python int, cast inside kernels)
        self.INVALID = 0xFFFFFFFF

        # Track if tree structure has been built (for refit)
        self.tree_built = False

        # Warp size for GPU-aligned operations
        self.WARP_SIZE = 32

    @ti.func
    def get_lower(self, idx: ti.i32) -> ti.math.vec3:
        """Get AABB lower bound from packed storage."""
        aabb = self.aabb[idx]
        return ti.Vector([aabb[0], aabb[1], aabb[2]])

    @ti.func
    def get_upper(self, idx: ti.i32) -> ti.math.vec3:
        """Get AABB upper bound from packed storage."""
        aabb = self.aabb[idx]
        return ti.Vector([aabb[3], aabb[4], aabb[5]])

    @ti.func
    def set_aabb(self, idx: ti.i32, lower: ti.math.vec3, upper: ti.math.vec3):
        """Set AABB bounds in packed storage."""
        self.aabb[idx] = ti.Vector([lower[0], lower[1], lower[2],
                                    upper[0], upper[1], upper[2]])

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
        """Set all node topology data in single write."""
        self.node_data[idx] = ti.Vector([parent, left, right, element], dt=ti.u32)

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        """Expand bits for Morton code computation (10 bits -> 30 bits).

        Optimized bit manipulation using magic numbers.
        """
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    @ti.func
    def morton_code_3d(self, x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
        """
        Compute 30-bit Morton code for a 3D point normalized to [0, 1].

        Optimized with fused multiply-clamp operations.
        """
        resolution = 1024.0
        # Fused clamp and scale
        xi = ti.cast(ti.min(ti.max(x * resolution, 0.0), 1023.0), ti.u32)
        yi = ti.cast(ti.min(ti.max(y * resolution, 0.0), 1023.0), ti.u32)
        zi = ti.cast(ti.min(ti.max(z * resolution, 0.0), 1023.0), ti.u32)

        xx = self.expand_bits(xi)
        yy = self.expand_bits(yi)
        zz = self.expand_bits(zi)

        return (xx << 2) | (yy << 1) | zz

    @ti.func
    def aabb_merge(self, idx1: ti.i32, idx2: ti.i32, dst: ti.i32):
        """Merge two AABBs into dst using packed storage."""
        aabb1 = self.aabb[idx1]
        aabb2 = self.aabb[idx2]
        # Vectorized min/max on packed AABBs
        lower = ti.min(ti.Vector([aabb1[0], aabb1[1], aabb1[2]]),
                       ti.Vector([aabb2[0], aabb2[1], aabb2[2]]))
        upper = ti.max(ti.Vector([aabb1[3], aabb1[4], aabb1[5]]),
                       ti.Vector([aabb2[3], aabb2[4], aabb2[5]]))
        self.aabb[dst] = ti.Vector([lower[0], lower[1], lower[2],
                                    upper[0], upper[1], upper[2]])

    @ti.func
    def aabb_overlap(self, idx1: ti.i32, idx2: ti.i32, gap: ti.f32) -> bool:
        """Check if two AABBs overlap with a gap tolerance.

        Optimized: Single load per AABB, then vectorized comparison.
        """
        aabb1 = self.aabb[idx1]
        aabb2 = self.aabb[idx2]

        # Direct boolean expression for better branch prediction
        return ((aabb1[3] - aabb2[0]) > -gap and (aabb2[3] - aabb1[0]) > -gap and
                (aabb1[4] - aabb2[1]) > -gap and (aabb2[4] - aabb1[1]) > -gap and
                (aabb1[5] - aabb2[2]) > -gap and (aabb2[5] - aabb1[2]) > -gap)

    @ti.func
    def aabb_overlap_point(self, point: ti.math.vec3, idx: ti.i32, gap: ti.f32) -> bool:
        """Check if a point's AABB overlaps with node's AABB.

        Optimized: Single AABB load, then vectorized comparison.
        """
        aabb = self.aabb[idx]

        # Direct boolean expression
        return ((aabb[3] - point[0]) > -gap and (point[0] - aabb[0]) > -gap and
                (aabb[4] - point[1]) > -gap and (point[1] - aabb[1]) > -gap and
                (aabb[5] - point[2]) > -gap and (point[2] - aabb[2]) > -gap)

    @ti.func
    def aabb_overlap_range(self, lower: ti.math.vec3, upper: ti.math.vec3,
                           idx: ti.i32, gap: ti.f32) -> bool:
        """Check if an AABB range overlaps with node's AABB.

        Used for edge-edge and triangle queries.
        """
        aabb = self.aabb[idx]

        return ((aabb[3] - lower[0]) > -gap and (upper[0] - aabb[0]) > -gap and
                (aabb[4] - lower[1]) > -gap and (upper[1] - aabb[1]) > -gap and
                (aabb[5] - lower[2]) > -gap and (upper[2] - aabb[2]) > -gap)

    @ti.func
    def common_upper_bits(self, lhs: ti.u64, rhs: ti.u64) -> ti.i32:
        """Count leading zeros of XOR of two values (common prefix length).

        Optimized binary search: 6 comparisons instead of 64 iterations.
        """
        xor_val = lhs ^ rhs
        count = 0
        if xor_val == 0:
            count = 64
        else:
            # Binary search for leading zeros
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
        """Determine the range of keys covered by an internal node."""
        first = 0
        last = num_leaves - 1

        if idx != 0:
            self_code = self.morton_codes[idx]
            L_delta = self.common_upper_bits(self_code, self.morton_codes[idx - 1])
            R_delta = self.common_upper_bits(self_code, self.morton_codes[idx + 1])

            d = 1 if R_delta > L_delta else -1
            delta_min = ti.min(L_delta, R_delta)

            # Compute upper bound for the range length
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

            # Binary search for the other end
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
        """Find the split position for an internal node."""
        first_code = self.morton_codes[first]
        last_code = self.morton_codes[last]

        split = (first + last) >> 1  # Default for equal codes

        if first_code != last_code:
            delta_node = self.common_upper_bits(first_code, last_code)

            # Binary search
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
class LBVH_Triangles_Optimized(LBVH_Optimized):
    """Optimized LBVH for triangle primitives (face BVH for vertex-face queries)."""

    def __init__(self, max_triangles: int):
        super().__init__(max_triangles)

    @ti.kernel
    def compute_leaf_aabbs(self,
                           vertices: ti.template(),
                           triangles: ti.template(),
                           n_triangles: ti.i32):
        """Compute AABBs for all triangle leaf nodes with packed storage."""
        self.num_primitives[None] = n_triangles

        for i in range(n_triangles):
            t0 = triangles[i, 0]
            t1 = triangles[i, 1]
            t2 = triangles[i, 2]

            v0 = vertices[t0]
            v1 = vertices[t1]
            v2 = vertices[t2]

            leaf_idx = i + n_triangles - 1
            lower = ti.min(ti.min(v0, v1), v2)
            upper = ti.max(ti.max(v0, v1), v2)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    @ti.kernel
    def compute_scene_aabb(self, n_triangles: ti.i32):
        """Compute the scene bounding box using warp-aligned block reduction.

        Optimized: Uses 32-element blocks matching GPU warp size for better
        occupancy and fewer atomic operations.
        """
        # Initialize scene bounds
        self.scene_bounds[None] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])

        ti.sync()

        # Warp-aligned reduction (WARP=32)
        WARP = 32
        n_warps = (n_triangles + WARP - 1) // WARP

        for warp_id in range(n_warps):
            # Local min/max for this warp
            local_min = ti.Vector([1e32, 1e32, 1e32])
            local_max = ti.Vector([-1e32, -1e32, -1e32])

            for lane in range(WARP):
                idx = warp_id * WARP + lane
                if idx < n_triangles:
                    leaf_idx = idx + n_triangles - 1
                    aabb = self.aabb[leaf_idx]
                    local_min = ti.min(local_min, ti.Vector([aabb[0], aabb[1], aabb[2]]))
                    local_max = ti.max(local_max, ti.Vector([aabb[3], aabb[4], aabb[5]]))

            # Single atomic per warp
            ti.atomic_min(self.scene_bounds[None][0], local_min[0])
            ti.atomic_min(self.scene_bounds[None][1], local_min[1])
            ti.atomic_min(self.scene_bounds[None][2], local_min[2])
            ti.atomic_max(self.scene_bounds[None][3], local_max[0])
            ti.atomic_max(self.scene_bounds[None][4], local_max[1])
            ti.atomic_max(self.scene_bounds[None][5], local_max[2])

    @ti.kernel
    def compute_morton_codes(self, n_triangles: ti.i32):
        """Compute Morton codes for all leaf nodes based on their centroids.

        Optimized: Uses packed AABB access for better cache utilization.
        """
        bounds = self.scene_bounds[None]
        scene_lower = ti.Vector([bounds[0], bounds[1], bounds[2]])
        scene_size = ti.Vector([bounds[3] - bounds[0],
                                bounds[4] - bounds[1],
                                bounds[5] - bounds[2]])

        # Avoid division by zero
        scene_size = ti.max(scene_size, ti.Vector([1e-10, 1e-10, 1e-10]))
        inv_scene_size = 1.0 / scene_size

        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            aabb = self.aabb[leaf_idx]
            # Compute center from packed AABB
            center = ti.Vector([(aabb[0] + aabb[3]) * 0.5,
                               (aabb[1] + aabb[4]) * 0.5,
                               (aabb[2] + aabb[5]) * 0.5])

            # Normalize to [0, 1] using precomputed inverse
            normalized = (center - scene_lower) * inv_scene_size

            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            # Pack Morton code with index for stable sorting
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = i

    @ti.kernel
    def init_nodes(self, n_triangles: ti.i32):
        """Initialize all nodes (leaf and internal) in single pass.

        Optimized: Fused initialization reduces kernel launch overhead.
        """
        INVALID = ti.u32(0xFFFFFFFF)

        for i in range(n_triangles):
            # Internal nodes (if exists)
            if i < n_triangles - 1:
                self.set_node(i, INVALID, INVALID, INVALID, INVALID)

            # Leaf nodes - use sorted index for element mapping
            leaf_idx = i + n_triangles - 1
            self.set_node(leaf_idx, INVALID, INVALID, INVALID, self.sorted_indices[i])

    @ti.kernel
    def build_internal_nodes(self, n_triangles: ti.i32):
        """Build internal nodes using the Morton code ordering."""
        for i in range(n_triangles - 1):
            range_ij = self.determine_range(i, n_triangles)
            first = range_ij[0]
            last = range_ij[1]

            gamma = self.find_split(first, last)

            left_child = gamma
            right_child = gamma + 1

            if ti.min(first, last) == gamma:
                left_child += n_triangles - 1
            if ti.max(first, last) == gamma + 1:
                right_child += n_triangles - 1

            # Update topology using packed node data
            node = self.node_data[i]
            self.node_data[i] = ti.Vector([node[0], ti.u32(left_child),
                                           ti.u32(right_child), node[3]], dt=ti.u32)

            # Set parent for children
            left_node = self.node_data[left_child]
            self.node_data[left_child] = ti.Vector([ti.u32(i), left_node[1],
                                                    left_node[2], left_node[3]], dt=ti.u32)
            right_node = self.node_data[right_child]
            self.node_data[right_child] = ti.Vector([ti.u32(i), right_node[1],
                                                     right_node[2], right_node[3]], dt=ti.u32)

    @ti.kernel
    def compute_internal_aabbs(self, n_triangles: ti.i32):
        """Compute AABBs for internal nodes bottom-up.

        Uses atomic synchronization to ensure both children are processed
        before computing parent AABB.
        """
        INVALID = ti.u32(0xFFFFFFFF)

        # Reset flags
        for i in range(n_triangles - 1):
            self.flags[i] = INVALID

        # Bottom-up traversal from leaves
        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            parent = self.get_parent(leaf_idx)

            while parent != INVALID:
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    # First thread to arrive, wait for sibling
                    break

                # Second thread to arrive, compute AABB
                left = ti.cast(self.get_left(parent), ti.i32)
                right = ti.cast(self.get_right(parent), ti.i32)
                self.aabb_merge(left, right, parent)

                ti.simt.block.mem_sync()

                parent = self.get_parent(parent)

    @ti.kernel
    def refit_leaf_aabbs(self, vertices: ti.template(),
                         triangles: ti.template(), n_triangles: ti.i32):
        """Refit leaf AABBs using current vertex positions (preserves tree structure)."""
        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            tri_idx = self.get_element(leaf_idx)  # Use existing mapping

            t0 = triangles[tri_idx, 0]
            t1 = triangles[tri_idx, 1]
            t2 = triangles[tri_idx, 2]

            v0 = vertices[t0]
            v1 = vertices[t1]
            v2 = vertices[t2]

            lower = ti.min(ti.min(v0, v1), v2)
            upper = ti.max(ti.max(v0, v1), v2)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    def refit(self, vertices, triangles, n_triangles: int):
        """
        Refit BVH by updating AABBs while preserving tree structure.
        Falls back to full build if tree not built or primitive count changed.
        """
        if not self.tree_built or self.num_primitives[None] != n_triangles:
            self.build(vertices, triangles, n_triangles)
            return

        # Update leaf AABBs from current vertex positions
        self.refit_leaf_aabbs(vertices, triangles, n_triangles)

        # Propagate changes up the tree
        self.compute_internal_aabbs(n_triangles)

    @ti.kernel
    def _prepare_sort(self, n_triangles: ti.i32):
        """Initialize indices and set unused morton codes to max for proper sorting."""
        max_prims = self.max_primitives
        for i in range(max_prims):
            self.sorted_indices[i] = ti.cast(i, ti.u32)
            # Set unused elements to max value so they sort to end
            if i >= n_triangles:
                self.morton_codes[i] = ti.u64(0xFFFFFFFFFFFFFFFF)

    def sort_morton_codes(self, n_triangles: int):
        """Sort Morton codes and indices using GPU parallel sort."""
        self._prepare_sort(n_triangles)
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def reorder_leaf_aabbs(self, n_triangles: ti.i32):
        """Reorder leaf AABBs according to sorted order.

        Optimized: Single-pass scatter using packed AABB storage.
        """
        # Scatter to temp buffer
        for i in range(n_triangles):
            orig_idx = n_triangles - 1 + self.sorted_indices[i]
            self.temp_aabb[i] = self.aabb[orig_idx]

        ti.sync()

        # Copy back to leaf positions
        for i in range(n_triangles):
            leaf_idx = n_triangles - 1 + i
            self.aabb[leaf_idx] = self.temp_aabb[i]

    def build(self, vertices, triangles, n_triangles: int):
        """
        Build the BVH tree.

        Args:
            vertices: Vertex positions field
            triangles: Triangle indices field (Nx3)
            n_triangles: Number of triangles
        """
        if n_triangles < 1:
            return

        # Step 1: Compute leaf AABBs
        self.compute_leaf_aabbs(vertices, triangles, n_triangles)

        # Step 2: Compute scene AABB (warp-aligned reduction)
        self.compute_scene_aabb(n_triangles)

        # Step 3: Compute Morton codes
        self.compute_morton_codes(n_triangles)

        # Step 4: Sort by Morton codes (GPU parallel sort)
        self.sort_morton_codes(n_triangles)

        # Step 5: Reorder leaf AABBs
        self.reorder_leaf_aabbs(n_triangles)

        # Step 6: Initialize all nodes (fused kernel)
        self.init_nodes(n_triangles)

        # Step 7: Build internal nodes
        self.build_internal_nodes(n_triangles)

        # Step 8: Compute internal AABBs
        self.compute_internal_aabbs(n_triangles)

        # Mark tree as built for refit
        self.tree_built = True


@ti.data_oriented
class LBVH_Edges_Optimized(LBVH_Optimized):
    """Optimized LBVH for edge primitives (edge BVH for edge-edge queries)."""

    def __init__(self, max_edges: int):
        super().__init__(max_edges)

    @ti.kernel
    def compute_leaf_aabbs(self,
                           vertices: ti.template(),
                           edges: ti.template(),
                           n_edges: ti.i32):
        """Compute AABBs for all edge leaf nodes with packed storage."""
        self.num_primitives[None] = n_edges

        for i in range(n_edges):
            e0 = edges[i, 0]
            e1 = edges[i, 1]

            v0 = vertices[e0]
            v1 = vertices[e1]

            leaf_idx = i + n_edges - 1
            lower = ti.min(v0, v1)
            upper = ti.max(v0, v1)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    @ti.kernel
    def compute_scene_aabb(self, n_edges: ti.i32):
        """Compute the scene bounding box using warp-aligned block reduction."""
        # Initialize scene bounds
        self.scene_bounds[None] = ti.Vector([1e32, 1e32, 1e32, -1e32, -1e32, -1e32])

        ti.sync()

        # Warp-aligned reduction (WARP=32)
        WARP = 32
        n_warps = (n_edges + WARP - 1) // WARP

        for warp_id in range(n_warps):
            local_min = ti.Vector([1e32, 1e32, 1e32])
            local_max = ti.Vector([-1e32, -1e32, -1e32])

            for lane in range(WARP):
                idx = warp_id * WARP + lane
                if idx < n_edges:
                    leaf_idx = idx + n_edges - 1
                    aabb = self.aabb[leaf_idx]
                    local_min = ti.min(local_min, ti.Vector([aabb[0], aabb[1], aabb[2]]))
                    local_max = ti.max(local_max, ti.Vector([aabb[3], aabb[4], aabb[5]]))

            ti.atomic_min(self.scene_bounds[None][0], local_min[0])
            ti.atomic_min(self.scene_bounds[None][1], local_min[1])
            ti.atomic_min(self.scene_bounds[None][2], local_min[2])
            ti.atomic_max(self.scene_bounds[None][3], local_max[0])
            ti.atomic_max(self.scene_bounds[None][4], local_max[1])
            ti.atomic_max(self.scene_bounds[None][5], local_max[2])

    @ti.kernel
    def compute_morton_codes(self, n_edges: ti.i32):
        """Compute Morton codes for all leaf nodes based on their centroids."""
        bounds = self.scene_bounds[None]
        scene_lower = ti.Vector([bounds[0], bounds[1], bounds[2]])
        scene_size = ti.Vector([bounds[3] - bounds[0],
                                bounds[4] - bounds[1],
                                bounds[5] - bounds[2]])

        scene_size = ti.max(scene_size, ti.Vector([1e-10, 1e-10, 1e-10]))
        inv_scene_size = 1.0 / scene_size

        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            aabb = self.aabb[leaf_idx]
            center = ti.Vector([(aabb[0] + aabb[3]) * 0.5,
                               (aabb[1] + aabb[4]) * 0.5,
                               (aabb[2] + aabb[5]) * 0.5])

            normalized = (center - scene_lower) * inv_scene_size

            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = i

    @ti.kernel
    def init_nodes(self, n_edges: ti.i32):
        """Initialize all nodes (leaf and internal) in single pass."""
        INVALID = ti.u32(0xFFFFFFFF)

        for i in range(n_edges):
            if i < n_edges - 1:
                self.set_node(i, INVALID, INVALID, INVALID, INVALID)

            leaf_idx = i + n_edges - 1
            self.set_node(leaf_idx, INVALID, INVALID, INVALID, self.sorted_indices[i])

    @ti.kernel
    def build_internal_nodes(self, n_edges: ti.i32):
        """Build internal nodes using the Morton code ordering."""
        for i in range(n_edges - 1):
            range_ij = self.determine_range(i, n_edges)
            first = range_ij[0]
            last = range_ij[1]

            gamma = self.find_split(first, last)

            left_child = gamma
            right_child = gamma + 1

            if ti.min(first, last) == gamma:
                left_child += n_edges - 1
            if ti.max(first, last) == gamma + 1:
                right_child += n_edges - 1

            node = self.node_data[i]
            self.node_data[i] = ti.Vector([node[0], ti.u32(left_child),
                                           ti.u32(right_child), node[3]], dt=ti.u32)

            left_node = self.node_data[left_child]
            self.node_data[left_child] = ti.Vector([ti.u32(i), left_node[1],
                                                    left_node[2], left_node[3]], dt=ti.u32)
            right_node = self.node_data[right_child]
            self.node_data[right_child] = ti.Vector([ti.u32(i), right_node[1],
                                                     right_node[2], right_node[3]], dt=ti.u32)

    @ti.kernel
    def compute_internal_aabbs(self, n_edges: ti.i32):
        """Compute AABBs for internal nodes bottom-up."""
        INVALID = ti.u32(0xFFFFFFFF)

        for i in range(n_edges - 1):
            self.flags[i] = INVALID

        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            parent = self.get_parent(leaf_idx)

            while parent != INVALID:
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    break

                left = ti.cast(self.get_left(parent), ti.i32)
                right = ti.cast(self.get_right(parent), ti.i32)
                self.aabb_merge(left, right, parent)

                ti.simt.block.mem_sync()

                parent = self.get_parent(parent)

    @ti.kernel
    def refit_leaf_aabbs(self, vertices: ti.template(),
                         edges: ti.template(), n_edges: ti.i32):
        """Refit leaf AABBs using current vertex positions."""
        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            edge_idx = self.get_element(leaf_idx)

            e0 = edges[edge_idx, 0]
            e1 = edges[edge_idx, 1]

            v0 = vertices[e0]
            v1 = vertices[e1]

            lower = ti.min(v0, v1)
            upper = ti.max(v0, v1)
            self.aabb[leaf_idx] = ti.Vector([lower[0], lower[1], lower[2],
                                             upper[0], upper[1], upper[2]])

    def refit(self, vertices, edges, n_edges: int):
        """Refit BVH by updating AABBs while preserving tree structure."""
        if not self.tree_built or self.num_primitives[None] != n_edges:
            self.build(vertices, edges, n_edges)
            return

        self.refit_leaf_aabbs(vertices, edges, n_edges)
        self.compute_internal_aabbs(n_edges)

    @ti.kernel
    def _prepare_sort(self, n_edges: ti.i32):
        """Initialize indices and set unused morton codes to max."""
        max_prims = self.max_primitives
        for i in range(max_prims):
            self.sorted_indices[i] = ti.cast(i, ti.u32)
            if i >= n_edges:
                self.morton_codes[i] = ti.u64(0xFFFFFFFFFFFFFFFF)

    def sort_morton_codes(self, n_edges: int):
        """Sort Morton codes and indices using GPU parallel sort."""
        self._prepare_sort(n_edges)
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def reorder_leaf_aabbs(self, n_edges: ti.i32):
        """Reorder leaf AABBs according to sorted order."""
        for i in range(n_edges):
            orig_idx = n_edges - 1 + self.sorted_indices[i]
            self.temp_aabb[i] = self.aabb[orig_idx]

        ti.sync()

        for i in range(n_edges):
            leaf_idx = n_edges - 1 + i
            self.aabb[leaf_idx] = self.temp_aabb[i]

    def build(self, vertices, edges, n_edges: int):
        """Build the BVH tree."""
        if n_edges < 1:
            return

        self.compute_leaf_aabbs(vertices, edges, n_edges)
        self.compute_scene_aabb(n_edges)
        self.compute_morton_codes(n_edges)
        self.sort_morton_codes(n_edges)
        self.reorder_leaf_aabbs(n_edges)
        self.init_nodes(n_edges)
        self.build_internal_nodes(n_edges)
        self.compute_internal_aabbs(n_edges)
        self.tree_built = True
