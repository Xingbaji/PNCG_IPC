"""
Linear Bounding Volume Hierarchy (LBVH) implementation for collision detection.
Based on the LBVH algorithm from Stiff-GIPC.

This module replaces the spatial hashing approach with a more efficient BVH-based
broad-phase collision detection.
"""

import taichi as ti
import numpy as np
from taichi.algorithms import parallel_sort


@ti.data_oriented
class LBVH:
    """
    Linear BVH implementation using Morton codes for fast construction.

    The tree structure uses 2N-1 nodes where N is the number of primitives:
    - Nodes 0 to N-2 are internal nodes
    - Nodes N-1 to 2N-2 are leaf nodes
    """

    def __init__(self, max_primitives: int):
        """
        Initialize BVH data structures.

        Args:
            max_primitives: Maximum number of primitives (triangles or edges)
        """
        self.max_primitives = max_primitives
        self.num_nodes = 2 * max_primitives - 1

        # AABB bounding volumes: lower (x,y,z), upper (x,y,z)
        self.bv_lower = ti.Vector.field(3, dtype=ti.f32, shape=self.num_nodes)
        self.bv_upper = ti.Vector.field(3, dtype=ti.f32, shape=self.num_nodes)

        # Temporary buffers for single-pass AABB reorder (avoids memory aliasing)
        self.temp_lower = ti.Vector.field(3, dtype=ti.f32, shape=max_primitives)
        self.temp_upper = ti.Vector.field(3, dtype=ti.f32, shape=max_primitives)

        # Node structure: parent_idx, left_idx, right_idx, element_idx
        self.parent_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.left_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.right_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.element_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)

        # Morton codes and indices for sorting
        self.morton_codes = ti.field(dtype=ti.u64, shape=max_primitives)
        self.sorted_indices = ti.field(dtype=ti.u32, shape=max_primitives)

        # Flags for bottom-up AABB computation
        self.flags = ti.field(dtype=ti.u32, shape=max_primitives)

        # Scene bounding box
        self.scene_lower = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.scene_upper = ti.Vector.field(3, dtype=ti.f32, shape=())

        # Current number of primitives
        self.num_primitives = ti.field(dtype=ti.i32, shape=())

        # Invalid index marker
        self.INVALID = 0xFFFFFFFF

        # Track if tree structure has been built (for refit)
        self.tree_built = False

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        """Expand bits for Morton code computation (10 bits -> 30 bits)."""
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    @ti.func
    def morton_code_3d(self, x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
        """
        Compute 30-bit Morton code for a 3D point normalized to [0, 1].
        """
        resolution = 1024.0
        x = ti.min(ti.max(x * resolution, 0.0), resolution - 1.0)
        y = ti.min(ti.max(y * resolution, 0.0), resolution - 1.0)
        z = ti.min(ti.max(z * resolution, 0.0), resolution - 1.0)

        xx = self.expand_bits(ti.cast(x, ti.u32))
        yy = self.expand_bits(ti.cast(y, ti.u32))
        zz = self.expand_bits(ti.cast(z, ti.u32))

        return (xx << 2) | (yy << 1) | zz

    @ti.func
    def aabb_merge(self, idx1: ti.i32, idx2: ti.i32, dst: ti.i32):
        """Merge two AABBs into dst."""
        self.bv_lower[dst] = ti.min(self.bv_lower[idx1], self.bv_lower[idx2])
        self.bv_upper[dst] = ti.max(self.bv_upper[idx1], self.bv_upper[idx2])

    @ti.func
    def aabb_overlap(self, idx1: ti.i32, idx2: ti.i32, gap: ti.f32) -> bool:
        """Check if two AABBs overlap with a gap tolerance.

        Uses direct boolean logic for better branch prediction and fewer operations.
        """
        lower1 = self.bv_lower[idx1]
        upper1 = self.bv_upper[idx1]
        lower2 = self.bv_lower[idx2]
        upper2 = self.bv_upper[idx2]

        # Direct boolean expression - allows compiler to optimize better
        return ((upper1[0] - lower2[0]) > -gap and (upper2[0] - lower1[0]) > -gap and
                (upper1[1] - lower2[1]) > -gap and (upper2[1] - lower1[1]) > -gap and
                (upper1[2] - lower2[2]) > -gap and (upper2[2] - lower1[2]) > -gap)

    @ti.func
    def aabb_overlap_point(self, point: ti.template(), idx: ti.i32, gap: ti.f32) -> bool:
        """Check if a point's AABB overlaps with node's AABB.

        Uses direct boolean logic for better branch prediction.
        """
        lower = self.bv_lower[idx]
        upper = self.bv_upper[idx]

        # Direct boolean expression
        return ((upper[0] - point[0]) > -gap and (point[0] - lower[0]) > -gap and
                (upper[1] - point[1]) > -gap and (point[1] - lower[1]) > -gap and
                (upper[2] - point[2]) > -gap and (point[2] - lower[2]) > -gap)

    @ti.func
    def common_upper_bits(self, lhs: ti.u64, rhs: ti.u64) -> ti.i32:
        """Count leading zeros of XOR of two values (common prefix length).

        Uses binary search instead of 64-iteration loop for better performance.
        Reduces from 64 iterations to 6 comparisons.
        """
        xor_val = lhs ^ rhs
        count = 0
        if xor_val == 0:
            count = 64
        else:
            # Binary search for leading zeros - 6 comparisons instead of 64 iterations
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
class LBVH_Triangles(LBVH):
    """LBVH for triangle primitives (face BVH for vertex-face queries)."""

    def __init__(self, max_triangles: int):
        super().__init__(max_triangles)

    @ti.kernel
    def compute_leaf_aabbs(self,
                           vertices: ti.template(),
                           triangles: ti.template(),
                           n_triangles: ti.i32):
        """Compute AABBs for all triangle leaf nodes."""
        self.num_primitives[None] = n_triangles

        for i in range(n_triangles):
            t0 = triangles[i, 0]
            t1 = triangles[i, 1]
            t2 = triangles[i, 2]

            v0 = vertices[t0]
            v1 = vertices[t1]
            v2 = vertices[t2]

            leaf_idx = i + n_triangles - 1
            self.bv_lower[leaf_idx] = ti.min(ti.min(v0, v1), v2)
            self.bv_upper[leaf_idx] = ti.max(ti.max(v0, v1), v2)

    @ti.kernel
    def compute_scene_aabb(self, n_triangles: ti.i32):
        """Compute the scene bounding box using block-wise reduction.

        Uses hierarchical reduction to minimize atomic contention:
        - Each block (256 elements) computes local min/max
        - Only one atomic operation per block (instead of per element)
        This reduces atomic operations by ~256x.
        """
        # Initialize scene bounds
        self.scene_lower[None] = ti.Vector([1e32, 1e32, 1e32])
        self.scene_upper[None] = ti.Vector([-1e32, -1e32, -1e32])

        ti.sync()

        # Block-wise reduction (BLOCK=256 as compile-time constant)
        n_blocks = (n_triangles + 255) // 256
        for block_id in range(n_blocks):
            # Local min/max for this block (no atomic operations here)
            local_min = ti.Vector([1e32, 1e32, 1e32])
            local_max = ti.Vector([-1e32, -1e32, -1e32])

            for i in ti.static(range(256)):
                idx = block_id * 256 + i
                if idx < n_triangles:
                    leaf_idx = idx + n_triangles - 1
                    local_min = ti.min(local_min, self.bv_lower[leaf_idx])
                    local_max = ti.max(local_max, self.bv_upper[leaf_idx])

            # Single atomic per block instead of per element
            ti.atomic_min(self.scene_lower[None], local_min)
            ti.atomic_max(self.scene_upper[None], local_max)

    @ti.kernel
    def compute_morton_codes(self, n_triangles: ti.i32):
        """Compute Morton codes for all leaf nodes based on their centroids."""
        scene_lower = self.scene_lower[None]
        scene_size = self.scene_upper[None] - scene_lower

        # Avoid division by zero
        scene_size = ti.max(scene_size, ti.Vector([1e-10, 1e-10, 1e-10]))

        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            center = (self.bv_lower[leaf_idx] + self.bv_upper[leaf_idx]) * 0.5

            # Normalize to [0, 1]
            normalized = (center - scene_lower) / scene_size

            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            # Pack Morton code with index for stable sorting
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = i

    @ti.kernel
    def init_leaf_nodes(self, n_triangles: ti.i32):
        """Initialize leaf nodes after sorting."""
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n_triangles):
            # Internal nodes
            if i < n_triangles - 1:
                self.left_idx[i] = INVALID
                self.right_idx[i] = INVALID
                self.parent_idx[i] = INVALID
                self.element_idx[i] = INVALID

            # Leaf nodes
            leaf_idx = i + n_triangles - 1
            self.left_idx[leaf_idx] = INVALID
            self.right_idx[leaf_idx] = INVALID
            self.parent_idx[leaf_idx] = INVALID
            self.element_idx[leaf_idx] = self.sorted_indices[i]

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

            self.left_idx[i] = left_child
            self.right_idx[i] = right_child
            self.parent_idx[left_child] = i
            self.parent_idx[right_child] = i

    @ti.kernel
    def compute_internal_aabbs(self, n_triangles: ti.i32):
        """Compute AABBs for internal nodes bottom-up."""
        INVALID = ti.u32(0xFFFFFFFF)
        # Reset flags
        for i in range(n_triangles - 1):
            self.flags[i] = INVALID

        # Bottom-up traversal from leaves
        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            parent = self.parent_idx[leaf_idx]

            while parent != INVALID:
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    # First thread to arrive, wait for sibling
                    break

                # Second thread to arrive, compute AABB
                left = self.left_idx[parent]
                right = self.right_idx[parent]
                self.aabb_merge(left, right, parent)

                ti.simt.block.mem_sync()

                parent = self.parent_idx[parent]

    @ti.kernel
    def refit_leaf_aabbs(self, vertices: ti.template(),
                         triangles: ti.template(), n_triangles: ti.i32):
        """Refit leaf AABBs using current vertex positions (preserves tree structure)."""
        for i in range(n_triangles):
            leaf_idx = i + n_triangles - 1
            tri_idx = self.element_idx[leaf_idx]  # Use existing mapping

            t0 = triangles[tri_idx, 0]
            t1 = triangles[tri_idx, 1]
            t2 = triangles[tri_idx, 2]

            v0 = vertices[t0]
            v1 = vertices[t1]
            v2 = vertices[t2]

            self.bv_lower[leaf_idx] = ti.min(ti.min(v0, v1), v2)
            self.bv_upper[leaf_idx] = ti.max(ti.max(v0, v1), v2)

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
        # Prepare indices and pad unused codes with max value
        self._prepare_sort(n_triangles)
        # GPU parallel sort - sorts morton_codes and reorders sorted_indices accordingly
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def reorder_leaf_aabbs(self, n_triangles: ti.i32):
        """Reorder leaf AABBs according to sorted order using temp buffers.

        Uses dedicated temp buffers instead of internal node area to avoid
        memory aliasing and enable cleaner single-pass operations.
        """
        # Single pass scatter to dedicated temp buffers
        for i in range(n_triangles):
            orig_idx = n_triangles - 1 + self.sorted_indices[i]
            self.temp_lower[i] = self.bv_lower[orig_idx]
            self.temp_upper[i] = self.bv_upper[orig_idx]

        ti.sync()

        # Copy back to leaf positions
        for i in range(n_triangles):
            leaf_idx = n_triangles - 1 + i
            self.bv_lower[leaf_idx] = self.temp_lower[i]
            self.bv_upper[leaf_idx] = self.temp_upper[i]

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

        # Step 2: Compute scene AABB
        self.compute_scene_aabb(n_triangles)

        # Step 3: Compute Morton codes
        self.compute_morton_codes(n_triangles)

        # Step 4: Sort by Morton codes (GPU parallel sort)
        self.sort_morton_codes(n_triangles)

        # Step 5: Reorder leaf AABBs
        self.reorder_leaf_aabbs(n_triangles)

        # Step 6: Initialize leaf nodes
        self.init_leaf_nodes(n_triangles)

        # Step 7: Build internal nodes
        self.build_internal_nodes(n_triangles)

        # Step 8: Compute internal AABBs
        self.compute_internal_aabbs(n_triangles)

        # Mark tree as built for refit
        self.tree_built = True


@ti.data_oriented
class LBVH_Edges(LBVH):
    """LBVH for edge primitives (edge BVH for edge-edge queries)."""

    def __init__(self, max_edges: int):
        super().__init__(max_edges)

    @ti.kernel
    def compute_leaf_aabbs(self,
                           vertices: ti.template(),
                           edges: ti.template(),
                           n_edges: ti.i32):
        """Compute AABBs for all edge leaf nodes."""
        self.num_primitives[None] = n_edges

        for i in range(n_edges):
            e0 = edges[i, 0]
            e1 = edges[i, 1]

            v0 = vertices[e0]
            v1 = vertices[e1]

            leaf_idx = i + n_edges - 1
            self.bv_lower[leaf_idx] = ti.min(v0, v1)
            self.bv_upper[leaf_idx] = ti.max(v0, v1)

    @ti.kernel
    def compute_scene_aabb(self, n_edges: ti.i32):
        """Compute the scene bounding box using block-wise reduction.

        Uses hierarchical reduction to minimize atomic contention:
        - Each block (256 elements) computes local min/max
        - Only one atomic operation per block (instead of per element)
        This reduces atomic operations by ~256x.
        """
        # Initialize scene bounds
        self.scene_lower[None] = ti.Vector([1e32, 1e32, 1e32])
        self.scene_upper[None] = ti.Vector([-1e32, -1e32, -1e32])

        ti.sync()

        # Block-wise reduction (BLOCK=256 as compile-time constant)
        n_blocks = (n_edges + 255) // 256
        for block_id in range(n_blocks):
            # Local min/max for this block (no atomic operations here)
            local_min = ti.Vector([1e32, 1e32, 1e32])
            local_max = ti.Vector([-1e32, -1e32, -1e32])

            for i in ti.static(range(256)):
                idx = block_id * 256 + i
                if idx < n_edges:
                    leaf_idx = idx + n_edges - 1
                    local_min = ti.min(local_min, self.bv_lower[leaf_idx])
                    local_max = ti.max(local_max, self.bv_upper[leaf_idx])

            # Single atomic per block instead of per element
            ti.atomic_min(self.scene_lower[None], local_min)
            ti.atomic_max(self.scene_upper[None], local_max)

    @ti.kernel
    def compute_morton_codes(self, n_edges: ti.i32):
        """Compute Morton codes for all leaf nodes based on their centroids."""
        scene_lower = self.scene_lower[None]
        scene_size = self.scene_upper[None] - scene_lower

        # Avoid division by zero
        scene_size = ti.max(scene_size, ti.Vector([1e-10, 1e-10, 1e-10]))

        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            center = (self.bv_lower[leaf_idx] + self.bv_upper[leaf_idx]) * 0.5

            # Normalize to [0, 1]
            normalized = (center - scene_lower) / scene_size

            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            # Pack Morton code with index for stable sorting
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = i

    @ti.kernel
    def init_leaf_nodes(self, n_edges: ti.i32):
        """Initialize leaf nodes after sorting."""
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n_edges):
            # Internal nodes
            if i < n_edges - 1:
                self.left_idx[i] = INVALID
                self.right_idx[i] = INVALID
                self.parent_idx[i] = INVALID
                self.element_idx[i] = INVALID

            # Leaf nodes
            leaf_idx = i + n_edges - 1
            self.left_idx[leaf_idx] = INVALID
            self.right_idx[leaf_idx] = INVALID
            self.parent_idx[leaf_idx] = INVALID
            self.element_idx[leaf_idx] = self.sorted_indices[i]

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

            self.left_idx[i] = left_child
            self.right_idx[i] = right_child
            self.parent_idx[left_child] = i
            self.parent_idx[right_child] = i

    @ti.kernel
    def compute_internal_aabbs(self, n_edges: ti.i32):
        """Compute AABBs for internal nodes bottom-up."""
        INVALID = ti.u32(0xFFFFFFFF)
        # Reset flags
        for i in range(n_edges - 1):
            self.flags[i] = INVALID

        # Bottom-up traversal from leaves
        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            parent = self.parent_idx[leaf_idx]

            while parent != INVALID:
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    # First thread to arrive, wait for sibling
                    break

                # Second thread to arrive, compute AABB
                left = self.left_idx[parent]
                right = self.right_idx[parent]
                self.aabb_merge(left, right, parent)

                ti.simt.block.mem_sync()

                parent = self.parent_idx[parent]

    @ti.kernel
    def refit_leaf_aabbs(self, vertices: ti.template(),
                         edges: ti.template(), n_edges: ti.i32):
        """Refit leaf AABBs using current vertex positions (preserves tree structure)."""
        for i in range(n_edges):
            leaf_idx = i + n_edges - 1
            edge_idx = self.element_idx[leaf_idx]  # Use existing mapping

            e0 = edges[edge_idx, 0]
            e1 = edges[edge_idx, 1]

            v0 = vertices[e0]
            v1 = vertices[e1]

            self.bv_lower[leaf_idx] = ti.min(v0, v1)
            self.bv_upper[leaf_idx] = ti.max(v0, v1)

    def refit(self, vertices, edges, n_edges: int):
        """
        Refit BVH by updating AABBs while preserving tree structure.
        Falls back to full build if tree not built or primitive count changed.
        """
        if not self.tree_built or self.num_primitives[None] != n_edges:
            self.build(vertices, edges, n_edges)
            return

        # Update leaf AABBs from current vertex positions
        self.refit_leaf_aabbs(vertices, edges, n_edges)

        # Propagate changes up the tree
        self.compute_internal_aabbs(n_edges)

    @ti.kernel
    def _prepare_sort(self, n_edges: ti.i32):
        """Initialize indices and set unused morton codes to max for proper sorting."""
        max_prims = self.max_primitives
        for i in range(max_prims):
            self.sorted_indices[i] = ti.cast(i, ti.u32)
            # Set unused elements to max value so they sort to end
            if i >= n_edges:
                self.morton_codes[i] = ti.u64(0xFFFFFFFFFFFFFFFF)

    def sort_morton_codes(self, n_edges: int):
        """Sort Morton codes and indices using GPU parallel sort."""
        # Prepare indices and pad unused codes with max value
        self._prepare_sort(n_edges)
        # GPU parallel sort - sorts morton_codes and reorders sorted_indices accordingly
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def reorder_leaf_aabbs(self, n_edges: ti.i32):
        """Reorder leaf AABBs according to sorted order using temp buffers.

        Uses dedicated temp buffers instead of internal node area to avoid
        memory aliasing and enable cleaner single-pass operations.
        """
        # Single pass scatter to dedicated temp buffers
        for i in range(n_edges):
            orig_idx = n_edges - 1 + self.sorted_indices[i]
            self.temp_lower[i] = self.bv_lower[orig_idx]
            self.temp_upper[i] = self.bv_upper[orig_idx]

        ti.sync()

        # Copy back to leaf positions
        for i in range(n_edges):
            leaf_idx = n_edges - 1 + i
            self.bv_lower[leaf_idx] = self.temp_lower[i]
            self.bv_upper[leaf_idx] = self.temp_upper[i]

    def build(self, vertices, edges, n_edges: int):
        """
        Build the BVH tree.

        Args:
            vertices: Vertex positions field
            edges: Edge indices field (Nx2)
            n_edges: Number of edges
        """
        if n_edges < 1:
            return

        # Step 1: Compute leaf AABBs
        self.compute_leaf_aabbs(vertices, edges, n_edges)

        # Step 2: Compute scene AABB
        self.compute_scene_aabb(n_edges)

        # Step 3: Compute Morton codes
        self.compute_morton_codes(n_edges)

        # Step 4: Sort by Morton codes (GPU parallel sort)
        self.sort_morton_codes(n_edges)

        # Step 5: Reorder leaf AABBs
        self.reorder_leaf_aabbs(n_edges)

        # Step 6: Initialize leaf nodes
        self.init_leaf_nodes(n_edges)

        # Step 7: Build internal nodes
        self.build_internal_nodes(n_edges)

        # Step 8: Compute internal AABBs
        self.compute_internal_aabbs(n_edges)

        # Mark tree as built for refit
        self.tree_built = True
