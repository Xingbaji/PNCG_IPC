"""
LBVH for triangle primitives (face BVH for vertex-face queries).
"""

import taichi as ti
from taichi.algorithms import parallel_sort

from .base import LBVH
from ...core.precision import PrecisionType


@ti.data_oriented
class LBVH_Triangles(LBVH):
    """
    LBVH for triangle primitives (face BVH for point-triangle queries).

    Inherits from LBVH base class and adds triangle-specific AABB computation
    and tree construction methods.
    """

    def __init__(self, max_triangles: int, precision: PrecisionType = 'f32'):
        """
        Initialize triangle BVH.

        Args:
            max_triangles: Maximum number of triangles
            precision: Float precision ('f32' or 'f64')
        """
        super().__init__(max_triangles, precision)

    @ti.kernel
    def compute_leaf_aabbs(
        self,
        vertices: ti.template(),
        triangles: ti.template(),
        n_triangles: ti.i32
    ):
        """
        Compute AABBs for all triangle leaf nodes.

        Args:
            vertices: Vertex positions field (Nx3)
            triangles: Triangle indices field (Mx3)
            n_triangles: Number of triangles
        """
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
        """
        Compute the scene bounding box using block-wise reduction.

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

            for i in range(256):
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
    def refit_leaf_aabbs(
        self,
        vertices: ti.template(),
        triangles: ti.template(),
        n_triangles: ti.i32
    ):
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
        """
        Reorder leaf AABBs according to sorted order using temp buffers.

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
