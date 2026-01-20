"""
LBVH for edge primitives (edge BVH for edge-edge queries).
"""

import taichi as ti
from taichi.algorithms import parallel_sort

from .base import LBVH
from ...core.precision import PrecisionType


@ti.data_oriented
class LBVH_Edges(LBVH):
    """
    LBVH for edge primitives (edge BVH for edge-edge queries).

    Inherits from LBVH base class and adds edge-specific AABB computation
    and tree construction methods.
    """

    def __init__(self, max_edges: int, precision: PrecisionType = 'f32'):
        """
        Initialize edge BVH.

        Args:
            max_edges: Maximum number of edges
            precision: Float precision ('f32' or 'f64')
        """
        super().__init__(max_edges, precision)

    @ti.kernel
    def compute_leaf_aabbs(
        self,
        vertices: ti.template(),
        edges: ti.template(),
        n_edges: ti.i32
    ):
        """
        Compute AABBs for all edge leaf nodes.

        Args:
            vertices: Vertex positions field (Nx3)
            edges: Edge indices field (Mx2)
            n_edges: Number of edges
        """
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
        n_blocks = (n_edges + 255) // 256
        for block_id in range(n_blocks):
            # Local min/max for this block (no atomic operations here)
            local_min = ti.Vector([1e32, 1e32, 1e32])
            local_max = ti.Vector([-1e32, -1e32, -1e32])

            for i in range(256):
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
    def refit_leaf_aabbs(
        self,
        vertices: ti.template(),
        edges: ti.template(),
        n_edges: ti.i32
    ):
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
        """
        Reorder leaf AABBs according to sorted order using temp buffers.

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
