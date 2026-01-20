"""
Base LBVH class with Morton code computation and tree construction.

Based on the LBVH algorithm from Stiff-GIPC, adapted to support f32/f64 precision.
"""

import taichi as ti
from ...core.precision import PrecisionType, PrecisionMixin, get_precision_config


@ti.data_oriented
class LBVH(PrecisionMixin):
    """
    Linear BVH implementation using Morton codes for fast construction.

    The tree structure uses 2N-1 nodes where N is the number of primitives:
    - Nodes 0 to N-2 are internal nodes
    - Nodes N-1 to 2N-2 are leaf nodes

    Supports both f32 and f64 precision for AABB storage.
    """

    def __init__(self, max_primitives: int, precision: PrecisionType = 'f32'):
        """
        Initialize BVH data structures.

        Args:
            max_primitives: Maximum number of primitives (triangles or edges)
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.max_primitives = max_primitives
        self.num_nodes = 2 * max_primitives - 1

        # Get float type from precision config
        float_type = self.cfg.float_type

        # AABB bounding volumes: lower (x,y,z), upper (x,y,z)
        self.bv_lower = ti.Vector.field(3, dtype=float_type, shape=self.num_nodes)
        self.bv_upper = ti.Vector.field(3, dtype=float_type, shape=self.num_nodes)

        # Temporary buffers for single-pass AABB reorder (avoids memory aliasing)
        self.temp_lower = ti.Vector.field(3, dtype=float_type, shape=max_primitives)
        self.temp_upper = ti.Vector.field(3, dtype=float_type, shape=max_primitives)

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
        self.scene_lower = ti.Vector.field(3, dtype=float_type, shape=())
        self.scene_upper = ti.Vector.field(3, dtype=float_type, shape=())

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
    def morton_code_3d(self, x: ti.template(), y: ti.template(), z: ti.template()) -> ti.u32:
        """
        Compute 30-bit Morton code for a 3D point normalized to [0, 1].
        """
        resolution = 1024.0
        x_clamped = ti.min(ti.max(x * resolution, 0.0), resolution - 1.0)
        y_clamped = ti.min(ti.max(y * resolution, 0.0), resolution - 1.0)
        z_clamped = ti.min(ti.max(z * resolution, 0.0), resolution - 1.0)

        xx = self.expand_bits(ti.cast(x_clamped, ti.u32))
        yy = self.expand_bits(ti.cast(y_clamped, ti.u32))
        zz = self.expand_bits(ti.cast(z_clamped, ti.u32))

        return (xx << 2) | (yy << 1) | zz

    @ti.func
    def aabb_merge(self, idx1: ti.i32, idx2: ti.i32, dst: ti.i32):
        """Merge two AABBs into dst."""
        self.bv_lower[dst] = ti.min(self.bv_lower[idx1], self.bv_lower[idx2])
        self.bv_upper[dst] = ti.max(self.bv_upper[idx1], self.bv_upper[idx2])

    @ti.func
    def aabb_overlap(self, idx1: ti.i32, idx2: ti.i32, gap: ti.template()) -> bool:
        """
        Check if two AABBs overlap with a gap tolerance.

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
    def aabb_overlap_point(self, point: ti.template(), idx: ti.i32, gap: ti.template()) -> bool:
        """
        Check if a point's AABB overlaps with node's AABB.

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
        """
        Count leading zeros of XOR of two values (common prefix length).

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

    @ti.func
    def is_leaf(self, idx: ti.i32, num_leaves: ti.i32) -> bool:
        """Check if a node index is a leaf node."""
        return idx >= num_leaves - 1

    @ti.func
    def get_leaf_primitive_idx(self, leaf_idx: ti.i32) -> ti.u32:
        """Get the primitive index stored in a leaf node."""
        return self.element_idx[leaf_idx]
