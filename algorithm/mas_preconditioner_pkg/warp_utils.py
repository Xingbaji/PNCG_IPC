"""
Warp-level utility functions for MAS Preconditioner.

This module provides bit manipulation functions and warp-level reduction
operations that emulate CUDA warp intrinsics for Taichi.

Reference: MASPreconditioner.cu (CUDA reference implementation)
"""

import taichi as ti

from .constants import BANKSIZE


# ==============================================================================
# Standalone Taichi Functions (can be used across modules)
# ==============================================================================

@ti.func
def bit_reverse_u32(x: ti.u32) -> ti.u32:
    """
    Reverse bits of a 32-bit unsigned integer.

    Equivalent to CUDA's __brev() intrinsic.

    Args:
        x: Input 32-bit unsigned integer

    Returns:
        Bit-reversed value
    """
    x = ((x & ti.u32(0x55555555)) << 1) | ((x & ti.u32(0xAAAAAAAA)) >> 1)
    x = ((x & ti.u32(0x33333333)) << 2) | ((x & ti.u32(0xCCCCCCCC)) >> 2)
    x = ((x & ti.u32(0x0F0F0F0F)) << 4) | ((x & ti.u32(0xF0F0F0F0)) >> 4)
    x = ((x & ti.u32(0x00FF00FF)) << 8) | ((x & ti.u32(0xFF00FF00)) >> 8)
    x = (x << 16) | (x >> 16)
    return x


@ti.func
def count_leading_zeros_u32(x: ti.u32) -> ti.i32:
    """
    Count leading zeros in a 32-bit unsigned integer.

    Equivalent to CUDA's __clz() intrinsic.

    Args:
        x: Input 32-bit unsigned integer

    Returns:
        Number of leading zeros (0-32)
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
def popcount_u32(x: ti.u32) -> ti.i32:
    """
    Count set bits (population count) in a 32-bit unsigned integer.

    Equivalent to CUDA's __popc() intrinsic.

    Uses the parallel bit counting algorithm (SWAR) for O(1) performance.

    Args:
        x: Input 32-bit unsigned integer

    Returns:
        Number of set bits (0-32)
    """
    x = x - ((x >> 1) & ti.u32(0x55555555))
    x = (x & ti.u32(0x33333333)) + ((x >> 2) & ti.u32(0x33333333))
    x = (x + (x >> 4)) & ti.u32(0x0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    return ti.i32(x & ti.u32(0x3F))


@ti.func
def find_first_set_u32(x: ti.u32) -> ti.i32:
    """
    Find position of first set bit (1-indexed, returns 0 if no bits set).

    Equivalent to CUDA's __ffs() intrinsic.

    Args:
        x: Input 32-bit unsigned integer

    Returns:
        Position of first set bit (1-32), or 0 if no bits set
    """
    result = ti.i32(0)
    if x != 0:
        # x & (~x + 1) isolates the lowest set bit
        lowest_bit = x & (~x + ti.u32(1))
        result = 32 - count_leading_zeros_u32(lowest_bit)
    return result


@ti.func
def find_first_set_zero_indexed(x: ti.u32) -> ti.i32:
    """
    Find position of first set bit (0-indexed), or -1 if none.

    Uses De Bruijn sequence for O(1) bit position lookup.

    Args:
        x: Input 32-bit unsigned integer

    Returns:
        Position of first set bit (0-31), or -1 if no bits set
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
    """
    Return bitmask of lanes less than lane_id.

    Equivalent to CUDA's __lanemask_lt() for BANKSIZE=16 warps.

    Args:
        lane_id: Current lane ID (0 to BANKSIZE-1)

    Returns:
        Bitmask with bits [0, lane_id) set
    """
    return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)


# ==============================================================================
# WarpReductionHelper Class
# ==============================================================================

@ti.data_oriented
class WarpReductionHelper:
    """
    Helper class for warp-level reduction operations.

    Since Taichi 1.7.4 doesn't support ti.simt warp primitives,
    this class provides field-based alternatives that emulate the
    behavior of CUDA warp intrinsics.

    Reference: MASPreconditioner.cu (lines 930-948)
    """

    def __init__(self, n_warps: int, warp_size: int = BANKSIZE):
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
                        curr_rep = find_first_set_u32(curr_conn) - 1
                        prev_rep = find_first_set_u32(prev_conn) - 1

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
                reversed_mask = bit_reverse_u32(mask)
                shifted = reversed_mask << ti.u32(lane_id + 1)
                clz = count_leading_zeros_u32(shifted)

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
