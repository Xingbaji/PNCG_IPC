"""
GPU Radix Sort implementation for LBVH Morton code sorting.

This custom implementation is optimized for:
1. 64-bit Morton codes (32-bit code + 32-bit index)
2. Reduced kernel launch overhead via fused operations
3. Block-level histogram computation
4. Efficient prefix sum using hierarchical reduction

Based on GPU Gems 3 Chapter 39: Parallel Prefix Sum (Scan) with CUDA
and "Introduction to GPU Radix Sort" by Duane Merrill.
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class RadixSortGPU:
    """
    GPU-accelerated Radix Sort for 64-bit keys with associated 32-bit values.

    Algorithm:
    1. For each radix digit (8 bits at a time, 8 passes for 64-bit):
       a. Compute per-block histograms
       b. Prefix sum across blocks
       c. Scatter elements to sorted positions

    Optimizations:
    - 8-bit radix (256 buckets) balances passes vs. histogram size
    - Block-level histogram reduces global memory traffic
    - Fused histogram + local sort where possible
    """

    def __init__(self, max_elements: int):
        """
        Initialize radix sort buffers.

        Args:
            max_elements: Maximum number of elements to sort
        """
        self.max_elements = max_elements

        # Radix parameters
        self.RADIX_BITS = 8
        self.RADIX_SIZE = 1 << self.RADIX_BITS  # 256
        self.NUM_PASSES = 8  # 64 bits / 8 bits per pass

        # Block parameters
        self.BLOCK_SIZE = 256
        self.max_blocks = (max_elements + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Double buffers for keys and values
        self.keys_in = ti.field(dtype=ti.u64, shape=max_elements)
        self.keys_out = ti.field(dtype=ti.u64, shape=max_elements)
        self.vals_in = ti.field(dtype=ti.u32, shape=max_elements)
        self.vals_out = ti.field(dtype=ti.u32, shape=max_elements)

        # Per-block histograms: [block_id, bucket]
        self.block_histograms = ti.field(dtype=ti.i32, shape=(self.max_blocks, self.RADIX_SIZE))

        # Global histogram (prefix sum result)
        self.global_offsets = ti.field(dtype=ti.i32, shape=self.RADIX_SIZE)

        # Block-level prefix sums for scatter
        self.block_offsets = ti.field(dtype=ti.i32, shape=(self.max_blocks, self.RADIX_SIZE))

        # Temporary for hierarchical prefix sum
        self.block_sums = ti.field(dtype=ti.i32, shape=self.max_blocks + 1)

    @ti.kernel
    def _compute_histograms(self, n: ti.i32, shift: ti.i32):
        """
        Compute per-block histograms for current radix digit.

        Each block computes histogram of its elements' radix digits.
        """
        RADIX_MASK = ti.u64(0xFF)

        # Clear histograms
        for bid in range(self.max_blocks):
            for bucket in range(self.RADIX_SIZE):
                self.block_histograms[bid, bucket] = 0

        ti.sync()

        # Compute histograms
        for i in range(n):
            key = self.keys_in[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // self.BLOCK_SIZE
            ti.atomic_add(self.block_histograms[block_id, digit], 1)

    @ti.kernel
    def _prefix_sum_histograms(self, n_blocks: ti.i32):
        """
        Compute prefix sum across all block histograms.

        Result: block_offsets[b, d] = sum of all histogram[b', d'] where
        (b' < b) or (b' == b and d' < d)

        This gives the global offset for each block's elements of each digit.
        """
        # First, compute global histogram (sum across blocks)
        for digit in range(self.RADIX_SIZE):
            total = 0
            for bid in range(n_blocks):
                total += self.block_histograms[bid, digit]
            self.global_offsets[digit] = total

        ti.sync()

        # Exclusive prefix sum on global histogram
        # global_offsets[d] = sum of counts for digits < d
        running_sum = 0
        for digit in range(self.RADIX_SIZE):
            count = self.global_offsets[digit]
            self.global_offsets[digit] = running_sum
            running_sum += count

        ti.sync()

        # Compute per-block offsets within each digit
        # block_offsets[b, d] = global_offsets[d] + sum of block_histograms[b', d] for b' < b
        for digit in range(self.RADIX_SIZE):
            offset = self.global_offsets[digit]
            for bid in range(n_blocks):
                self.block_offsets[bid, digit] = offset
                offset += self.block_histograms[bid, digit]

    @ti.kernel
    def _scatter(self, n: ti.i32, shift: ti.i32):
        """
        Scatter elements to their sorted positions based on current radix digit.

        Uses per-block offsets and atomics for conflict resolution within blocks.
        """
        RADIX_MASK = ti.u64(0xFF)

        for i in range(n):
            key = self.keys_in[i]
            val = self.vals_in[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // self.BLOCK_SIZE

            # Get destination index using atomic increment
            dest = ti.atomic_add(self.block_offsets[block_id, digit], 1)

            self.keys_out[dest] = key
            self.vals_out[dest] = val

    @ti.kernel
    def _swap_buffers(self, n: ti.i32):
        """Swap input and output buffers for next pass."""
        for i in range(n):
            self.keys_in[i] = self.keys_out[i]
            self.vals_in[i] = self.vals_out[i]

    def sort(self, keys: ti.Field, values: ti.Field, n: int):
        """
        Sort keys and reorder values accordingly.

        Args:
            keys: Field of uint64 keys (Morton codes)
            values: Field of uint32 values (indices)
            n: Number of elements to sort
        """
        if n <= 1:
            return

        n_blocks = (n + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Copy input to internal buffers
        self._copy_input(keys, values, n)

        # Perform radix sort passes (LSB first)
        for pass_idx in range(self.NUM_PASSES):
            shift = pass_idx * self.RADIX_BITS

            # Compute per-block histograms
            self._compute_histograms(n, shift)

            # Prefix sum across histograms
            self._prefix_sum_histograms(n_blocks)

            # Scatter to sorted positions
            self._scatter(n, shift)

            # Swap buffers for next pass (except last)
            if pass_idx < self.NUM_PASSES - 1:
                self._swap_buffers(n)

        # Copy result back (result is in keys_out/vals_out after even passes, swapped otherwise)
        if self.NUM_PASSES % 2 == 0:
            self._copy_output(keys, values, n)
        else:
            self._copy_output_swapped(keys, values, n)

    @ti.kernel
    def _copy_input(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            self.keys_in[i] = keys[i]
            self.vals_in[i] = values[i]

    @ti.kernel
    def _copy_output(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            keys[i] = self.keys_out[i]
            values[i] = self.vals_out[i]

    @ti.kernel
    def _copy_output_swapped(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            keys[i] = self.keys_in[i]
            values[i] = self.vals_in[i]


@ti.data_oriented
class RadixSortGPU_Optimized:
    """
    Optimized GPU Radix Sort with reduced passes and better memory access.

    Optimizations over basic version:
    1. Flattened histogram layout for better cache behavior
    2. Separate kernels for A->B and B->A passes (avoid template issues)
    3. Coalesced memory access patterns
    """

    def __init__(self, max_elements: int):
        self.max_elements = max_elements

        # Use 8-bit radix: 8 passes for 64-bit
        self.RADIX_BITS = 8
        self.RADIX_SIZE = 256
        self.NUM_PASSES = 8

        self.BLOCK_SIZE = 256
        self.max_blocks = (max_elements + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Double buffers
        self.keys_a = ti.field(dtype=ti.u64, shape=max_elements)
        self.keys_b = ti.field(dtype=ti.u64, shape=max_elements)
        self.vals_a = ti.field(dtype=ti.u32, shape=max_elements)
        self.vals_b = ti.field(dtype=ti.u32, shape=max_elements)

        # Histograms: flatten to 1D for better cache behavior
        # Layout: block_histograms[block_id * RADIX_SIZE + bucket]
        self.block_histograms = ti.field(dtype=ti.i32, shape=self.max_blocks * self.RADIX_SIZE)

        # Global prefix sums
        self.global_prefix = ti.field(dtype=ti.i32, shape=self.RADIX_SIZE)

        # Per-block offsets (for scatter)
        self.block_prefix = ti.field(dtype=ti.i32, shape=self.max_blocks * self.RADIX_SIZE)

    # ========== A -> B kernels ==========

    @ti.kernel
    def _histogram_pass_a(self, n: ti.i32, shift: ti.i32):
        """Compute histograms from buffer A."""
        RADIX_MASK = ti.u64(0xFF)
        RADIX_SIZE = 256
        n_blocks = (n + 255) // 256

        # Clear histograms
        for idx in range(n_blocks * RADIX_SIZE):
            self.block_histograms[idx] = 0

        ti.sync()

        # Count elements per bucket per block
        for i in range(n):
            key = self.keys_a[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // 256
            hist_idx = block_id * RADIX_SIZE + digit
            ti.atomic_add(self.block_histograms[hist_idx], 1)

    @ti.kernel
    def _scatter_pass_a_to_b(self, n: ti.i32, shift: ti.i32):
        """Scatter elements from A to B."""
        RADIX_MASK = ti.u64(0xFF)
        RADIX_SIZE = 256

        for i in range(n):
            key = self.keys_a[i]
            val = self.vals_a[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // 256
            prefix_idx = block_id * RADIX_SIZE + digit

            dest = ti.atomic_add(self.block_prefix[prefix_idx], 1)

            self.keys_b[dest] = key
            self.vals_b[dest] = val

    # ========== B -> A kernels ==========

    @ti.kernel
    def _histogram_pass_b(self, n: ti.i32, shift: ti.i32):
        """Compute histograms from buffer B."""
        RADIX_MASK = ti.u64(0xFF)
        RADIX_SIZE = 256
        n_blocks = (n + 255) // 256

        # Clear histograms
        for idx in range(n_blocks * RADIX_SIZE):
            self.block_histograms[idx] = 0

        ti.sync()

        # Count elements per bucket per block
        for i in range(n):
            key = self.keys_b[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // 256
            hist_idx = block_id * RADIX_SIZE + digit
            ti.atomic_add(self.block_histograms[hist_idx], 1)

    @ti.kernel
    def _scatter_pass_b_to_a(self, n: ti.i32, shift: ti.i32):
        """Scatter elements from B to A."""
        RADIX_MASK = ti.u64(0xFF)
        RADIX_SIZE = 256

        for i in range(n):
            key = self.keys_b[i]
            val = self.vals_b[i]
            digit = ti.cast((key >> shift) & RADIX_MASK, ti.i32)
            block_id = i // 256
            prefix_idx = block_id * RADIX_SIZE + digit

            dest = ti.atomic_add(self.block_prefix[prefix_idx], 1)

            self.keys_a[dest] = key
            self.vals_a[dest] = val

    # ========== Common kernels ==========

    @ti.kernel
    def _prefix_sum_pass(self, n_blocks: ti.i32):
        """Compute global and per-block prefix sums."""
        RADIX_SIZE = 256

        # Sum across blocks for each digit
        for digit in range(RADIX_SIZE):
            total = 0
            for bid in range(n_blocks):
                total += self.block_histograms[bid * RADIX_SIZE + digit]
            self.global_prefix[digit] = total

        ti.sync()

        # Exclusive prefix sum on global counts
        running = 0
        for digit in range(RADIX_SIZE):
            count = self.global_prefix[digit]
            self.global_prefix[digit] = running
            running += count

        ti.sync()

        # Per-block prefix sums
        for digit in range(RADIX_SIZE):
            offset = self.global_prefix[digit]
            for bid in range(n_blocks):
                self.block_prefix[bid * RADIX_SIZE + digit] = offset
                offset += self.block_histograms[bid * RADIX_SIZE + digit]

    @ti.kernel
    def _copy_to_a(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            self.keys_a[i] = keys[i]
            self.vals_a[i] = values[i]

    @ti.kernel
    def _copy_from_a(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            keys[i] = self.keys_a[i]
            values[i] = self.vals_a[i]

    @ti.kernel
    def _copy_from_b(self, keys: ti.template(), values: ti.template(), n: ti.i32):
        for i in range(n):
            keys[i] = self.keys_b[i]
            values[i] = self.vals_b[i]

    def sort(self, keys: ti.Field, values: ti.Field, n: int):
        """Sort keys with associated values."""
        if n <= 1:
            return

        n_blocks = (n + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Copy input to buffer A
        self._copy_to_a(keys, values, n)

        # Perform radix sort passes
        # Pass 0, 2, 4, 6: A -> B
        # Pass 1, 3, 5, 7: B -> A
        for pass_idx in range(self.NUM_PASSES):
            shift = pass_idx * self.RADIX_BITS

            if pass_idx % 2 == 0:
                # A -> B
                self._histogram_pass_a(n, shift)
                self._prefix_sum_pass(n_blocks)
                self._scatter_pass_a_to_b(n, shift)
            else:
                # B -> A
                self._histogram_pass_b(n, shift)
                self._prefix_sum_pass(n_blocks)
                self._scatter_pass_b_to_a(n, shift)

        # After 8 passes (even number), result is in B
        # (A->B, B->A, A->B, B->A, A->B, B->A, A->B, B->A)
        # Pass 0: A->B, Pass 1: B->A, ..., Pass 7: B->A
        # After pass 7, result is in A
        self._copy_from_a(keys, values, n)


# Convenience function to create sorter
def create_radix_sorter(max_elements: int, optimized: bool = True):
    """
    Create a radix sorter instance.

    Args:
        max_elements: Maximum number of elements
        optimized: Use optimized version (default True)

    Returns:
        RadixSort instance with sort() method
    """
    if optimized:
        return RadixSortGPU_Optimized(max_elements)
    else:
        return RadixSortGPU(max_elements)
