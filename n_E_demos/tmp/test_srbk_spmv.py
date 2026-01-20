"""
Test script for SRBK SpMV and Warp Reduction implementations.

This script verifies the correctness of:
1. SRBKSpMV: Symmetric Reduce-By-Key Sparse Matrix-Vector Multiplication
2. WarpReductionHelper: Warp-level segmented reduction

Reference implementations from:
- /root/Stiff-GIPC_init/StiffGIPC/linear_system/utils/spmv.cu
- /root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np
import time

# Initialize Taichi
try:
    ti.init(arch=ti.cuda)
    print("[Taichi] Initialized with CUDA backend")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Initialized with CPU backend")

from algorithm.mas_preconditioner import (
    SRBKSpMV, WarpReductionHelper,
    _bit_reverse_u32, _count_leading_zeros_u32, _popcount_u32, _find_first_set_u32,
    BANKSIZE
)


def test_bit_operations():
    """Test CUDA intrinsic equivalents."""
    print("\n=== Testing Bit Operations ===")

    @ti.kernel
    def test_brev(x: ti.u32) -> ti.u32:
        return _bit_reverse_u32(x)

    @ti.kernel
    def test_clz(x: ti.u32) -> ti.i32:
        return _count_leading_zeros_u32(x)

    @ti.kernel
    def test_popc(x: ti.u32) -> ti.i32:
        return _popcount_u32(x)

    @ti.kernel
    def test_ffs(x: ti.u32) -> ti.i32:
        return _find_first_set_u32(x)

    # Test cases
    test_values = [0, 1, 0x80000000, 0xFFFFFFFF, 0x12345678, 0b10110100]

    print("\nBit reverse (__brev equivalent):")
    for v in test_values:
        result = test_brev(v)
        # Manual verification for simple cases
        print(f"  brev(0x{v:08X}) = 0x{result:08X}")

    print("\nCount leading zeros (__clz equivalent):")
    expected_clz = [32, 31, 0, 0, 3, 24]  # Expected values
    for v, exp in zip(test_values, expected_clz):
        result = test_clz(v)
        status = "PASS" if result == exp else f"FAIL (expected {exp})"
        print(f"  clz(0x{v:08X}) = {result} [{status}]")

    print("\nPopulation count (__popc equivalent):")
    expected_popc = [0, 1, 1, 32, 13, 4]
    for v, exp in zip(test_values, expected_popc):
        result = test_popc(v)
        status = "PASS" if result == exp else f"FAIL (expected {exp})"
        print(f"  popc(0x{v:08X}) = {result} [{status}]")

    print("\nFind first set (__ffs equivalent, 1-indexed):")
    expected_ffs = [0, 1, 32, 1, 4, 3]  # 1-indexed position of first set bit
    for v, exp in zip(test_values, expected_ffs):
        result = test_ffs(v)
        status = "PASS" if result == exp else f"FAIL (expected {exp})"
        print(f"  ffs(0x{v:08X}) = {result} [{status}]")


def test_srbk_spmv(n_verts=100, sparsity=0.02, verbose=True):
    """Test SRBK SpMV implementation.

    Args:
        n_verts: Number of vertices (matrix size = n_verts * 3)
        sparsity: Fraction of off-diagonal blocks to fill (0-1)
        verbose: Whether to print detailed results
    """
    if verbose:
        print(f"\n=== Testing SRBK SpMV (n_verts={n_verts}, sparsity={sparsity:.1%}) ===")

    # Estimate max triplets: diagonal + sparse off-diagonal
    max_off_diag = int(n_verts * (n_verts - 1) / 2 * sparsity) + n_verts
    max_triplets = n_verts + max_off_diag + 1000  # Extra buffer

    # Initialize SpMV
    spmv = SRBKSpMV(max_triplets, n_verts * 3)

    # Create a simple test matrix: tridiagonal + some random entries
    np.random.seed(42)

    # For large matrices, skip dense verification
    use_dense_verify = n_verts <= 2000

    if use_dense_verify:
        A_dense = np.zeros((n_verts * 3, n_verts * 3))
    else:
        A_dense = None

    spmv.clear()

    # Add diagonal blocks
    for i in range(n_verts):
        diag_block = np.eye(3) * (2.0 + np.random.rand())
        if use_dense_verify:
            A_dense[i*3:(i+1)*3, i*3:(i+1)*3] = diag_block

        # Convert to taichi matrix and add
        diag_ti = ti.Matrix(diag_block.tolist(), dt=ti.f32)
        spmv.add_triplet(i, i, diag_ti)

    # Add off-diagonal blocks based on sparsity pattern
    # Use a random sparse pattern
    n_off_diag = int(n_verts * (n_verts - 1) / 2 * sparsity)

    # Generate random (i, j) pairs for upper triangle
    if n_off_diag > 0:
        all_pairs = []
        # For efficiency, sample from chunks
        for i in range(n_verts - 1):
            for j in range(i + 1, min(i + 50, n_verts)):  # Neighbor band
                all_pairs.append((i, j))

        # Also add some random long-range connections
        for _ in range(min(n_off_diag // 2, 10000)):
            i = np.random.randint(0, n_verts - 1)
            j = np.random.randint(i + 1, n_verts)
            all_pairs.append((i, j))

        # Remove duplicates and sample
        all_pairs = list(set(all_pairs))
        np.random.shuffle(all_pairs)
        selected_pairs = all_pairs[:n_off_diag]

        for i, j in selected_pairs:
            off_diag = np.random.randn(3, 3) * 0.5
            off_diag = (off_diag + off_diag.T) / 2  # Make symmetric

            if use_dense_verify:
                A_dense[i*3:(i+1)*3, j*3:(j+1)*3] = off_diag
                A_dense[j*3:(j+1)*3, i*3:(i+1)*3] = off_diag.T

            # Add to sparse (upper only)
            off_diag_ti = ti.Matrix(off_diag.tolist(), dt=ti.f32)
            spmv.add_triplet(i, j, off_diag_ti)

    n_triplets = spmv.n_triplets[None]
    if verbose:
        print(f"  Created matrix with {n_triplets} triplets ({n_triplets / (n_verts * n_verts) * 100:.2f}% density)")

    # Sort triplets
    spmv.sort_by_row()
    if verbose:
        print("  Sorted triplets by row")

    # Create test vectors
    x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    y_naive = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    y_row = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    # Initialize x with random values
    x_np = np.random.randn(n_verts, 3)
    x.from_numpy(x_np)

    # Compute using dense matrix (ground truth) if small enough
    if use_dense_verify:
        x_flat = x_np.flatten()
        y_dense = A_dense @ x_flat
        y_dense_reshaped = y_dense.reshape(n_verts, 3)

    # Compute using naive SpMV
    spmv.spmv_naive(x, y_naive, 1.0, 0.0)
    y_naive_np = y_naive.to_numpy()

    # Compute using row-parallel SpMV
    spmv.spmv_row_parallel(x, y_row, 1.0, 0.0)
    y_row_np = y_row.to_numpy()

    # Compare results
    if use_dense_verify:
        naive_error = np.max(np.abs(y_naive_np - y_dense_reshaped))
        row_error = np.max(np.abs(y_row_np - y_dense_reshaped))
    else:
        # Compare naive and row-parallel against each other
        naive_error = 0.0
        row_error = np.max(np.abs(y_row_np - y_naive_np))

    if verbose:
        print(f"\n  Results:")
        if use_dense_verify:
            print(f"    Dense (ground truth):  ||y|| = {np.linalg.norm(y_dense_reshaped):.6f}")
        print(f"    Naive SpMV:            ||y|| = {np.linalg.norm(y_naive_np):.6f}", end="")
        if use_dense_verify:
            print(f", max error = {naive_error:.2e}")
        else:
            print()
        print(f"    Row-parallel SpMV:     ||y|| = {np.linalg.norm(y_row_np):.6f}, max error vs naive = {row_error:.2e}")

    # Performance comparison
    n_iters = 100 if n_verts <= 5000 else 50
    if verbose:
        print(f"\n  Performance ({n_iters} iterations):")

    # Warm up
    for _ in range(10):
        spmv.spmv_naive(x, y_naive, 1.0, 0.0)
        spmv.spmv_row_parallel(x, y_row, 1.0, 0.0)
    ti.sync()

    # Time naive
    start = time.perf_counter()
    for _ in range(n_iters):
        spmv.spmv_naive(x, y_naive, 1.0, 0.0)
    ti.sync()
    naive_time = (time.perf_counter() - start) / n_iters * 1000

    # Time row-parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        spmv.spmv_row_parallel(x, y_row, 1.0, 0.0)
    ti.sync()
    row_time = (time.perf_counter() - start) / n_iters * 1000

    if verbose:
        print(f"    Naive SpMV:        {naive_time:.3f} ms")
        print(f"    Row-parallel SpMV: {row_time:.3f} ms")
        print(f"    Speedup:           {naive_time/row_time:.2f}x")

    # Verify correctness
    tol = 1e-10
    passed = (naive_error < tol if use_dense_verify else True) and row_error < tol
    if verbose:
        if passed:
            print("\n  [PASS] SRBK SpMV implementations are correct")
        else:
            print("\n  [FAIL] SRBK SpMV implementations have errors")

    return passed, naive_time, row_time, n_triplets


def test_warp_reduction():
    """Test WarpReductionHelper implementation."""
    print("\n=== Testing Warp Reduction ===")

    n_verts = 64  # 4 warps of size 16
    n_warps = (n_verts + BANKSIZE - 1) // BANKSIZE

    # Create warp reduction helper
    helper = WarpReductionHelper(n_warps, BANKSIZE)

    # Create test data and connectivity mask
    data = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    output = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    connect_mask = ti.field(dtype=ti.u32, shape=n_verts)

    # Initialize data: each vertex has value [idx, idx*2, idx*3]
    data_np = np.zeros((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        data_np[i] = [i, i*2, i*3]
    data.from_numpy(data_np)

    # Test case 1: All vertices in same warp are fully connected
    print("\n  Test 1: Fully connected warps")
    connect_mask_np = np.zeros(n_verts, dtype=np.uint32)
    for warp_id in range(n_warps):
        # All lanes connected to all others in this warp
        full_mask = (1 << BANKSIZE) - 1  # 0xFFFF for BANKSIZE=16
        for lane_id in range(BANKSIZE):
            idx = warp_id * BANKSIZE + lane_id
            if idx < n_verts:
                connect_mask_np[idx] = full_mask
    connect_mask.from_numpy(connect_mask_np)

    # Perform reduction
    helper.reduce(data, output, n_verts, connect_mask)
    output_np = output.to_numpy()

    # Verify: lane 0 of each warp should have sum of all lanes
    for warp_id in range(n_warps):
        warp_start = warp_id * BANKSIZE
        warp_end = min(warp_start + BANKSIZE, n_verts)

        expected_sum = np.sum(data_np[warp_start:warp_end], axis=0)
        actual_sum = output_np[warp_start]

        error = np.max(np.abs(expected_sum - actual_sum))
        status = "PASS" if error < 1e-5 else f"FAIL (error={error:.2e})"
        print(f"    Warp {warp_id}: expected={expected_sum}, got={actual_sum} [{status}]")

    # Test case 2: Each vertex is its own component (no reduction)
    print("\n  Test 2: No connectivity (each vertex is its own component)")
    for i in range(n_verts):
        lane_id = i % BANKSIZE
        connect_mask_np[i] = 1 << lane_id  # Only self-connected
    connect_mask.from_numpy(connect_mask_np)

    helper.reduce(data, output, n_verts, connect_mask)
    output_np = output.to_numpy()

    # Verify: each vertex should have its original value
    max_error = np.max(np.abs(output_np - data_np))
    status = "PASS" if max_error < 1e-5 else f"FAIL (max_error={max_error:.2e})"
    print(f"    Max error from identity: {max_error:.2e} [{status}]")

    # Test case 3: Pairs of vertices connected
    print("\n  Test 3: Pairwise connectivity")
    for i in range(n_verts):
        warp_id = i // BANKSIZE
        lane_id = i % BANKSIZE
        partner_lane = lane_id ^ 1  # XOR with 1 to get partner (0<->1, 2<->3, etc.)
        connect_mask_np[i] = (1 << lane_id) | (1 << partner_lane)
    connect_mask.from_numpy(connect_mask_np)

    helper.reduce(data, output, n_verts, connect_mask)
    output_np = output.to_numpy()

    # Verify: even lanes should have sum of pair
    errors = []
    for i in range(0, n_verts, 2):
        expected_sum = data_np[i] + data_np[i+1] if i+1 < n_verts else data_np[i]
        actual_sum = output_np[i]
        errors.append(np.max(np.abs(expected_sum - actual_sum)))

    max_error = max(errors)
    status = "PASS" if max_error < 1e-5 else f"FAIL (max_error={max_error:.2e})"
    print(f"    Max error from pairwise reduction: {max_error:.2e} [{status}]")

    return max_error < 1e-5


def test_srbk_spmv_scaling():
    """Test SRBK SpMV at different scales to evaluate performance scaling."""
    print("\n" + "=" * 60)
    print("SRBK SpMV Scaling Test")
    print("=" * 60)

    # Test configurations: (n_verts, sparsity)
    test_configs = [
        (100, 0.05),      # Small
        (500, 0.02),      # Medium-small
        (1000, 0.01),     # Medium
        (2000, 0.005),    # Medium-large
        (5000, 0.002),    # Large
        (10000, 0.001),   # Very large
    ]

    results = []
    print(f"\n{'n_verts':>8} {'triplets':>10} {'naive(ms)':>10} {'row(ms)':>10} {'speedup':>8} {'status':>8}")
    print("-" * 60)

    for n_verts, sparsity in test_configs:
        try:
            passed, naive_time, row_time, n_triplets = test_srbk_spmv(
                n_verts=n_verts, sparsity=sparsity, verbose=False
            )
            speedup = naive_time / row_time if row_time > 0 else 0
            status = "PASS" if passed else "FAIL"
            print(f"{n_verts:>8} {n_triplets:>10} {naive_time:>10.3f} {row_time:>10.3f} {speedup:>8.2f}x {status:>8}")
            results.append((n_verts, n_triplets, naive_time, row_time, speedup, passed))
        except Exception as e:
            print(f"{n_verts:>8} {'ERROR':>10} {str(e)[:40]}")
            results.append((n_verts, 0, 0, 0, 0, False))

    return results


def test_warp_reduction_scaling():
    """Test Warp Reduction at different scales."""
    print("\n" + "=" * 60)
    print("Warp Reduction Scaling Test")
    print("=" * 60)

    test_sizes = [64, 256, 1024, 4096, 16384]

    print(f"\n{'n_verts':>8} {'n_warps':>8} {'time(ms)':>10} {'status':>8}")
    print("-" * 40)

    for n_verts in test_sizes:
        n_warps = (n_verts + BANKSIZE - 1) // BANKSIZE

        # Create warp reduction helper
        helper = WarpReductionHelper(n_warps, BANKSIZE)

        # Create test data
        data = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        output = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        connect_mask = ti.field(dtype=ti.u32, shape=n_verts)

        # Initialize data
        data_np = np.random.randn(n_verts, 3).astype(np.float32)
        data.from_numpy(data_np)

        # Full connectivity within warps
        connect_mask_np = np.zeros(n_verts, dtype=np.uint32)
        full_mask = (1 << BANKSIZE) - 1
        for i in range(n_verts):
            connect_mask_np[i] = full_mask
        connect_mask.from_numpy(connect_mask_np)

        # Warm up
        for _ in range(10):
            helper.reduce(data, output, n_verts, connect_mask)
        ti.sync()

        # Benchmark
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            helper.reduce(data, output, n_verts, connect_mask)
        ti.sync()
        avg_time = (time.perf_counter() - start) / n_iters * 1000

        # Verify
        output_np = output.to_numpy()
        errors = []
        for warp_id in range(n_warps):
            warp_start = warp_id * BANKSIZE
            warp_end = min(warp_start + BANKSIZE, n_verts)
            expected_sum = np.sum(data_np[warp_start:warp_end], axis=0)
            actual_sum = output_np[warp_start]
            errors.append(np.max(np.abs(expected_sum - actual_sum)))

        passed = max(errors) < 1e-4
        status = "PASS" if passed else "FAIL"

        print(f"{n_verts:>8} {n_warps:>8} {avg_time:>10.3f} {status:>8}")

    return True


def main():
    print("=" * 60)
    print("SRBK SpMV and Warp Reduction Test Suite")
    print("=" * 60)

    results = []

    # Test bit operations
    test_bit_operations()

    # Test SRBK SpMV with small size for correctness
    print("\n--- Correctness Test (small scale) ---")
    passed, _, _, _ = test_srbk_spmv(n_verts=100, sparsity=0.05)
    results.append(("SRBK SpMV (correctness)", passed))

    # Test Warp Reduction
    results.append(("Warp Reduction", test_warp_reduction()))

    # Scaling tests
    print("\n--- Scaling Tests ---")
    test_srbk_spmv_scaling()
    test_warp_reduction_scaling()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        all_passed = all_passed and passed

    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
