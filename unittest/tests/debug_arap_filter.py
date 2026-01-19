"""
Debug ARAP_filter to find the source of negative eigenvalues.
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

# Debug fields
debug_s = ti.field(dtype=ti.f32, shape=3)
debug_lambda = ti.field(dtype=ti.f32, shape=3)
debug_sums = ti.field(dtype=ti.f32, shape=3)
debug_filter_triggered = ti.field(dtype=ti.i32, shape=3)


@ti.func
def ssvd(F):
    """SVD using Taichi built-in."""
    U, sig, V = ti.svd(F, ti.f32)
    return U, sig, V


@ti.kernel
def debug_arap_filter(F_in: ti.types.matrix(3, 3, ti.f32), mu: ti.f32, la: ti.f32):
    """Debug version of ARAP_filter."""
    U, sig, V = ssvd(F_in)

    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]

    debug_s[0] = s0
    debug_s[1] = s1
    debug_s[2] = s2

    # Compute sums
    sum_12 = s1 + s2
    sum_02 = s0 + s2
    sum_01 = s0 + s1

    debug_sums[0] = sum_12
    debug_sums[1] = sum_02
    debug_sums[2] = sum_01

    # Compute raw lambdas
    lambda0 = 2.0 / sum_12
    lambda1 = 2.0 / sum_02
    lambda2 = 2.0 / sum_01

    # Apply filter
    debug_filter_triggered[0] = 0
    debug_filter_triggered[1] = 0
    debug_filter_triggered[2] = 0

    if sum_12 < 2.0:
        lambda0 = 1.0
        debug_filter_triggered[0] = 1
    if sum_02 < 2.0:
        lambda1 = 1.0
        debug_filter_triggered[1] = 1
    if sum_01 < 2.0:
        lambda2 = 1.0
        debug_filter_triggered[2] = 1

    debug_lambda[0] = lambda0
    debug_lambda[1] = lambda1
    debug_lambda[2] = lambda2


def main():
    print("=" * 60)
    print("ARAP_filter Debug")
    print("=" * 60)

    # Test with perturbed F
    np.random.seed(42)
    F = np.eye(3, dtype=np.float32) + 0.0001 * np.random.randn(3, 3).astype(np.float32)

    print(f"\nInput F:")
    print(F)

    # Run debug kernel
    F_ti = ti.Matrix(F.tolist())
    debug_arap_filter(F_ti, 1.0, 1.0)

    # Get results
    s = debug_s.to_numpy()
    sums = debug_sums.to_numpy()
    lambdas = debug_lambda.to_numpy()
    filters = debug_filter_triggered.to_numpy()

    print(f"\nTaichi SVD singular values:")
    print(f"  s = {s}")

    print(f"\nSums (Taichi f32):")
    print(f"  s1 + s2 = {sums[0]:.10f}")
    print(f"  s0 + s2 = {sums[1]:.10f}")
    print(f"  s0 + s1 = {sums[2]:.10f}")

    print(f"\nFilter triggered:")
    print(f"  λ0 (s1+s2 < 2.0): {bool(filters[0])}")
    print(f"  λ1 (s0+s2 < 2.0): {bool(filters[1])}")
    print(f"  λ2 (s0+s1 < 2.0): {bool(filters[2])}")

    print(f"\nLambda values (after filter):")
    print(f"  λ0 = {lambdas[0]:.10f}")
    print(f"  λ1 = {lambdas[1]:.10f}")
    print(f"  λ2 = {lambdas[2]:.10f}")

    # Compare with NumPy
    print(f"\n--- NumPy comparison ---")
    U_np, s_np, Vh_np = np.linalg.svd(F)
    print(f"NumPy SVD singular values: {s_np}")

    # Check if lambdas are > 1.0 (which would cause negative eigenvalues)
    print(f"\n--- Analysis ---")
    for i, (lam, filt) in enumerate(zip(lambdas, filters)):
        if lam > 1.0 and not filt:
            print(f"⚠️ λ{i} = {lam:.10f} > 1.0 but filter NOT triggered!")
        elif lam > 1.0:
            print(f"⚠️ λ{i} = {lam:.10f} > 1.0 (filter triggered but still > 1.0?)")

    # Expected eigenvalues on twist modes: 2 - 2λ
    print(f"\nExpected twist mode eigenvalues (for μ=1):")
    for i, lam in enumerate(lambdas):
        twist_eig = 2.0 - 2.0 * lam
        sign = "≥0 ✓" if twist_eig >= 0 else "<0 ⚠️"
        print(f"  Mode {i}: 2 - 2*{lam:.8f} = {twist_eig:.6e} {sign}")


if __name__ == '__main__':
    main()
