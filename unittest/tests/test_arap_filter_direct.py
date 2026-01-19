"""
Direct test of ARAP_filter Hessian computation.
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter


# Store Hessian result
H_result = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())


@ti.kernel
def compute_hessian_at_identity(mu: ti.f32, la: ti.f32):
    """Compute ARAP_filter Hessian at F = I."""
    F = ti.Matrix.identity(ti.f32, 3)
    H = compute_d2PsidF2_ARAP_filter(F, mu, la)
    H_result[None] = H


@ti.kernel
def compute_hessian_at_F(F_in: ti.types.matrix(3, 3, ti.f32), mu: ti.f32, la: ti.f32):
    """Compute ARAP_filter Hessian at given F."""
    H = compute_d2PsidF2_ARAP_filter(F_in, mu, la)
    H_result[None] = H


def main():
    mu = 3571428.571428572
    la = 14285714.28571429

    print("=" * 60)
    print("Direct ARAP_filter Hessian Test")
    print("=" * 60)

    # Test 1: Identity F
    print("\n1. Hessian at F = I:")
    compute_hessian_at_identity(mu, la)
    H = H_result[None].to_numpy()
    H_sym = (H + H.T) / 2
    eigvals = np.linalg.eigvalsh(H_sym)
    print(f"   Eigenvalues: min={np.min(eigvals):.4e}, max={np.max(eigvals):.4e}")
    print(f"   Negative count: {np.sum(eigvals < 0)}")

    # Test 2: Near-identity F (slightly perturbed)
    print("\n2. Hessian at F ≈ I (perturbed):")
    F_perturbed = np.eye(3, dtype=np.float32) + 0.0001 * np.random.randn(3, 3).astype(np.float32)
    F_ti = ti.Matrix(F_perturbed.tolist())
    compute_hessian_at_F(F_ti, mu, la)
    H = H_result[None].to_numpy()
    H_sym = (H + H.T) / 2
    eigvals = np.linalg.eigvalsh(H_sym)
    print(f"   Eigenvalues: min={np.min(eigvals):.4e}, max={np.max(eigvals):.4e}")
    print(f"   Negative count: {np.sum(eigvals < 0)}")

    # Test 3: mu = 1.0 to isolate scaling
    print("\n3. Hessian at F = I with μ = 1.0:")
    compute_hessian_at_identity(1.0, 1.0)
    H = H_result[None].to_numpy()
    H_sym = (H + H.T) / 2
    eigvals = np.linalg.eigvalsh(H_sym)
    print(f"   Eigenvalues: {np.sort(eigvals)}")
    print(f"   Negative count: {np.sum(eigvals < 0)}")

    # Expected: [0, 0, 0, 2, 2, 2, 2, 2, 2] for mu = 1.0

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    if np.sum(eigvals < -1e-6) > 0:
        print("⚠️ ARAP_filter produces negative eigenvalues!")
        print("   This suggests a bug in the Taichi implementation.")
    else:
        print("✓ ARAP_filter produces correct PSD Hessian at F = I")


if __name__ == '__main__':
    main()
