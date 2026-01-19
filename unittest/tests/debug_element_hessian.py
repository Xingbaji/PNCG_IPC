"""
Debug element Hessian H_e = dFdx^T @ d2PsidF2 @ dFdx transformation.
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter, ssvd
from math_utils.matrix_util import compute_dFdx


# Storage for debug
H_e_result = ti.Matrix.field(12, 12, dtype=ti.f32, shape=())
d2PsidF2_result = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
dFdx_result = ti.Matrix.field(9, 12, dtype=ti.f32, shape=())


@ti.kernel
def compute_element_hessian(B_in: ti.types.matrix(3, 3, ti.f32),
                            F_in: ti.types.matrix(3, 3, ti.f32),
                            mu: ti.f32, la: ti.f32, para: ti.f32):
    """Compute element Hessian."""
    # Compute d2PsidF2
    d2PsidF2 = compute_d2PsidF2_ARAP_filter(F_in, mu, la)
    d2PsidF2_result[None] = d2PsidF2

    # Compute dFdx
    dFdx = compute_dFdx(B_in)
    dFdx_result[None] = dFdx

    # Compute H_e = dFdx^T @ d2PsidF2 @ dFdx
    temp = d2PsidF2 @ dFdx  # 9x12
    H_e = dFdx.transpose() @ temp  # 12x12
    H_e = para * H_e

    H_e_result[None] = H_e


def main():
    print("=" * 60)
    print("Element Hessian Debug")
    print("=" * 60)

    # Create a simple B matrix (reference configuration inverse)
    # For unit tetrahedron
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    # F at identity
    F = np.eye(3, dtype=np.float32)

    mu = 3571428.571428572
    para = 1.0  # W * dt^2, simplified for testing

    print(f"\nB (reference config inverse):")
    print(B)
    print(f"\nF (deformation gradient):")
    print(F)
    print(f"\nμ = {mu:.4e}")
    print(f"para (W * dt²) = {para}")

    # Run Taichi computation
    B_ti = ti.Matrix(B.tolist())
    F_ti = ti.Matrix(F.tolist())
    compute_element_hessian(B_ti, F_ti, mu, 1.0, para)

    # Get results
    H_e = H_e_result[None].to_numpy()
    d2PsidF2 = d2PsidF2_result[None].to_numpy()
    dFdx = dFdx_result[None].to_numpy()

    # Check d2PsidF2
    d2PsidF2_sym = (d2PsidF2 + d2PsidF2.T) / 2
    eigvals_F = np.linalg.eigvalsh(d2PsidF2_sym)
    print(f"\nd2PsidF2 (F-space Hessian):")
    print(f"  Min eigenvalue: {np.min(eigvals_F):.4e}")
    print(f"  Negative count: {np.sum(eigvals_F < 0)}")

    # Check H_e
    H_e_sym = (H_e + H_e.T) / 2
    eigvals_x = np.linalg.eigvalsh(H_e_sym)
    print(f"\nH_e (x-space element Hessian):")
    print(f"  Eigenvalues: {np.sort(eigvals_x)}")
    print(f"  Min eigenvalue: {np.min(eigvals_x):.4e}")
    print(f"  Negative count: {np.sum(eigvals_x < 0)}")

    # Key insight: If d2PsidF2 is PSD, is H_e also PSD?
    # H_e = dFdx^T @ d2PsidF2 @ dFdx
    # This is a congruence transformation, which preserves positive semi-definiteness!
    # If d2PsidF2 >= 0, then for any x: x^T H_e x = x^T dFdx^T @ d2PsidF2 @ dFdx x
    #                                           = (dFdx x)^T @ d2PsidF2 @ (dFdx x) >= 0

    print(f"\n--- Analysis ---")
    if np.min(eigvals_F) >= -1e-6 and np.min(eigvals_x) >= -1e-6:
        print("✓ Both d2PsidF2 and H_e are PSD (as expected)")
    elif np.min(eigvals_F) >= -1e-6 and np.min(eigvals_x) < -1e-6:
        print("⚠️ d2PsidF2 is PSD but H_e is NOT!")
        print("   This should not happen (congruence preserves PSD)")
    elif np.min(eigvals_F) < -1e-6:
        print("⚠️ d2PsidF2 is NOT PSD!")

    # Check asymmetry
    asym_F = np.linalg.norm(d2PsidF2 - d2PsidF2.T)
    asym_x = np.linalg.norm(H_e - H_e.T)
    print(f"\nAsymmetry errors:")
    print(f"  d2PsidF2: {asym_F:.4e}")
    print(f"  H_e: {asym_x:.4e}")

    # Check sub-blocks of H_e (4x4 vertex blocks, each 3x3)
    print(f"\nH_e vertex sub-blocks (diagonal):")
    for i in range(4):
        sub_block = H_e[i*3:(i+1)*3, i*3:(i+1)*3]
        sub_sym = (sub_block + sub_block.T) / 2
        eigvals_sub = np.linalg.eigvalsh(sub_sym)
        print(f"  Vertex {i}: min_eig={np.min(eigvals_sub):.4e}, trace={np.trace(sub_block):.4e}")


if __name__ == '__main__':
    main()
