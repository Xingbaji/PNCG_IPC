"""
Simple test for H_e symmetry using numpy.
"""

import numpy as np


def compute_dFdx(B):
    """Compute derivative of deformation gradient w.r.t. vertex positions.
    B is 3x3 inverse rest deformation gradient.
    Returns 9x12 matrix.
    """
    dFdx = np.zeros((9, 12))
    for i in range(3):
        for j in range(3):
            # Vertex 0 contribution (negative sum of others)
            dFdx[i * 3 + j, i] = -(B[j, 0] + B[j, 1] + B[j, 2])
            # Vertices 1, 2, 3 contributions
            for k in range(3):
                dFdx[i * 3 + j, (k + 1) * 3 + i] = B[j, k]
    return dFdx


def main():
    print("="*70)
    print("H_e SYMMETRY TEST (NumPy)")
    print("="*70)

    # Create a random B matrix (inverse rest deformation gradient)
    np.random.seed(42)
    B = np.random.randn(3, 3)

    print(f"\nB matrix:\n{B}")

    # Compute dFdx
    dFdx = compute_dFdx(B)
    print(f"\ndFdx shape: {dFdx.shape}")

    # For ARAP: d2PsidF2 = 2 * mu * I_9x9
    mu = 1.0
    d2PsidF2 = 2.0 * mu * np.eye(9)

    # H_e = dFdx^T @ d2PsidF2 @ dFdx
    H_e = dFdx.T @ d2PsidF2 @ dFdx
    print(f"H_e shape: {H_e.shape}")

    # Check symmetry
    sym_err = np.linalg.norm(H_e - H_e.T)
    print(f"\n||H_e - H_e^T|| = {sym_err:.6e}")

    # Since d2PsidF2 is symmetric PSD, and H_e = dFdx^T @ d2PsidF2 @ dFdx
    # H_e should be symmetric by construction:
    # H_e^T = (dFdx^T @ d2PsidF2 @ dFdx)^T = dFdx^T @ d2PsidF2^T @ dFdx = dFdx^T @ d2PsidF2 @ dFdx = H_e

    print("\nH_e is symmetric by construction:")
    print("  H_e = dFdx^T @ d2PsidF2 @ dFdx")
    print("  H_e^T = dFdx^T @ d2PsidF2^T @ dFdx = dFdx^T @ d2PsidF2 @ dFdx = H_e")
    print("  (since d2PsidF2 is symmetric)")

    # Check 3x3 diagonal sub-blocks
    print("\n3x3 diagonal sub-block symmetry:")
    for i in range(4):
        sub_block = H_e[i*3:(i+1)*3, i*3:(i+1)*3]
        sub_sym_err = np.linalg.norm(sub_block - sub_block.T)
        print(f"  H_e[{i},{i}] symmetry error: {sub_sym_err:.6e}")

    # Check 3x3 off-diagonal sub-blocks
    print("\n3x3 off-diagonal sub-block symmetry (H_e[i,j] should equal H_e[j,i]^T):")
    for i in range(4):
        for j in range(i+1, 4):
            block_ij = H_e[i*3:(i+1)*3, j*3:(j+1)*3]
            block_ji = H_e[j*3:(j+1)*3, i*3:(i+1)*3]
            diff = np.linalg.norm(block_ij - block_ji.T)
            print(f"  ||H_e[{i},{j}] - H_e[{j},{i}]^T|| = {diff:.6e}")

    # THE KEY INSIGHT:
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
The 3x3 diagonal sub-block H_e[i,i] IS symmetric by construction.
The 3x3 off-diagonal sub-block H_e[i,j] = H_e[j,i]^T (transpose relation).

When assembling to block storage:
- For (i,j) with i < j: store H_e[i,j] at sym_idx(lane_i, lane_j)
- The storage at sym_idx represents block(min_lane, max_lane)

If lane_i < lane_j:
  - Store H_e[i,j] directly

If lane_i > lane_j (reordering needed):
  - We want block(lane_j, lane_i) which should equal block(lane_i, lane_j)^T
  - But we have H_e[i,j] from element
  - H_e[i,j] is the coupling between element vertex i and j
  - In the global matrix, this goes to position (vi, vj)
  - If lane_i > lane_j, we store at position (lane_j, lane_i) in sym storage
  - The value stored should be H_e[i,j]^T to maintain symmetry

The current code stores H_e[i,j].T (sub_block[dj, di]) which is CORRECT!
""")

    # But wait - let's verify this more carefully
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    # For element with 4 vertices, consider (i=0, j=1)
    # H_e[0:3, 3:6] is the coupling between v0 and v1

    block_01 = H_e[0:3, 3:6]
    block_10 = H_e[3:6, 0:3]

    print(f"\nH_e[0,1] (coupling v0-v1):\n{block_01}")
    print(f"\nH_e[1,0] (coupling v1-v0):\n{block_10}")
    print(f"\nH_e[0,1] - H_e[1,0]^T:\n{block_01 - block_10.T}")
    print(f"\nAre they transposes? {np.allclose(block_01, block_10.T)}")


if __name__ == '__main__':
    main()
