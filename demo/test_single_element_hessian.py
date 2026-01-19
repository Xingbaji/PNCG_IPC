"""
Test that a single element Hessian is PSD.
"""

import sys
import os

# Setup path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)

from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter
from math_utils.matrix_util import compute_dFdx


@ti.kernel
def test_element_hessian() -> ti.i32:
    """
    Test a single element with F = I (identity deformation).
    Returns 1 if all diagonal entries of H_e are positive, 0 otherwise.
    """
    mu = ti.f32(1e6)
    la = ti.f32(1e5)
    dt = ti.f32(0.01)
    W = ti.f32(0.001)  # Small element volume

    para = W * dt * dt

    # F = I (identity)
    F = ti.Matrix([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])

    # Compute B matrix for a regular tetrahedron
    # For simplicity, use a reference element
    B = ti.Matrix([[-1.0, -1.0, -1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    B = B.transpose()  # Make it 3x3

    # Compute dFdx
    dFdx = compute_dFdx(B)

    # Compute d2PsidF2
    d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

    # Compute H_e
    temp = d2PsidF2 @ dFdx
    H_e = dFdx.transpose() @ temp
    H_e = para * H_e

    # Check diagonals
    all_positive = 1
    for i in ti.static(range(12)):
        if H_e[i, i] < 0:
            print(f"H_e[{i},{i}] = ", H_e[i, i])
            all_positive = 0

    # Print first diagonal entry
    print("H_e[0,0] = ", H_e[0, 0])
    print("d2PsidF2[0,0] = ", d2PsidF2[0, 0])

    return all_positive


def main():
    print("=" * 60)
    print("Test Single Element Hessian")
    print("=" * 60)

    result = test_element_hessian()

    if result == 1:
        print("\nAll diagonal entries are POSITIVE - Element Hessian is likely PSD")
    else:
        print("\nSome diagonal entries are NEGATIVE - Problem detected!")


if __name__ == '__main__':
    main()
