"""
Debug full ARAP_filter Hessian computation step by step.
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

from math_utils.elastic_util import ssvd, flatten_matrix


# Store intermediate results
H_result = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
q0_result = ti.Vector.field(9, dtype=ti.f32, shape=())
q1_result = ti.Vector.field(9, dtype=ti.f32, shape=())
q2_result = ti.Vector.field(9, dtype=ti.f32, shape=())
lambda_result = ti.field(dtype=ti.f32, shape=3)


@ti.kernel
def compute_arap_filter_debug(F_in: ti.types.matrix(3, 3, ti.f32), mu: ti.f32, la: ti.f32):
    """Compute ARAP_filter Hessian with debug output."""
    U, sig, V = ssvd(F_in)

    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]

    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)

    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda2 = 1.0

    lambda_result[0] = lambda0
    lambda_result[1] = lambda1
    lambda_result[2] = lambda2

    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]

    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)

    q0 = flatten_matrix(Q0)
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    q0_result[None] = q0
    q1_result[None] = q1
    q2_result[None] = q2

    # Build Hessian
    d2PsidF2 = -mu * (lambda0 * q0.outer_product(q0) +
                      lambda1 * q1.outer_product(q1) +
                      lambda2 * q2.outer_product(q2))

    for i in ti.static(range(9)):
        d2PsidF2[i, i] += 2.0 * mu

    H_result[None] = d2PsidF2


def main():
    print("=" * 60)
    print("Full ARAP_filter Hessian Debug")
    print("=" * 60)

    # Test with perturbed F
    np.random.seed(42)
    F = np.eye(3, dtype=np.float32) + 0.0001 * np.random.randn(3, 3).astype(np.float32)

    mu = 3571428.571428572

    print(f"\nInput F:")
    print(F)
    print(f"\nμ = {mu:.4e}")

    # Run Taichi computation
    F_ti = ti.Matrix(F.tolist())
    compute_arap_filter_debug(F_ti, mu, 1.0)

    # Get results
    H_ti = H_result[None].to_numpy()
    q0 = q0_result[None].to_numpy()
    q1 = q1_result[None].to_numpy()
    q2 = q2_result[None].to_numpy()
    lambdas = lambda_result.to_numpy()

    print(f"\nLambda values:")
    print(f"  λ0 = {lambdas[0]:.10f}")
    print(f"  λ1 = {lambdas[1]:.10f}")
    print(f"  λ2 = {lambdas[2]:.10f}")

    print(f"\nTwist mode norms:")
    print(f"  ||q0||² = {np.dot(q0, q0):.6f}")
    print(f"  ||q1||² = {np.dot(q1, q1):.6f}")
    print(f"  ||q2||² = {np.dot(q2, q2):.6f}")

    print(f"\nOrthogonality:")
    print(f"  q0·q1 = {np.dot(q0, q1):.6e}")
    print(f"  q0·q2 = {np.dot(q0, q2):.6e}")
    print(f"  q1·q2 = {np.dot(q1, q2):.6e}")

    # Check Hessian symmetry
    H_sym = (H_ti + H_ti.T) / 2
    asym_err = np.linalg.norm(H_ti - H_ti.T)
    print(f"\nHessian asymmetry error: {asym_err:.4e}")

    # Eigenvalue analysis
    eigvals_ti = np.linalg.eigvalsh(H_sym)
    print(f"\nTaichi Hessian eigenvalues:")
    print(f"  Sorted: {np.sort(eigvals_ti)}")
    print(f"  Min: {np.min(eigvals_ti):.4e}")
    print(f"  Negative count: {np.sum(eigvals_ti < 0)}")

    # Compare with NumPy implementation
    print(f"\n--- NumPy comparison ---")
    U_np, s_np, Vh_np = np.linalg.svd(F)
    V_np = Vh_np.T

    s0, s1, s2 = s_np

    # Filter
    lambda0_np = 1.0 if s1 + s2 < 2.0 else 2.0 / (s1 + s2)
    lambda1_np = 1.0 if s0 + s2 < 2.0 else 2.0 / (s0 + s2)
    lambda2_np = 1.0 if s0 + s1 < 2.0 else 2.0 / (s0 + s1)

    U0, U1, U2 = U_np[:, 0], U_np[:, 1], U_np[:, 2]
    V0, V1, V2 = V_np[:, 0], V_np[:, 1], V_np[:, 2]

    Q0_np = np.outer(V1, U2) - np.outer(V2, U1)
    Q1_np = np.outer(V2, U0) - np.outer(V0, U2)
    Q2_np = np.outer(V1, U0) - np.outer(V0, U1)

    q0_np = Q0_np.flatten()
    q1_np = Q1_np.flatten()
    q2_np = Q2_np.flatten()

    H_np = 2.0 * mu * np.eye(9)
    H_np -= mu * lambda0_np * np.outer(q0_np, q0_np)
    H_np -= mu * lambda1_np * np.outer(q1_np, q1_np)
    H_np -= mu * lambda2_np * np.outer(q2_np, q2_np)

    H_np = (H_np + H_np.T) / 2
    eigvals_np = np.linalg.eigvalsh(H_np)

    print(f"\nNumPy Hessian eigenvalues:")
    print(f"  Sorted: {np.sort(eigvals_np)}")
    print(f"  Min: {np.min(eigvals_np):.4e}")
    print(f"  Negative count: {np.sum(eigvals_np < 0)}")

    # Key comparison
    print(f"\n--- Key differences ---")
    print(f"Eigenvalue diff: {np.linalg.norm(np.sort(eigvals_ti) - np.sort(eigvals_np)):.4e}")

    # Check twist modes
    print(f"\nTwist mode vector comparison:")
    print(f"  ||q0_ti - q0_np|| = {np.linalg.norm(q0 - q0_np):.4e}")
    print(f"  ||q1_ti - q1_np|| = {np.linalg.norm(q1 - q1_np):.4e}")
    print(f"  ||q2_ti - q2_np|| = {np.linalg.norm(q2 - q2_np):.4e}")

    # Check if sign flip
    print(f"\nSign check (dot products with NumPy version):")
    print(f"  q0_ti · q0_np = {np.dot(q0, q0_np):.4f}")
    print(f"  q1_ti · q1_np = {np.dot(q1, q1_np):.4f}")
    print(f"  q2_ti · q2_np = {np.dot(q2, q2_np):.4f}")


if __name__ == '__main__':
    main()
