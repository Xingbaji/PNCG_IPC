"""
ABD Shape Energy for rigidity regularization.

Penalizes non-rigid deformations by measuring deviation from orthogonality:
  V_shape = κ * v * ||A @ A^T - I₃||²_F

For a rigid body, A @ A^T = I (rotation matrix property).
"""

import taichi as ti


class ABDShapeEnergy:
    """
    ABD Shape Energy for regularization.

    Energy formula:
      V = κ * v * Σ((aᵢ·aᵢ - 1)² + 2*Σᵢ<ⱼ(aᵢ·aⱼ)²)

    where:
      aᵢ = rows of affine matrix A (from state q)
      κ = shape stiffness
      v = body volume
    """

    @staticmethod
    @ti.func
    def energy(q):
        """
        Compute shape energy (without κ*v factor).

        Args:
            q: 12D ABD state vector

        Returns:
            Shape energy (scalar)
        """
        a1 = ti.Vector([q[3], q[4], q[5]], dt=q.dtype)
        a2 = ti.Vector([q[6], q[7], q[8]], dt=q.dtype)
        a3 = ti.Vector([q[9], q[10], q[11]], dt=q.dtype)

        # Diagonal terms: (|aᵢ|² - 1)²
        E = (a1.norm_sqr() - 1.0) ** 2
        E += (a2.norm_sqr() - 1.0) ** 2
        E += (a3.norm_sqr() - 1.0) ** 2

        # Off-diagonal terms: 2 * (aᵢ·aⱼ)²
        E += 2.0 * (a1.dot(a2)) ** 2
        E += 2.0 * (a2.dot(a3)) ** 2
        E += 2.0 * (a3.dot(a1)) ** 2

        return E

    @staticmethod
    @ti.func
    def gradient(q):
        """
        Compute shape energy gradient (9D, affine part only).

        ∂V/∂aᵢ = 4(aᵢ·aᵢ - 1)aᵢ + 4*Σⱼ≠ᵢ(aⱼ·aᵢ)aⱼ

        Args:
            q: 12D ABD state vector

        Returns:
            9D gradient [∂V/∂a1; ∂V/∂a2; ∂V/∂a3]
        """
        a1 = ti.Vector([q[3], q[4], q[5]], dt=q.dtype)
        a2 = ti.Vector([q[6], q[7], q[8]], dt=q.dtype)
        a3 = ti.Vector([q[9], q[10], q[11]], dt=q.dtype)

        grad = ti.Vector([q.dtype(0.0)] * 9, dt=q.dtype)

        # ∂V/∂a1 = 4(a1·a1 - 1)a1 + 4(a2·a1)a2 + 4(a3·a1)a3
        dEda1 = (4.0 * (a1.norm_sqr() - 1.0) * a1 +
                 4.0 * a2.dot(a1) * a2 +
                 4.0 * a3.dot(a1) * a3)

        # ∂V/∂a2 = 4(a2·a2 - 1)a2 + 4(a3·a2)a3 + 4(a1·a2)a1
        dEda2 = (4.0 * (a2.norm_sqr() - 1.0) * a2 +
                 4.0 * a3.dot(a2) * a3 +
                 4.0 * a1.dot(a2) * a1)

        # ∂V/∂a3 = 4(a3·a3 - 1)a3 + 4(a1·a3)a1 + 4(a2·a3)a2
        dEda3 = (4.0 * (a3.norm_sqr() - 1.0) * a3 +
                 4.0 * a1.dot(a3) * a1 +
                 4.0 * a2.dot(a3) * a2)

        for i in ti.static(range(3)):
            grad[i] = dEda1[i]
            grad[3 + i] = dEda2[i]
            grad[6 + i] = dEda3[i]

        return grad

    @staticmethod
    @ti.func
    def gradient_12d(q):
        """
        Compute shape energy gradient in full 12D space.

        Translation part is zero since shape energy doesn't depend on position.

        Args:
            q: 12D ABD state vector

        Returns:
            12D gradient
        """
        grad9 = ABDShapeEnergy.gradient(q)
        grad12 = ti.Vector([q.dtype(0.0)] * 12, dt=q.dtype)

        # Copy 9D gradient to affine part
        for i in ti.static(range(9)):
            grad12[3 + i] = grad9[i]

        return grad12

    @staticmethod
    @ti.func
    def hessian(q):
        """
        Compute shape energy Hessian (9×9, affine part only).

        Diagonal blocks:
          ∂²V/∂aᵢ² = 8*aᵢ@aᵢ^T + 4*(|aᵢ|² - 1)*I + 4*aⱼ@aⱼ^T + 4*aₖ@aₖ^T

        Off-diagonal blocks:
          ∂²V/∂aᵢ∂aⱼ = 4*aⱼ@aᵢ^T + 4*(aᵢ·aⱼ)*I

        Args:
            q: 12D ABD state vector

        Returns:
            9×9 Hessian matrix
        """
        a1 = ti.Vector([q[3], q[4], q[5]], dt=q.dtype)
        a2 = ti.Vector([q[6], q[7], q[8]], dt=q.dtype)
        a3 = ti.Vector([q[9], q[10], q[11]], dt=q.dtype)

        H = ti.Matrix.zero(q.dtype, 9, 9)
        I3 = ti.Matrix.identity(q.dtype, 3)

        # Diagonal blocks
        ddV_da1 = (8.0 * a1.outer_product(a1) +
                   4.0 * (a1.norm_sqr() - 1.0) * I3 +
                   4.0 * a2.outer_product(a2) +
                   4.0 * a3.outer_product(a3))

        ddV_da2 = (8.0 * a2.outer_product(a2) +
                   4.0 * (a2.norm_sqr() - 1.0) * I3 +
                   4.0 * a3.outer_product(a3) +
                   4.0 * a1.outer_product(a1))

        ddV_da3 = (8.0 * a3.outer_product(a3) +
                   4.0 * (a3.norm_sqr() - 1.0) * I3 +
                   4.0 * a1.outer_product(a1) +
                   4.0 * a2.outer_product(a2))

        # Off-diagonal blocks
        ddV_da1_da2 = 4.0 * a2.outer_product(a1) + 4.0 * a1.dot(a2) * I3
        ddV_da1_da3 = 4.0 * a3.outer_product(a1) + 4.0 * a1.dot(a3) * I3
        ddV_da2_da3 = 4.0 * a3.outer_product(a2) + 4.0 * a2.dot(a3) * I3

        # Fill 9×9 matrix
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                H[i, j] = ddV_da1[i, j]
                H[3 + i, 3 + j] = ddV_da2[i, j]
                H[6 + i, 6 + j] = ddV_da3[i, j]

                H[i, 3 + j] = ddV_da1_da2[i, j]
                H[3 + i, j] = ddV_da1_da2[j, i]  # Transpose

                H[i, 6 + j] = ddV_da1_da3[i, j]
                H[6 + i, j] = ddV_da1_da3[j, i]

                H[3 + i, 6 + j] = ddV_da2_da3[i, j]
                H[6 + i, 3 + j] = ddV_da2_da3[j, i]

        return H

    @staticmethod
    @ti.func
    def hessian_12x12(q):
        """
        Compute shape energy Hessian in full 12×12 space.

        Translation part is zero block.

        Args:
            q: 12D ABD state vector

        Returns:
            12×12 Hessian matrix
        """
        H9 = ABDShapeEnergy.hessian(q)
        H12 = ti.Matrix.zero(q.dtype, 12, 12)

        # Copy 9×9 Hessian to affine-affine block
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                H12[3 + i, 3 + j] = H9[i, j]

        return H12
