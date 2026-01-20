"""
ABD Dyadic Mass matrix computation and application.

The 12×12 mass matrix M for an ABD body is built from point masses:
  M = Σ_i m_i * J_i^T @ J_i

Instead of storing the full matrix, we store three compact components:
  - m: total mass (scalar)
  - m_x_bar: Σ m_i * x̄_i (3D vector)
  - m_dyadic: Σ m_i * x̄_i @ x̄_i^T (3×3 matrix)
"""

import taichi as ti
import numpy as np


class ABDDyadicMass:
    """
    ABD Dyadic Mass utilities.

    The 12×12 mass matrix has structure:
    M = [m*I₃        m_x̄^T      m_x̄^T      m_x̄^T    ]
        [m_x̄        m_dyadic   0          0         ]
        [m_x̄        0          m_dyadic   0         ]
        [m_x̄        0          0          m_dyadic  ]

    (with appropriate transposes and placements)
    """

    @staticmethod
    @ti.func
    def compute_contribution(mass, x_bar):
        """
        Compute mass matrix contribution from a single point.

        Args:
            mass: Point mass (scalar)
            x_bar: Rest position relative to COM (3D)

        Returns:
            m, m_x_bar, m_dyadic (components to accumulate)
        """
        m = mass
        m_x_bar = mass * x_bar
        m_dyadic = mass * x_bar.outer_product(x_bar)
        return m, m_x_bar, m_dyadic

    @staticmethod
    @ti.func
    def apply(m, m_x_bar, m_dyadic, p):
        """
        Apply mass matrix: result = M @ p

        Efficient implementation without forming full 12×12 matrix.

        Args:
            m: Total mass (scalar)
            m_x_bar: Mass-weighted position (3D)
            m_dyadic: Mass-weighted dyadic (3×3)
            p: 12D input vector

        Returns:
            12D output vector M @ p
        """
        result = ti.Vector([p.dtype(0.0)] * 12, dt=p.dtype)

        # Extract components of p
        p_p = ti.Vector([p[0], p[1], p[2]], dt=p.dtype)
        p_a1 = ti.Vector([p[3], p[4], p[5]], dt=p.dtype)
        p_a2 = ti.Vector([p[6], p[7], p[8]], dt=p.dtype)
        p_a3 = ti.Vector([p[9], p[10], p[11]], dt=p.dtype)

        # Translation rows: M[0:3, :] @ p
        # = m * p_p + m_x_bar * (sum of affine components dotted appropriately)
        D_p_a1 = m_dyadic @ p_a1
        D_p_a2 = m_dyadic @ p_a2
        D_p_a3 = m_dyadic @ p_a3

        # Row 0: m * p[0] + m_x̄ · p_a1
        # Row 1: m * p[1] + m_x̄ · p_a2
        # Row 2: m * p[2] + m_x̄ · p_a3
        result[0] = m * p_p[0] + m_x_bar.dot(p_a1)
        result[1] = m * p_p[1] + m_x_bar.dot(p_a2)
        result[2] = m * p_p[2] + m_x_bar.dot(p_a3)

        # Affine rows
        # a1 block: m_dyadic @ p_a1 + m_x̄ * p[0]
        for i in ti.static(range(3)):
            result[3 + i] = D_p_a1[i] + m_x_bar[i] * p_p[0]

        # a2 block: m_dyadic @ p_a2 + m_x̄ * p[1]
        for i in ti.static(range(3)):
            result[6 + i] = D_p_a2[i] + m_x_bar[i] * p_p[1]

        # a3 block: m_dyadic @ p_a3 + m_x̄ * p[2]
        for i in ti.static(range(3)):
            result[9 + i] = D_p_a3[i] + m_x_bar[i] * p_p[2]

        return result

    @staticmethod
    def to_matrix_numpy(m, m_x_bar_np, m_dyadic_np):
        """
        Build full 12×12 mass matrix (numpy, CPU).

        Useful for matrix inversion and debugging.

        Args:
            m: Total mass (scalar)
            m_x_bar_np: Mass-weighted position (3D numpy array)
            m_dyadic_np: Mass-weighted dyadic (3×3 numpy array)

        Returns:
            12×12 numpy mass matrix
        """
        M = np.zeros((12, 12))

        # Translation block (3×3)
        M[0:3, 0:3] = m * np.eye(3)

        # Translation-affine coupling (3×9, top-right)
        # M[0, 3:6] = m_x̄^T (affects a1)
        # etc.
        for k in range(3):  # which affine block
            M[k, 3 + k * 3: 3 + (k + 1) * 3] = m_x_bar_np

        # Affine-translation coupling (9×3, bottom-left)
        for k in range(3):
            M[3 + k * 3: 3 + (k + 1) * 3, k] = m_x_bar_np

        # Affine diagonal blocks (9×9 with 3 diagonal 3×3 blocks)
        M[3:6, 3:6] = m_dyadic_np
        M[6:9, 6:9] = m_dyadic_np
        M[9:12, 9:12] = m_dyadic_np

        return M

    @staticmethod
    @ti.func
    def to_matrix(m, m_x_bar, m_dyadic):
        """
        Build full 12×12 mass matrix (Taichi).

        Args:
            m: Total mass (scalar)
            m_x_bar: Mass-weighted position (3D vector)
            m_dyadic: Mass-weighted dyadic (3×3 matrix)

        Returns:
            12×12 mass matrix
        """
        M = ti.Matrix.zero(m_dyadic.dtype, 12, 12)

        # Translation block
        for i in ti.static(range(3)):
            M[i, i] = m

        # Translation-affine coupling
        for k in ti.static(range(3)):
            for i in ti.static(range(3)):
                M[k, 3 + k * 3 + i] = m_x_bar[i]
                M[3 + k * 3 + i, k] = m_x_bar[i]

        # Affine diagonal blocks
        for k in ti.static(range(3)):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    M[3 + k * 3 + i, 3 + k * 3 + j] = m_dyadic[i, j]

        return M
