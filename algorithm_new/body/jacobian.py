"""
ABD Jacobian transformations for 3D <-> 12D mapping.

The ABD state vector q ∈ R¹² consists of:
  q[0:3]   = p     (center of mass position)
  q[3:6]   = a1    (first row of affine matrix A)
  q[6:9]   = a2    (second row of affine matrix A)
  q[9:12]  = a3    (third row of affine matrix A)

Vertex position: x_i = p + A @ x̄_i = J_i @ q
where x̄_i is the rest position relative to center of mass.
"""

import taichi as ti


class ABDJacobian:
    """
    Static methods for ABD Jacobian transformations.

    All methods are @ti.func for use in GPU kernels.
    """

    @staticmethod
    @ti.func
    def apply_J(x_bar, q):
        """
        Forward mapping: x = J @ q = p + A @ x̄

        The Jacobian matrix J is:
        J = [I₃ | x̄₁I₃ | x̄₂I₃ | x̄₃I₃]  (3×12 matrix)

        Args:
            x_bar: Rest position relative to COM (3D vector)
            q: ABD state vector (12D)

        Returns:
            Vertex position (3D vector)
        """
        # Extract components
        p = ti.Vector([q[0], q[1], q[2]], dt=q.dtype)
        a1 = ti.Vector([q[3], q[4], q[5]], dt=q.dtype)
        a2 = ti.Vector([q[6], q[7], q[8]], dt=q.dtype)
        a3 = ti.Vector([q[9], q[10], q[11]], dt=q.dtype)

        # x = p + [a1·x̄, a2·x̄, a3·x̄]
        x = p + ti.Vector([
            a1.dot(x_bar),
            a2.dot(x_bar),
            a3.dot(x_bar)
        ], dt=q.dtype)

        return x

    @staticmethod
    @ti.func
    def apply_JT(x_bar, g):
        """
        Transpose mapping: g_q = J^T @ g

        Projects 3D gradient to 12D state space.

        J^T = [I₃    ]
              [x̄₁I₃  ]
              [x̄₂I₃  ]
              [x̄₃I₃  ]

        Result: g_q = [g; x̄₁*g[0]; x̄₂*g[0]; x̄₃*g[0];
                         x̄₁*g[1]; x̄₂*g[1]; x̄₃*g[1];
                         x̄₁*g[2]; x̄₂*g[2]; x̄₃*g[2]]

        Actually simpler: g_q = [g; x̄⊗g] where ⊗ is outer product flattened

        Args:
            x_bar: Rest position relative to COM (3D vector)
            g: 3D gradient

        Returns:
            12D gradient in state space
        """
        g12 = ti.Vector([g.dtype(0.0)] * 12, dt=g.dtype)

        # Translation part: direct copy
        g12[0] = g[0]
        g12[1] = g[1]
        g12[2] = g[2]

        # Affine part: x̄_i * g_j (outer product, row-major)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                g12[3 + i * 3 + j] = x_bar[i] * g[j]

        return g12

    @staticmethod
    @ti.func
    def JT_H_J(x_bar_i, H, x_bar_j):
        """
        Transform 3×3 Hessian to 12×12 state space.

        H_12×12 = J_i^T @ H_3×3 @ J_j

        Used for contact Hessians between vertices i and j
        belonging to the same or different ABD bodies.

        Args:
            x_bar_i: Rest position of vertex i (3D)
            H: 3×3 Hessian matrix
            x_bar_j: Rest position of vertex j (3D)

        Returns:
            12×12 Hessian in state space
        """
        result = ti.Matrix.zero(H.dtype, 12, 12)

        # Block (0,0): H [3×3 translation block]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                result[i, j] = H[i, j]

        # Block (0, 3:12): H columns * y^T [translation-affine coupling]
        for i in ti.static(range(3)):
            for k in ti.static(range(3)):  # column in H
                for j in ti.static(range(3)):  # y component
                    result[i, 3 + k * 3 + j] = H[i, k] * x_bar_j[j]

        # Block (3:12, 0): x * H rows [affine-translation coupling]
        for k in ti.static(range(3)):  # row in H
            for i in ti.static(range(3)):  # x component
                for j in ti.static(range(3)):
                    result[3 + k * 3 + i, j] = x_bar_i[i] * H[k, j]

        # Block (3:12, 3:12): H ⊗ (x @ y^T) [Kronecker structure]
        for ki in ti.static(range(3)):
            for kj in ti.static(range(3)):
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        result[3 + ki * 3 + i, 3 + kj * 3 + j] = (
                            H[ki, kj] * x_bar_i[i] * x_bar_j[j]
                        )

        return result

    @staticmethod
    @ti.func
    def apply_J_mat(x_bar, q):
        """
        Build the 3×12 Jacobian matrix explicitly.

        J = [I₃ | x̄[0]*I₃ | x̄[1]*I₃ | x̄[2]*I₃]

        Rarely needed since apply_J/apply_JT are more efficient.

        Args:
            x_bar: Rest position (3D)
            q: ABD state (12D, used only for dtype)

        Returns:
            3×12 Jacobian matrix
        """
        J = ti.Matrix.zero(q.dtype, 3, 12)

        # Identity block for translation
        for i in ti.static(range(3)):
            J[i, i] = q.dtype(1.0)

        # Scaled identity blocks for affine part
        for k in ti.static(range(3)):  # which affine row
            scale = x_bar[k]
            for i in ti.static(range(3)):
                J[i, 3 + k * 3 + i] = scale

        return J
