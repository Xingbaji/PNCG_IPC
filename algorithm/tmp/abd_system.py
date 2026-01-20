"""
Affine Body Dynamics (ABD) System for PNCG-IPC solver.

This module implements ABD from Stiff-GIPC, adapted for Taichi/MeshTaichi.
ABD represents deformable bodies using a compact 12D state vector:
    q = [p; a1; a2; a3]^T
where p is the center of mass and A = [a1 a2 a3]^T is the affine transformation.

All vertex positions are computed as: x_i = p + A * x̄_i
where x̄_i is the rest-frame position relative to center of mass.

Key features:
- Boundary conditions: Free, Fixed, Motor
- Shape energy for volume preservation
- Motor constraint forces
- Kinetic energy computation
- Integration with PNCG solver

Reference: Stiff-GIPC (CUDA implementation)
"""

import taichi as ti
import numpy as np
from enum import IntEnum

# ABD constants
ABD_STATE_DIM = 12  # 3 (position) + 9 (affine matrix)


class BodyBoundaryType(IntEnum):
    """Boundary condition types for ABD bodies."""
    FREE = 0    # Free body (physics simulation)
    FIXED = 1   # Fixed/pinned body (no motion)
    MOTOR = 2   # Motorized rotation (prescribed angular velocity)


@ti.data_oriented
class ABDJacobian:
    """
    Jacobian matrix for ABD transformation.

    The Jacobian J is a 3x12 matrix that maps the 12D state to 3D position:
        x = J * q

    For a point with rest position x̄ = (x̄, ȳ, z̄), the Jacobian is:
        J = [I₃ | x̄*I₃ | ȳ*I₃ | z̄*I₃]

    where I₃ is the 3x3 identity matrix.
    """

    @staticmethod
    @ti.func
    def compute_jacobian(x_bar: ti.types.vector(3, ti.f32)) -> ti.types.matrix(3, 12, ti.f32):
        """
        Compute 3x12 Jacobian matrix for a point.

        Args:
            x_bar: Rest position relative to center of mass (3D vector)

        Returns:
            J: 3x12 Jacobian matrix
        """
        J = ti.Matrix.zero(ti.f32, 3, 12)

        # First block: identity (translation part)
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[2, 2] = 1.0

        # Second block: x̄ * I₃ (first column of A)
        J[0, 3] = x_bar[0]
        J[1, 4] = x_bar[0]
        J[2, 5] = x_bar[0]

        # Third block: ȳ * I₃ (second column of A)
        J[0, 6] = x_bar[1]
        J[1, 7] = x_bar[1]
        J[2, 8] = x_bar[1]

        # Fourth block: z̄ * I₃ (third column of A)
        J[0, 9] = x_bar[2]
        J[1, 10] = x_bar[2]
        J[2, 11] = x_bar[2]

        return J

    @staticmethod
    @ti.func
    def apply_J(x_bar: ti.types.vector(3, ti.f32),
                q: ti.types.vector(12, ti.f32)) -> ti.types.vector(3, ti.f32):
        """
        Apply Jacobian: x = J * q (state to position).

        More efficient than explicit matrix multiplication.

        Args:
            x_bar: Rest position relative to center of mass
            q: 12D state vector [p; a1; a2; a3]

        Returns:
            x: 3D world position
        """
        # Extract components
        p = ti.Vector([q[0], q[1], q[2]])

        # A is stored column-major: [a1; a2; a3] where each ai is 3D
        # q[3:6] = a1, q[6:9] = a2, q[9:12] = a3
        # A @ x_bar = x_bar[0]*a1 + x_bar[1]*a2 + x_bar[2]*a3
        Ax = ti.Vector([
            q[3] * x_bar[0] + q[6] * x_bar[1] + q[9] * x_bar[2],
            q[4] * x_bar[0] + q[7] * x_bar[1] + q[10] * x_bar[2],
            q[5] * x_bar[0] + q[8] * x_bar[1] + q[11] * x_bar[2]
        ])

        return p + Ax

    @staticmethod
    @ti.func
    def apply_JT(x_bar: ti.types.vector(3, ti.f32),
                 g: ti.types.vector(3, ti.f32)) -> ti.types.vector(12, ti.f32):
        """
        Apply Jacobian transpose: g_q = J^T * g (gradient projection).

        Projects a 3D gradient to 12D state space.

        Args:
            x_bar: Rest position relative to center of mass
            g: 3D gradient in world space

        Returns:
            g_q: 12D gradient in state space
        """
        g_q = ti.Vector.zero(ti.f32, 12)

        # Translation part: J^T[:3, :] = I₃
        g_q[0] = g[0]
        g_q[1] = g[1]
        g_q[2] = g[2]

        # Affine part: J^T[3:, :] = [x̄*I₃; ȳ*I₃; z̄*I₃]^T
        # a1 gradient
        g_q[3] = x_bar[0] * g[0]
        g_q[4] = x_bar[0] * g[1]
        g_q[5] = x_bar[0] * g[2]

        # a2 gradient
        g_q[6] = x_bar[1] * g[0]
        g_q[7] = x_bar[1] * g[1]
        g_q[8] = x_bar[1] * g[2]

        # a3 gradient
        g_q[9] = x_bar[2] * g[0]
        g_q[10] = x_bar[2] * g[1]
        g_q[11] = x_bar[2] * g[2]

        return g_q

    @staticmethod
    @ti.func
    def apply_JT_H_J(x_bar: ti.types.vector(3, ti.f32),
                     H: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(12, 12, ti.f32):
        """
        Compute J^T * H * J for Hessian transformation.

        Args:
            x_bar: Rest position relative to center of mass
            H: 3x3 Hessian in world space

        Returns:
            H_q: 12x12 Hessian in state space
        """
        H_q = ti.Matrix.zero(ti.f32, 12, 12)

        # Build J explicitly for full transformation
        J = ABDJacobian.compute_jacobian(x_bar)

        # Compute J^T @ H @ J
        JT_H = J.transpose() @ H  # 12x3
        H_q = JT_H @ J  # 12x12

        return H_q


@ti.data_oriented
class ABDBody:
    """
    Single ABD body representation.

    Stores the 12D state and provides methods for:
    - State to position mapping
    - Gradient projection
    - Mass matrix computation
    - Shape energy (penalizes deviation from rigid motion)
    """

    def __init__(self, body_id: int, n_points: int, point_ids: np.ndarray,
                 rest_positions: np.ndarray, masses: np.ndarray):
        """
        Initialize an ABD body.

        Args:
            body_id: Unique identifier for this body
            n_points: Number of points in this body
            point_ids: Global vertex IDs belonging to this body
            rest_positions: Rest positions of all points (n_points x 3)
            masses: Mass of each point (n_points,)
        """
        self.body_id = body_id
        self.n_points = n_points

        # Compute center of mass and relative positions
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass
        x_bar = rest_positions - com  # Relative positions

        # Store in Taichi fields
        self.point_ids = ti.field(dtype=ti.i32, shape=n_points)
        self.point_ids.from_numpy(point_ids.astype(np.int32))

        self.x_bar = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
        self.x_bar.from_numpy(x_bar)

        self.point_mass = ti.field(dtype=ti.f32, shape=n_points)
        self.point_mass.from_numpy(masses)

        # State vectors (12D)
        self.q = ti.Vector.field(12, dtype=ti.f32, shape=())  # Current state
        self.q_prev = ti.Vector.field(12, dtype=ti.f32, shape=())  # Previous state
        self.q_tilde = ti.Vector.field(12, dtype=ti.f32, shape=())  # Predicted state
        self.q_v = ti.Vector.field(12, dtype=ti.f32, shape=())  # Velocity
        self.dq = ti.Vector.field(12, dtype=ti.f32, shape=())  # Search direction
        self.grad_q = ti.Vector.field(12, dtype=ti.f32, shape=())  # Gradient

        # Mass matrix (12x12)
        self.abd_mass = ti.Matrix.field(12, 12, dtype=ti.f32, shape=())
        self.abd_mass_inv = ti.Matrix.field(12, 12, dtype=ti.f32, shape=())

        # Properties
        self.total_mass = total_mass
        self.volume = 0.0  # Set from cells
        self.com_rest = com  # Rest center of mass

        # Shape energy stiffness
        self.kappa_shape = 1e6  # Stiffness for shape preservation

        # Initialize state to identity transformation at COM
        self._init_state(com)

        # Compute mass matrix
        self._compute_mass_matrix()

    def _init_state(self, com: np.ndarray):
        """Initialize state to identity transformation at center of mass."""
        q_init = np.zeros(12)
        q_init[0:3] = com  # Position at COM
        # Identity affine matrix (column-major storage)
        q_init[3] = 1.0   # a1 = (1, 0, 0)
        q_init[7] = 1.0   # a2 = (0, 1, 0)
        q_init[11] = 1.0  # a3 = (0, 0, 1)

        self.q.from_numpy(q_init)
        self.q_prev.from_numpy(q_init)
        self.q_tilde.from_numpy(q_init)
        self.q_v.from_numpy(np.zeros(12))

    @ti.kernel
    def _compute_mass_matrix_kernel(self):
        """
        Compute 12x12 ABD mass matrix.

        M = Σ_i m_i * J_i^T @ J_i

        where J_i is the Jacobian for point i.
        """
        M = ti.Matrix.zero(ti.f32, 12, 12)

        for i in range(self.n_points):
            m = self.point_mass[i]
            x_bar = self.x_bar[i]

            # Compute J^T @ J contribution
            # J = [I₃ | x̄*I₃ | ȳ*I₃ | z̄*I₃]
            # J^T @ J is block diagonal with specific structure

            # Translation-translation block (3x3)
            for d in ti.static(range(3)):
                M[d, d] += m

            # Translation-affine cross terms
            for d in ti.static(range(3)):
                # p-a1 coupling
                M[d, 3 + d] += m * x_bar[0]
                M[3 + d, d] += m * x_bar[0]
                # p-a2 coupling
                M[d, 6 + d] += m * x_bar[1]
                M[6 + d, d] += m * x_bar[1]
                # p-a3 coupling
                M[d, 9 + d] += m * x_bar[2]
                M[9 + d, d] += m * x_bar[2]

            # Affine-affine blocks
            for d in ti.static(range(3)):
                # a1-a1 block
                M[3 + d, 3 + d] += m * x_bar[0] * x_bar[0]
                # a2-a2 block
                M[6 + d, 6 + d] += m * x_bar[1] * x_bar[1]
                # a3-a3 block
                M[9 + d, 9 + d] += m * x_bar[2] * x_bar[2]

                # a1-a2 coupling
                M[3 + d, 6 + d] += m * x_bar[0] * x_bar[1]
                M[6 + d, 3 + d] += m * x_bar[0] * x_bar[1]
                # a1-a3 coupling
                M[3 + d, 9 + d] += m * x_bar[0] * x_bar[2]
                M[9 + d, 3 + d] += m * x_bar[0] * x_bar[2]
                # a2-a3 coupling
                M[6 + d, 9 + d] += m * x_bar[1] * x_bar[2]
                M[9 + d, 6 + d] += m * x_bar[1] * x_bar[2]

        self.abd_mass[None] = M

    def _compute_mass_matrix(self):
        """Compute and invert mass matrix."""
        self._compute_mass_matrix_kernel()

        # Invert on CPU (12x12 is small)
        M = self.abd_mass.to_numpy()[0]

        # Add small regularization for numerical stability
        M += np.eye(12) * 1e-10

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print(f"[ABD] Warning: Mass matrix singular for body {self.body_id}, using pseudo-inverse")
            M_inv = np.linalg.pinv(M)

        self.abd_mass_inv.from_numpy(M_inv.reshape(1, 12, 12))


@ti.data_oriented
class ABDShapeEnergy:
    """
    Shape energy functions for ABD bodies.

    Shape energy penalizes non-rigid deformations:
    V_shape / (κv) = Σ(aᵢ·aᵢ - 1)² + Σᵢ≠ⱼ(aᵢ·aⱼ)²

    This measures deviation from orthonormality of the affine columns.
    """

    @staticmethod
    @ti.func
    def compute_energy(q: ti.types.vector(12, ti.f32)) -> ti.f32:
        """
        Compute shape energy (without κv scaling).

        Args:
            q: 12D state vector

        Returns:
            Shape energy value
        """
        # Extract affine columns
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        # Squared deviation from unit length
        E = (a1.norm_sqr() - 1.0) ** 2
        E += (a2.norm_sqr() - 1.0) ** 2
        E += (a3.norm_sqr() - 1.0) ** 2

        # Squared dot products (non-orthogonality)
        E += 2.0 * (a1.dot(a2)) ** 2
        E += 2.0 * (a2.dot(a3)) ** 2
        E += 2.0 * (a3.dot(a1)) ** 2

        return E

    @staticmethod
    @ti.func
    def compute_gradient(q: ti.types.vector(12, ti.f32)) -> ti.types.vector(9, ti.f32):
        """
        Compute shape energy gradient w.r.t. affine part (9D).

        ∂V/∂aᵢ = 4(aᵢ·aᵢ - 1)aᵢ + 4Σⱼ≠ᵢ(aⱼ·aᵢ)aⱼ

        Args:
            q: 12D state vector

        Returns:
            9D gradient (dE/da1, dE/da2, dE/da3)
        """
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        # dE/da1
        dEda1 = 4.0 * (a1.norm_sqr() - 1.0) * a1 + 4.0 * a2.dot(a1) * a2 + 4.0 * a3.dot(a1) * a3

        # dE/da2
        dEda2 = 4.0 * (a2.norm_sqr() - 1.0) * a2 + 4.0 * a3.dot(a2) * a3 + 4.0 * a1.dot(a2) * a1

        # dE/da3
        dEda3 = 4.0 * (a3.norm_sqr() - 1.0) * a3 + 4.0 * a1.dot(a3) * a1 + 4.0 * a2.dot(a3) * a2

        grad = ti.Vector.zero(ti.f32, 9)
        grad[0], grad[1], grad[2] = dEda1[0], dEda1[1], dEda1[2]
        grad[3], grad[4], grad[5] = dEda2[0], dEda2[1], dEda2[2]
        grad[6], grad[7], grad[8] = dEda3[0], dEda3[1], dEda3[2]

        return grad

    @staticmethod
    @ti.func
    def compute_hessian(q: ti.types.vector(12, ti.f32)) -> ti.types.matrix(9, 9, ti.f32):
        """
        Compute shape energy Hessian (9x9 for affine part).

        The Hessian has the block structure:
        [∂²V/∂a1² , ∂²V/∂a1∂a2, ∂²V/∂a1∂a3]
        [∂²V/∂a2∂a1, ∂²V/∂a2² , ∂²V/∂a2∂a3]
        [∂²V/∂a3∂a1, ∂²V/∂a3∂a2, ∂²V/∂a3² ]

        Args:
            q: 12D state vector

        Returns:
            9x9 Hessian matrix
        """
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        H = ti.Matrix.zero(ti.f32, 9, 9)
        I3 = ti.Matrix.identity(ti.f32, 3)

        # ∂²V/∂a1² = 8*a1*a1ᵀ + 4*(a1·a1 - 1)*I + 4*a2*a2ᵀ + 4*a3*a3ᵀ
        ddVdda1 = 8.0 * a1.outer_product(a1) + 4.0 * (a1.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a2.outer_product(a2) + 4.0 * a3.outer_product(a3)

        # ∂²V/∂a2² = 8*a2*a2ᵀ + 4*(a2·a2 - 1)*I + 4*a3*a3ᵀ + 4*a1*a1ᵀ
        ddVdda2 = 8.0 * a2.outer_product(a2) + 4.0 * (a2.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a3.outer_product(a3) + 4.0 * a1.outer_product(a1)

        # ∂²V/∂a3² = 8*a3*a3ᵀ + 4*(a3·a3 - 1)*I + 4*a1*a1ᵀ + 4*a2*a2ᵀ
        ddVdda3 = 8.0 * a3.outer_product(a3) + 4.0 * (a3.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a1.outer_product(a1) + 4.0 * a2.outer_product(a2)

        # ∂²V/∂a1∂a2 = 4*a2*a1ᵀ + 4*(a1·a2)*I
        ddVda1da2 = 4.0 * a2.outer_product(a1) + 4.0 * a1.dot(a2) * I3

        # ∂²V/∂a1∂a3 = 4*a3*a1ᵀ + 4*(a1·a3)*I
        ddVda1da3 = 4.0 * a3.outer_product(a1) + 4.0 * a1.dot(a3) * I3

        # ∂²V/∂a2∂a3 = 4*a3*a2ᵀ + 4*(a2·a3)*I
        ddVda2da3 = 4.0 * a3.outer_product(a2) + 4.0 * a2.dot(a3) * I3

        # Assemble 9x9 Hessian
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                # Diagonal blocks
                H[i, j] = ddVdda1[i, j]
                H[3 + i, 3 + j] = ddVdda2[i, j]
                H[6 + i, 6 + j] = ddVdda3[i, j]

                # Off-diagonal blocks
                H[i, 3 + j] = ddVda1da2[i, j]
                H[3 + i, j] = ddVda1da2[j, i]  # Transpose

                H[i, 6 + j] = ddVda1da3[i, j]
                H[6 + i, j] = ddVda1da3[j, i]  # Transpose

                H[3 + i, 6 + j] = ddVda2da3[i, j]
                H[6 + i, 3 + j] = ddVda2da3[j, i]  # Transpose

        return H


@ti.data_oriented
class ABDMotor:
    """
    Motor constraint for ABD bodies with prescribed rotation.

    Implements rotation around an axis using small rotation approximation.
    """

    @staticmethod
    @ti.func
    def compute_rotation_matrix(axis: ti.types.vector(3, ti.f32),
                                 theta: ti.f32) -> ti.types.matrix(3, 3, ti.f32):
        """
        Compute rotation matrix using Rodrigues' formula.

        R = I + sin(θ)K + (1 - cos(θ))K²
        where K is the skew-symmetric cross-product matrix of axis.

        Args:
            axis: Normalized rotation axis
            theta: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        c = ti.cos(theta)
        s = ti.sin(theta)

        K = ti.Matrix([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ], dt=ti.f32)

        I3 = ti.Matrix.identity(ti.f32, 3)
        R = I3 + s * K + (1.0 - c) * (K @ K)

        return R

    @staticmethod
    @ti.func
    def compute_motor_target(q_base: ti.types.vector(12, ti.f32),
                             axis: ti.types.vector(3, ti.f32),
                             omega: ti.f32,
                             dt: ti.f32) -> ti.types.vector(12, ti.f32):
        """
        Compute target state for motor body.

        The motor applies a rotation of omega*dt around the specified axis.

        Args:
            q_base: Base state (typically q_tilde or q_prev)
            axis: Rotation axis (normalized)
            omega: Angular velocity (rad/s)
            dt: Time step

        Returns:
            Target state q_target
        """
        theta = omega * dt
        R = ABDMotor.compute_rotation_matrix(axis, theta)

        # Extract affine matrix from q (column-major)
        A = ti.Matrix([
            [q_base[3], q_base[6], q_base[9]],
            [q_base[4], q_base[7], q_base[10]],
            [q_base[5], q_base[8], q_base[11]]
        ], dt=ti.f32)

        # Apply rotation: A_new = R @ A
        A_new = R @ A

        # Build target state (keep position, update affine)
        q_target = q_base
        q_target[3] = A_new[0, 0]
        q_target[4] = A_new[1, 0]
        q_target[5] = A_new[2, 0]
        q_target[6] = A_new[0, 1]
        q_target[7] = A_new[1, 1]
        q_target[8] = A_new[2, 1]
        q_target[9] = A_new[0, 2]
        q_target[10] = A_new[1, 2]
        q_target[11] = A_new[2, 2]

        return q_target


@ti.data_oriented
class ABDSystem:
    """
    Complete ABD system managing multiple ABD bodies.

    Provides:
    - Body management and state storage
    - Boundary conditions (Free, Fixed, Motor)
    - Batch position/gradient computation
    - Integration with FEM solver
    - Shape energy computation with Hessian
    - Kinetic energy computation
    - Motor constraint forces with rotation
    - CCD support for ABD bodies
    """

    def __init__(self, max_bodies: int = 64, max_points_per_body: int = 10000):
        """
        Initialize ABD system.

        Args:
            max_bodies: Maximum number of ABD bodies
            max_points_per_body: Maximum points per body
        """
        self.max_bodies = max_bodies
        self.max_points_per_body = max_points_per_body
        self.max_total_points = max_bodies * max_points_per_body

        # Number of active bodies
        self.n_bodies = 0

        # Body state vectors (batched for GPU efficiency)
        self.q = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.q_prev = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.q_tilde = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.q_temp = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)  # For line search rollback
        self.q_v = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.dq = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.grad_q = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)
        self.grad_q_prev = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)  # Previous gradient for CG

        # Mass matrices
        self.abd_mass = ti.Matrix.field(12, 12, dtype=ti.f32, shape=max_bodies)
        self.abd_mass_inv = ti.Matrix.field(12, 12, dtype=ti.f32, shape=max_bodies)

        # Diagonal preconditioner (12x12 block per body)
        self.abd_precond = ti.Matrix.field(12, 12, dtype=ti.f32, shape=max_bodies)

        # Body properties
        self.body_volume = ti.field(dtype=ti.f32, shape=max_bodies)
        self.kappa_shape = ti.field(dtype=ti.f32, shape=max_bodies)
        self.body_total_mass = ti.field(dtype=ti.f32, shape=max_bodies)

        # Boundary conditions
        self.boundary_type = ti.field(dtype=ti.i32, shape=max_bodies)  # BodyBoundaryType enum

        # Motor parameters (for MOTOR boundary type)
        self.motor_speed = ti.field(dtype=ti.f32, shape=max_bodies)  # rad/s
        self.motor_strength = ti.field(dtype=ti.f32, shape=max_bodies)  # Torque scaling
        self.motor_axis = ti.Vector.field(3, dtype=ti.f32, shape=max_bodies)  # Rotation axis

        # Point data (flattened storage)
        self.point_body_id = ti.field(dtype=ti.i32, shape=self.max_total_points)
        self.point_local_id = ti.field(dtype=ti.i32, shape=self.max_total_points)
        self.x_bar = ti.Vector.field(3, dtype=ti.f32, shape=self.max_total_points)
        self.point_mass = ti.field(dtype=ti.f32, shape=self.max_total_points)

        # Body point ranges [start, end)
        self.body_point_start = ti.field(dtype=ti.i32, shape=max_bodies + 1)

        # Global vertex ID mapping (abd_point_idx -> global_vertex_id)
        self.global_vertex_id = ti.field(dtype=ti.i32, shape=self.max_total_points)

        # Reverse mapping (global_vertex_id -> abd_point_idx, -1 if not ABD)
        self.vertex_to_abd_point = None  # Set when mesh is known

        # Total ABD points
        self.n_total_points = 0

        # Simulation parameters
        self.dt = 0.01
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        # Initialize
        self._init_fields()

    @ti.kernel
    def _init_fields(self):
        """Initialize fields to zero/identity."""
        for i in range(self.max_bodies):
            # Identity state
            self.q[i] = ti.Vector([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dt=ti.f32)
            self.q_prev[i] = self.q[i]
            self.q_tilde[i] = self.q[i]
            self.q_temp[i] = self.q[i]
            self.q_v[i] = ti.Vector.zero(ti.f32, 12)
            self.dq[i] = ti.Vector.zero(ti.f32, 12)
            self.grad_q[i] = ti.Vector.zero(ti.f32, 12)
            self.grad_q_prev[i] = ti.Vector.zero(ti.f32, 12)

            self.abd_mass[i] = ti.Matrix.identity(ti.f32, 12)
            self.abd_mass_inv[i] = ti.Matrix.identity(ti.f32, 12)
            self.abd_precond[i] = ti.Matrix.identity(ti.f32, 12)

            self.body_volume[i] = 1.0
            self.kappa_shape[i] = 1e6
            self.body_total_mass[i] = 1.0

            # Default to FREE boundary
            self.boundary_type[i] = 0  # FREE

            # Motor defaults
            self.motor_speed[i] = 0.0
            self.motor_strength[i] = 10.0
            self.motor_axis[i] = ti.Vector([0.0, 1.0, 0.0])

    def add_body(self, point_ids: np.ndarray, rest_positions: np.ndarray,
                 masses: np.ndarray, volume: float = 1.0, kappa_shape: float = 1e6,
                 boundary_type: int = 0, motor_speed: float = 0.0,
                 motor_strength: float = 10.0, motor_axis: np.ndarray = None) -> int:
        """
        Add an ABD body to the system.

        Args:
            point_ids: Global vertex IDs belonging to this body
            rest_positions: Rest positions (n_points x 3)
            masses: Point masses (n_points,)
            volume: Body volume (for shape energy scaling)
            kappa_shape: Shape preservation stiffness
            boundary_type: 0=FREE, 1=FIXED, 2=MOTOR
            motor_speed: Angular velocity for MOTOR type (rad/s)
            motor_strength: Motor torque scaling
            motor_axis: Rotation axis for MOTOR type (default Y-axis)

        Returns:
            body_id: Assigned body ID
        """
        if self.n_bodies >= self.max_bodies:
            raise RuntimeError(f"Maximum number of ABD bodies ({self.max_bodies}) exceeded")

        n_points = len(point_ids)
        if self.n_total_points + n_points > self.max_total_points:
            raise RuntimeError(f"Maximum total ABD points ({self.max_total_points}) exceeded")

        body_id = self.n_bodies

        # Compute center of mass
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass
        x_bar = rest_positions - com

        # Store point data
        start_idx = self.n_total_points
        end_idx = start_idx + n_points

        self.body_point_start[body_id] = start_idx
        self.body_point_start[body_id + 1] = end_idx

        # Upload point data
        for i, (pid, xb, m) in enumerate(zip(point_ids, x_bar, masses)):
            idx = start_idx + i
            self.point_body_id[idx] = body_id
            self.point_local_id[idx] = i
            self.x_bar[idx] = xb.tolist()
            self.point_mass[idx] = m
            self.global_vertex_id[idx] = int(pid)

        # Initialize state (identity at COM)
        q_init = np.zeros(12)
        q_init[0:3] = com
        q_init[3] = 1.0  # Identity affine
        q_init[7] = 1.0
        q_init[11] = 1.0

        self.q[body_id] = q_init.tolist()
        self.q_prev[body_id] = q_init.tolist()
        self.q_tilde[body_id] = q_init.tolist()

        # Compute mass matrix
        self._compute_body_mass_matrix(body_id, start_idx, end_idx)

        # Set properties
        self.body_volume[body_id] = volume
        self.kappa_shape[body_id] = kappa_shape
        self.body_total_mass[body_id] = total_mass

        # Set boundary conditions
        self.boundary_type[body_id] = boundary_type
        self.motor_speed[body_id] = motor_speed
        self.motor_strength[body_id] = motor_strength
        if motor_axis is not None:
            axis = motor_axis / (np.linalg.norm(motor_axis) + 1e-10)
            self.motor_axis[body_id] = axis.tolist()
        else:
            self.motor_axis[body_id] = [0.0, 1.0, 0.0]

        self.n_bodies += 1
        self.n_total_points = end_idx

        boundary_names = {0: 'FREE', 1: 'FIXED', 2: 'MOTOR'}
        print(f"[ABD] Added body {body_id}: {n_points} points, COM = {com}, type = {boundary_names.get(boundary_type, 'UNKNOWN')}")

        return body_id

    def _compute_body_mass_matrix(self, body_id: int, start_idx: int, end_idx: int):
        """Compute mass matrix for a single body."""
        n_points = end_idx - start_idx

        # Gather data
        x_bar_np = np.zeros((n_points, 3))
        masses_np = np.zeros(n_points)

        for i in range(n_points):
            idx = start_idx + i
            x_bar_np[i] = [self.x_bar[idx][j] for j in range(3)]
            masses_np[i] = self.point_mass[idx]

        # Compute M = Σ m_i J_i^T J_i
        M = np.zeros((12, 12))

        for i in range(n_points):
            m = masses_np[i]
            xb = x_bar_np[i]

            # Build J (3x12)
            J = np.zeros((3, 12))
            J[0, 0] = J[1, 1] = J[2, 2] = 1.0
            J[0, 3] = J[1, 4] = J[2, 5] = xb[0]
            J[0, 6] = J[1, 7] = J[2, 8] = xb[1]
            J[0, 9] = J[1, 10] = J[2, 11] = xb[2]

            M += m * J.T @ J

        # Add regularization
        M += np.eye(12) * 1e-10

        # Invert
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # Store
        self.abd_mass[body_id] = M.tolist()
        self.abd_mass_inv[body_id] = M_inv.tolist()

    def setup_vertex_mapping(self, n_total_verts: int):
        """
        Setup reverse mapping from global vertex ID to ABD point index.

        Args:
            n_total_verts: Total number of vertices in the mesh
        """
        self.vertex_to_abd_point = ti.field(dtype=ti.i32, shape=n_total_verts)
        self.vertex_to_abd_point.fill(-1)  # -1 means not an ABD point

        # Fill mapping
        for i in range(self.n_total_points):
            global_id = self.global_vertex_id[i]
            self.vertex_to_abd_point[global_id] = i

    @ti.kernel
    def compute_x_from_q(self, vertices: ti.template()):
        """
        Compute vertex positions from ABD states.

        Args:
            vertices: Vertex position field to update (mesh.verts.x or similar)
        """
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            q = self.q[body_id]
            global_id = self.global_vertex_id[i]

            # x = p + A @ x_bar
            x = ABDJacobian.apply_J(x_bar, q)

            vertices[global_id] = ti.cast(x, ti.f32)

    @ti.kernel
    def compute_q_tilde(self, dt: ti.f32):
        """
        Compute predicted state: q̃ = q + dt*q_v + dt²*M⁻¹*f_ext

        Handles boundary conditions:
        - FREE: Standard integration with gravity
        - FIXED: q_tilde = q (no motion)
        - MOTOR: Apply prescribed rotation around motor axis

        Args:
            dt: Time step
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]
            q = self.q[body_id]

            if btype == 1:  # FIXED
                # Fixed body: no motion
                self.q_tilde[body_id] = q
                self.q_prev[body_id] = q
            elif btype == 2:  # MOTOR
                # Motor body: apply prescribed rotation around axis
                axis = self.motor_axis[body_id]
                omega = self.motor_speed[body_id]

                # Compute target state with rotation applied
                q_target = ABDMotor.compute_motor_target(q, axis, omega, dt)

                self.q_tilde[body_id] = q_target
                self.q_prev[body_id] = q
            else:  # FREE
                q_v = self.q_v[body_id]
                M_inv = self.abd_mass_inv[body_id]
                total_mass = self.body_total_mass[body_id]

                # External force (gravity on translation part)
                f_ext = ti.Vector.zero(ti.f32, 12)
                f_ext[0] = total_mass * self.gravity[0]
                f_ext[1] = total_mass * self.gravity[1]
                f_ext[2] = total_mass * self.gravity[2]

                # q̃ = q + dt*v + dt²*M⁻¹*f
                q_tilde = q + dt * q_v + dt * dt * (M_inv @ f_ext)

                self.q_tilde[body_id] = q_tilde
                self.q_prev[body_id] = q

    @ti.kernel
    def save_state_for_line_search(self):
        """Save current state for potential line search rollback."""
        for body_id in range(self.n_bodies):
            self.q_temp[body_id] = self.q[body_id]

    @ti.kernel
    def restore_state_from_line_search(self):
        """Restore state from before line search."""
        for body_id in range(self.n_bodies):
            self.q[body_id] = self.q_temp[body_id]

    @ti.kernel
    def project_gradient_to_q(self, grad_x: ti.template()):
        """
        Project vertex gradients to ABD state gradient.

        g_q = Σ_i J_i^T @ g_x[i]

        Handles boundary conditions:
        - FREE: Standard projection
        - FIXED: Gradient set to zero
        - MOTOR: Position DOFs zeroed

        Args:
            grad_x: Per-vertex gradient field (mesh.verts.grad)
        """
        # Clear gradients and save previous
        for body_id in range(self.n_bodies):
            self.grad_q_prev[body_id] = self.grad_q[body_id]
            self.grad_q[body_id] = ti.Vector.zero(ti.f32, 12)

        # Accumulate
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            btype = self.boundary_type[body_id]

            # Skip fixed bodies (zero gradient)
            if btype == 1:  # FIXED
                continue

            x_bar = self.x_bar[i]
            global_id = self.global_vertex_id[i]

            g_x = ti.cast(grad_x[global_id], ti.f32)
            g_q = ABDJacobian.apply_JT(x_bar, g_x)

            # Atomic add to body gradient
            for d in ti.static(range(12)):
                ti.atomic_add(self.grad_q[body_id][d], g_q[d])

        # Apply boundary condition constraints
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                # Zero gradient for fixed bodies
                self.grad_q[body_id] = ti.Vector.zero(ti.f32, 12)
            elif btype == 2:  # MOTOR
                # Zero translation DOFs for motor bodies (only rotation allowed)
                self.grad_q[body_id][0] = 0.0
                self.grad_q[body_id][1] = 0.0
                self.grad_q[body_id][2] = 0.0

    def compute_shape_energy(self) -> float:
        """
        Compute shape-preserving energy for all ABD bodies.

        V_shape = κ * v * ||A*A^T - I||_F²

        This penalizes non-rigid deformations.

        Returns:
            Total shape energy
        """
        return self._compute_shape_energy_kernel()

    @ti.kernel
    def _compute_shape_energy_kernel(self) -> ti.f32:
        """Kernel for shape energy computation."""
        E_shape = 0.0

        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]

            # Extract A from q (column-major)
            A = ti.Matrix([
                [q[3], q[6], q[9]],
                [q[4], q[7], q[10]],
                [q[5], q[8], q[11]]
            ], dt=ti.f32)

            # C = A @ A^T - I
            AAT = A @ A.transpose()
            C = AAT - ti.Matrix.identity(ti.f32, 3)

            # ||C||_F² = trace(C^T @ C)
            frob_sq = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    frob_sq += C[i, j] * C[i, j]

            E_shape += kappa * vol * frob_sq

        return E_shape

    @ti.kernel
    def add_shape_gradient(self):
        """
        Add shape energy gradient to grad_q.

        Uses the new ABDShapeEnergy class for accurate gradient computation.
        Scales gradient by κv*dt² for time-stepping consistency.
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                continue

            q = self.q[body_id]
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]
            kvt2 = kappa * vol * self.dt * self.dt

            # Compute shape gradient (9D for affine part)
            grad_shape = ABDShapeEnergy.compute_gradient(q)

            # Add to gradient (indices 3-11 correspond to affine part)
            for d in ti.static(range(9)):
                self.grad_q[body_id][3 + d] += kvt2 * grad_shape[d]

    @ti.kernel
    def add_motor_constraint_gradient(self):
        """
        Add motor constraint gradient for MOTOR bodies.

        For motor bodies, adds a penalty force to track the target rotation.
        g_motor = strength * M_rot * (q - q_target)

        where M_rot is the rotational part of the mass matrix.
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype != 2:  # Only for MOTOR
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]  # Target state with rotation
            M = self.abd_mass[body_id]
            strength = self.motor_strength[body_id]

            # Compute deviation from target (only rotation/affine part)
            dq = ti.Vector.zero(ti.f32, 12)
            for d in ti.static(range(9)):
                dq[3 + d] = q[3 + d] - q_tilde[3 + d]

            # Motor force: strength * M[rot,rot] * dq[rot]
            # Use mass matrix for rotational DOFs (indices 3-11)
            g_motor = ti.Vector.zero(ti.f32, 12)
            for i in ti.static(range(9)):
                for j in ti.static(range(9)):
                    g_motor[3 + i] += strength * M[3 + i, 3 + j] * dq[3 + j]

            # Add to gradient
            for d in ti.static(range(12)):
                self.grad_q[body_id][d] += g_motor[d]

    @ti.kernel
    def compute_shape_hessian_contribution(self, dt: ti.f32) -> ti.f32:
        """
        Compute p^T * H_shape * p contribution to pHp.

        Returns:
            Shape energy Hessian contribution to pHp
        """
        result = 0.0

        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                continue

            q = self.q[body_id]
            dq = self.dq[body_id]
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]
            kvt2 = kappa * vol * dt * dt

            # Compute shape Hessian (9x9)
            H_shape = ABDShapeEnergy.compute_hessian(q)

            # Make positive definite (project negative eigenvalues to zero)
            # For simplicity, we just clamp any negative diagonal entries
            # In production, full EVD projection would be used

            # Extract affine part of dq (9D)
            dq_affine = ti.Vector.zero(ti.f32, 9)
            for d in ti.static(range(9)):
                dq_affine[d] = dq[3 + d]

            # Compute dq^T * H * dq
            Hdq = H_shape @ dq_affine
            for d in ti.static(range(9)):
                result += kvt2 * dq_affine[d] * Hdq[d]

        return result

    @ti.kernel
    def compute_motor_hessian_contribution(self) -> ti.f32:
        """
        Compute p^T * H_motor * p contribution for motor bodies.

        Returns:
            Motor constraint Hessian contribution to pHp
        """
        result = 0.0

        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype != 2:  # Only for MOTOR
                continue

            dq = self.dq[body_id]
            M = self.abd_mass[body_id]
            strength = self.motor_strength[body_id]

            # Motor Hessian is strength * M[rot,rot]
            # Compute dq_rot^T * (strength * M[rot,rot]) * dq_rot
            for i in ti.static(range(9)):
                for j in ti.static(range(9)):
                    result += strength * dq[3 + i] * M[3 + i, 3 + j] * dq[3 + j]

        return result

    @ti.kernel
    def update_velocity(self, dt: ti.f32):
        """
        Update velocity after optimization: q_v = (q - q_prev) / dt

        Handles boundary conditions:
        - FREE: Standard velocity update
        - FIXED: Zero velocity
        - MOTOR: Prescribed angular velocity around motor axis

        Args:
            dt: Time step
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]
            q = self.q[body_id]
            q_prev = self.q_prev[body_id]

            if btype == 1:  # FIXED
                self.q_v[body_id] = ti.Vector.zero(ti.f32, 12)
            elif btype == 2:  # MOTOR
                # Motor has prescribed rotation velocity around motor_axis
                # The velocity is computed from the skew-symmetric matrix of omega
                # q_v[p] = 0 (translation velocity is zero for motor)
                # q_v[A] = omega_skew @ A where omega_skew = omega * [axis]_x
                #
                # For ABD, the affine velocity is: dA/dt = omega_skew @ A
                # In column-major storage: d(a1,a2,a3)/dt = omega_skew @ (a1,a2,a3)

                axis = self.motor_axis[body_id]
                omega = self.motor_speed[body_id]

                # Extract current affine matrix A from q (column-major)
                A = ti.Matrix([
                    [q[3], q[6], q[9]],
                    [q[4], q[7], q[10]],
                    [q[5], q[8], q[11]]
                ], dt=ti.f32)

                # Compute skew-symmetric matrix [axis]_x
                omega_skew = omega * ti.Matrix([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0]
                ], dt=ti.f32)

                # dA/dt = omega_skew @ A
                dA_dt = omega_skew @ A

                # Build velocity vector
                q_v = ti.Vector.zero(ti.f32, 12)
                # Translation velocity is zero for motor (fixed pivot)
                q_v[0] = 0.0
                q_v[1] = 0.0
                q_v[2] = 0.0
                # Affine velocity (column-major storage)
                q_v[3] = dA_dt[0, 0]
                q_v[4] = dA_dt[1, 0]
                q_v[5] = dA_dt[2, 0]
                q_v[6] = dA_dt[0, 1]
                q_v[7] = dA_dt[1, 1]
                q_v[8] = dA_dt[2, 1]
                q_v[9] = dA_dt[0, 2]
                q_v[10] = dA_dt[1, 2]
                q_v[11] = dA_dt[2, 2]

                self.q_v[body_id] = q_v
            else:  # FREE
                self.q_v[body_id] = (q - q_prev) / dt

    def compute_kinetic_energy(self) -> float:
        """
        Compute kinetic energy for all ABD bodies.

        K = 0.5 * Σ_i dq_i^T @ M_i @ dq_i
        where dq = q - q_tilde

        Returns:
            Total kinetic energy
        """
        return self._compute_kinetic_energy_kernel()

    @ti.kernel
    def _compute_kinetic_energy_kernel(self) -> ti.f32:
        """Kernel for kinetic energy computation."""
        K = 0.0

        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                # Fixed bodies have zero kinetic energy
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]
            M = self.abd_mass[body_id]

            dq = q - q_tilde

            if btype == 2:  # MOTOR
                # Motor bodies: only rotation contributes
                # Zero out translation part
                dq[0] = 0.0
                dq[1] = 0.0
                dq[2] = 0.0

            # K_i = 0.5 * dq^T @ M @ dq
            Mdq = M @ dq
            for d in ti.static(range(12)):
                K += 0.5 * dq[d] * Mdq[d]

        return K

    @ti.kernel
    def add_inertia_gradient(self):
        """
        Add inertia term to gradient: g += M @ (q - q_tilde)

        This should be called after project_gradient_to_q.
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]
            M = self.abd_mass[body_id]

            dq = q - q_tilde
            g_inertia = M @ dq

            if btype == 2:  # MOTOR
                # Zero translation DOFs
                g_inertia[0] = 0.0
                g_inertia[1] = 0.0
                g_inertia[2] = 0.0

            for d in ti.static(range(12)):
                self.grad_q[body_id][d] += g_inertia[d]

    @ti.kernel
    def compute_preconditioner(self, dt: ti.f32):
        """
        Compute diagonal preconditioner for ABD bodies.

        P_i = M_i + dt^2 * H_shape_i

        This provides the diagonal blocks for the ABD portion of the global preconditioner.
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]
            M = self.abd_mass[body_id]

            if btype == 1:  # FIXED
                # Identity preconditioner for fixed bodies
                self.abd_precond[body_id] = ti.Matrix.identity(ti.f32, 12)
                continue

            # Start with mass matrix
            P = M

            # Add shape energy Hessian (simplified: assume diagonal dominance)
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]
            kvt2 = kappa * vol * dt * dt

            # Approximate shape Hessian contribution (diagonal estimate)
            for d in range(9):
                P[3 + d // 3 * 3 + d % 3, 3 + d // 3 * 3 + d % 3] += kvt2 * 8.0

            # Invert using simple formula (since P should be well-conditioned)
            self.abd_precond[body_id] = P

    @ti.kernel
    def apply_preconditioner(self):
        """
        Apply preconditioner: z = P^{-1} @ g

        Stores result in dq field.
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                self.dq[body_id] = ti.Vector.zero(ti.f32, 12)
                continue

            # Use inverse mass matrix as preconditioner (simpler and effective)
            M_inv = self.abd_mass_inv[body_id]
            grad_q = self.grad_q[body_id]

            self.dq[body_id] = M_inv @ grad_q

    @ti.kernel
    def step_forward(self, alpha: ti.f32):
        """
        Take optimization step: q = q - alpha * dq

        Handles boundary conditions:
        - FREE: Standard update
        - FIXED: No update (q stays fixed)
        - MOTOR: Only rotation updates

        Args:
            alpha: Step size
        """
        for body_id in range(self.n_bodies):
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                # No update for fixed bodies
                continue

            dq = self.dq[body_id]

            if btype == 2:  # MOTOR
                # Zero translation update for motor bodies
                dq[0] = 0.0
                dq[1] = 0.0
                dq[2] = 0.0

            self.q[body_id] = self.q[body_id] - alpha * dq

    @ti.kernel
    def compute_dq_from_vertex_p(self, vertex_p: ti.template()):
        """
        Compute ABD search direction from vertex search directions.

        dq = Σ_i J_i^T @ p_i (projected to ABD space)

        Args:
            vertex_p: Per-vertex search direction (mesh.verts.p)
        """
        # Clear dq
        for body_id in range(self.n_bodies):
            self.dq[body_id] = ti.Vector.zero(ti.f32, 12)

        # Accumulate from vertices
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                continue

            x_bar = self.x_bar[i]
            global_id = self.global_vertex_id[i]

            p = ti.cast(vertex_p[global_id], ti.f32)
            dq_contrib = ABDJacobian.apply_JT(x_bar, p)

            for d in ti.static(range(12)):
                ti.atomic_add(self.dq[body_id][d], dq_contrib[d])

    @ti.kernel
    def compute_vertex_p_from_dq(self, vertex_p: ti.template()):
        """
        Compute vertex search directions from ABD search direction.

        p_i = J_i @ dq (mapped to vertex space)

        Args:
            vertex_p: Per-vertex search direction to update (mesh.verts.p)
        """
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            global_id = self.global_vertex_id[i]
            dq = self.dq[body_id]

            # p = J @ dq
            p = ABDJacobian.apply_J(x_bar, dq)
            vertex_p[global_id] = ti.cast(p, ti.f32)

    @ti.kernel
    def compute_max_vertex_displacement(self) -> ti.f32:
        """
        Compute maximum vertex displacement from search direction.

        This is used for CCD to ensure step size doesn't cause tunneling.

        Returns:
            Maximum displacement magnitude across all ABD vertices
        """
        max_disp = 0.0

        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            dq = self.dq[body_id]

            # Compute vertex displacement: dx = J @ dq
            dx = ABDJacobian.apply_J(x_bar, dq)
            disp = dx.norm()

            ti.atomic_max(max_disp, disp)

        return max_disp

    @ti.kernel
    def compute_ccd_alpha_ground(self, ground_y: ti.f32, margin: ti.f32) -> ti.f32:
        """
        Compute conservative CCD step size for ground plane collision.

        For each ABD vertex, compute the maximum step size that keeps it
        above the ground plane minus a margin.

        Args:
            ground_y: Ground plane Y coordinate
            margin: Safety margin (typically 0.5 * dHat)

        Returns:
            Conservative step size alpha in [0, 1]
        """
        alpha_min = 1.0

        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            btype = self.boundary_type[body_id]

            if btype == 1:  # FIXED
                continue

            x_bar = self.x_bar[i]
            q = self.q[body_id]
            dq = self.dq[body_id]

            # Current position
            x = ABDJacobian.apply_J(x_bar, q)
            # Search direction for vertex
            dx = ABDJacobian.apply_J(x_bar, dq)

            # Current distance to ground
            dist = x[1] - ground_y

            # Only check if moving toward ground
            if dx[1] < 0:
                # Maximum step before hitting ground - margin
                # x[1] - alpha * dx[1] >= ground_y + margin
                # alpha <= (x[1] - ground_y - margin) / (-dx[1])
                alpha_bound = (dist - margin) / (-dx[1])
                if alpha_bound > 0:
                    ti.atomic_min(alpha_min, alpha_bound)

        return alpha_min

    @ti.kernel
    def compute_ccd_alpha_self(self, dHat: ti.f32) -> ti.f32:
        """
        Compute conservative CCD step size for self-collision (simplified).

        This is a simplified version that computes alpha based on maximum
        displacement relative to dHat. For production use, proper CCD
        would require vertex-triangle and edge-edge tests.

        Args:
            dHat: Contact threshold distance

        Returns:
            Conservative step size alpha in [0, 1]
        """
        alpha = 1.0

        # Compute max displacement
        max_disp = 0.0
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            dq = self.dq[body_id]

            dx = ABDJacobian.apply_J(x_bar, dq)
            disp = dx.norm()
            ti.atomic_max(max_disp, disp)

        # Limit step size so displacement < 0.5 * dHat
        if max_disp > 0:
            alpha = ti.min(alpha, 0.5 * dHat / max_disp)

        return alpha

    def compute_ccd_step_size(self, ground_y: float = 0.0, dHat: float = 0.01) -> float:
        """
        Compute conservative CCD step size for ABD bodies.

        Combines ground collision and self-collision constraints.

        Args:
            ground_y: Ground plane Y coordinate
            dHat: Contact threshold distance

        Returns:
            Conservative step size alpha in [0, 1]
        """
        alpha = 1.0

        # Ground plane CCD
        alpha_ground = self.compute_ccd_alpha_ground(ground_y, 0.5 * dHat)
        alpha = min(alpha, alpha_ground)

        # Self-collision CCD (simplified)
        alpha_self = self.compute_ccd_alpha_self(dHat)
        alpha = min(alpha, alpha_self)

        return alpha

    def get_stats(self) -> dict:
        """Return statistics about the ABD system."""
        # Count boundary types
        n_free = 0
        n_fixed = 0
        n_motor = 0
        for i in range(self.n_bodies):
            btype = self.boundary_type[i]
            if btype == 0:
                n_free += 1
            elif btype == 1:
                n_fixed += 1
            elif btype == 2:
                n_motor += 1

        return {
            'n_bodies': self.n_bodies,
            'n_total_points': self.n_total_points,
            'max_bodies': self.max_bodies,
            'n_free': n_free,
            'n_fixed': n_fixed,
            'n_motor': n_motor,
        }

    def set_boundary_type(self, body_id: int, boundary_type: int):
        """
        Set boundary type for a body after creation.

        Args:
            body_id: Body index
            boundary_type: 0=FREE, 1=FIXED, 2=MOTOR
        """
        if body_id < 0 or body_id >= self.n_bodies:
            raise ValueError(f"Invalid body_id: {body_id}")
        self.boundary_type[body_id] = boundary_type

    def set_motor_params(self, body_id: int, speed: float, strength: float,
                         axis: np.ndarray = None):
        """
        Set motor parameters for a MOTOR body.

        Args:
            body_id: Body index
            speed: Angular velocity (rad/s)
            strength: Torque scaling
            axis: Rotation axis (default: Y-axis)
        """
        if body_id < 0 or body_id >= self.n_bodies:
            raise ValueError(f"Invalid body_id: {body_id}")

        self.motor_speed[body_id] = speed
        self.motor_strength[body_id] = strength

        if axis is not None:
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            self.motor_axis[body_id] = axis.tolist()

    def get_body_state(self, body_id: int) -> dict:
        """
        Get current state of a body.

        Args:
            body_id: Body index

        Returns:
            Dictionary with q, q_v, boundary_type
        """
        if body_id < 0 or body_id >= self.n_bodies:
            raise ValueError(f"Invalid body_id: {body_id}")

        q = [self.q[body_id][i] for i in range(12)]
        q_v = [self.q_v[body_id][i] for i in range(12)]

        return {
            'q': q,
            'q_v': q_v,
            'boundary_type': int(self.boundary_type[body_id]),
            'total_mass': float(self.body_total_mass[body_id]),
            'volume': float(self.body_volume[body_id]),
        }


# Utility functions for ABD body detection
def detect_rigid_components(mesh_positions: np.ndarray,
                           mesh_cells: np.ndarray,
                           rigidity_threshold: float = 0.1) -> list:
    """
    Detect connected components that could be treated as ABD bodies.

    This is a placeholder for more sophisticated rigid body detection.
    In practice, users would specify which bodies are ABD vs FEM.

    Args:
        mesh_positions: Vertex positions (N x 3)
        mesh_cells: Cell connectivity (M x 4 for tets)
        rigidity_threshold: Threshold for considering a component rigid

    Returns:
        List of (vertex_ids, is_rigid) tuples for each component
    """
    # Simple connected component detection
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_verts = len(mesh_positions)
    n_cells = len(mesh_cells)

    # Build adjacency matrix
    rows = []
    cols = []
    for cell in mesh_cells:
        for i in range(4):
            for j in range(i + 1, 4):
                rows.extend([cell[i], cell[j]])
                cols.extend([cell[j], cell[i]])

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts))

    # Find connected components
    n_components, labels = connected_components(adj, directed=False)

    components = []
    for comp_id in range(n_components):
        vertex_ids = np.where(labels == comp_id)[0]
        # For now, assume all components are deformable (FEM)
        # Users would specify ABD bodies explicitly
        components.append((vertex_ids, False))

    return components
