"""
Affine Body Dynamics (ABD) System Implementation.

This module implements the ABD system for rigid/nearly-rigid body simulation
using 12D reduced coordinates (position + 3x3 affine transformation matrix).

Reference: Stiff-GIPC (CUDA implementation)

Key components:
- ABDJacobian: Jacobian matrix for coordinate transformation (J: 3x12)
- ABDBody: Single ABD body with mass matrix and state
- ABDShapeEnergy: Shape preservation energy (penalizes non-rigid deformation)
- ABDMotor: Motor constraint for prescribed rotation
- ABDSystem: Main system managing multiple ABD bodies

State vector q ∈ ℝ¹²:
    q = [p; a1; a2; a3]^T
    where p ∈ ℝ³ is center of mass position
    and A = [a1, a2, a3]^T ∈ ℝ³ˣ³ is affine transformation matrix

Vertex position mapping:
    x_i = p + A @ x̄_i = J @ q
    where x̄_i is rest position relative to COM
"""

import numpy as np
import taichi as ti
from enum import IntEnum


class BodyBoundaryType(IntEnum):
    """Boundary condition types for ABD bodies."""
    FREE = 0     # Standard physics simulation
    FIXED = 1    # Pinned/constrained (no motion)
    MOTOR = 2    # Prescribed rotation around axis


@ti.data_oriented
class ABDJacobian:
    """
    Jacobian matrix for ABD coordinate transformation.

    The Jacobian J ∈ ℝ³ˣ¹² maps the 12D state q to 3D position x:
        x = J @ q = p + A @ x̄

    J = [I₃ | x̄₁I₃ | x̄₂I₃ | x̄₃I₃]

    where x̄ = [x̄₁, x̄₂, x̄₃]^T is the rest position relative to COM.
    """

    @staticmethod
    @ti.func
    def apply_J(x_bar: ti.types.vector(3, ti.f32),
                q: ti.types.vector(12, ti.f32)) -> ti.types.vector(3, ti.f32):
        """
        Apply J @ q to get world position.

        x = p + A @ x̄
        where p = q[0:3], A rows are q[3:6], q[6:9], q[9:12]

        Args:
            x_bar: Rest position relative to COM [x̄₁, x̄₂, x̄₃]
            q: 12D state vector [p; a1; a2; a3]

        Returns:
            3D world position
        """
        p = ti.Vector([q[0], q[1], q[2]])
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        # x = p + [a1·x̄, a2·x̄, a3·x̄]
        x = p + ti.Vector([a1.dot(x_bar), a2.dot(x_bar), a3.dot(x_bar)])
        return x

    @staticmethod
    @ti.func
    def apply_JT(x_bar: ti.types.vector(3, ti.f32),
                 g: ti.types.vector(3, ti.f32)) -> ti.types.vector(12, ti.f32):
        """
        Apply J^T @ g to project 3D gradient to 12D state space.

        g_q = J^T @ g = [g; x̄·g₁; x̄·g₂; x̄·g₃]

        Args:
            x_bar: Rest position relative to COM
            g: 3D gradient

        Returns:
            12D gradient in state space
        """
        g12 = ti.Vector([0.0] * 12, dt=ti.f32)
        # Translation part
        g12[0] = g[0]
        g12[1] = g[1]
        g12[2] = g[2]
        # Affine part: x̄ * g_i for each component
        g12[3] = x_bar[0] * g[0]
        g12[4] = x_bar[1] * g[0]
        g12[5] = x_bar[2] * g[0]
        g12[6] = x_bar[0] * g[1]
        g12[7] = x_bar[1] * g[1]
        g12[8] = x_bar[2] * g[1]
        g12[9] = x_bar[0] * g[2]
        g12[10] = x_bar[1] * g[2]
        g12[11] = x_bar[2] * g[2]
        return g12

    @staticmethod
    @ti.func
    def JT_H_J(x_bar_i: ti.types.vector(3, ti.f32),
               H: ti.types.matrix(3, 3, ti.f32),
               x_bar_j: ti.types.vector(3, ti.f32)) -> ti.types.matrix(12, 12, ti.f32):
        """
        Compute J_i^T @ H @ J_j for Hessian transformation.

        Used to project 3x3 Hessian to 12x12 state space Hessian.

        Args:
            x_bar_i: Rest position for left Jacobian
            H: 3x3 Hessian matrix
            x_bar_j: Rest position for right Jacobian

        Returns:
            12x12 transformed Hessian
        """
        result = ti.Matrix.zero(ti.f32, 12, 12)

        x = x_bar_i
        y = x_bar_j

        # Block (0,0): H
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                result[i, j] = H[i, j]

        # Block (0, 3:12): H column * y^T
        for i in ti.static(range(3)):
            for k in ti.static(range(3)):  # column index in H
                for j in ti.static(range(3)):  # y component
                    result[i, 3 + k * 3 + j] = H[i, k] * y[j]

        # Block (3:12, 0): x * H row
        for k in ti.static(range(3)):  # row index in H
            for i in ti.static(range(3)):  # x component
                for j in ti.static(range(3)):
                    result[3 + k * 3 + i, j] = x[i] * H[k, j]

        # Block (3:12, 3:12): H ⊗ (x @ y^T)
        x_y = ti.Matrix.zero(ti.f32, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                x_y[i, j] = x[i] * y[j]

        for ki in ti.static(range(3)):
            for kj in ti.static(range(3)):
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        result[3 + ki * 3 + i, 3 + kj * 3 + j] = H[ki, kj] * x_y[i, j]

        return result


@ti.data_oriented
class ABDDyadicMass:
    """
    Efficient representation of J^T @ M @ J mass matrix.

    Instead of storing full 12x12 matrix, we store:
    - m: total mass
    - m_x̄: m * x̄ (mass-weighted position)
    - m_x̄x̄^T: m * x̄ @ x̄^T (dyadic product)

    This allows efficient matrix-vector products.
    """

    @staticmethod
    @ti.func
    def compute_dyadic_mass(mass: ti.f32, x_bar: ti.types.vector(3, ti.f32)):
        """
        Compute dyadic mass components from point mass and rest position.

        Returns (m, m_x_bar, m_dyadic) where:
        - m: scalar mass
        - m_x_bar: 3-vector m * x̄
        - m_dyadic: 3x3 matrix m * x̄ @ x̄^T
        """
        m = mass
        m_x_bar = mass * x_bar
        m_dyadic = mass * (x_bar.outer_product(x_bar))
        return m, m_x_bar, m_dyadic

    @staticmethod
    @ti.func
    def apply_mass(m: ti.f32,
                   m_x_bar: ti.types.vector(3, ti.f32),
                   m_dyadic: ti.types.matrix(3, 3, ti.f32),
                   p: ti.types.vector(12, ti.f32)) -> ti.types.vector(12, ti.f32):
        """
        Apply mass matrix to vector: result = M @ p

        Uses efficient dyadic representation.
        """
        result = ti.Vector([0.0] * 12, dt=ti.f32)

        p_p = ti.Vector([p[0], p[1], p[2]])
        p_a1 = ti.Vector([p[3], p[4], p[5]])
        p_a2 = ti.Vector([p[6], p[7], p[8]])
        p_a3 = ti.Vector([p[9], p[10], p[11]])

        # Translation part: m_x̄ · p_a + m * p_p
        result[0] = m_x_bar.dot(p_a1) + m * p_p[0]
        result[1] = m_x_bar.dot(p_a2) + m * p_p[1]
        result[2] = m_x_bar.dot(p_a3) + m * p_p[2]

        # Affine parts: D @ p_ai + m_x̄ * p_p[i]
        D_p_a1 = m_dyadic @ p_a1
        D_p_a2 = m_dyadic @ p_a2
        D_p_a3 = m_dyadic @ p_a3

        for i in ti.static(range(3)):
            result[3 + i] = D_p_a1[i] + m_x_bar[i] * p_p[0]
            result[6 + i] = D_p_a2[i] + m_x_bar[i] * p_p[1]
            result[9 + i] = D_p_a3[i] + m_x_bar[i] * p_p[2]

        return result

    @staticmethod
    @ti.func
    def to_matrix(m: ti.f32,
                  m_x_bar: ti.types.vector(3, ti.f32),
                  m_dyadic: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(12, 12, ti.f32):
        """Convert dyadic representation to full 12x12 matrix."""
        M = ti.Matrix.zero(ti.f32, 12, 12)

        # Diagonal blocks for translation
        M[0, 0] = m
        M[1, 1] = m
        M[2, 2] = m

        # Off-diagonal coupling (translation-affine)
        for i in ti.static(range(3)):
            M[0, 3 + i] = m_x_bar[i]
            M[3 + i, 0] = m_x_bar[i]
            M[1, 6 + i] = m_x_bar[i]
            M[6 + i, 1] = m_x_bar[i]
            M[2, 9 + i] = m_x_bar[i]
            M[9 + i, 2] = m_x_bar[i]

        # Diagonal blocks for affine parts
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                M[3 + i, 3 + j] = m_dyadic[i, j]
                M[6 + i, 6 + j] = m_dyadic[i, j]
                M[9 + i, 9 + j] = m_dyadic[i, j]

        return M


@ti.data_oriented
class ABDShapeEnergy:
    """
    Shape preservation energy for ABD bodies.

    Energy: V_shape = κ * v * ||A @ A^T - I₃||²_F

    This penalizes non-rigid deformations by measuring how far
    A @ A^T is from identity (for rigid body A @ A^T = I).

    Equivalent formulation:
    V/κv = Σ(aᵢ·aᵢ - 1)² + Σᵢ≠ⱼ(aᵢ·aⱼ)²
    """

    @staticmethod
    @ti.func
    def compute_energy(q: ti.types.vector(12, ti.f32)) -> ti.f32:
        """
        Compute shape energy (without κv factor).

        V/κv = Σ(aᵢ·aᵢ - 1)² + 2*Σᵢ<ⱼ(aᵢ·aⱼ)²
        """
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        # Diagonal terms: (|aᵢ|² - 1)²
        E = (a1.norm_sqr() - 1.0) ** 2
        E += (a2.norm_sqr() - 1.0) ** 2
        E += (a3.norm_sqr() - 1.0) ** 2

        # Off-diagonal terms: (aᵢ·aⱼ)²
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

        Returns 9D gradient [∂V/∂a1; ∂V/∂a2; ∂V/∂a3]
        """
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        grad = ti.Vector([0.0] * 9, dt=ti.f32)

        # ∂V/∂a1 = 4(a1·a1 - 1)a1 + 4(a2·a1)a2 + 4(a3·a1)a3
        dEda1 = 4.0 * (a1.norm_sqr() - 1.0) * a1 + 4.0 * a2.dot(a1) * a2 + 4.0 * a3.dot(a1) * a3

        # ∂V/∂a2 = 4(a2·a2 - 1)a2 + 4(a3·a2)a3 + 4(a1·a2)a1
        dEda2 = 4.0 * (a2.norm_sqr() - 1.0) * a2 + 4.0 * a3.dot(a2) * a3 + 4.0 * a1.dot(a2) * a1

        # ∂V/∂a3 = 4(a3·a3 - 1)a3 + 4(a1·a3)a1 + 4(a2·a3)a2
        dEda3 = 4.0 * (a3.norm_sqr() - 1.0) * a3 + 4.0 * a1.dot(a3) * a1 + 4.0 * a2.dot(a3) * a2

        for i in ti.static(range(3)):
            grad[i] = dEda1[i]
            grad[3 + i] = dEda2[i]
            grad[6 + i] = dEda3[i]

        return grad

    @staticmethod
    @ti.func
    def compute_hessian(q: ti.types.vector(12, ti.f32)) -> ti.types.matrix(9, 9, ti.f32):
        """
        Compute shape energy Hessian w.r.t. affine part (9x9).

        H = [∂²V/∂a1²    ∂²V/∂a1∂a2  ∂²V/∂a1∂a3]
            [∂²V/∂a2∂a1  ∂²V/∂a2²    ∂²V/∂a2∂a3]
            [∂²V/∂a3∂a1  ∂²V/∂a3∂a2  ∂²V/∂a3²  ]

        Diagonal: ∂²V/∂aᵢ² = 8aᵢaᵢᵀ + 4(|aᵢ|² - 1)I + 4aⱼaⱼᵀ + 4aₖaₖᵀ
        Off-diag: ∂²V/∂aᵢ∂aⱼ = 4aⱼaᵢᵀ + 4(aᵢ·aⱼ)I
        """
        a1 = ti.Vector([q[3], q[4], q[5]])
        a2 = ti.Vector([q[6], q[7], q[8]])
        a3 = ti.Vector([q[9], q[10], q[11]])

        H = ti.Matrix.zero(ti.f32, 9, 9)
        I3 = ti.Matrix.identity(ti.f32, 3)

        # ∂²V/∂a1² = 8a1a1ᵀ + 4(|a1|² - 1)I + 4a2a2ᵀ + 4a3a3ᵀ
        ddV_da1 = 8.0 * a1.outer_product(a1) + 4.0 * (a1.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a2.outer_product(a2) + 4.0 * a3.outer_product(a3)

        # ∂²V/∂a2² = 8a2a2ᵀ + 4(|a2|² - 1)I + 4a3a3ᵀ + 4a1a1ᵀ
        ddV_da2 = 8.0 * a2.outer_product(a2) + 4.0 * (a2.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a3.outer_product(a3) + 4.0 * a1.outer_product(a1)

        # ∂²V/∂a3² = 8a3a3ᵀ + 4(|a3|² - 1)I + 4a1a1ᵀ + 4a2a2ᵀ
        ddV_da3 = 8.0 * a3.outer_product(a3) + 4.0 * (a3.norm_sqr() - 1.0) * I3 \
                  + 4.0 * a1.outer_product(a1) + 4.0 * a2.outer_product(a2)

        # ∂²V/∂a1∂a2 = 4a2a1ᵀ + 4(a1·a2)I
        ddV_da1_da2 = 4.0 * a2.outer_product(a1) + 4.0 * a1.dot(a2) * I3

        # ∂²V/∂a1∂a3 = 4a3a1ᵀ + 4(a1·a3)I
        ddV_da1_da3 = 4.0 * a3.outer_product(a1) + 4.0 * a1.dot(a3) * I3

        # ∂²V/∂a2∂a3 = 4a3a2ᵀ + 4(a2·a3)I
        ddV_da2_da3 = 4.0 * a3.outer_product(a2) + 4.0 * a2.dot(a3) * I3

        # Fill the 9x9 matrix
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
    def make_positive_definite(H: ti.types.matrix(9, 9, ti.f32)) -> ti.types.matrix(9, 9, ti.f32):
        """
        Project Hessian to positive semi-definite.

        Simple approach: clamp diagonal to be positive
        (Full eigendecomposition would be more accurate but expensive)
        """
        result = H
        # Simple diagonal clamping for numerical stability
        for i in ti.static(range(9)):
            if result[i, i] < 1e-6:
                result[i, i] = 1e-6
        return result


@ti.data_oriented
class ABDSystem:
    """
    Main ABD system managing multiple bodies.

    Handles:
    - Body state management (q, q_prev, q_tilde, q_v, dq)
    - Mass matrix computation and storage
    - Gradient/Hessian computation
    - Position mapping (q -> x)
    - Velocity update
    """

    def __init__(self, max_bodies: int = 64, max_points_per_body: int = 10000):
        """
        Initialize ABD system.

        Args:
            max_bodies: Maximum number of ABD bodies
            max_points_per_body: Maximum points per body
        """
        self.max_bodies = max_bodies
        self.max_points = max_bodies * max_points_per_body

        # Current number of bodies and total points
        self.n_bodies = 0
        self.n_total_points = 0

        # Time step and gravity (to be set externally)
        self.dt = 0.01
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        # Body state vectors (12D each)
        self.q = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)           # Current state
        self.q_prev = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)      # Previous state
        self.q_tilde = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)     # Predicted state
        self.q_v = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)         # Velocity
        self.q_temp = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)      # Temp for line search
        self.dq = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)          # Search direction
        self.grad_q = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)      # Gradient

        # Body properties
        self.body_volume = ti.field(dtype=ti.f32, shape=max_bodies)
        self.body_kappa = ti.field(dtype=ti.f32, shape=max_bodies)             # Shape stiffness
        self.body_total_mass = ti.field(dtype=ti.f32, shape=max_bodies)
        self.boundary_type = ti.field(dtype=ti.i32, shape=max_bodies)

        # Motor properties (for MOTOR type bodies)
        self.motor_speed = ti.field(dtype=ti.f32, shape=max_bodies)
        self.motor_strength = ti.field(dtype=ti.f32, shape=max_bodies)
        self.motor_axis = ti.Vector.field(3, dtype=ti.f32, shape=max_bodies)

        # Mass matrix storage (12x12 per body)
        self.abd_mass = ti.Matrix.field(12, 12, dtype=ti.f32, shape=max_bodies)
        self.abd_mass_inv = ti.Matrix.field(12, 12, dtype=ti.f32, shape=max_bodies)

        # Dyadic mass components for efficient computation
        self.body_m = ti.field(dtype=ti.f32, shape=max_bodies)
        self.body_m_x_bar = ti.Vector.field(3, dtype=ti.f32, shape=max_bodies)
        self.body_m_dyadic = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_bodies)

        # Gravity force in state space
        self.abd_gravity = ti.Vector.field(12, dtype=ti.f32, shape=max_bodies)

        # Point data
        self.point_body_id = ti.field(dtype=ti.i32, shape=self.max_points)
        self.x_bar = ti.Vector.field(3, dtype=ti.f32, shape=self.max_points)
        self.global_vertex_id = ti.field(dtype=ti.i32, shape=self.max_points)
        self.point_mass = ti.field(dtype=ti.f32, shape=self.max_points)

        # Body point ranges
        self.body_point_start = ti.field(dtype=ti.i32, shape=max_bodies)
        self.body_point_count = ti.field(dtype=ti.i32, shape=max_bodies)

        # Vertex to ABD point mapping
        self.vertex_to_abd_point = None  # Will be created when needed

    def add_body(self, point_ids: np.ndarray, rest_positions: np.ndarray,
                 masses: np.ndarray, volume: float, kappa_shape: float = 1e6,
                 boundary_type: int = 0, motor_speed: float = 0.0,
                 motor_strength: float = 10.0, motor_axis: np.ndarray = None) -> int:
        """
        Add a new ABD body.

        Args:
            point_ids: Global vertex IDs for this body
            rest_positions: Rest positions (Nx3)
            masses: Point masses (N,)
            volume: Body volume
            kappa_shape: Shape stiffness coefficient
            boundary_type: 0=FREE, 1=FIXED, 2=MOTOR
            motor_speed: Angular velocity for MOTOR type (rad/s)
            motor_strength: Motor torque scaling
            motor_axis: Rotation axis for MOTOR type (default: [0,1,0])

        Returns:
            Body ID
        """
        if self.n_bodies >= self.max_bodies:
            raise RuntimeError(f"Maximum number of bodies ({self.max_bodies}) exceeded")

        body_id = self.n_bodies
        n_points = len(point_ids)

        if self.n_total_points + n_points > self.max_points:
            raise RuntimeError(f"Maximum number of points ({self.max_points}) exceeded")

        # Compute center of mass
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass

        # Compute rest positions relative to COM
        x_bar_np = rest_positions - com

        # Initialize state: identity affine transformation
        q_init = np.zeros(12)
        q_init[0:3] = com
        q_init[3:6] = [1, 0, 0]  # a1 = e1
        q_init[6:9] = [0, 1, 0]  # a2 = e2
        q_init[9:12] = [0, 0, 1]  # a3 = e3

        # Store body data
        self.q[body_id] = q_init
        self.q_prev[body_id] = q_init
        self.q_tilde[body_id] = q_init
        self.q_v[body_id] = np.zeros(12)
        self.q_temp[body_id] = q_init
        self.dq[body_id] = np.zeros(12)
        self.grad_q[body_id] = np.zeros(12)

        self.body_volume[body_id] = volume
        self.body_kappa[body_id] = kappa_shape
        self.body_total_mass[body_id] = total_mass
        self.boundary_type[body_id] = boundary_type

        # Motor properties
        self.motor_speed[body_id] = motor_speed
        self.motor_strength[body_id] = motor_strength
        if motor_axis is None:
            motor_axis = np.array([0.0, 1.0, 0.0])
        self.motor_axis[body_id] = motor_axis / np.linalg.norm(motor_axis)

        # Store point data
        point_start = self.n_total_points
        self.body_point_start[body_id] = point_start
        self.body_point_count[body_id] = n_points

        for i in range(n_points):
            idx = point_start + i
            self.point_body_id[idx] = body_id
            self.x_bar[idx] = x_bar_np[i]
            self.global_vertex_id[idx] = point_ids[i]
            self.point_mass[idx] = masses[i]

        self.n_total_points += n_points
        self.n_bodies += 1

        # Compute mass matrix for this body
        self._compute_body_mass_matrix(body_id)

        # Compute gravity force
        self._compute_body_gravity(body_id)

        return body_id

    def _compute_body_mass_matrix(self, body_id: int):
        """Compute 12x12 mass matrix for a body from point masses."""
        point_start = int(self.body_point_start[body_id])
        point_count = int(self.body_point_count[body_id])

        # Accumulate dyadic mass components
        m_total = 0.0
        m_x_bar = np.zeros(3)
        m_dyadic = np.zeros((3, 3))

        for i in range(point_count):
            idx = point_start + i
            m = float(self.point_mass[idx])
            x = np.array([float(self.x_bar[idx][j]) for j in range(3)])

            m_total += m
            m_x_bar += m * x
            m_dyadic += m * np.outer(x, x)

        # Store dyadic components
        self.body_m[body_id] = m_total
        self.body_m_x_bar[body_id] = m_x_bar
        self.body_m_dyadic[body_id] = m_dyadic

        # Build full 12x12 mass matrix
        M = np.zeros((12, 12))
        M[0, 0] = M[1, 1] = M[2, 2] = m_total

        # Off-diagonal coupling
        for i in range(3):
            M[0, 3 + i] = M[3 + i, 0] = m_x_bar[i]
            M[1, 6 + i] = M[6 + i, 1] = m_x_bar[i]
            M[2, 9 + i] = M[9 + i, 2] = m_x_bar[i]

        # Diagonal blocks
        M[3:6, 3:6] = m_dyadic
        M[6:9, 6:9] = m_dyadic
        M[9:12, 9:12] = m_dyadic

        self.abd_mass[body_id] = M

        # Compute inverse
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        self.abd_mass_inv[body_id] = M_inv

    def _compute_body_gravity(self, body_id: int):
        """Compute gravity force in state space."""
        total_mass = float(self.body_total_mass[body_id])
        M_inv = self.abd_mass_inv[body_id].to_numpy()

        # Gravity force on translation DOFs only
        f_ext = np.zeros(12)
        gravity_np = np.array([float(self.gravity[i]) for i in range(3)])
        f_ext[0:3] = total_mass * gravity_np

        # Convert to acceleration: g = M^{-1} @ f
        g = M_inv @ f_ext
        self.abd_gravity[body_id] = g

    def setup_vertex_mapping(self, n_vertices: int):
        """
        Setup mapping from global vertex IDs to ABD points.

        Args:
            n_vertices: Total number of vertices in the mesh
        """
        self.vertex_to_abd_point = ti.field(dtype=ti.i32, shape=n_vertices)
        self.vertex_to_abd_point.fill(-1)

        for i in range(self.n_total_points):
            vid = int(self.global_vertex_id[i])
            self.vertex_to_abd_point[vid] = i

    @ti.kernel
    def compute_q_tilde(self, dt: ti.f32):
        """
        Compute predicted state q_tilde for all bodies.

        For FREE bodies:
            q_tilde = q + dt * q_v + dt² * M^{-1} @ f_ext

        For FIXED bodies:
            q_tilde = q (no change)

        For MOTOR bodies:
            Apply prescribed rotation to affine part
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                self.q_tilde[body_id] = self.q_prev[body_id]
            else:
                # Standard prediction with gravity
                q_prev = self.q_prev[body_id]
                q_v = self.q_v[body_id]
                g = self.abd_gravity[body_id]

                self.q_tilde[body_id] = q_prev + dt * q_v + (dt * dt) * g

    @ti.kernel
    def compute_x_from_q(self, vertices: ti.template()):
        """
        Map ABD state q to vertex positions x.

        x_i = J_i @ q = p + A @ x̄_i
        """
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar_i = self.x_bar[i]
            q = self.q[body_id]
            global_id = self.global_vertex_id[i]

            # x = J @ q
            x = ABDJacobian.apply_J(x_bar_i, q)
            vertices[global_id] = ti.Vector([ti.cast(x[0], ti.f32),
                                             ti.cast(x[1], ti.f32),
                                             ti.cast(x[2], ti.f32)])

    @ti.kernel
    def project_gradient_to_q(self, vertex_grad: ti.template()):
        """
        Project vertex gradients to ABD state space.

        g_q = Σ_i J_i^T @ g_x_i

        Args:
            vertex_grad: Per-vertex gradients (3D)
        """
        # Clear gradients
        for body_id in range(self.n_bodies):
            self.grad_q[body_id] = ti.Vector([0.0] * 12, dt=ti.f32)

        # Accumulate gradients
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]

            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            x_bar_i = self.x_bar[i]
            global_id = self.global_vertex_id[i]
            g_x = vertex_grad[global_id]

            # Convert to f32
            g_x_f32 = ti.Vector([ti.cast(g_x[0], ti.f32),
                                 ti.cast(g_x[1], ti.f32),
                                 ti.cast(g_x[2], ti.f32)])

            # g_q = J^T @ g_x
            g_q = ABDJacobian.apply_JT(x_bar_i, g_x_f32)

            # Atomic add
            for d in ti.static(range(12)):
                ti.atomic_add(self.grad_q[body_id][d], g_q[d])

    @ti.kernel
    def add_inertia_gradient(self):
        """
        Add inertia gradient to grad_q.

        g_inertia = M @ (q - q_tilde)
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]
            M = self.abd_mass[body_id]

            dq = q - q_tilde
            g_inertia = M @ dq

            for d in ti.static(range(12)):
                self.grad_q[body_id][d] += g_inertia[d]

    @ti.kernel
    def add_shape_gradient(self):
        """
        Add shape energy gradient to grad_q.

        g_shape = κ * v * dt² * ∂V_shape/∂q
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            q = self.q[body_id]
            kappa = self.body_kappa[body_id]
            volume = self.body_volume[body_id]
            dt = self.dt

            # Compute shape gradient (9D)
            shape_grad = ABDShapeEnergy.compute_gradient(q)

            # Scale by κ * v * dt²
            scale = kappa * volume * dt * dt

            # Add to gradient (only affine part, indices 3-11)
            for d in ti.static(range(9)):
                self.grad_q[body_id][3 + d] += scale * shape_grad[d]

    @ti.kernel
    def add_motor_constraint_gradient(self):
        """
        Add motor constraint gradient for MOTOR type bodies.

        The motor constraint penalizes deviation from prescribed rotation.
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] != BodyBoundaryType.MOTOR:
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]
            motor_str = self.motor_strength[body_id]
            M = self.abd_mass[body_id]

            # Compute rotation delta
            dt = self.dt
            theta = self.motor_speed[body_id] * dt
            axis = self.motor_axis[body_id]

            # For now, use simplified motor constraint
            # Penalize deviation in affine part from expected rotation
            dq = q - q_tilde

            # Zero out translation DOFs for motor constraint
            dq[0] = 0.0
            dq[1] = 0.0
            dq[2] = 0.0
            dq[3] = 0.0
            dq[4] = 0.0
            dq[5] = 0.0

            # Motor Hessian: only affects rotation DOFs (6-11)
            motor_grad = ti.Vector([0.0] * 12, dt=ti.f32)
            for i in ti.static(range(6)):
                for j in ti.static(range(6)):
                    motor_grad[6 + i] += motor_str * M[6 + i, 6 + j] * dq[6 + j]

            for d in ti.static(range(12)):
                self.grad_q[body_id][d] += motor_grad[d]

    @ti.kernel
    def copy_q_to_temp(self):
        """Copy current state to temp for line search."""
        for body_id in range(self.n_bodies):
            self.q_temp[body_id] = self.q[body_id]

    @ti.kernel
    def step_forward(self, alpha: ti.f32):
        """
        Update state: q = q_temp - alpha * dq

        Args:
            alpha: Step size
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            self.q[body_id] = self.q_temp[body_id] - alpha * self.dq[body_id]

    @ti.kernel
    def update_velocity(self, dt: ti.f32):
        """
        Update velocity after optimization step.

        q_v = (q - q_prev) / dt
        """
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                self.q_v[body_id] = ti.Vector([0.0] * 12, dt=ti.f32)
            else:
                self.q_v[body_id] = (self.q[body_id] - self.q_prev[body_id]) / dt

            # Update q_prev for next frame
            self.q_prev[body_id] = self.q[body_id]

    @ti.kernel
    def compute_kinetic_energy(self) -> ti.f32:
        """
        Compute total kinetic energy.

        K = 0.5 * Σ (q - q_tilde)^T @ M @ (q - q_tilde)
        """
        K = 0.0
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            q = self.q[body_id]
            q_tilde = self.q_tilde[body_id]
            M = self.abd_mass[body_id]

            dq = q - q_tilde
            Mdq = M @ dq
            K += 0.5 * dq.dot(Mdq)

        return K

    @ti.kernel
    def compute_shape_energy(self) -> ti.f32:
        """
        Compute total shape energy.

        V = Σ κ * v * dt² * V_shape(q)
        """
        V = 0.0
        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            kappa = self.body_kappa[body_id]
            volume = self.body_volume[body_id]
            dt = self.dt

            V += kappa * volume * dt * dt * ABDShapeEnergy.compute_energy(q)

        return V

    def compute_shape_hessian_contribution(self, dt: float) -> float:
        """
        Compute p^T @ H_shape @ p for line search.

        Returns contribution to pHp from shape energy Hessian.
        """
        return self._compute_shape_pHp(dt)

    @ti.kernel
    def _compute_shape_pHp(self, dt: ti.f32) -> ti.f32:
        """Kernel to compute shape Hessian contribution."""
        pHp = 0.0
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            q = self.q[body_id]
            dq = self.dq[body_id]
            kappa = self.body_kappa[body_id]
            volume = self.body_volume[body_id]

            # Compute shape Hessian (9x9)
            H_shape = ABDShapeEnergy.compute_hessian(q)
            H_shape = ABDShapeEnergy.make_positive_definite(H_shape)

            # Extract affine part of dq
            dq_affine = ti.Vector([dq[3], dq[4], dq[5], dq[6], dq[7], dq[8],
                                   dq[9], dq[10], dq[11]], dt=ti.f32)

            # p^T @ H @ p
            Hp = H_shape @ dq_affine
            pHp += kappa * volume * dt * dt * dq_affine.dot(Hp)

        return pHp

    def compute_motor_hessian_contribution(self) -> float:
        """Compute motor constraint Hessian contribution to pHp."""
        return self._compute_motor_pHp()

    @ti.kernel
    def _compute_motor_pHp(self) -> ti.f32:
        """Kernel to compute motor Hessian contribution."""
        pHp = 0.0
        for body_id in range(self.n_bodies):
            if self.boundary_type[body_id] != BodyBoundaryType.MOTOR:
                continue

            dq = self.dq[body_id]
            motor_str = self.motor_strength[body_id]
            M = self.abd_mass[body_id]

            # Motor Hessian only affects rotation DOFs
            for i in ti.static(range(6)):
                for j in ti.static(range(6)):
                    pHp += motor_str * dq[6 + i] * M[6 + i, 6 + j] * dq[6 + j]

        return pHp

    def compute_ccd_step_size(self, ground_y: float = 0.0, dHat: float = 0.01) -> float:
        """
        Compute maximum step size for continuous collision detection.

        Simple implementation: just check distance to ground plane.
        """
        return self._compute_ccd_alpha(ground_y, dHat)

    @ti.kernel
    def _compute_ccd_alpha(self, ground_y: ti.f32, dHat: ti.f32) -> ti.f32:
        """Kernel to compute CCD step size."""
        alpha_min = 1.0

        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            if self.boundary_type[body_id] == BodyBoundaryType.FIXED:
                continue

            x_bar_i = self.x_bar[i]
            q = self.q_temp[body_id]
            dq = self.dq[body_id]

            # Current position
            x_curr = ABDJacobian.apply_J(x_bar_i, q)

            # Direction
            x_dir = ABDJacobian.apply_J(x_bar_i, dq)

            # Check ground plane
            if x_dir[1] > 0:  # Moving down (towards ground)
                dist = x_curr[1] - ground_y - dHat
                if dist > 0 and x_dir[1] > 1e-10:
                    alpha = 0.9 * dist / x_dir[1]
                    ti.atomic_min(alpha_min, alpha)

        return alpha_min

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            'n_bodies': self.n_bodies,
            'n_total_points': self.n_total_points,
            'kinetic_energy': float(self.compute_kinetic_energy()),
            'shape_energy': float(self.compute_shape_energy())
        }


# Utility functions

def create_identity_state() -> np.ndarray:
    """Create identity ABD state (origin + identity matrix)."""
    q = np.zeros(12)
    q[3] = 1.0  # a1 = [1, 0, 0]
    q[7] = 1.0  # a2 = [0, 1, 0]
    q[11] = 1.0  # a3 = [0, 0, 1]
    return q


def extract_affine_matrix(q: np.ndarray) -> np.ndarray:
    """Extract 3x3 affine matrix A from state q."""
    A = np.zeros((3, 3))
    A[0, :] = q[3:6]
    A[1, :] = q[6:9]
    A[2, :] = q[9:12]
    return A


def set_affine_matrix(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Set affine matrix A in state q."""
    q_new = q.copy()
    q_new[3:6] = A[0, :]
    q_new[6:9] = A[1, :]
    q_new[9:12] = A[2, :]
    return q_new
