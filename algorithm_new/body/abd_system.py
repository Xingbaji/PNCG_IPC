"""
ABD (Affine Body Dynamics) System.

Manages multiple affine bodies with 12D state representation.
Implements the BodySystem protocol for integration with FEM solvers.
"""

import taichi as ti
import numpy as np
from enum import IntEnum
from typing import Optional, List

from ..core.precision import PrecisionType, PrecisionMixin
from .jacobian import ABDJacobian
from .mass import ABDDyadicMass
from .shape_energy import ABDShapeEnergy


class BodyBoundaryType(IntEnum):
    """Boundary condition types for ABD bodies."""
    FREE = 0    # Standard physics (gravity + inertia)
    FIXED = 1   # Pinned/constrained (no motion)
    MOTOR = 2   # Prescribed rotation around axis


@ti.data_oriented
class ABDSystem(PrecisionMixin):
    """
    ABD System for managing affine bodies.

    Each body has a 12D state vector q:
      q[0:3]   = p     (center of mass position)
      q[3:6]   = a1    (first row of affine matrix)
      q[6:9]   = a2    (second row of affine matrix)
      q[9:12]  = a3    (third row of affine matrix)

    Vertex positions: x_i = p + A @ x̄_i where x̄_i is rest position.

    Usage:
        system = ABDSystem(max_bodies=64, max_points=10000)
        body_id = system.add_body(point_ids, rest_positions, masses, volume)
        system.compute_predicted_state(dt)
        system.compute_x_from_q(vertices)
        system.project_gradient(vertex_grad)
        system.add_body_gradient()
        system.step_forward(alpha)
    """

    def __init__(
        self,
        max_bodies: int = 64,
        max_points_per_body: int = 10000,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize ABD system.

        Args:
            max_bodies: Maximum number of ABD bodies
            max_points_per_body: Maximum points per body (for allocation)
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_BODIES = max_bodies
        self.MAX_POINTS = max_bodies * max_points_per_body

        float_type = self.cfg.float_type

        # Body count
        self._n_bodies = ti.field(dtype=ti.i32, shape=())
        self._n_bodies[None] = 0
        self._n_total_points = ti.field(dtype=ti.i32, shape=())
        self._n_total_points[None] = 0

        # Body start indices in point arrays
        self._body_point_start = ti.field(dtype=ti.i32, shape=max_bodies)
        self._body_point_count = ti.field(dtype=ti.i32, shape=max_bodies)

        # State vectors (12D each)
        self._q = ti.Vector.field(12, dtype=float_type, shape=max_bodies)
        self._q_prev = ti.Vector.field(12, dtype=float_type, shape=max_bodies)
        self._q_tilde = ti.Vector.field(12, dtype=float_type, shape=max_bodies)
        self._q_v = ti.Vector.field(12, dtype=float_type, shape=max_bodies)
        self._dq = ti.Vector.field(12, dtype=float_type, shape=max_bodies)
        self._grad_q = ti.Vector.field(12, dtype=float_type, shape=max_bodies)

        # Mass matrices (stored as components + full inverse)
        self._body_m = ti.field(dtype=float_type, shape=max_bodies)
        self._body_m_x_bar = ti.Vector.field(3, dtype=float_type, shape=max_bodies)
        self._body_m_dyadic = ti.Matrix.field(3, 3, dtype=float_type, shape=max_bodies)
        self._body_mass_inv = ti.Matrix.field(12, 12, dtype=float_type, shape=max_bodies)

        # Body properties
        self._body_volume = ti.field(dtype=float_type, shape=max_bodies)
        self._body_kappa = ti.field(dtype=float_type, shape=max_bodies)
        self._boundary_type = ti.field(dtype=ti.i32, shape=max_bodies)

        # Gravity
        self._gravity = ti.Vector.field(3, dtype=float_type, shape=())
        self._gravity[None] = ti.Vector([0.0, -9.8, 0.0], dt=float_type)

        # Point data
        self._point_body_id = ti.field(dtype=ti.i32, shape=self.MAX_POINTS)
        self._x_bar = ti.Vector.field(3, dtype=float_type, shape=self.MAX_POINTS)
        self._global_vertex_id = ti.field(dtype=ti.i32, shape=self.MAX_POINTS)
        self._point_mass = ti.field(dtype=float_type, shape=self.MAX_POINTS)

        # Time step (set before integration)
        self._dt = ti.field(dtype=float_type, shape=())

        # CPU storage for mass matrix inversion
        self._body_mass_np: List[Optional[np.ndarray]] = [None] * max_bodies
        self._body_mass_inv_np: List[Optional[np.ndarray]] = [None] * max_bodies

    @property
    def n_bodies(self) -> int:
        """Get number of ABD bodies."""
        return self._n_bodies[None]

    @property
    def n_total_points(self) -> int:
        """Get total number of points across all bodies."""
        return self._n_total_points[None]

    def set_gravity(self, gravity: tuple):
        """Set gravity vector."""
        self._gravity[None] = ti.Vector(gravity)

    def add_body(
        self,
        point_ids: np.ndarray,
        rest_positions: np.ndarray,
        masses: np.ndarray,
        volume: float,
        kappa_shape: float = 1e6,
        boundary_type: int = BodyBoundaryType.FREE,
    ) -> int:
        """
        Add a new ABD body to the system.

        Args:
            point_ids: Global vertex IDs for this body (N,)
            rest_positions: Rest positions in world space (Nx3)
            masses: Point masses (N,)
            volume: Body volume
            kappa_shape: Shape stiffness for rigidity
            boundary_type: Boundary condition type

        Returns:
            Body ID
        """
        body_id = self._n_bodies[None]
        n_points = len(point_ids)
        point_start = self._n_total_points[None]

        # Compute center of mass
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass

        # Rest positions relative to COM
        x_bar_np = rest_positions - com

        # Initialize state: identity affine
        q_init = np.zeros(12, dtype=np.float32 if self.precision == 'f32' else np.float64)
        q_init[0:3] = com
        q_init[3:6] = [1, 0, 0]
        q_init[6:9] = [0, 1, 0]
        q_init[9:12] = [0, 0, 1]

        # Set body properties
        self._q[body_id] = q_init
        self._q_prev[body_id] = q_init
        self._q_tilde[body_id] = q_init
        self._q_v[body_id] = np.zeros(12)
        self._body_volume[body_id] = volume
        self._body_kappa[body_id] = kappa_shape
        self._boundary_type[body_id] = boundary_type
        self._body_point_start[body_id] = point_start
        self._body_point_count[body_id] = n_points

        # Set point data
        for i in range(n_points):
            idx = point_start + i
            self._point_body_id[idx] = body_id
            self._x_bar[idx] = x_bar_np[i]
            self._global_vertex_id[idx] = point_ids[i]
            self._point_mass[idx] = masses[i]

        # Compute mass matrix
        self._compute_body_mass_matrix(body_id, x_bar_np, masses)

        self._n_bodies[None] += 1
        self._n_total_points[None] += n_points

        return body_id

    def _compute_body_mass_matrix(self, body_id: int, x_bar_np: np.ndarray, masses: np.ndarray):
        """Compute 12×12 mass matrix and its inverse for a body."""
        # Accumulate dyadic components
        m_total = float(np.sum(masses))
        m_x_bar_np = np.sum(x_bar_np * masses[:, np.newaxis], axis=0)
        m_dyadic_np = np.zeros((3, 3))
        for i in range(len(masses)):
            m_dyadic_np += masses[i] * np.outer(x_bar_np[i], x_bar_np[i])

        # Store components
        self._body_m[body_id] = m_total
        self._body_m_x_bar[body_id] = m_x_bar_np
        self._body_m_dyadic[body_id] = m_dyadic_np

        # Build full matrix and invert
        M_np = ABDDyadicMass.to_matrix_numpy(m_total, m_x_bar_np, m_dyadic_np)
        M_inv_np = np.linalg.inv(M_np)

        self._body_mass_np[body_id] = M_np
        self._body_mass_inv_np[body_id] = M_inv_np
        self._body_mass_inv[body_id] = M_inv_np

    @ti.kernel
    def compute_predicted_state(self, dt: ti.template()):
        """
        Compute predicted state q̃ for implicit Euler.

        q̃ = q + dt * v + dt² * M⁻¹ * f_ext

        Args:
            dt: Time step
        """
        self._dt[None] = dt

        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] == ti.i32(BodyBoundaryType.FIXED):
                # Fixed body: no change
                self._q_tilde[body_id] = self._q_prev[body_id]
            else:
                q = self._q_prev[body_id]
                v = self._q_v[body_id]
                M_inv = self._body_mass_inv[body_id]

                # Gravity force (acts on translation part)
                g = self._gravity[None]
                m = self._body_m[body_id]
                f_ext = ti.Vector([m * g[0], m * g[1], m * g[2],
                                   0, 0, 0, 0, 0, 0, 0, 0, 0], dt=q.dtype)

                # Predicted state
                q_tilde = q + dt * v + dt * dt * (M_inv @ f_ext)
                self._q_tilde[body_id] = q_tilde
                self._q[body_id] = q_tilde

    @ti.kernel
    def compute_x_from_q(self, vertices: ti.template()):
        """
        Map ABD state q to vertex positions.

        x_i = J_i @ q = p + A @ x̄_i

        Args:
            vertices: Vertex position field to update
        """
        for i in range(self._n_total_points[None]):
            body_id = self._point_body_id[i]
            x_bar_i = self._x_bar[i]
            q = self._q[body_id]
            global_id = self._global_vertex_id[i]
            x = ABDJacobian.apply_J(x_bar_i, q)
            vertices[global_id] = x

    @ti.kernel
    def reset_gradient(self):
        """Reset body gradients to zero."""
        for body_id in range(self._n_bodies[None]):
            self._grad_q[body_id] = ti.Vector([0.0] * 12, dt=self._q[0].dtype)

    @ti.kernel
    def project_gradient(self, vertex_grad: ti.template()):
        """
        Project vertex gradients to ABD state space.

        g_q = Σ_i J_i^T @ g_x_i

        Args:
            vertex_grad: Vertex gradient field
        """
        for i in range(self._n_total_points[None]):
            body_id = self._point_body_id[i]
            x_bar_i = self._x_bar[i]
            global_id = self._global_vertex_id[i]
            g_x = vertex_grad[global_id]
            g_q = ABDJacobian.apply_JT(x_bar_i, g_x)
            for d in ti.static(range(12)):
                ti.atomic_add(self._grad_q[body_id][d], g_q[d])

    @ti.kernel
    def add_inertia_gradient(self):
        """
        Add inertia gradient: g += M @ (q - q̃)

        This is the gradient of the implicit Euler inertia term.
        """
        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] != ti.i32(BodyBoundaryType.FIXED):
                q = self._q[body_id]
                q_tilde = self._q_tilde[body_id]
                m = self._body_m[body_id]
                m_x_bar = self._body_m_x_bar[body_id]
                m_dyadic = self._body_m_dyadic[body_id]

                diff = q - q_tilde
                g_inertia = ABDDyadicMass.apply(m, m_x_bar, m_dyadic, diff)

                for d in ti.static(range(12)):
                    self._grad_q[body_id][d] += g_inertia[d]

    @ti.kernel
    def add_shape_gradient(self):
        """
        Add shape energy gradient: g += κ * v * dt² * ∂V_shape/∂q

        This penalizes non-rigid deformations.
        """
        dt_sq = self._dt[None] * self._dt[None]

        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] != ti.i32(BodyBoundaryType.FIXED):
                q = self._q[body_id]
                kappa = self._body_kappa[body_id]
                volume = self._body_volume[body_id]

                shape_grad = ABDShapeEnergy.gradient_12d(q)
                scale = kappa * volume * dt_sq

                for d in ti.static(range(12)):
                    self._grad_q[body_id][d] += scale * shape_grad[d]

    @ti.kernel
    def compute_search_direction(self):
        """
        Compute search direction: dq = -M⁻¹ @ grad_q

        Simple gradient descent with mass preconditioning.
        """
        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] != ti.i32(BodyBoundaryType.FIXED):
                M_inv = self._body_mass_inv[body_id]
                grad = self._grad_q[body_id]
                self._dq[body_id] = -M_inv @ grad
            else:
                self._dq[body_id] = ti.Vector([0.0] * 12, dt=self._q[0].dtype)

    @ti.kernel
    def step_forward(self, alpha: ti.template()):
        """
        Take a step: q += alpha * dq

        Args:
            alpha: Step size
        """
        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] != ti.i32(BodyBoundaryType.FIXED):
                self._q[body_id] = self._q[body_id] + alpha * self._dq[body_id]

    @ti.kernel
    def update_velocity(self, dt: ti.template()):
        """
        Update velocity after time step: v = (q - q_prev) / dt

        Args:
            dt: Time step
        """
        for body_id in range(self._n_bodies[None]):
            if self._boundary_type[body_id] != ti.i32(BodyBoundaryType.FIXED):
                self._q_v[body_id] = (self._q[body_id] - self._q_prev[body_id]) / dt
                self._q_prev[body_id] = self._q[body_id]

    @ti.kernel
    def compute_kinetic_energy(self) -> ti.template():
        """
        Compute total kinetic energy: E = 0.5 * Σ v^T M v

        Returns:
            Total kinetic energy
        """
        E = self._q[0].dtype(0.0)
        for body_id in range(self._n_bodies[None]):
            v = self._q_v[body_id]
            m = self._body_m[body_id]
            m_x_bar = self._body_m_x_bar[body_id]
            m_dyadic = self._body_m_dyadic[body_id]
            Mv = ABDDyadicMass.apply(m, m_x_bar, m_dyadic, v)
            E += 0.5 * v.dot(Mv)
        return E

    @ti.kernel
    def compute_shape_energy(self) -> ti.template():
        """
        Compute total shape energy.

        Returns:
            Total shape energy
        """
        E = self._q[0].dtype(0.0)
        for body_id in range(self._n_bodies[None]):
            q = self._q[body_id]
            kappa = self._body_kappa[body_id]
            volume = self._body_volume[body_id]
            E += kappa * volume * ABDShapeEnergy.energy(q)
        return E
