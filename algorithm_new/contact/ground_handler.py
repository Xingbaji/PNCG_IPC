"""
Ground Contact Handler for handling contact with ground plane.

Provides specialized contact handling for ground plane collisions.
"""

import taichi as ti
from typing import Optional

from ..core.precision import PrecisionType, PrecisionMixin
from .barrier import LogBarrier, CubicBarrier


@ti.data_oriented
class GroundContactHandler(PrecisionMixin):
    """
    Ground Contact Handler for ground plane collision.

    Handles contact between vertices and an infinite ground plane at y = ground_y.
    Uses the same barrier functions as IPC contact but with simplified geometry.

    Usage:
        handler = GroundContactHandler(n_vertices, ground_y=-0.5)
        handler.add_gradient(vertices, grad, dt, kappa, dHat)
        handler.add_diagonal_hessian(vertices, diagH, dt, kappa, dHat)
    """

    def __init__(
        self,
        max_vertices: int,
        ground_y: float = 0.0,
        ground_normal: tuple = (0.0, 1.0, 0.0),
        barrier_type: str = 'log',
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize ground contact handler.

        Args:
            max_vertices: Maximum number of vertices
            ground_y: Y-coordinate of ground plane
            ground_normal: Normal vector of ground plane (default: pointing up)
            barrier_type: Type of barrier function ('log' or 'cubic')
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_VERTICES = max_vertices
        self.barrier_type = barrier_type

        float_type = self.cfg.float_type

        # Ground plane parameters
        self._ground_y = ti.field(dtype=float_type, shape=())
        self._ground_y[None] = ground_y

        self._ground_normal = ti.Vector.field(3, dtype=float_type, shape=())
        self._ground_normal[None] = ti.Vector(ground_normal)

        # Vertex selection mask (optional)
        self._vertex_mask = ti.field(dtype=ti.i32, shape=max_vertices)
        self._use_mask = False

    @property
    def ground_y(self) -> float:
        """Get ground Y coordinate."""
        return self._ground_y[None]

    @ground_y.setter
    def ground_y(self, value: float):
        """Set ground Y coordinate."""
        self._ground_y[None] = value

    def set_vertex_mask(self, mask):
        """
        Set which vertices to check for ground contact.

        Args:
            mask: Array-like of vertex indices, or None to check all
        """
        if mask is None:
            self._use_mask = False
        else:
            self._use_mask = True
            for i, idx in enumerate(mask):
                self._vertex_mask[i] = idx

    @ti.func
    def _barrier_energy(self, d, dHat, kappa):
        """Dispatch to appropriate barrier energy function."""
        E = d * 0.0
        if ti.static(self.barrier_type == 'cubic'):
            E = CubicBarrier.energy(d, dHat, kappa)
        else:
            E = LogBarrier.energy(d, dHat, kappa)
        return E

    @ti.func
    def _barrier_gradient(self, d, dHat, kappa):
        """Dispatch to appropriate barrier gradient function."""
        g = d * 0.0
        if ti.static(self.barrier_type == 'cubic'):
            g = CubicBarrier.gradient(d, dHat, kappa)
        else:
            g = LogBarrier.gradient(d, dHat, kappa)
        return g

    @ti.func
    def _barrier_hessian(self, d, dHat, kappa):
        """Dispatch to appropriate barrier hessian function."""
        H = d * 0.0
        if ti.static(self.barrier_type == 'cubic'):
            H = CubicBarrier.hessian(d, dHat, kappa)
        else:
            H = LogBarrier.hessian(d, dHat, kappa)
        return H

    @ti.kernel
    def compute_energy(
        self,
        vertices: ti.template(),
        n_vertices: ti.i32,
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ) -> ti.template():
        """
        Compute total ground contact energy.

        Args:
            vertices: Vertex positions field (n_vertices x 3)
            n_vertices: Number of vertices
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold

        Returns:
            Total ground contact energy
        """
        dt_inv_sq = 1.0 / (dt * dt)
        ground_y = self._ground_y[None]
        total_E = kappa * 0.0  # Preserve type

        for i in range(n_vertices):
            # Distance to ground plane
            d = vertices[i][1] - ground_y
            if d < dHat:
                E_contact = self._barrier_energy(d, dHat, kappa)
                total_E += dt_inv_sq * E_contact

        return total_E

    @ti.kernel
    def add_gradient(
        self,
        vertices: ti.template(),
        grad: ti.template(),
        n_vertices: ti.i32,
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Add ground contact gradient to vertex gradients.

        For ground contact, the gradient is in the Y direction only:
        grad[i] += (1/dt^2) * barrier_g(d) * [0, 1, 0]

        Args:
            vertices: Vertex positions field (n_vertices x 3)
            grad: Gradient field to add to (n_vertices x 3)
            n_vertices: Number of vertices
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        ground_y = self._ground_y[None]

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            if d < dHat and d > 1e-10:
                bg = self._barrier_gradient(d, dHat, kappa)
                grad[i][1] += dt_inv_sq * bg

    @ti.kernel
    def add_diagonal_hessian(
        self,
        vertices: ti.template(),
        diagH: ti.template(),
        n_vertices: ti.i32,
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Add ground contact diagonal Hessian.

        For ground contact, only the Y component is affected:
        diagH[i][1] += (1/dt^2) * barrier_H(d)

        Args:
            vertices: Vertex positions field (n_vertices x 3)
            diagH: Diagonal Hessian field to add to (n_vertices x 3)
            n_vertices: Number of vertices
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        ground_y = self._ground_y[None]

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            if d < dHat and d > 1e-10:
                bH = self._barrier_hessian(d, dHat, kappa)
                # Only Y component affected, and ensure non-negative
                diagH[i][1] += ti.max(dt_inv_sq * bH, vertices[i].dtype(0.0))

    @ti.kernel
    def compute_pHp(
        self,
        vertices: ti.template(),
        p: ti.template(),
        n_vertices: ti.i32,
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ) -> ti.template():
        """
        Compute p^T * H_ground * p.

        For ground contact:
        pHp += (1/dt^2) * barrier_H(d) * p[i][1]^2

        Args:
            vertices: Vertex positions field
            p: Search direction field (n_vertices x 3)
            n_vertices: Number of vertices
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold

        Returns:
            Scalar p^T * H * p
        """
        dt_inv_sq = 1.0 / (dt * dt)
        ground_y = self._ground_y[None]
        pHp = kappa * 0.0

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            if d < dHat and d > 1e-10:
                bH = self._barrier_hessian(d, dHat, kappa)
                py = p[i][1]
                pHp += dt_inv_sq * bH * py * py

        return pHp

    @ti.kernel
    def apply_hessian(
        self,
        vertices: ti.template(),
        v: ti.template(),
        result: ti.template(),
        n_vertices: ti.i32,
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Apply ground contact Hessian: result += H_ground * v

        For ground contact:
        result[i][1] += (1/dt^2) * barrier_H(d) * v[i][1]

        Args:
            vertices: Vertex positions field
            v: Input vector field (n_vertices x 3)
            result: Output field to add to (n_vertices x 3)
            n_vertices: Number of vertices
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        ground_y = self._ground_y[None]

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            if d < dHat and d > 1e-10:
                bH = self._barrier_hessian(d, dHat, kappa)
                result[i][1] += dt_inv_sq * bH * v[i][1]

    @ti.kernel
    def compute_min_ground_distance(
        self,
        vertices: ti.template(),
        n_vertices: ti.i32,
    ) -> ti.template():
        """
        Compute minimum distance to ground.

        Args:
            vertices: Vertex positions field
            n_vertices: Number of vertices

        Returns:
            Minimum distance to ground
        """
        ground_y = self._ground_y[None]
        min_d = vertices[0].dtype(1e10)

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            ti.atomic_min(min_d, d)

        return min_d

    @ti.kernel
    def count_ground_contacts(
        self,
        vertices: ti.template(),
        n_vertices: ti.i32,
        dHat: ti.template(),
    ) -> ti.i32:
        """
        Count vertices in contact with ground.

        Args:
            vertices: Vertex positions field
            n_vertices: Number of vertices
            dHat: Contact threshold

        Returns:
            Number of ground contacts
        """
        ground_y = self._ground_y[None]
        count = 0

        for i in range(n_vertices):
            d = vertices[i][1] - ground_y
            if d < dHat:
                ti.atomic_add(count, 1)

        return count
