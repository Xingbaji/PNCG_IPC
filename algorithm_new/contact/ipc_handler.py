"""
IPC Contact Handler implementing the ContactHandler protocol.

Provides contact force computation using IPC barrier functions.
"""

import taichi as ti
from typing import Literal

from ..core.precision import PrecisionType, PrecisionMixin
from .barrier import LogBarrier, CubicBarrier
from .utils.jacobian import compute_dtdx_t


BarrierType = Literal['log', 'cubic']


@ti.data_oriented
class IPCContactHandler(PrecisionMixin):
    """
    IPC Contact Handler for computing contact forces and Hessians.

    Implements the ContactHandler protocol with support for:
    - Log barrier (default IPC)
    - Cubic barrier (simpler alternative)
    - Adaptive kappa computation
    - SPD Hessian projection

    Usage:
        handler = IPCContactHandler(max_contacts=2**18, barrier_type='log')
        handler.set_contacts(contact_storage, n_contacts)
        handler.add_gradient(vertices, grad, dt, kappa, dHat)
        handler.add_diagonal_hessian(vertices, diagH, dt, kappa, dHat)
    """

    def __init__(
        self,
        max_contacts: int = 2**18,
        barrier_type: BarrierType = 'log',
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize IPC contact handler.

        Args:
            max_contacts: Maximum number of contact pairs
            barrier_type: Type of barrier function ('log' or 'cubic')
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_CONTACTS = max_contacts
        self.barrier_type = barrier_type

        float_type = self.cfg.float_type

        # Contact pair storage reference (set via set_contacts)
        self._contact_pairs = None
        self._n_contacts = ti.field(dtype=ti.i32, shape=())
        self._n_contacts[None] = 0

        # Per-contact precomputed values for efficiency
        # barrier_g / dist
        self._para = ti.field(dtype=float_type, shape=max_contacts)
        # (barrier_H - para) / dist^2
        self._para0 = ti.field(dtype=float_type, shape=max_contacts)

    def set_contacts(self, contact_storage, n_contacts: int = None):
        """
        Set the contact pairs for processing.

        Args:
            contact_storage: ContactPairStorage instance
            n_contacts: Number of contacts (if None, uses storage.count)
        """
        self._contact_pairs = contact_storage.contact_pairs
        if n_contacts is not None:
            self._n_contacts[None] = n_contacts
        else:
            self._n_contacts[None] = contact_storage.count

    @property
    def n_contacts(self) -> int:
        """Get current number of contacts."""
        return self._n_contacts[None]

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

    @ti.func
    def _barrier_curvature(self, d, dHat, kappa):
        """Dispatch to appropriate barrier curvature function."""
        c = d * 0.0
        if ti.static(self.barrier_type == 'cubic'):
            c = CubicBarrier.curvature(d, dHat, kappa)
        else:
            c = LogBarrier.curvature(d, dHat, kappa)
        return c

    @ti.kernel
    def compute_energy(
        self,
        vertices: ti.template(),
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ) -> ti.template():
        """
        Compute total contact barrier energy.

        Args:
            vertices: Vertex positions field (not used, distances from contacts)
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold

        Returns:
            Total contact energy
        """
        dt_inv_sq = 1.0 / (dt * dt)
        total_E = kappa * 0.0  # Preserve type
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            dist = pair.b
            E_contact = self._barrier_energy(dist, dHat, kappa)
            total_E += dt_inv_sq * E_contact
        return total_E

    @ti.kernel
    def add_gradient(
        self,
        vertices: ti.template(),
        grad: ti.template(),
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Add contact gradient to vertex gradients.

        The contact force on vertex i is: f_i = (1/dt^2) * (barrier_g / dist) * cord[i] * t

        Args:
            vertices: Vertex positions field
            grad: Gradient field to add to (n_vertices x 3)
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist > 1e-10:
                bg = self._barrier_gradient(dist, dHat, kappa)
                para = bg / dist

                for i in ti.static(range(4)):
                    CORD = cord[i]
                    ID = ti.cast(ids[i], ti.i32)
                    force = dt_inv_sq * para * CORD * t
                    grad[ID] += force

    @ti.kernel
    def add_diagonal_hessian(
        self,
        vertices: ti.template(),
        diagH: ti.template(),
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Add contact diagonal Hessian approximation.

        The diagonal Hessian contribution for vertex i is:
            diag_i = (1/dt^2) * cord[i]^2 * (para0 * t*t + para)

        Where:
            para = barrier_g / dist
            para0 = (barrier_H - para) / dist^2

        Args:
            vertices: Vertex positions field
            diagH: Diagonal Hessian field to add to (n_vertices x 3)
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist > 1e-10:
                dist2 = dist * dist
                bg = self._barrier_gradient(dist, dHat, kappa)
                bH = self._barrier_hessian(dist, dHat, kappa)

                para = bg / dist
                para0 = (bH - para) / dist2

                for i in ti.static(range(4)):
                    CORD = cord[i]
                    CORD_sq = CORD * CORD
                    ID = ti.cast(ids[i], ti.i32)
                    # diag_tmp = CORD^2 * (para0 * t*t + para * ones)
                    one_vec = ti.Vector([1.0, 1.0, 1.0], dt=t.dtype)
                    diag_tmp = dt_inv_sq * CORD_sq * (para0 * t * t + para * one_vec)
                    # Project to SPD (clamp negatives)
                    diag_spd = ti.max(diag_tmp, t.dtype(0.0) * one_vec)
                    diagH[ID] += diag_spd

    @ti.kernel
    def compute_pHp(
        self,
        vertices: ti.template(),
        p: ti.template(),
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ) -> ti.template():
        """
        Compute p^T * H_contact * p for conjugate gradient.

        This is the contact Hessian quadratic form, needed for
        preconditioned conjugate gradient.

        Args:
            vertices: Vertex positions field
            p: Search direction field (n_vertices x 3)
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold

        Returns:
            Scalar p^T * H * p
        """
        dt_inv_sq = 1.0 / (dt * dt)
        pHp = kappa * 0.0  # Preserve type
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist > 1e-10:
                dist2 = dist * dist
                bg = self._barrier_gradient(dist, dHat, kappa)
                bH = self._barrier_hessian(dist, dHat, kappa)

                para = bg / dist
                para0 = (bH - para) / dist2

                # Compute p^T * (dtdx * dtdx^T) * p = (dtdx^T * p)^2
                # where dtdx[3i:3i+3] = cord[i] * t
                dtdx_dot_p = t.dtype(0.0)
                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    dtdx_dot_p += cord[i] * t.dot(p[ID])

                # Full Hessian contribution:
                # H_contact = para0 * dtdx * dtdx^T + para * diag(cord^2 * I)
                # p^T * H * p = para0 * (dtdx^T * p)^2 + para * sum(cord[i]^2 * ||p[i]||^2)
                pHp_contact = para0 * dtdx_dot_p * dtdx_dot_p

                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    pHp_contact += para * cord[i] * cord[i] * p[ID].norm_sqr()

                pHp += dt_inv_sq * pHp_contact

        return pHp

    @ti.kernel
    def apply_hessian(
        self,
        vertices: ti.template(),
        v: ti.template(),
        result: ti.template(),
        dt: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Apply contact Hessian: result += H_contact * v

        Args:
            vertices: Vertex positions field
            v: Input vector field (n_vertices x 3)
            result: Output field to add to (n_vertices x 3)
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        dt_inv_sq = 1.0 / (dt * dt)
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist > 1e-10:
                dist2 = dist * dist
                bg = self._barrier_gradient(dist, dHat, kappa)
                bH = self._barrier_hessian(dist, dHat, kappa)

                para = bg / dist
                para0 = (bH - para) / dist2

                # Compute dtdx^T * v
                dtdx_dot_v = t.dtype(0.0)
                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    dtdx_dot_v += cord[i] * t.dot(v[ID])

                # Apply H * v = para0 * dtdx * (dtdx^T * v) + para * diag(cord^2) * v
                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    Hv_i = para0 * cord[i] * t * dtdx_dot_v
                    Hv_i += para * cord[i] * cord[i] * v[ID]
                    result[ID] += dt_inv_sq * Hv_i


def compute_adaptive_kappa(
    avg_mass: float,
    gap: float,
    hessian_diag: float = 0.0,
) -> float:
    """
    Compute adaptive barrier stiffness.

    kappa_bar = m / g^2 + H_n

    Where:
        m = average vertex mass
        g = gap distance
        H_n = Hessian in contact normal direction (optional)

    This makes the barrier infinitely stiff as gap -> 0.

    Args:
        avg_mass: Average vertex mass
        gap: Current gap distance
        hessian_diag: Hessian contribution in normal direction

    Returns:
        Adaptive kappa value
    """
    if gap < 1e-10:
        return 1e12  # Very large but finite

    kappa_bar = avg_mass / (gap * gap)
    kappa_bar += abs(hessian_diag)

    return kappa_bar
