"""
GCP Contact Handler implementing the ContactHandler protocol.

Provides contact force computation using GCP barrier functions with
automatic adjacency filtering via directional factors.
"""

import taichi as ti
from typing import Optional

from ...core.precision import PrecisionType, PrecisionMixin
from .config import GCPConfig
from .gcp_barrier import GCPBarrier
from .directional_factor import compute_gamma_PT, compute_gamma_EE


@ti.data_oriented
class GCPContactHandler(PrecisionMixin):
    """
    GCP Contact Handler for computing contact forces and Hessians.

    Implements the ContactHandler protocol with GCP-specific features:
    - Automatic adjacent element filtering via directional factor
    - C2 smooth mollification for robust optimization
    - Optional per-primitive adaptive epsilon

    Usage:
        config = GCPConfig(epsilon_target=0.1, kappa=1e4)
        handler = GCPContactHandler(max_contacts=2**18, config=config)
        handler.set_contacts(contact_storage, n_contacts)
        handler.compute_gamma(vertices)  # Precompute directional factors
        handler.add_gradient(vertices, grad, dt)
    """

    def __init__(
        self,
        max_contacts: int = 2**18,
        config: Optional[GCPConfig] = None,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize GCP contact handler.

        Args:
            max_contacts: Maximum number of contact pairs
            config: GCP configuration (uses default if None)
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_CONTACTS = max_contacts
        self.config = config if config is not None else GCPConfig()

        float_type = self.cfg.float_type

        # Contact pair storage reference
        self._contact_pairs = None
        self._n_contacts = ti.field(dtype=ti.i32, shape=())
        self._n_contacts[None] = 0

        # Per-contact directional factor (computed each frame)
        self._gamma = ti.field(dtype=float_type, shape=max_contacts)

        # Per-contact epsilon (if adaptive)
        self._epsilon = ti.field(dtype=float_type, shape=max_contacts)

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

    @ti.kernel
    def compute_gamma_kernel(
        self,
        vertices: ti.template(),
        alpha: ti.f32,
        epsilon_target: ti.f32,
    ):
        """
        Compute directional factors for all contacts.

        Should be called after contact detection and before gradient/hessian
        computation.

        Args:
            vertices: Vertex positions field
            alpha: Smooth step parameter
            epsilon_target: Default epsilon value
        """
        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c

            # Get vertex positions
            v0 = vertices[ti.cast(ids[0], ti.i32)]
            v1 = vertices[ti.cast(ids[1], ti.i32)]
            v2 = vertices[ti.cast(ids[2], ti.i32)]
            v3 = vertices[ti.cast(ids[3], ti.i32)]

            # Determine if PT or EE based on barycentric pattern
            is_PT = ti.abs(cord[0] - 1.0) < 0.01

            gamma = v0.dtype(0.0)
            if is_PT:
                # PT: v0 = point, v1/v2/v3 = triangle
                cord0 = -cord[1]
                cord1 = -cord[2]
                cord2 = -cord[3]
                gamma = compute_gamma_PT(v0, v1, v2, v3, cord0, cord1, cord2, alpha)
            else:
                # EE: v0/v1 = edge A, v2/v3 = edge B
                sc = -cord[1]
                tc = cord[3]
                gamma = compute_gamma_EE(v0, v1, v2, v3, sc, tc, alpha)

            self._gamma[idx] = gamma
            self._epsilon[idx] = epsilon_target

    def compute_gamma(self, vertices):
        """
        Compute directional factors for all contacts.

        Wrapper for compute_gamma_kernel with config parameters.

        Args:
            vertices: Vertex positions field
        """
        self.compute_gamma_kernel(
            vertices,
            float(self.config.alpha),
            float(self.config.epsilon_target),
        )

    @ti.kernel
    def compute_energy(
        self,
        vertices: ti.template(),
        dt: ti.template(),
    ) -> ti.template():
        """
        Compute total GCP contact barrier energy.

        Args:
            vertices: Vertex positions field
            dt: Time step

        Returns:
            Total contact energy
        """
        kappa = self.cfg.float_type(self.config.kappa)
        dt_inv_sq = 1.0 / (dt * dt)
        total_E = dt * 0.0  # Preserve type

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            dist = pair.b
            gamma = self._gamma[idx]
            epsilon = self._epsilon[idx]

            E_contact = GCPBarrier.energy(dist, epsilon, gamma, kappa)
            total_E += dt_inv_sq * E_contact

        return total_E

    @ti.kernel
    def add_gradient(
        self,
        vertices: ti.template(),
        grad: ti.template(),
        dt: ti.template(),
    ):
        """
        Add GCP contact gradient to vertex gradients.

        Args:
            vertices: Vertex positions field
            grad: Gradient field to add to (n_vertices x 3)
            dt: Time step
        """
        kappa = self.cfg.float_type(self.config.kappa)
        dt_inv_sq = 1.0 / (dt * dt)

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = self._gamma[idx]
            epsilon = self._epsilon[idx]

            if dist > 1e-10 and gamma > 1e-8:
                bg = GCPBarrier.gradient(dist, epsilon, gamma, kappa)
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
    ):
        """
        Add GCP contact diagonal Hessian approximation.

        Args:
            vertices: Vertex positions field
            diagH: Diagonal Hessian field to add to (n_vertices x 3)
            dt: Time step
        """
        kappa = self.cfg.float_type(self.config.kappa)
        dt_inv_sq = 1.0 / (dt * dt)

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = self._gamma[idx]
            epsilon = self._epsilon[idx]

            if dist > 1e-10 and gamma > 1e-8:
                dist2 = dist * dist
                bg = GCPBarrier.gradient(dist, epsilon, gamma, kappa)
                bH = GCPBarrier.hessian(dist, epsilon, gamma, kappa)

                para = bg / dist
                para0 = (bH - para) / dist2

                for i in ti.static(range(4)):
                    CORD = cord[i]
                    CORD_sq = CORD * CORD
                    ID = ti.cast(ids[i], ti.i32)
                    one_vec = ti.Vector([1.0, 1.0, 1.0], dt=t.dtype)
                    diag_tmp = dt_inv_sq * CORD_sq * (para0 * t * t + para * one_vec)
                    diag_spd = ti.max(diag_tmp, t.dtype(0.0) * one_vec)
                    diagH[ID] += diag_spd

    @ti.kernel
    def compute_pHp(
        self,
        vertices: ti.template(),
        p: ti.template(),
        dt: ti.template(),
    ) -> ti.template():
        """
        Compute p^T * H_contact * p for conjugate gradient.

        Args:
            vertices: Vertex positions field
            p: Search direction field (n_vertices x 3)
            dt: Time step

        Returns:
            Scalar p^T * H * p
        """
        kappa = self.cfg.float_type(self.config.kappa)
        dt_inv_sq = 1.0 / (dt * dt)
        pHp = dt * 0.0

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = self._gamma[idx]
            epsilon = self._epsilon[idx]

            if dist > 1e-10 and gamma > 1e-8:
                dist2 = dist * dist
                bg = GCPBarrier.gradient(dist, epsilon, gamma, kappa)
                bH = GCPBarrier.hessian(dist, epsilon, gamma, kappa)

                para = bg / dist
                para0 = (bH - para) / dist2

                # p^T * (dtdx * dtdx^T) * p = (dtdx^T * p)^2
                dtdx_dot_p = t.dtype(0.0)
                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    dtdx_dot_p += cord[i] * t.dot(p[ID])

                pHp_contact = para0 * dtdx_dot_p * dtdx_dot_p

                for i in ti.static(range(4)):
                    ID = ti.cast(ids[i], ti.i32)
                    pHp_contact += para * cord[i] * cord[i] * p[ID].norm_sqr()

                pHp += dt_inv_sq * pHp_contact

        return pHp

    @ti.kernel
    def count_active_contacts(self, gamma_threshold: ti.f32) -> ti.i32:
        """
        Count contacts with gamma above threshold.

        This gives the number of "true" contacts after filtering
        adjacent elements.

        Args:
            gamma_threshold: Minimum gamma to count as active

        Returns:
            Number of active contacts
        """
        count = 0
        for idx in range(self._n_contacts[None]):
            if self._gamma[idx] > gamma_threshold:
                ti.atomic_add(count, 1)
        return count

    def get_active_contact_ratio(self, gamma_threshold: float = 0.1) -> float:
        """
        Get ratio of active contacts to total contacts.

        Useful for debugging and monitoring adjacency filtering.

        Args:
            gamma_threshold: Minimum gamma for active contact

        Returns:
            Ratio in [0, 1]
        """
        n = self._n_contacts[None]
        if n == 0:
            return 0.0
        active = self.count_active_contacts(gamma_threshold)
        return float(active) / float(n)
