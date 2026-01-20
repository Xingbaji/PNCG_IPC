"""
Per-Subdomain CCD (Algorithm 2 from MAS-PNCG paper).

Key insight from paper Section 3.5:
"By independently evaluating distance functions and applying safe steps
{α_d} locally for each subdomain d, regions with fewer constraints can
maintain their optimal descent speed while only the critical subdomains
are conservatively damped."

Position update formula:
x_{k+1} = x_k + Σ_d S_d^T α_d S_d p_{k+1}
"""

import taichi as ti
from typing import Optional
from ..core.precision import PrecisionType, PrecisionMixin
from ..collision.queries.ccd import (
    point_triangle_ccd_lower_bound,
    edge_edge_ccd_lower_bound,
)


# Default banksize for subdomain partitioning
DEFAULT_BANKSIZE = 16


@ti.data_oriented
class SubdomainCCD(PrecisionMixin):
    """
    Per-Subdomain Continuous Collision Detection.

    Implements Algorithm 2 from the MAS-PNCG paper:
    - Each subdomain (block of BANKSIZE vertices) has its own step size α_d
    - Contacts constrain only the involved subdomains
    - Non-constrained subdomains maintain full step size
    - Prevents "numerical locking" where a single contact slows entire mesh

    Usage:
        subdomain_ccd = SubdomainCCD(n_verts=1000, banksize=16)
        subdomain_ccd.set_contacts(contact_pairs, n_contacts)
        subdomain_ccd.compute_subdomain_alphas(x, p, dHat)
        subdomain_ccd.update_x_subdomain(x, p)  # Uses per-subdomain alphas
    """

    def __init__(
        self,
        n_verts: int,
        banksize: int = DEFAULT_BANKSIZE,
        max_contacts: int = 2**18,
        safety_factor: float = 0.9,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize per-subdomain CCD.

        Args:
            n_verts: Number of vertices in mesh
            banksize: Number of vertices per subdomain (typically 16)
            max_contacts: Maximum number of contact pairs
            safety_factor: Safety factor for CCD results (0 < factor < 1)
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.n_verts = n_verts
        self.banksize = banksize
        self.n_subdomains = (n_verts + banksize - 1) // banksize
        self.max_contacts = max_contacts
        self.safety_factor = safety_factor

        float_type = self.cfg.float_type

        # Per-subdomain step sizes
        self.subdomain_alpha = ti.field(dtype=float_type, shape=self.n_subdomains)

        # Contact pairs reference
        self._contact_pairs = None
        self._n_contacts = ti.field(dtype=ti.i32, shape=())
        self._n_contacts[None] = 0

        # Ground plane
        self._ground_y = ti.field(dtype=float_type, shape=())
        self._ground_y[None] = -1e10  # Default: no ground
        self._dHat = ti.field(dtype=float_type, shape=())
        self._dHat[None] = 0.01
        self._ground_enabled = ti.field(dtype=ti.i32, shape=())
        self._ground_enabled[None] = 0

        # Global minimum alpha (for monitoring)
        self._min_alpha = ti.field(dtype=float_type, shape=())

        print(f'[SubdomainCCD] Initialized: {n_verts} verts, {self.n_subdomains} subdomains, '
              f'banksize={banksize}')

    def set_contacts(self, contact_storage, n_contacts: int = None):
        """
        Set contact pairs for CCD computation.

        Args:
            contact_storage: ContactPairStorage instance
            n_contacts: Number of contacts (if None, uses storage.count)
        """
        self._contact_pairs = contact_storage.contact_pairs
        if n_contacts is not None:
            self._n_contacts[None] = n_contacts
        else:
            self._n_contacts[None] = contact_storage.count

    def set_ground(self, ground_y: float, enabled: bool = True):
        """
        Set ground plane parameters.

        Args:
            ground_y: Y-coordinate of ground plane
            enabled: Whether ground collision is enabled
        """
        self._ground_y[None] = ground_y
        self._ground_enabled[None] = 1 if enabled else 0

    def set_dHat(self, dHat: float):
        """Set barrier threshold for CCD margin."""
        self._dHat[None] = dHat

    @ti.kernel
    def _init_subdomain_alpha(self):
        """Initialize all subdomain alphas to 1.0."""
        for d in range(self.n_subdomains):
            self.subdomain_alpha[d] = 1.0
        self._min_alpha[None] = 1.0

    @ti.kernel
    def _compute_subdomain_ccd_contacts(
        self,
        x: ti.template(),
        p: ti.template(),
        banksize: ti.i32,
        safety_factor: ti.f32,
    ):
        """
        Compute per-subdomain CCD for all contacts (PT and EE).

        For each contact:
        1. Compute CCD lower bound using cubic polynomial method
        2. Update subdomain_alpha for all involved subdomains

        Args:
            x: Current vertex positions
            p: Search direction
            banksize: Vertices per subdomain
            safety_factor: Safety factor to apply
        """
        dHat = self._dHat[None]

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a
            cord = pair.c

            # Get vertex indices
            v0_id = ti.cast(ids[0], ti.i32)
            v1_id = ti.cast(ids[1], ti.i32)
            v2_id = ti.cast(ids[2], ti.i32)
            v3_id = ti.cast(ids[3], ti.i32)

            # Current positions
            x0 = x[v0_id]
            x1 = x[v1_id]
            x2 = x[v2_id]
            x3 = x[v3_id]

            # Search direction (displacements)
            p0 = p[v0_id]
            p1 = p[v1_id]
            p2 = p[v2_id]
            p3 = p[v3_id]

            # Check if this is a PT or EE contact
            # PT: cord = [1, -c0, -c1, -c2] where first coord is ~1.0
            is_pt = ti.abs(cord[0] - 1.0) < 0.01

            # Compute CCD lower bound
            alpha_l = 1.0
            if is_pt:
                # Point-Triangle CCD
                # p0 = point, x1/x2/x3 = triangle
                alpha_l = point_triangle_ccd_lower_bound(
                    x0, x1, x2, x3,  # Current positions
                    p0, p1, p2, p3,  # Displacements
                )
            else:
                # Edge-Edge CCD
                # x0/x1 = edge A, x2/x3 = edge B
                alpha_l = edge_edge_ccd_lower_bound(
                    x0, x1, x2, x3,  # Current positions
                    p0, p1, p2, p3,  # Displacements
                )

            # Apply safety factor and clamp
            alpha_safe = safety_factor * alpha_l
            alpha_safe = ti.max(alpha_safe, 1e-6)

            # Update alpha for all subdomains involved in this contact
            subdomain_0 = v0_id // banksize
            subdomain_1 = v1_id // banksize
            subdomain_2 = v2_id // banksize
            subdomain_3 = v3_id // banksize

            ti.atomic_min(self.subdomain_alpha[subdomain_0], alpha_safe)
            ti.atomic_min(self.subdomain_alpha[subdomain_1], alpha_safe)
            ti.atomic_min(self.subdomain_alpha[subdomain_2], alpha_safe)
            ti.atomic_min(self.subdomain_alpha[subdomain_3], alpha_safe)

            ti.atomic_min(self._min_alpha[None], alpha_safe)

    @ti.kernel
    def _compute_subdomain_ccd_ground(
        self,
        x: ti.template(),
        p: ti.template(),
        n_verts: ti.i32,
        banksize: ti.i32,
        safety_factor: ti.f32,
    ):
        """
        Compute per-subdomain CCD for ground plane collision.

        For vertices moving toward the ground, compute safe step size
        and update corresponding subdomain's alpha.

        Args:
            x: Current vertex positions
            p: Search direction
            n_verts: Number of vertices
            banksize: Vertices per subdomain
            safety_factor: Safety factor to apply
        """
        ground_y = self._ground_y[None]
        dHat = self._dHat[None]
        safety_margin = 0.1 * dHat

        for vid in range(n_verts):
            subdomain_id = vid // banksize

            # Current distance to ground
            d = x[vid][1] - ground_y

            # Velocity toward ground (negative p[1] means moving down)
            v_y = p[vid][1]

            # Only check if moving toward ground and currently above safety margin
            if v_y < -1e-10 and d > safety_margin:
                # Time to reach safety margin: d + alpha * v_y = safety_margin
                # alpha = (d - safety_margin) / (-v_y)
                toc = (d - safety_margin) / (-v_y)
                alpha_safe = safety_factor * toc
                alpha_safe = ti.max(alpha_safe, 1e-6)

                ti.atomic_min(self.subdomain_alpha[subdomain_id], alpha_safe)
                ti.atomic_min(self._min_alpha[None], alpha_safe)

    def compute_subdomain_alphas(self, x, p):
        """
        Compute per-subdomain CCD step sizes.

        Implements Algorithm 2 from the paper:
        1. Initialize all subdomain alphas to 1.0
        2. For each contact, compute CCD and update involved subdomains
        3. For ground collision, update subdomain alphas

        After this call, self.subdomain_alpha[d] contains the safe step size
        for subdomain d.

        Args:
            x: Current vertex positions field
            p: Search direction field

        Returns:
            Minimum alpha across all subdomains (for monitoring)
        """
        # Step 1: Initialize all alphas to 1.0
        self._init_subdomain_alpha()

        # Step 2: Compute CCD for all contacts
        if self._contact_pairs is not None and self._n_contacts[None] > 0:
            self._compute_subdomain_ccd_contacts(
                x, p, self.banksize, self.safety_factor
            )

        # Step 3: Compute CCD for ground
        if self._ground_enabled[None] == 1:
            self._compute_subdomain_ccd_ground(
                x, p, self.n_verts, self.banksize, self.safety_factor
            )

        return float(self._min_alpha[None])

    @ti.kernel
    def _update_x_subdomain_kernel(
        self,
        x: ti.template(),
        p: ti.template(),
        n_verts: ti.i32,
        banksize: ti.i32,
    ):
        """
        Update positions using per-subdomain step sizes.

        x[v] += subdomain_alpha[v // banksize] * p[v]

        Args:
            x: Vertex positions (modified in-place)
            p: Search direction
            n_verts: Number of vertices
            banksize: Vertices per subdomain
        """
        for vid in range(n_verts):
            subdomain_id = vid // banksize
            alpha_d = self.subdomain_alpha[subdomain_id]
            x[vid] += alpha_d * p[vid]

    def update_x_subdomain(self, x, p):
        """
        Update positions using per-subdomain step sizes.

        Implements the position update formula from the paper:
        x_{k+1} = x_k + Σ_d S_d^T α_d S_d p_{k+1}

        Each vertex uses the alpha from its subdomain:
        x[v] += subdomain_alpha[v // banksize] * p[v]

        This allows regions with fewer constraints to maintain optimal descent
        while critical subdomains are conservatively damped.

        Args:
            x: Vertex positions field (modified in-place)
            p: Search direction field
        """
        self._update_x_subdomain_kernel(x, p, self.n_verts, self.banksize)

    @ti.kernel
    def get_min_alpha(self) -> ti.f32:
        """Get minimum alpha across all subdomains."""
        alpha_min = ti.f32(1.0)
        for d in range(self.n_subdomains):
            ti.atomic_min(alpha_min, ti.f32(self.subdomain_alpha[d]))
        return alpha_min

    def get_alpha_statistics(self) -> dict:
        """
        Get statistics about subdomain step sizes.

        Returns:
            dict with min/max/mean alpha values
        """
        import numpy as np
        alphas = self.subdomain_alpha.to_numpy()
        return {
            'min': float(np.min(alphas)),
            'max': float(np.max(alphas)),
            'mean': float(np.mean(alphas)),
            'n_constrained': int(np.sum(alphas < 0.99)),
            'n_subdomains': self.n_subdomains,
        }
