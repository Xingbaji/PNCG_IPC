"""
CCD-based Step Size Computer implementing StepSizeComputer protocol.

Provides continuous collision detection (CCD) for safe line search step sizes.
"""

import taichi as ti
from typing import Optional

from ..core.precision import PrecisionType, PrecisionMixin
from ..collision.queries.ccd import (
    point_triangle_ccd_lower_bound,
    edge_edge_ccd_lower_bound,
)


@ti.data_oriented
class CCDStepSizeComputer(PrecisionMixin):
    """
    CCD-based Step Size Computer for collision-free line search.

    Computes the maximum step size that guarantees no collisions occur
    during the line search. Uses continuous collision detection (CCD)
    to find the first time of impact.

    Implements the StepSizeComputer protocol.

    Usage:
        computer = CCDStepSizeComputer(max_contacts=2**18)
        alpha_safe = computer.compute_safe_step(
            vertices, velocities, contact_pairs, n_contacts, dHat
        )
    """

    def __init__(
        self,
        max_contacts: int = 2**18,
        safety_factor: float = 0.8,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize CCD step size computer.

        Args:
            max_contacts: Maximum number of contact pairs
            safety_factor: Factor to scale down the CCD result (0 < factor <= 1)
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_CONTACTS = max_contacts
        self.safety_factor = safety_factor

        float_type = self.cfg.float_type

        # Contact pair storage reference
        self._contact_pairs = None
        self._n_contacts = ti.field(dtype=ti.i32, shape=())
        self._n_contacts[None] = 0

        # Ground plane parameters
        self._ground_y = ti.field(dtype=float_type, shape=())
        self._ground_y[None] = -1e10  # Default: no ground

        # Minimum step size result
        self._min_toi = ti.field(dtype=float_type, shape=())

    def set_contacts(self, contact_storage, n_contacts: int = None):
        """
        Set the contact pairs for CCD computation.

        Args:
            contact_storage: ContactPairStorage instance
            n_contacts: Number of contacts (if None, uses storage.count)
        """
        self._contact_pairs = contact_storage.contact_pairs
        if n_contacts is not None:
            self._n_contacts[None] = n_contacts
        else:
            self._n_contacts[None] = contact_storage.count

    def set_ground(self, ground_y: float):
        """
        Set ground plane Y coordinate for ground CCD.

        Args:
            ground_y: Y-coordinate of ground plane
        """
        self._ground_y[None] = ground_y

    @ti.kernel
    def _compute_pt_ee_ccd(
        self,
        x: ti.template(),
        p: ti.template(),
        boundary_triangles: ti.template(),
        boundary_edges: ti.template(),
        n_triangles: ti.i32,
        n_edges: ti.i32,
        dHat: ti.template(),
    ):
        """
        Compute minimum TOI across all PT and EE pairs using stored contacts.

        Args:
            x: Current vertex positions
            p: Search direction (velocity)
            boundary_triangles: Triangle indices (n_triangles x 3)
            boundary_edges: Edge indices (n_edges x 2)
            n_triangles: Number of boundary triangles
            n_edges: Number of boundary edges
            dHat: Barrier threshold
        """
        self._min_toi[None] = 1.0

        for idx in range(self._n_contacts[None]):
            pair = self._contact_pairs[idx]
            ids = pair.a

            # Determine if PT or EE based on vertex pattern
            # PT: ids = [p, t0, t1, t2] where p != t0, t1, t2
            # EE: ids = [a0, a1, b0, b1] where edges are separate

            # Get vertex positions and velocities
            v0 = x[ti.cast(ids[0], ti.i32)]
            v1 = x[ti.cast(ids[1], ti.i32)]
            v2 = x[ti.cast(ids[2], ti.i32)]
            v3 = x[ti.cast(ids[3], ti.i32)]

            p0 = p[ti.cast(ids[0], ti.i32)]
            p1 = p[ti.cast(ids[1], ti.i32)]
            p2 = p[ti.cast(ids[2], ti.i32)]
            p3 = p[ti.cast(ids[3], ti.i32)]

            # Check if this is a PT or EE contact based on index pattern
            # PT: first vertex is the point, others form triangle
            # EE: first two form edge A, last two form edge B
            dist = pair.b
            cord = pair.c

            # Use the barycentric coordinate pattern to determine type
            # PT: cord = [1, -c0, -c1, -c2] where c0+c1+c2=1
            # EE: cord = [s-1, -s, 1-t, t]

            # Simple heuristic: check if first coordinate is exactly 1.0
            is_PT = ti.abs(cord[0] - 1.0) < 0.01

            toi = dHat * 0.0 + 1.0  # Initialize to 1.0

            if is_PT:
                # Point-Triangle CCD
                # v0 = point, v1/v2/v3 = triangle
                toi = point_triangle_ccd_lower_bound(
                    v0, v1, v2, v3,  # Current positions
                    v0 + p0, v1 + p1, v2 + p2, v3 + p3,  # End positions
                    dHat,
                )
            else:
                # Edge-Edge CCD
                # v0/v1 = edge A, v2/v3 = edge B
                toi = edge_edge_ccd_lower_bound(
                    v0, v1, v2, v3,  # Current positions
                    v0 + p0, v1 + p1, v2 + p2, v3 + p3,  # End positions
                    dHat,
                )

            ti.atomic_min(self._min_toi[None], toi)

    @ti.kernel
    def _compute_ground_ccd(
        self,
        x: ti.template(),
        p: ti.template(),
        n_vertices: ti.i32,
        dHat: ti.template(),
    ):
        """
        Compute minimum TOI for ground collision.

        A vertex moving from y to y + p_y will hit ground at:
        toi = (ground_y + dHat - y) / p_y  (if p_y < 0)

        Args:
            x: Current vertex positions
            p: Search direction (velocity)
            n_vertices: Number of vertices
            dHat: Barrier threshold
        """
        ground_y = self._ground_y[None]

        for i in range(n_vertices):
            y = x[i][1]
            py = p[i][1]

            # Only check if moving towards ground
            if py < -1e-10:
                # Time to reach (ground_y + dHat)
                target_y = ground_y + dHat
                if y > target_y:  # Currently above target
                    toi = (target_y - y) / py
                    if toi > 0.0 and toi < 1.0:
                        ti.atomic_min(self._min_toi[None], toi)

    def compute_safe_step(
        self,
        vertices,
        search_dir,
        boundary_triangles=None,
        boundary_edges=None,
        n_triangles: int = 0,
        n_edges: int = 0,
        n_vertices: int = 0,
        dHat: float = 0.01,
        check_ground: bool = True,
    ) -> float:
        """
        Compute the maximum safe step size.

        Uses CCD to find the first time of impact (TOI) and returns
        a conservative step size.

        Args:
            vertices: Current vertex positions field
            search_dir: Search direction field
            boundary_triangles: Triangle indices (optional)
            boundary_edges: Edge indices (optional)
            n_triangles: Number of boundary triangles
            n_edges: Number of boundary edges
            n_vertices: Number of vertices
            dHat: Barrier threshold
            check_ground: Whether to check ground collision

        Returns:
            Safe step size in (0, 1]
        """
        self._min_toi[None] = 1.0

        # Contact CCD (PT and EE)
        if self._contact_pairs is not None and self._n_contacts[None] > 0:
            self._compute_pt_ee_ccd(
                vertices, search_dir,
                boundary_triangles, boundary_edges,
                n_triangles, n_edges,
                dHat,
            )

        # Ground CCD
        if check_ground and n_vertices > 0:
            self._compute_ground_ccd(vertices, search_dir, n_vertices, dHat)

        # Apply safety factor
        toi = self._min_toi[None]
        return float(toi * self.safety_factor)

    @ti.kernel
    def _compute_safe_step_simple_kernel(
        self,
        x: ti.template(),
        p: ti.template(),
        n_vertices: ti.i32,
        dHat: ti.f32,
    ):
        """
        Compute safe step size with ground CCD only (kernel).

        Stores result in self._min_toi.
        """
        ground_y = self._ground_y[None]
        self._min_toi[None] = 1.0

        for i in range(n_vertices):
            y = x[i][1]
            py = p[i][1]

            # Only check if moving towards ground
            if py < -1e-10:
                target_y = ground_y + dHat
                if y > target_y:
                    toi = (target_y - y) / py
                    if toi > 0.0 and toi < 1.0:
                        ti.atomic_min(self._min_toi[None], toi)

    def compute_safe_step_simple(
        self,
        x,
        p,
        n_vertices: int,
        dHat: float,
    ) -> float:
        """
        Compute safe step size with ground CCD only.

        This is a simplified version that only checks ground collision,
        useful when contact pairs are not available.

        Args:
            x: Current vertex positions
            p: Search direction
            n_vertices: Number of vertices
            dHat: Barrier threshold

        Returns:
            Safe step size
        """
        self._compute_safe_step_simple_kernel(x, p, n_vertices, dHat)
        return float(self._min_toi[None] * self.safety_factor)


@ti.func
def compute_vertex_ground_toi(
    y: ti.template(),
    vy: ti.template(),
    ground_y: ti.template(),
    dHat: ti.template(),
) -> ti.template():
    """
    Compute time of impact for a single vertex with ground.

    Args:
        y: Current Y position
        vy: Y velocity (negative = moving down)
        ground_y: Ground Y coordinate
        dHat: Barrier threshold

    Returns:
        Time of impact (0-1), or 1.0 if no collision
    """
    toi = y * 0.0 + 1.0  # Default: no collision

    if vy < -1e-10:
        target_y = ground_y + dHat
        if y > target_y:
            t = (target_y - y) / vy
            if t > 0.0 and t < 1.0:
                toi = t

    return toi
