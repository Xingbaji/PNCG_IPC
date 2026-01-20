"""
Directional factor (Gamma) computation for GCP.

The directional factor is key to GCP's ability to automatically filter
adjacent elements without explicit adjacency checks.
"""

import taichi as ti
from .smooth_step import smooth_step_cubic


@ti.func
def compute_triangle_normal(x0, x1, x2):
    """
    Compute normalized outward normal of a triangle.

    Args:
        x0, x1, x2: Triangle vertices (3D vectors)

    Returns:
        Normalized normal vector (3D)
    """
    e1 = x1 - x0
    e2 = x2 - x0
    n = e1.cross(e2)
    n_len = n.norm()
    if n_len > 1e-10:
        n = n / n_len
    else:
        # Degenerate triangle, use default normal
        n = ti.Vector([0.0, 1.0, 0.0], dt=x0.dtype)
    return n


@ti.func
def compute_gamma_PT(xp, x0, x1, x2, cord0, cord1, cord2, alpha):
    """
    Compute directional factor for Point-Triangle contact.

    The directional factor gamma encodes two constraints:
    1. phi_m (local minimum): Measures tangential deviation
       - gamma_m = h(phi_m, 0, alpha*dist) where phi_m = ||d - (d·n)n||
       - For adjacent elements: phi_m large → gamma_m ≈ 0
       - For true contacts: phi_m small → gamma_m ≈ 1

    2. phi_e (exterior direction): Measures if point is on correct side
       - gamma_e = 1 - h(-phi_e, -alpha*dist, 0) where phi_e = d·n
       - For same-side points: phi_e < 0 → gamma_e ≈ 0
       - For opposite-side points: phi_e > 0 → gamma_e ≈ 1

    Final gamma = gamma_m * gamma_e

    Args:
        xp: Point position (3D)
        x0, x1, x2: Triangle vertices (3D)
        cord0, cord1, cord2: Barycentric coordinates of closest point
        alpha: Smooth step transition parameter

    Returns:
        Directional factor in [0, 1]
    """
    # Closest point on triangle
    xt = cord0 * x0 + cord1 * x1 + cord2 * x2
    d_vec = xp - xt
    dist = d_vec.norm()
    gamma = d_vec.dtype(0.0)

    if dist > 1e-10:
        # Triangle normal
        n = compute_triangle_normal(x0, x1, x2)

        # Decompose contact direction into normal and tangential components
        d_normal = d_vec.dot(n)  # Signed normal distance
        d_tangent = d_vec - d_normal * n  # Tangential component
        phi_m = d_tangent.norm()  # Tangential deviation magnitude
        phi_e = d_normal  # Normal component (positive = exterior)

        # Scale alpha by current distance
        alpha_scaled = alpha * dist

        # Local minimum constraint: small tangential deviation
        gamma_m = smooth_step_cubic(phi_m, d_vec.dtype(0.0), alpha_scaled)

        # Exterior constraint: point should be on exterior side of triangle
        # h(-phi_e, -alpha_scaled, 0) goes from 1 to 0 as phi_e goes from 0 to alpha_scaled
        gamma_e = 1.0 - smooth_step_cubic(-phi_e, -alpha_scaled, d_vec.dtype(0.0))

        # Combined directional factor
        gamma = gamma_m * gamma_e

    return gamma


@ti.func
def compute_gamma_EE(ea0, ea1, eb0, eb1, sc, tc, alpha):
    """
    Compute directional factor for Edge-Edge contact.

    Similar to PT but uses cross product of edge tangents for normal.
    Special handling for near-parallel edges where cross product is small.

    Args:
        ea0, ea1: First edge endpoints (3D)
        eb0, eb1: Second edge endpoints (3D)
        sc: Parameter on first edge (closest point = (1-sc)*ea0 + sc*ea1)
        tc: Parameter on second edge (closest point = (1-tc)*eb0 + tc*eb1)
        alpha: Smooth step transition parameter

    Returns:
        Directional factor in [0, 1]
    """
    # Closest points on each edge
    pa = (1.0 - sc) * ea0 + sc * ea1
    pb = (1.0 - tc) * eb0 + tc * eb1
    d_vec = pa - pb
    dist = d_vec.norm()
    gamma = d_vec.dtype(0.0)

    if dist > 1e-10:
        # Edge tangent vectors
        ta = ea1 - ea0
        tb = eb1 - eb0
        ta_len = ta.norm()
        tb_len = tb.norm()

        if ta_len > 1e-10 and tb_len > 1e-10:
            ta = ta / ta_len
            tb = tb / tb_len

            # Normal from cross product of edge tangents
            n = ta.cross(tb)
            n_len = n.norm()

            if n_len > 1e-6:
                n = n / n_len

                # Orient normal to point from edge B to edge A
                if n.dot(d_vec) < 0:
                    n = -n

                # Decompose into normal and tangential
                d_normal = d_vec.dot(n)
                d_tangent = d_vec - d_normal * n
                phi_m = d_tangent.norm()
                phi_e = d_normal

                alpha_scaled = alpha * dist
                gamma_m = smooth_step_cubic(phi_m, d_vec.dtype(0.0), alpha_scaled)
                gamma_e = 1.0 - smooth_step_cubic(-phi_e, -alpha_scaled, d_vec.dtype(0.0))
                gamma = gamma_m * gamma_e
            else:
                # Near-parallel edges: use conservative value
                # These contacts are geometrically ambiguous
                gamma = d_vec.dtype(0.5)

    return gamma


@ti.func
def compute_gamma_from_contact(
    ids,
    x,
    cord,
    t,
    alpha,
    is_PT: ti.template(),
):
    """
    Compute directional factor from contact pair data.

    This is a convenience function that extracts vertex positions
    and calls the appropriate gamma computation.

    Args:
        ids: Vertex indices (4 element vector)
        x: Vertex positions field
        cord: Barycentric coordinates (4 element vector)
        t: Contact direction vector
        alpha: Smooth step parameter
        is_PT: True for PT contact, False for EE contact

    Returns:
        Directional factor gamma
    """
    gamma = t.dtype(0.0)

    v0 = x[ti.cast(ids[0], ti.i32)]
    v1 = x[ti.cast(ids[1], ti.i32)]
    v2 = x[ti.cast(ids[2], ti.i32)]
    v3 = x[ti.cast(ids[3], ti.i32)]

    if ti.static(is_PT):
        # PT: v0 = point, v1/v2/v3 = triangle
        # cord = [1, -c0, -c1, -c2]
        cord0 = -cord[1]
        cord1 = -cord[2]
        cord2 = -cord[3]
        gamma = compute_gamma_PT(v0, v1, v2, v3, cord0, cord1, cord2, alpha)
    else:
        # EE: v0/v1 = edge A, v2/v3 = edge B
        # cord = [s-1, -s, 1-t, t]
        sc = -cord[1]
        tc = cord[3]
        gamma = compute_gamma_EE(v0, v1, v2, v3, sc, tc, alpha)

    return gamma
