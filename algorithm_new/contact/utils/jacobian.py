"""
Contact direction Jacobian computation utilities.

Provides functions to compute the mapping between contact distance and vertex positions.
"""

import taichi as ti


@ti.func
def compute_dtdx_t(t, cord):
    """
    Construct 12x1 vector: dt/dx^T with barycentric weighting.

    For contact between 4 vertices with barycentric coordinates cord[0..3],
    and contact direction t, this computes how the distance changes with
    vertex positions.

    For each vertex i in 0..3:
        dtdx_t[3i : 3i+3] = cord[i] * t

    Args:
        t: Contact direction vector (3D)
        cord: Barycentric coordinates (4D)

    Returns:
        12x1 vector representing dt/dx^T
    """
    # Use the type from t to determine float type
    dtdx_t = ti.Vector.zero(t.dtype, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            dtdx_t[i * 3 + j] = cord[i] * t[j]
    return dtdx_t


@ti.func
def compute_d_dtdx(d, cord):
    """
    Compute distance gradient: sum(cord[i] * d[3i:3i+3]).

    This computes the weighted sum of position displacements,
    effectively computing the distance gradient with respect to
    a 12D position vector.

    Args:
        d: 12D displacement vector
        cord: Barycentric coordinates (4D)

    Returns:
        3D vector representing the weighted displacement
    """
    # Determine type from d
    result = ti.Vector.zero(d.dtype, 3)
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            result[j] += cord[i] * d[i * 3 + j]
    return result


@ti.func
def compute_contact_gradient_contribution(
    para,
    t,
    cord,
    vertex_idx: ti.i32,
):
    """
    Compute contact force contribution for a single vertex.

    The contact force on vertex i is: f_i = para * cord[i] * t

    Where:
        para = barrier_g / dist (normalized barrier gradient)
        cord[i] = barycentric coordinate for vertex i
        t = contact direction vector

    Args:
        para: Normalized barrier gradient (barrier_g / dist)
        t: Contact direction vector (3D)
        cord: Barycentric coordinates (4D)
        vertex_idx: Which vertex (0-3)

    Returns:
        3D force vector for the specified vertex
    """
    return para * cord[vertex_idx] * t


@ti.func
def compute_contact_diagonal_contribution(
    para0,
    para,
    t,
    cord,
    vertex_idx: ti.i32,
):
    """
    Compute diagonal Hessian contribution for a single vertex.

    The diagonal Hessian contribution for vertex i is:
        diag_i = cord[i]^2 * (para0 * t*t + para * I_3)

    Where:
        para0 = (barrier_H - barrier_g/dist) / dist^2
        para = barrier_g / dist
        t = contact direction vector
        t*t = element-wise t[j]^2

    Args:
        para0: Curvature term
        para: Gradient term
        t: Contact direction vector (3D)
        cord: Barycentric coordinates (4D)
        vertex_idx: Which vertex (0-3)

    Returns:
        3D diagonal contribution vector for the specified vertex
    """
    CORD = cord[vertex_idx]
    CORD_sq = CORD * CORD
    # diag_tmp = CORD^2 * (para0 * t*t + para * ones)
    diag_tmp = CORD_sq * (para0 * t * t + para * ti.Vector([1.0, 1.0, 1.0], dt=t.dtype))
    return diag_tmp
