"""
Intersection test functions for collision detection.

Provides segment-triangle intersection tests for penetration detection.
"""

import taichi as ti


@ti.func
def mat3_determinant(col0: ti.template(), col1: ti.template(), col2: ti.template()):
    """Compute determinant of 3x3 matrix formed by column vectors."""
    return (col0[0] * (col1[1] * col2[2] - col1[2] * col2[1]) -
            col1[0] * (col0[1] * col2[2] - col0[2] * col2[1]) +
            col2[0] * (col0[1] * col1[2] - col0[2] * col1[1]))


@ti.func
def segment_triangle_intersect_cramer(ve0, ve1, vt0, vt1, vt2):
    """
    Exact segment-triangle intersection test using Cramer's rule.
    Based on GIPC.cu segTriIntersect implementation.

    Args:
        ve0, ve1: Segment endpoints
        vt0, vt1, vt2: Triangle vertices

    Returns:
        1 if segment intersects triangle interior, 0 otherwise
    """
    result = 0

    # Triangle edges
    col0 = vt1 - vt0
    col1 = vt2 - vt0
    col2 = ve0 - ve1  # Segment direction (reversed)

    # Triangle normal
    n = col0.cross(col1)

    # Early exit: check if segment endpoints are on the same side of triangle plane
    d0 = n.dot(ve0 - vt0)
    d1 = n.dot(ve1 - vt0)
    if d0 * d1 > 0.0:
        result = 0
    else:
        # Compute determinant of coefficient matrix
        det = mat3_determinant(col0, col1, col2)

        if ti.abs(det) < 1e-20:
            # Degenerate case: segment parallel to triangle or coplanar
            result = 0
        else:
            # Solve for barycentric coordinates and segment parameter using Cramer's rule
            b = ve0 - vt0

            u = mat3_determinant(b, col1, col2) / det
            v = mat3_determinant(col0, b, col2) / det
            t = mat3_determinant(col0, col1, b) / det

            # Check if intersection is inside triangle and within segment
            if u >= 0.0 and v >= 0.0 and u + v <= 1.0 and t >= 0.0 and t <= 1.0:
                result = 1

    return result


@ti.func
def segment_intersect_triangle(P, Q, A, B, C):
    """
    Alternative segment-triangle intersection test.

    Args:
        P, Q: Segment endpoints
        A, B, C: Triangle vertices

    Returns:
        1 if segment intersects triangle, 0 otherwise
    """
    RLen = (Q - P).norm()
    RDir = (Q - P) / RLen
    ROrigin = P
    E1 = B - A
    E2 = C - A
    N = E1.cross(E2)
    det = -RDir.dot(N)
    invdet = 1.0 / det
    AO = ROrigin - A
    DAO = AO.cross(RDir)
    u = E2.dot(DAO) * invdet
    v = -E1.dot(DAO) * invdet
    t = AO.dot(N) * invdet
    ret = 0
    if det >= 1e-5 and t >= 1e-6 and u >= 1e-6 and v >= 1e-6 and (u + v) <= 1.0 - 1e-6 and t <= RLen:
        ret = 1
    return ret
