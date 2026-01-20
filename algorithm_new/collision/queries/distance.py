"""
Distance computation functions for collision detection.

Provides point-triangle, edge-edge, and other geometric distance computations.
"""

import taichi as ti


@ti.func
def dist3D_Segment_to_Segment(A0, A1, B0, B1):
    """
    Compute the closest points and distance between two 3D line segments.

    Args:
        A0, A1: Endpoints of segment A
        B0, B1: Endpoints of segment B

    Returns:
        dP: Direction vector from closest point on A to closest point on B
        sc: Parameter for closest point on A (in [0,1])
        tc: Parameter for closest point on B (in [0,1])
    """
    u = A1 - A0
    v = B1 - B0
    w = A0 - B0
    a = u.norm_sqr()
    b = u.dot(v)
    c = v.norm_sqr()
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < 1e-7:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if -d + b < 0.0:
            sN = 0.0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    if ti.abs(sN) < 1e-7:
        sc = 0.0
    else:
        sc = sN / sD
    if ti.abs(tN) < 1e-7:
        tc = 0.0
    else:
        tc = tN / tD

    dP = -w - (sc * u) + (tc * v)  # Qc - Pc
    return dP, sc, tc


@ti.func
def dist3D_Point_Triangle(P, V0, V1, V2):
    """
    Compute barycentric coordinates of the closest point on triangle to point P.

    Args:
        P: Query point
        V0, V1, V2: Triangle vertices

    Returns:
        cord0, cord1, cord2: Barycentric coordinates (sum to 1.0)
    """
    cord0 = 0.0
    cord1 = 0.0
    cord2 = 0.0
    v = V2 - V0
    u = V1 - V0
    nVec = u.cross(v)
    s_p = (nVec.dot(P - V0)) / (nVec.dot(nVec))
    P0 = P - s_p * nVec  # P project to plane
    w = P0 - V0
    n_cross_v = nVec.cross(v)
    n_cross_u = nVec.cross(u)
    s = w.dot(n_cross_v) / (u.dot(n_cross_v))
    t = w.dot(n_cross_u) / (v.dot(n_cross_u))

    if s >= 0.0 and t >= 0.0:
        if s + t <= 1.0:
            cord0 = 1.0 - s - t
            cord1 = s
            cord2 = t
        else:
            q = V2 - V1
            k = (P - V1).dot(q) / (q.dot(q))
            if k > 1.0:
                cord2 = 1.0
            elif k < 0.0:
                cord1 = 1.0
            else:
                cord1 = 1.0 - k
                cord2 = k
    elif s >= 0.0 and t < 0.0:
        k = w.dot(u) / (u.dot(u))
        if k > 1.0:
            cord1 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord1 = k
    elif s < 0.0 and t >= 0.0:
        k = w.dot(v) / (v.dot(v))
        if k > 1.0:
            cord2 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord2 = k
    else:  # s < 0 and t < 0
        cord0 = 1.0

    return cord0, cord1, cord2


@ti.func
def _dType_point_triangle(v0, v1, v2, v3):
    """
    Determine the distance type for point-triangle pair.

    Args:
        v0: Point
        v1, v2, v3: Triangle vertices

    Returns:
        dtype: Distance type
            0: PP v1, 1: PP v2, 2: PP v3
            3: PE v1v2, 4: PE v2v3, 5: PE v3v1
            6: PT (point-triangle face)
    """
    dtype = 6  # default to PT
    basis0 = v2 - v1
    basis1 = v3 - v1
    basis2 = v0 - v1

    nVec = basis0.cross(basis1)
    basis1_new = basis0.cross(nVec)

    # Compute param[0] using Cramer's rule
    D = ti.Matrix([[basis0[0], basis1_new[0], nVec[0]],
                   [basis0[1], basis1_new[1], nVec[1]],
                   [basis0[2], basis1_new[2], nVec[2]]])
    D1 = ti.Matrix([[basis2[0], basis1_new[0], nVec[0]],
                    [basis2[1], basis1_new[1], nVec[1]],
                    [basis2[2], basis1_new[2], nVec[2]]])
    D2 = ti.Matrix([[basis0[0], basis2[0], nVec[0]],
                    [basis0[1], basis2[1], nVec[1]],
                    [basis0[2], basis2[2], nVec[2]]])

    det_D = D.determinant()
    param0_x = D1.determinant() / det_D
    param0_y = D2.determinant() / det_D

    if param0_x > 0.0 and param0_x < 1.0 and param0_y >= 0.0:
        dtype = 3  # PE v1v2
    else:
        # Check edge v2v3
        basis0 = v3 - v2
        basis1_new = basis0.cross(nVec)
        basis2 = v0 - v2

        D = ti.Matrix([[basis0[0], basis1_new[0], nVec[0]],
                       [basis0[1], basis1_new[1], nVec[1]],
                       [basis0[2], basis1_new[2], nVec[2]]])
        D1 = ti.Matrix([[basis2[0], basis1_new[0], nVec[0]],
                        [basis2[1], basis1_new[1], nVec[1]],
                        [basis2[2], basis1_new[2], nVec[2]]])
        D2 = ti.Matrix([[basis0[0], basis2[0], nVec[0]],
                        [basis0[1], basis2[1], nVec[1]],
                        [basis0[2], basis2[2], nVec[2]]])

        det_D = D.determinant()
        param1_x = D1.determinant() / det_D
        param1_y = D2.determinant() / det_D

        if param1_x > 0.0 and param1_x < 1.0 and param1_y >= 0.0:
            dtype = 4  # PE v2v3
        else:
            # Check edge v3v1
            basis0 = v1 - v3
            basis1_new = basis0.cross(nVec)
            basis2 = v0 - v3

            D = ti.Matrix([[basis0[0], basis1_new[0], nVec[0]],
                           [basis0[1], basis1_new[1], nVec[1]],
                           [basis0[2], basis1_new[2], nVec[2]]])
            D1 = ti.Matrix([[basis2[0], basis1_new[0], nVec[0]],
                            [basis2[1], basis1_new[1], nVec[1]],
                            [basis2[2], basis1_new[2], nVec[2]]])
            D2 = ti.Matrix([[basis0[0], basis2[0], nVec[0]],
                            [basis0[1], basis2[1], nVec[1]],
                            [basis0[2], basis2[2], nVec[2]]])

            det_D = D.determinant()
            param2_x = D1.determinant() / det_D
            param2_y = D2.determinant() / det_D

            if param2_x > 0.0 and param2_x < 1.0 and param2_y >= 0.0:
                dtype = 5  # PE v3v1
            else:
                if param0_x <= 0.0 and param2_x >= 1.0:
                    dtype = 0  # PP v1
                elif param1_x <= 0.0 and param0_x >= 1.0:
                    dtype = 1  # PP v2
                elif param2_x <= 0.0 and param1_x >= 1.0:
                    dtype = 2  # PP v3
                else:
                    dtype = 6  # PT

    return dtype


@ti.func
def _dType_edge_edge(v0, v1, v2, v3):
    """
    Determine the distance type for edge-edge pair.

    Args:
        v0, v1: Edge a endpoints
        v2, v3: Edge b endpoints

    Returns:
        dtype: Distance type
            0: PP ea0-eb0, 1: PP ea0-eb1, 2: PE ea0-eb
            3: PP ea1-eb0, 4: PP ea1-eb1, 5: PE ea1-eb
            6: PE eb0-ea, 7: PE eb1-ea, 8: EE (edge-edge)
    """
    u = v1 - v0
    v = v3 - v2
    w = v0 - v2

    a = u.norm_sqr()
    b = u.dot(v)
    c = v.norm_sqr()
    d = u.dot(w)
    e = v.dot(w)

    D = a * c - b * b
    tD = D
    tN = D
    defaultCase = 8
    sN = b * e - c * d

    if sN <= 0.0:
        tN = e
        tD = c
        defaultCase = 2
    elif sN >= D:
        tN = e + b
        tD = c
        defaultCase = 5
    else:
        tN = a * e - b * d
        cross_uv = u.cross(v)
        if tN > 0.0 and tN < tD and (w.dot(cross_uv) == 0.0 or cross_uv.norm_sqr() < 1.0e-20 * a * c):
            if sN < D / 2.0:
                tN = e
                tD = c
                defaultCase = 2
            else:
                tN = e + b
                tD = c
                defaultCase = 5

    dtype = defaultCase
    if tN <= 0.0:
        if -d <= 0.0:
            dtype = 0
        elif -d >= a:
            dtype = 3
        else:
            dtype = 6
    elif tN >= tD:
        if (-d + b) <= 0.0:
            dtype = 1
        elif (-d + b) >= a:
            dtype = 4
        else:
            dtype = 7

    return dtype


@ti.func
def point_point_distance(v0, v1):
    """Compute squared distance between two points."""
    return (v0 - v1).norm_sqr()


@ti.func
def point_edge_distance(v0, v1, v2):
    """Compute squared distance from point v0 to edge v1-v2."""
    return (v1 - v0).cross(v2 - v0).norm_sqr() / (v2 - v1).norm_sqr()


@ti.func
def point_triangle_distance(v0, v1, v2, v3):
    """Compute squared distance from point v0 to triangle v1-v2-v3."""
    b = (v2 - v1).cross(v3 - v1)
    aTb = (v0 - v1).dot(b)
    return aTb * aTb / b.norm_sqr()


@ti.func
def edge_edge_distance(v0, v1, v2, v3):
    """Compute squared distance between edge v0-v1 and edge v2-v3."""
    b = (v1 - v0).cross(v3 - v2)
    aTb = (v2 - v0).dot(b)
    return aTb * aTb / b.norm_sqr()


@ti.func
def point_triangle_distance_unclassified(p, t0, t1, t2):
    """Compute squared point-triangle distance with automatic type classification."""
    dtype = _dType_point_triangle(p, t0, t1, t2)
    dist2 = 0.0
    if dtype == 0:
        dist2 = point_point_distance(p, t0)
    elif dtype == 1:
        dist2 = point_point_distance(p, t1)
    elif dtype == 2:
        dist2 = point_point_distance(p, t2)
    elif dtype == 3:
        dist2 = point_edge_distance(p, t0, t1)
    elif dtype == 4:
        dist2 = point_edge_distance(p, t1, t2)
    elif dtype == 5:
        dist2 = point_edge_distance(p, t2, t0)
    else:  # dtype == 6
        dist2 = point_triangle_distance(p, t0, t1, t2)
    return dist2


@ti.func
def edge_edge_distance_unclassified(ea0, ea1, eb0, eb1):
    """Compute squared edge-edge distance with automatic type classification."""
    dtype = _dType_edge_edge(ea0, ea1, eb0, eb1)
    dist2 = 0.0
    if dtype == 0:
        dist2 = point_point_distance(ea0, eb0)
    elif dtype == 1:
        dist2 = point_point_distance(ea0, eb1)
    elif dtype == 2:
        dist2 = point_edge_distance(ea0, eb0, eb1)
    elif dtype == 3:
        dist2 = point_point_distance(ea1, eb0)
    elif dtype == 4:
        dist2 = point_point_distance(ea1, eb1)
    elif dtype == 5:
        dist2 = point_edge_distance(ea1, eb0, eb1)
    elif dtype == 6:
        dist2 = point_edge_distance(eb0, ea0, ea1)
    elif dtype == 7:
        dist2 = point_edge_distance(eb1, ea0, ea1)
    else:  # dtype == 8
        dist2 = edge_edge_distance(ea0, ea1, eb0, eb1)
    return dist2
