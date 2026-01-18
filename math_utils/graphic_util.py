import taichi as ti
@ti.func
def dist3D_Segment_to_Segment(A0,A1,B0,B1):
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
    dP = - w - (sc * u) + (tc * v) # Qc - Pc
    return dP, sc, tc

@ti.func
def dist3D_Point_Triangle(P, V0, V1, V2):
    cord0 = 0.0
    cord1 = 0.0
    cord2 = 0.0
    v = V2 - V0
    u = V1 - V0
    nVec = u.cross(v)
    s_p = (nVec.dot(P - V0)) / (nVec.dot(nVec))
    P0 = P - s_p * nVec # P project to plane
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
    else: # s < 0 and t < 0
        cord0 = 1.0
    return cord0, cord1, cord2


@ti.func
def dist3D_Point_Triangle_type(P, V0, V1, V2):
    cord0 = 0.0
    cord1 = 0.0
    cord2 = 0.0
    v = V2 - V0
    u = V1 - V0
    nVec = u.cross(v)
    s_p = (nVec.dot(P - V0)) / (nVec.dot(nVec))
    P0 = P - s_p * nVec # P project to plane
    w = P0 - V0
    n_cross_v = nVec.cross(v)
    n_cross_u = nVec.cross(u)
    s = w.dot(n_cross_v) / (u.dot(n_cross_v))
    t = w.dot(n_cross_u) / (v.dot(n_cross_u))
    type = 0 # 0: PP, 1: PE, 2: PT
    if s >= 0.0 and t >= 0.0:
        if s + t <= 1.0:
            cord0 = 1.0 - s - t
            cord1 = s
            cord2 = t
            type = 2
        else:
            q = V2 - V1
            k = (P - V1).dot(q) / (q.dot(q))
            if k > 1.0:
                cord2 = 1.0
                type = 0
            elif k < 0.0:
                cord1 = 1.0
                type = 0
            else:
                cord1 = 1.0 - k
                cord2 = k
                type = 1
    elif s >= 0.0 and t < 0.0:
        k = w.dot(u) / (u.dot(u))
        if k > 1.0:
            cord1 = 1.0
            type = 0
        elif k < 0.0:
            cord0 = 1.0
            type = 0
        else:
            cord0 = 1.0 - k
            cord1 = k
            type = 1
    elif s < 0.0 and t >= 0.0:
        k = w.dot(v) / (v.dot(v))
        if k > 1.0:
            cord2 = 1.0
            type = 0
        elif k < 0.0:
            cord0 = 1.0
            type = 0
        else:
            cord0 = 1.0 - k
            cord2 = k
            type = 1
    else: # s < 0 and t < 0
        cord0 = 1.0
        type = 0
    return cord0, cord1, cord2, type


@ti.func
def dcd_line_triangle(xa, xb, x0, x1, x2):
    ret = 0
    x10 = x1 - x0
    x20 = x2 - x0
    N = x10.cross(x20)
    x0a = x0 - xa
    xba = xb - xa
    t = x0a.dot(N) / xba.dot(N)
    if t >= 0.0 and t <= 1.0:
        xt = (1-t) * x0 + t * x1
        ret0 = ((x0 - xt).cross(x1-xt)).dot(N)
        ret1 = ((x1 - xt).cross(x2-xt)).dot(N)
        ret2 = ((x2 - xt).cross(x0-xt)).dot(N)
        if ret0 >= 0 and ret1 >= 0 and ret2 >= 0:
            ret = 1
    return ret

@ti.func
def segment_intersect_triangle_new(P0, P1, V0, V1, V2):
    ret = 0
    u = V1 - V0
    v = V2 - V0
    n = u.cross(v)
    if n.norm() > 1e-6: # triangle is not degenerate
        dir = P1 - P0
        w0 = P0 - V0
        a = - n.dot(w0)
        b = n.dot(dir)
        if ti.abs(b) > 1e-6: #ray is not parallel to triangle plane
            r = a / b
            if r >= 0.0 and r <= 1.0:
                I = P0 + r * dir # intersection point
                uu = u.dot(u)
                uv = u.dot(v)
                vv = v.dot(v)
                w = I - V0
                wu = w.dot(u)
                wv = w.dot(v)
                D = uv * uv - uu * vv
                s = (uv * wv - vv * wu) / D
                t = (uv * wu - uu * wv) / D
                if s< 0.0 or s > 1.0:
                    ret = 0
                elif t < 0.0 or (s+t) > 1.0:
                    ret = 0
                else:
                    ret = 1
    return ret





@ti.func
def segment_intersect_triangle(P, Q, A, B, C):
    RLen = (Q - P).norm()
    RDir = (Q - P) / RLen
    ROrigin = P
    E1 = B - A
    E2 = C - A
    N = E1.cross(E2)
    det = -RDir.dot(N)
    invdet = 1.0 / det
    AO  = ROrigin - A
    DAO = AO.cross(RDir)
    u = E2.dot(DAO) * invdet
    v = -E1.dot(DAO) * invdet
    t = AO.dot(N) * invdet
    ret = 0
    if det >= 1e-5 and t >= 1e-6 and u >= 1e-6 and v >= 1e-6 and (u+v) <= 1.0-1e-6 and t <= RLen:
        ret = 1
        # print(det,det>=1e-5, t, t>=0.0, u, u>=0.0, v, v>=0.0, (u+v) <= 1.0, t <= RLen)

    return ret
    # return det >= 1e-12 and t >= 0.0 and u >= 0.0 and v >= 0.0 and (u+v) <= 1.0 and t <= RLen

@ti.func
def point_triangle_ccd_broadphase(p0, t0, t1, t2, dHat):
    min_t = ti.min(ti.min(t0, t1), t2)
    max_t = ti.max(ti.max(t0, t1), t2)
    return (p0 < max_t + dHat).all() and (min_t - dHat < p0).all()

@ti.func
def edge_edge_ccd_broadphase(a0, a1, b0, b1, dHat):
    max_a = ti.max(a0, a1)
    min_a = ti.min(a0, a1)
    max_b = ti.max(b0, b1)
    min_b = ti.min(b0, b1)
    return (min_a < max_b + dHat).all() and (min_b - dHat < max_a).all()


# ===================== ACCD (Additive CCD) Functions =====================
# These implement conservative CCD with improved lower bounds from the paper
# "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework
# for Incremental Potential Contact"

@ti.func
def _dType_point_triangle(v0, v1, v2, v3):
    """
    Determine the distance type for point-triangle pair.
    v0: point, v1, v2, v3: triangle vertices
    Returns:
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
    v0, v1: edge a, v2, v3: edge b
    Returns:
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
    tN = D  # Initialize tN
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


# ===================== CubicNoRootRegionPrecise Helper Functions =====================
# Based on cyPolynomial.h by Cem Yuksel - high-performance polynomial root finding
# Reference: Cem Yuksel. 2022. High-Performance Polynomial Root Finding for Graphics.

@ti.func
def _is_different_sign(a, b):
    """Check if two numbers have different signs."""
    return (a < 0.0) != (b < 0.0)


@ti.func
def _polynomial_eval3(a, b, c, d, x):
    """Evaluate cubic polynomial f(x) = a*x^3 + b*x^2 + c*x + d using Horner's method."""
    return ((a * x + b) * x + c) * x + d


@ti.func
def _mult_sign(v, sign):
    """Return v with the sign of 'sign': sign(sign) * v"""
    result = v
    if sign < 0.0:
        result = -v
    return result


@ti.func
def _find_safe_region_bisection(a, b, c, d, x_safe, x_unsafe, y_safe, y_unsafe):
    """
    Find safe region boundary using bisection in interval [x_safe, x_unsafe].
    Returns the largest x where the polynomial doesn't cross zero.
    """
    xs = x_safe
    xu = x_unsafe
    ys = y_safe

    # 20 iterations gives ~1e-6 precision
    for _ in range(20):
        interval = xu - xs
        if interval < 1e-10:
            break

        x_mid = 0.5 * (xs + xu)
        y_mid = _polynomial_eval3(a, b, c, d, x_mid)

        if _is_different_sign(ys, y_mid):
            xu = x_mid
        else:
            xs = x_mid
            ys = y_mid

    return xs


@ti.func
def _cubic_no_root_region_precise(a, b, c, d, alpha_l):
    """
    Find the safe region boundary for cubic polynomial f(x) = a*x^3 + b*x^2 + c*x + d.
    Returns alpha such that f(x) has no root in [alpha_l, alpha].

    This is the precise version that uses bisection to find exact boundaries
    instead of conservatively returning extrema points.
    """
    alpha = 1.0
    found = False

    if alpha_l < 1.0:
        x_current = alpha_l
        if x_current < 0.0:
            x_current = 0.0

        y_current = _polynomial_eval3(a, b, c, d, x_current)

        if y_current == 0.0:
            alpha = x_current
            found = True

        if not found:
            # f(1) = a + b + c + d
            y1 = a + b + c + d

            # Compute derivative f'(x) = 3a*x^2 + 2b*x + c critical points
            deriv_a = a * 3.0      # 3a
            deriv_b_2 = b          # b (actually 2b/2 = b)
            deriv_c = c            # c

            # Discriminant: delta_4 = b^2 - 3a*c
            delta_4 = deriv_b_2 * deriv_b_2 - deriv_a * deriv_c

            # Only when a != 0 and discriminant > 0 do we have two extrema
            if ti.abs(deriv_a) > 1e-12 and delta_4 > 0.0:
                d_2 = ti.sqrt(delta_4)

                # Use numerically stable root formula (avoid catastrophic cancellation)
                q = -(deriv_b_2 + _mult_sign(d_2, deriv_b_2))

                if ti.abs(q) > 1e-12:
                    rv0 = q / deriv_a
                    rv1 = deriv_c / q

                    # Ensure xa < xb
                    xa = ti.min(rv0, rv1)
                    xb = ti.max(rv0, rv1)

                    # Check first extremum
                    if not found and xa > x_current and xa < 1.0:
                        ya = _polynomial_eval3(a, b, c, d, xa)
                        if _is_different_sign(y_current, ya):
                            alpha = _find_safe_region_bisection(a, b, c, d, x_current, xa, y_current, ya)
                            found = True
                        else:
                            x_current = xa
                            y_current = ya

                    # Check second extremum
                    if not found and xb > x_current and xb < 1.0:
                        yb = _polynomial_eval3(a, b, c, d, xb)
                        if _is_different_sign(y_current, yb):
                            alpha = _find_safe_region_bisection(a, b, c, d, x_current, xb, y_current, yb)
                            found = True
                        else:
                            x_current = xb
                            y_current = yb
                else:
                    # q ≈ 0: use direct formula (when b ≈ 0)
                    # f'(x) = 3ax² + c = 0 => x = ±√(-c/(3a))
                    if deriv_a * deriv_c < 0.0:  # Ensure -c/(3a) > 0
                        x_extreme = ti.sqrt(-deriv_c / deriv_a)

                        # Check positive extremum
                        if not found and x_extreme > x_current and x_extreme < 1.0:
                            y_extreme = _polynomial_eval3(a, b, c, d, x_extreme)
                            if _is_different_sign(y_current, y_extreme):
                                alpha = _find_safe_region_bisection(a, b, c, d, x_current, x_extreme, y_current, y_extreme)
                                found = True
                            else:
                                x_current = x_extreme
                                y_current = y_extreme

                        # Check negative extremum (if in valid range)
                        x_extreme_neg = -x_extreme
                        if not found and x_extreme_neg > x_current and x_extreme_neg < 1.0:
                            y_extreme_neg = _polynomial_eval3(a, b, c, d, x_extreme_neg)
                            if _is_different_sign(y_current, y_extreme_neg):
                                alpha = _find_safe_region_bisection(a, b, c, d, x_current, x_extreme_neg, y_current, y_extreme_neg)
                                found = True
                            else:
                                x_current = x_extreme_neg
                                y_current = y_extreme_neg

            # a=0: degenerates to quadratic, derivative f'(x) = 2bx + c has one extremum
            elif ti.abs(deriv_a) <= 1e-12 and ti.abs(b) > 1e-12:
                x_extreme = -c / (2.0 * b)

                if not found and x_extreme > x_current and x_extreme < 1.0:
                    y_extreme = _polynomial_eval3(a, b, c, d, x_extreme)
                    if _is_different_sign(y_current, y_extreme):
                        alpha = _find_safe_region_bisection(a, b, c, d, x_current, x_extreme, y_current, y_extreme)
                        found = True
                    else:
                        x_current = x_extreme
                        y_current = y_extreme

            # Check final interval [x_current, 1]
            if not found and _is_different_sign(y_current, y1):
                alpha = _find_safe_region_bisection(a, b, c, d, x_current, 1.0, y_current, y1)

    return alpha


@ti.func
def _equate_cubic_vf(a0, ad, b0, bd, c0, cd, p0, pd):
    """
    Compute cubic polynomial coefficients for vertex-face (point-triangle) coplanarity.
    The polynomial f(t) = a*t^3 + b*t^2 + c*t + d represents the signed volume
    of the tetrahedron formed by the point and triangle at time t.
    f(t) = 0 means the point lies on the triangle plane.
    """
    dab = bd - ad
    dac = cd - ad
    dap = pd - ad
    oab = b0 - a0
    oac = c0 - a0
    oap = p0 - a0

    dabXdac = dab.cross(dac)
    dabXoac = dab.cross(oac)
    oabXdac = oab.cross(dac)
    oabXoac = oab.cross(oac)

    a = dap.dot(dabXdac)
    b = oap.dot(dabXdac) + dap.dot(dabXoac + oabXdac)
    c = dap.dot(oabXoac) + oap.dot(dabXoac + oabXdac)
    d = oap.dot(oabXoac)

    return a, b, c, d


@ti.func
def _equate_cubic_ee(pa, pad, qa, qad, pb, pbd, qb, qbd):
    """
    Compute cubic polynomial coefficients for edge-edge coplanarity.
    The polynomial f(t) = a*t^3 + b*t^2 + c*t + d represents the signed volume
    of the tetrahedron formed by the four edge endpoints at time t.
    f(t) = 0 means the two edges are coplanar.
    """
    u0 = qa - pa
    v0 = qb - pb
    w0 = pb - pa

    ud = qad - pad
    vd = qbd - pbd
    wd = pbd - pad

    u0_x_v0 = u0.cross(v0)
    u0_x_vd = u0.cross(vd)
    ud_x_v0 = ud.cross(v0)
    ud_x_vd = ud.cross(vd)

    a = wd.dot(ud_x_vd) + ud.dot(w0.cross(vd)) + ud.dot(wd.cross(v0)) + w0.dot(ud_x_vd)
    b = w0.dot(u0_x_vd) + w0.dot(ud_x_v0) + wd.dot(u0_x_v0) + wd.dot(ud_x_v0) + ud.dot(w0.cross(v0)) + u0.dot(wd.cross(v0)) + u0.dot(w0.cross(vd))
    c = w0.dot(u0_x_v0) + wd.dot(u0_x_v0) + u0.dot(w0.cross(v0))
    d = w0.dot(u0_x_v0)

    return a, b, c, d


# ===================== CCD Lower Bound Functions =====================

@ti.func
def point_triangle_ccd_lower_bound(p, t0, t1, t2, dp, dt0, dt1, dt2):
    """
    Compute a conservative lower bound on the time of contact for point-triangle CCD.
    Uses CubicNoRootRegionPrecise for tight bounds.

    Args:
        p, t0, t1, t2: Current positions (point and triangle vertices)
        dp, dt0, dt1, dt2: Displacements over the time step

    Returns:
        toc: Time of contact (0 to 1), where 1 means no collision in this time step
    """
    # Compute relative displacements for initial lower bound
    rel0 = dp - dt0
    rel1 = dp - dt1
    rel2 = dp - dt2

    sq_norm0 = rel0.norm_sqr()
    sq_norm1 = rel1.norm_sqr()
    sq_norm2 = rel2.norm_sqr()

    max_sq_norm = ti.max(sq_norm0, ti.max(sq_norm1, sq_norm2))
    max_disp_mag = ti.sqrt(max_sq_norm)

    toc = 1.0
    if max_disp_mag > 0.0:
        dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2)
        dist_cur = ti.sqrt(dist2_cur)
        alpha_l = dist_cur / max_disp_mag

        # Center velocities (move center of mass to origin)
        mov = -0.25 * (dt0 + dt1 + dt2 + dp)
        dt0_c = dt0 + mov
        dt1_c = dt1 + mov
        dt2_c = dt2 + mov
        dp_c = dp + mov

        # Compute cubic polynomial coefficients
        a, b, c, d = _equate_cubic_vf(t0, dt0_c, t1, dt1_c, t2, dt2_c, p, dp_c)

        # Find safe region using precise cubic root finding
        toc = _cubic_no_root_region_precise(a, b, c, d, alpha_l)

    return toc


@ti.func
def edge_edge_ccd_lower_bound(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1):
    """
    Compute a conservative lower bound on the time of contact for edge-edge CCD.
    Uses CubicNoRootRegionPrecise for tight bounds.

    Args:
        ea0, ea1: Current positions of edge a endpoints
        eb0, eb1: Current positions of edge b endpoints
        dea0, dea1: Displacements of edge a endpoints
        deb0, deb1: Displacements of edge b endpoints

    Returns:
        toc: Time of contact (0 to 1), where 1 means no collision in this time step
    """
    # Compute relative displacements for initial lower bound
    rel00 = dea0 - deb0
    rel01 = dea0 - deb1
    rel10 = dea1 - deb0
    rel11 = dea1 - deb1

    sq_norm00 = rel00.norm_sqr()
    sq_norm01 = rel01.norm_sqr()
    sq_norm10 = rel10.norm_sqr()
    sq_norm11 = rel11.norm_sqr()

    max_sq_norm = ti.max(ti.max(sq_norm00, sq_norm01), ti.max(sq_norm10, sq_norm11))
    max_disp_mag = ti.sqrt(max_sq_norm)

    toc = 1.0
    if max_disp_mag > 0.0:
        dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1)
        dist_cur = ti.sqrt(dist2_cur)
        alpha_l = dist_cur / max_disp_mag

        # Center velocities (move center of mass to origin)
        mov = -0.25 * (dea0 + dea1 + deb0 + deb1)
        dea0_c = dea0 + mov
        dea1_c = dea1 + mov
        deb0_c = deb0 + mov
        deb1_c = deb1 + mov

        # Compute cubic polynomial coefficients
        a, b, c, d = _equate_cubic_ee(ea0, dea0_c, ea1, dea1_c, eb0, deb0_c, eb1, deb1_c)

        # Find safe region using precise cubic root finding
        toc = _cubic_no_root_region_precise(a, b, c, d, alpha_l)

    return toc


@ti.func
def point_triangle_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness=0.0):
    """
    Compute a conservative time of contact for point-triangle CCD.
    Uses centering approach (original ACCD algorithm).

    Args:
        p, t0, t1, t2: Current positions (point and triangle vertices)
        dp, dt0, dt1, dt2: Displacements over the time step
        eta: Safety coefficient (typically 0.1-0.5)
        thickness: Minimum separation distance (default 0.0)

    Returns:
        toc: Time of contact (0 to 1), where 1 means no collision in this time step
    """
    # Use local copies
    p_cur = p
    t0_cur = t0
    t1_cur = t1
    t2_cur = t2

    # Center the displacements (move center of mass to origin)
    mov = -0.25 * (dt0 + dt1 + dt2 + dp)
    dp_centered = dp + mov
    dt0_centered = dt0 + mov
    dt1_centered = dt1 + mov
    dt2_centered = dt2 + mov

    # Compute max displacement magnitude
    disp_mag2_vec0 = dt0_centered.norm_sqr()
    disp_mag2_vec1 = dt1_centered.norm_sqr()
    disp_mag2_vec2 = dt2_centered.norm_sqr()

    max_disp_mag = dp_centered.norm() + ti.sqrt(ti.max(disp_mag2_vec0, ti.max(disp_mag2_vec1, disp_mag2_vec2)))

    toc = 1.0
    if max_disp_mag > 0.0:
        dist2_cur = point_triangle_distance_unclassified(p_cur, t0_cur, t1_cur, t2_cur)
        dist_cur = ti.sqrt(dist2_cur)
        gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness)
        toc = 0.0

        for _ in range(50000):
            dFunc = dist2_cur - thickness * thickness
            toc_lower_bound = (1.0 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag)

            p_cur = p_cur + dp_centered * toc_lower_bound
            t0_cur = t0_cur + dt0_centered * toc_lower_bound
            t1_cur = t1_cur + dt1_centered * toc_lower_bound
            t2_cur = t2_cur + dt2_centered * toc_lower_bound

            dist2_cur = point_triangle_distance_unclassified(p_cur, t0_cur, t1_cur, t2_cur)
            dist_cur = ti.sqrt(dist2_cur)

            if toc > 0.0 and ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap):
                break

            toc += toc_lower_bound
            if toc > 1.0:
                toc = 1.0
                break

    return toc


@ti.func
def edge_edge_ccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness=0.0):
    """
    Compute a conservative time of contact for edge-edge CCD.
    Uses centering approach (original ACCD algorithm).

    Args:
        ea0, ea1: Current positions of edge a endpoints
        eb0, eb1: Current positions of edge b endpoints
        dea0, dea1: Displacements of edge a endpoints
        deb0, deb1: Displacements of edge b endpoints
        eta: Safety coefficient (typically 0.1-0.5)
        thickness: Minimum separation distance (default 0.0)

    Returns:
        toc: Time of contact (0 to 1), where 1 means no collision in this time step
    """
    # Use local copies
    ea0_cur = ea0
    ea1_cur = ea1
    eb0_cur = eb0
    eb1_cur = eb1

    # Center the displacements (move center of mass to origin)
    mov = -0.25 * (dea0 + dea1 + deb0 + deb1)
    dea0_centered = dea0 + mov
    dea1_centered = dea1 + mov
    deb0_centered = deb0 + mov
    deb1_centered = deb1 + mov

    # Compute max displacement magnitude
    max_disp_mag = (ti.sqrt(ti.max(dea0_centered.norm_sqr(), dea1_centered.norm_sqr())) +
                    ti.sqrt(ti.max(deb0_centered.norm_sqr(), deb1_centered.norm_sqr())))

    toc = 1.0
    if max_disp_mag > 0.0:
        dist2_cur = edge_edge_distance_unclassified(ea0_cur, ea1_cur, eb0_cur, eb1_cur)

        dFunc = dist2_cur - thickness * thickness
        if dFunc <= 0.0:
            # Use minimum point-point distance as fallback
            dists0 = (ea0_cur - eb0_cur).norm_sqr()
            dists1 = (ea0_cur - eb1_cur).norm_sqr()
            dists2 = (ea1_cur - eb0_cur).norm_sqr()
            dists3 = (ea1_cur - eb1_cur).norm_sqr()
            dist2_cur = ti.min(ti.min(dists0, dists1), ti.min(dists2, dists3))
            dFunc = dist2_cur - thickness * thickness

        dist_cur = ti.sqrt(dist2_cur)
        gap = eta * dFunc / (dist_cur + thickness)
        toc = 0.0

        for _ in range(50000):
            toc_lower_bound = (1.0 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag)

            ea0_cur = ea0_cur + dea0_centered * toc_lower_bound
            ea1_cur = ea1_cur + dea1_centered * toc_lower_bound
            eb0_cur = eb0_cur + deb0_centered * toc_lower_bound
            eb1_cur = eb1_cur + deb1_centered * toc_lower_bound

            dist2_cur = edge_edge_distance_unclassified(ea0_cur, ea1_cur, eb0_cur, eb1_cur)
            dFunc = dist2_cur - thickness * thickness

            if dFunc <= 0.0:
                dists0 = (ea0_cur - eb0_cur).norm_sqr()
                dists1 = (ea0_cur - eb1_cur).norm_sqr()
                dists2 = (ea1_cur - eb0_cur).norm_sqr()
                dists3 = (ea1_cur - eb1_cur).norm_sqr()
                dist2_cur = ti.min(ti.min(dists0, dists1), ti.min(dists2, dists3))
                dFunc = dist2_cur - thickness * thickness

            dist_cur = ti.sqrt(dist2_cur)

            if toc > 0.0 and (dFunc / (dist_cur + thickness) < gap):
                break

            toc += toc_lower_bound
            if toc > 1.0:
                toc = 1.0
                break

    return toc