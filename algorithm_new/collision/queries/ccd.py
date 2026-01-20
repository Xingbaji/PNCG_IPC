"""
Continuous Collision Detection (CCD) functions.

Provides conservative CCD with improved lower bounds for point-triangle
and edge-edge collision pairs.
"""

import taichi as ti
from .distance import (
    point_triangle_distance_unclassified,
    edge_edge_distance_unclassified,
)


# ===================== Broadphase Functions =====================

@ti.func
def point_triangle_ccd_broadphase(p0, t0, t1, t2, dHat):
    """
    Broadphase test for point-triangle CCD.

    Args:
        p0: Point position
        t0, t1, t2: Triangle vertex positions
        dHat: Distance threshold

    Returns:
        True if the point's AABB overlaps with the triangle's AABB (expanded by dHat)
    """
    min_t = ti.min(ti.min(t0, t1), t2)
    max_t = ti.max(ti.max(t0, t1), t2)
    return (p0 < max_t + dHat).all() and (min_t - dHat < p0).all()


@ti.func
def edge_edge_ccd_broadphase(a0, a1, b0, b1, dHat):
    """
    Broadphase test for edge-edge CCD.

    Args:
        a0, a1: First edge endpoints
        b0, b1: Second edge endpoints
        dHat: Distance threshold

    Returns:
        True if the two edges' AABBs overlap (expanded by dHat)
    """
    max_a = ti.max(a0, a1)
    min_a = ti.min(a0, a1)
    max_b = ti.max(b0, b1)
    min_b = ti.min(b0, b1)
    return (min_a < max_b + dHat).all() and (min_b - dHat < max_a).all()


# ===================== Cubic Polynomial Helper Functions =====================

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
            deriv_a = a * 3.0
            deriv_b_2 = b
            deriv_c = c

            # Discriminant: delta_4 = b^2 - 3a*c
            delta_4 = deriv_b_2 * deriv_b_2 - deriv_a * deriv_c

            # Only when a != 0 and discriminant > 0 do we have two extrema
            if ti.abs(deriv_a) > 1e-12 and delta_4 > 0.0:
                d_2 = ti.sqrt(delta_4)

                # Use numerically stable root formula
                q = -(deriv_b_2 + _mult_sign(d_2, deriv_b_2))

                if ti.abs(q) > 1e-12:
                    rv0 = q / deriv_a
                    rv1 = deriv_c / q

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
                    # q ~ 0: use direct formula (when b ~ 0)
                    if deriv_a * deriv_c < 0.0:
                        x_extreme = ti.sqrt(-deriv_c / deriv_a)

                        if not found and x_extreme > x_current and x_extreme < 1.0:
                            y_extreme = _polynomial_eval3(a, b, c, d, x_extreme)
                            if _is_different_sign(y_current, y_extreme):
                                alpha = _find_safe_region_bisection(a, b, c, d, x_current, x_extreme, y_current, y_extreme)
                                found = True
                            else:
                                x_current = x_extreme
                                y_current = y_extreme

                        x_extreme_neg = -x_extreme
                        if not found and x_extreme_neg > x_current and x_extreme_neg < 1.0:
                            y_extreme_neg = _polynomial_eval3(a, b, c, d, x_extreme_neg)
                            if _is_different_sign(y_current, y_extreme_neg):
                                alpha = _find_safe_region_bisection(a, b, c, d, x_current, x_extreme_neg, y_current, y_extreme_neg)
                                found = True
                            else:
                                x_current = x_extreme_neg
                                y_current = y_extreme_neg

            # a=0: degenerates to quadratic
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


# ===================== Cubic Polynomial Coefficients =====================

@ti.func
def _equate_cubic_vf(a0, ad, b0, bd, c0, cd, p0, pd):
    """
    Compute cubic polynomial coefficients for vertex-face (point-triangle) coplanarity.
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

        # Center velocities
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

        # Center velocities
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


# ===================== ACCD (Additive CCD) Functions =====================

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
    p_cur = p
    t0_cur = t0
    t1_cur = t1
    t2_cur = t2

    # Center the displacements
    mov = -0.25 * (dt0 + dt1 + dt2 + dp)
    dp_centered = dp + mov
    dt0_centered = dt0 + mov
    dt1_centered = dt1 + mov
    dt2_centered = dt2 + mov

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
    ea0_cur = ea0
    ea1_cur = ea1
    eb0_cur = eb0
    eb1_cur = eb1

    # Center the displacements
    mov = -0.25 * (dea0 + dea1 + deb0 + deb1)
    dea0_centered = dea0 + mov
    dea1_centered = dea1 + mov
    deb0_centered = deb0 + mov
    deb1_centered = deb1 + mov

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
