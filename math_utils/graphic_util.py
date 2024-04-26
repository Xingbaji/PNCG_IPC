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