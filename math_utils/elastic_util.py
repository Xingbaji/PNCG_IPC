# The functions for elastic deformation
# dPsidF : first derivative of Psi w.r.t F
# d2PsidF2 : second derivative of Psi w.r.t F

import taichi as ti
from math_utils.matrix_util import *

# ARAP
@ti.func
def compute_Psi_ARAP(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    Psi = mu * (F - R).norm_sqr()
    return Psi

@ti.func
def compute_dPsidF_ARAP(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    dPsidF_3x3 = 2.0 * mu * (F - R)
    return dPsidF_3x3

@ti.func
def compute_dPsidx_ARAP(F, B, mu, la):
    dPsidF = compute_dPsidF_ARAP(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx

@ti.func
def compute_diag_d2Psidx2_ARAP(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    return mu * (2.0 * X - diag_h4)


@ti.func
def compute_pHp_ARAP(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret = mu * (2.0 * ret0 - ret1)
    return ret


# ARAP filter
@ti.func
def compute_diag_d2Psidx2_ARAP_filter(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    return mu * (2.0 * X - diag_h4)


@ti.func
def compute_pHp_ARAP_filter(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret = mu * (2.0 * ret0 - ret1)
    return ret


# FCR
@ti.func
def compute_Psi_FCR(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    J = F.determinant()
    Psi = mu * (F - R).norm_sqr() + 0.5 * la * (J - 1.0) ** 2
    return Psi


@ti.func
def compute_dPsidF_FCR(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    dPsidF_3x3_ARAP = 2.0 * mu * (F - R)
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    dPsidF_3x3 = dPsidF_3x3_ARAP + la * (J - 1) * dJdF
    return dPsidF_3x3


@ti.func
def compute_dPsidx_FCR(F, B, mu, la):
    dPsidF = compute_dPsidF_FCR(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_FCR(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    diag_ARAP = mu * (2.0 * X - diag_h4)
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diagJ = la * dFdx_T_g3 ** 2
    return diag_ARAP + diagJ


@ti.func
def compute_diag_d2Psidx2_FCR_filter(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    diag_ARAP = mu * (2.0 * X - diag_h4)
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diagJ = la * dFdx_T_g3 ** 2
    return diag_ARAP + diagJ


@ti.func
def compute_pHp_FCR(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret_ARAP = mu * (2.0 * ret0 - ret1)
    g_3 = compute_vec_dJdF(F)
    ret_2 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret_3 = la * (J - 1.0) * compute_d_H3_d(F, dFdx_p)
    return ret_ARAP + ret_2 + ret_3


@ti.func
def compute_pHp_FCR_filter(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret_ARAP = mu * (2.0 * ret0 - ret1)
    g_3 = compute_vec_dJdF(F)
    ret_2 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret_3 = la * (J - 1.0) * compute_d_H3_d(F, dFdx_p)
    if ret_3 < 0.0:
        ret_3 = 0.0
    return ret_ARAP + ret_2 + ret_3


# SNH
@ti.func
def compute_Psi_SNH(F, mu, la):
    J = F.determinant()
    return 0.5 * mu * (F.norm_sqr() - 3) - mu * (J - 1) + 0.5 * la * (J - 1) ** 2


@ti.func
def compute_dPsidF_SNH(F, mu, la):
    # return 3x3
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    dPsidF = mu * F + (- mu + la * (J - 1)) * dJdF
    return dPsidF


@ti.func
def compute_dPsidx_SNH(F, B, mu, la):
    dPsidF = compute_dPsidF_SNH(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_SNH(F, B, mu, la):
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diag1 = la * dFdx_T_g3 ** 2
    diag2 = mu * compute_diag_dFdx_T_dFdx(B)
    return diag1 + diag2


@ti.func
def compute_pHp_SNH(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = mu * dFdx_p.norm_sqr()
    g_3 = compute_vec_dJdF(F)
    ret1 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret2 = (la * (J - 1.0) - mu) * compute_d_H3_d(F, dFdx_p)
    return ret0 + ret1 + ret2

@ti.func
def compute_Psi_NH(F, mu, la):
    J = F.determinant()
    return 0.5 * mu * (F.norm_sqr() - 3) - mu * ti.log(J) + 0.5 * la * ti.log(J) ** 2


@ti.func
def compute_dPsidF_NH(F, mu, la):
    # return 3x3
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    para = (- mu + la * ti.log(J)) / J
    dPsidF = mu * F + para * dJdF
    return dPsidF.transpose()


@ti.func
def compute_dPsidx_NH(F, B, mu, la):
    dPsidF = compute_dPsidF_SNH(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_NH(F, B, mu, la):
    J = F.determinant()
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    para = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    diag1 = para * dFdx_T_g3 ** 2
    diag2 = mu * compute_diag_dFdx_T_dFdx(B)
    return diag1 + diag2


@ti.func
def compute_pHp_NH(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = mu * dFdx_p.norm_sqr()
    g_3 = compute_vec_dJdF(F)
    J = F.determinant()
    para1 = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    ret1 = para1 * ((g_3.dot(dFdx_p)) ** 2)
    para2 = (la * ti.log(J) - mu) / J
    ret2 = para2 * compute_d_H3_d(F, dFdx_p)
    return ret0 + ret1 + ret2


@ti.func
def compute_d2PsidF2_ARAP(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    # remove 1/sqrt(2) here
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    d2PsidF2 = - mu *( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    # remove 2.0 here
    for i in ti.static(range(9)):
        d2PsidF2[i,i] += 2 * mu
    return d2PsidF2

@ti.func
def compute_d2PsidF2_ARAP_filter(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0

    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    # remove 1/sqrt(2) here
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    d2PsidF2 = - mu * ( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    # remove 2.0 here
    for i in ti.static(range(9)):
        d2PsidF2[i,i] += 2.0 * mu
    return d2PsidF2

@ti.func
def compute_d2PsidF2_SNH(F, mu, la):
    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    ret = mu * g3.outer_product(g3) + (la * (J - 1.0) - mu) * H3
    for i in range(9):
        ret[i,i] += mu
    return ret

@ti.func
def compute_d2PsidF2_NH(F, mu, la):
    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    para1 = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    para2 = (la * ti.log(J) - mu) / J
    ret = para1 * g3.outer_product(g3) + para2 * H3
    for i in range(9):
        ret += mu
    return ret

@ti.func
def compute_d2PsidF2_FCR(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    # ARAP
    d2PsidF2_ARAP = - mu *( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    for i in ti.static(range(9)):
        d2PsidF2_ARAP[i,i] += 2 * mu

    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    d2PsidF2_FCR = d2PsidF2_ARAP + la * g3.outer_product(g3) + la * (J-1) * H3
    return d2PsidF2_FCR


@ti.func
def compute_d2PsidF2_FCR_filter(F, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    # ARAP
    d2PsidF2_ARAP = - mu * (
                lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)))
    for i in ti.static(range(9)):
        d2PsidF2_ARAP[i, i] += 2 * mu

    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    d2PsidF2_FCR = d2PsidF2_ARAP + la * g3.outer_product(g3) + la * (J - 1) * H3
    return d2PsidF2_FCR
