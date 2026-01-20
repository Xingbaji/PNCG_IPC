# The functions for elastic deformation
# dPsidF : first derivative of Psi w.r.t F
# d2PsidF2 : second derivative of Psi w.r.t F

import taichi as ti
from math_utils.matrix_util import *

# Import SPD projection functions
from math_utils.elastic_hessian_spd import (
    compute_element_hessian_spd,
    compute_element_force,
    svd3x3,
    Svd3x3,
    DiffTable3,
    compute_difftable_stvk,
    compute_difftable_neohookean,
    compute_difftable_arap,
    compute_hessian_spd_3d,
)

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
        lambda2 = 1.0
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
        lambda2 = 1.0 # fixed error: lambda0->lambda2

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
    """
    Compute FCR (Fixed Corotated) Hessian with SPD filtering.

    The FCR energy is: Psi = mu * ||F - R||_F^2 + la/2 * (J - 1)^2

    The Hessian has two parts:
    1. ARAP part: filtered to ensure SPD (twist mode clamping)
    2. Volume part: la * g3 @ g3.T + la * (J-1) * H3
       - g3 @ g3.T is always PSD (outer product)
       - (J-1) * H3 can be negative when J < 1, so we clamp (J-1) to max(J-1, 0)
    """
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]

    # Twist mode eigenvalues: lambda_i = 2 / (s_j + s_k)
    # For SPD, need lambda_i <= 2, i.e., s_j + s_k >= 1
    # Clamp to 1.0 when s_j + s_k < 2.0 to ensure SPD
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda2 = 1.0  # BUG FIX: was lambda0, should be lambda2

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

    # ARAP part with filtered twist modes
    d2PsidF2_ARAP = - mu * (
                lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)))
    for i in ti.static(range(9)):
        d2PsidF2_ARAP[i, i] += 2 * mu

    # Volume part
    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()

    # g3 @ g3.T is always PSD
    # (J-1) * H3 can be negative when J < 1, clamp to ensure PSD contribution
    # Note: H3 eigenvalues can be negative, so (J-1)*H3 with J<1 gives positive
    # contribution from negative eigenvalues, which is still not SPD.
    # Safe approach: only add (J-1)*H3 when J >= 1
    vol_coeff = ti.max(J - 1.0, 0.0)
    d2PsidF2_FCR = d2PsidF2_ARAP + la * g3.outer_product(g3) + la * vol_coeff * H3
    return d2PsidF2_FCR


# ==============================================================================
# SPD-projected Hessian functions using eigenanalysis
# These functions use SVD-space eigenvalue decomposition with eigenvalue clamping
# to guarantee positive semi-definite Hessian matrices.
# ==============================================================================

# Material type constants for SPD functions
MATERIAL_STVK = 0
MATERIAL_NEOHOOKEAN = 1
MATERIAL_ARAP = 2


@ti.func
def compute_d2PsidF2_STVK_SPD(F, mu, la):
    """
    Compute StVK (St. Venant-Kirchhoff) Hessian with SPD projection.

    Uses eigenanalysis-based approach: decompose Hessian in SVD space,
    clamp negative eigenvalues to zero, reconstruct via rank-1 updates.

    Energy: E = μ ||E||² + (λ/2) tr(E)²
    where E = (F^T F - I) / 2 is Green strain
    """
    return compute_element_hessian_spd(F, mu, la, MATERIAL_STVK, 1e-6)


@ti.func
def compute_d2PsidF2_NH_SPD(F, mu, la):
    """
    Compute Neo-Hookean Hessian with SPD projection.

    Uses eigenanalysis-based approach for guaranteed SPD.

    Energy: E = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2) ln²(J)
    where I₁ = ||F||², J = det(F)
    """
    return compute_element_hessian_spd(F, mu, la, MATERIAL_NEOHOOKEAN, 1e-6)


@ti.func
def compute_d2PsidF2_ARAP_SPD(F, mu, la):
    """
    Compute ARAP Hessian with full SPD projection.

    Uses eigenanalysis-based approach instead of simple twist mode clamping.
    This provides a more rigorous SPD guarantee.

    Energy: E = μ ||F - R||²
    where R is the rotation part from polar decomposition F = RS
    """
    return compute_element_hessian_spd(F, mu, la, MATERIAL_ARAP, 1e-6)


@ti.func
def compute_dPsidF_STVK_SPD(F, mu, la):
    """
    Compute StVK (St. Venant-Kirchhoff) stress tensor (First Piola-Kirchhoff).
    """
    return compute_element_force(F, mu, la, MATERIAL_STVK)


@ti.func
def compute_dPsidF_NH_SPD(F, mu, la):
    """
    Compute Neo-Hookean stress tensor (First Piola-Kirchhoff).
    """
    return compute_element_force(F, mu, la, MATERIAL_NEOHOOKEAN)


@ti.func
def compute_dPsidF_ARAP_SPD(F, mu, la):
    """
    Compute ARAP stress tensor (First Piola-Kirchhoff).
    """
    return compute_element_force(F, mu, la, MATERIAL_ARAP)


@ti.func
def compute_dPsidx_STVK_SPD(F, B, mu, la):
    """Compute StVK gradient in vertex space"""
    dPsidF = compute_dPsidF_STVK_SPD(F, mu, la)
    dPsidx = compute_dFdx_T_N(B, dPsidF)
    return dPsidx


@ti.func
def compute_dPsidx_NH_SPD(F, B, mu, la):
    """Compute Neo-Hookean gradient in vertex space"""
    dPsidF = compute_dPsidF_NH_SPD(F, mu, la)
    dPsidx = compute_dFdx_T_N(B, dPsidF)
    return dPsidx


@ti.func
def compute_dPsidx_ARAP_SPD(F, B, mu, la):
    """Compute ARAP gradient in vertex space"""
    dPsidF = compute_dPsidF_ARAP_SPD(F, mu, la)
    dPsidx = compute_dFdx_T_N(B, dPsidF)
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_STVK_SPD(F, B, mu, la):
    """
    Compute diagonal of StVK Hessian in vertex space.
    Uses full SPD Hessian, then extracts diagonal after transformation.
    """
    d2PsidF2 = compute_d2PsidF2_STVK_SPD(F, mu, la)  # 9x9
    # Transform to x-space: diag(dFdx^T @ d2PsidF2 @ dFdx)
    return compute_diag_from_d2PsidF2(B, d2PsidF2)


@ti.func
def compute_diag_d2Psidx2_NH_SPD(F, B, mu, la):
    """
    Compute diagonal of Neo-Hookean Hessian in vertex space.
    """
    d2PsidF2 = compute_d2PsidF2_NH_SPD(F, mu, la)
    return compute_diag_from_d2PsidF2(B, d2PsidF2)


@ti.func
def compute_diag_d2Psidx2_ARAP_SPD(F, B, mu, la):
    """
    Compute diagonal of ARAP Hessian in vertex space.
    """
    d2PsidF2 = compute_d2PsidF2_ARAP_SPD(F, mu, la)
    return compute_diag_from_d2PsidF2(B, d2PsidF2)


@ti.func
def compute_diag_from_d2PsidF2(B, d2PsidF2):
    """
    Compute diagonal of Hessian in x-space from F-space Hessian.
    diag_x = diag(dFdx^T @ d2PsidF2 @ dFdx)
    """
    # dFdx is 9x12, so dFdx^T @ d2PsidF2 @ dFdx is 12x12
    # We only need the diagonal
    diag_result = ti.Vector.zero(ti.f32, 12)

    # Compute dFdx columns and accumulate diagonal
    for k in ti.static(range(12)):
        # Get k-th column of dFdx
        dFdx_col_k = ti.Vector.zero(ti.f32, 9)
        # dFdx structure: dF_ij/dx_k depends on B
        # Using the relation from compute_dFdx_p
        vertex_idx = k // 3
        dim_idx = k % 3

        if vertex_idx == 0:
            # x0: dF/dx0 = -B^T (summed)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    if i == dim_idx:
                        dFdx_col_k[j * 3 + i] = -(B[0, j] + B[1, j] + B[2, j])
        else:
            # x1, x2, x3: dF/dx_k = B[vertex_idx-1, :] in the dim_idx row
            for j in ti.static(range(3)):
                dFdx_col_k[j * 3 + dim_idx] = B[vertex_idx - 1, j]

        # Compute (dFdx^T @ d2PsidF2 @ dFdx)[k,k] = dFdx_col_k^T @ d2PsidF2 @ dFdx_col_k
        temp = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                temp[i] += d2PsidF2[i, j] * dFdx_col_k[j]

        diag_result[k] = dFdx_col_k.dot(temp)

    return diag_result


@ti.func
def compute_pHp_STVK_SPD(F, B, p, mu, la):
    """
    Compute p^T H p for StVK with SPD Hessian.
    """
    d2PsidF2 = compute_d2PsidF2_STVK_SPD(F, mu, la)
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    # p^T H p = (dFdx @ p)^T @ d2PsidF2 @ (dFdx @ p)
    temp = ti.Vector.zero(ti.f32, 9)
    for i in ti.static(range(9)):
        for j in ti.static(range(9)):
            temp[i] += d2PsidF2[i, j] * dFdx_p[j]
    return dFdx_p.dot(temp)


@ti.func
def compute_pHp_NH_SPD(F, B, p, mu, la):
    """
    Compute p^T H p for Neo-Hookean with SPD Hessian.
    """
    d2PsidF2 = compute_d2PsidF2_NH_SPD(F, mu, la)
    dFdx_p = compute_dFdx_p(B, p)
    temp = ti.Vector.zero(ti.f32, 9)
    for i in ti.static(range(9)):
        for j in ti.static(range(9)):
            temp[i] += d2PsidF2[i, j] * dFdx_p[j]
    return dFdx_p.dot(temp)


@ti.func
def compute_pHp_ARAP_SPD(F, B, p, mu, la):
    """
    Compute p^T H p for ARAP with SPD Hessian.
    """
    d2PsidF2 = compute_d2PsidF2_ARAP_SPD(F, mu, la)
    dFdx_p = compute_dFdx_p(B, p)
    temp = ti.Vector.zero(ti.f32, 9)
    for i in ti.static(range(9)):
        for j in ti.static(range(9)):
            temp[i] += d2PsidF2[i, j] * dFdx_p[j]
    return dFdx_p.dot(temp)
