"""
Contact Assembly Utilities for MAS Preconditioner.

This module provides barrier function implementations and utility functions
for contact Hessian assembly.

Barrier functions:
- Log barrier (default): -kappa * (d - dHat)^2 * log(d / dHat)
- Cubic barrier: -2*kappa/(3*dHat) * (d - dHat)^3
"""

import taichi as ti


# ============================================================================
# Log Barrier Functions
# ============================================================================

@ti.func
def barrier_E_log(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Log barrier energy: E = -kappa * (d - dHat)^2 * log(d / dHat)

    Active when d < dHat.
    """
    E = 0.0
    if d < dHat and d > 1e-10:
        t2 = d - dHat
        E = -kappa * t2 * t2 * ti.log(d / dHat)
    return E


@ti.func
def barrier_g_log(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Log barrier gradient: g = dE/dd

    g = kappa * [(d-dHat) * (-2*log(d/dHat)) - (d-dHat)^2 / d]
    """
    g = 0.0
    if d < dHat and d > 1e-10:
        t2 = d - dHat
        g = kappa * (t2 * ti.log(d / dHat) * (-2.0) - (t2 * t2) / d)
    return g


@ti.func
def barrier_H_log(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Log barrier Hessian: H = d^2E/dd^2

    H = kappa * [(-2) * log(d/dHat) - 4 + 4*dHat/d + (d-dHat)^2 / d^2]
    """
    H = 0.0
    if d < dHat and d > 1e-10:
        H = kappa * ((-2.0) * ti.log(d / dHat) - 4.0 + 4.0 * dHat / d + (d - dHat) ** 2 / (d * d))
    return H


# ============================================================================
# Cubic Barrier Functions
# ============================================================================

@ti.func
def barrier_E_cubic(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Cubic barrier energy: E = -2*kappa/(3*dHat) * (d - dHat)^3

    Active when d < dHat.
    """
    E = 0.0
    if d < dHat:
        y = d - dHat  # y is negative when d < dHat
        E = -2.0 * kappa * (y * y * y) / (3.0 * dHat)
    return E


@ti.func
def barrier_g_cubic(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Cubic barrier gradient: g = dE/dd = -2*kappa/dHat * (d - dHat)^2

    Active when d < dHat.
    """
    g = 0.0
    if d < dHat:
        y = d - dHat
        g = -2.0 * kappa * (y * y) / dHat
    return g


@ti.func
def barrier_H_cubic(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Cubic barrier Hessian: H = d^2E/dd^2 = 4*kappa * (1 - d/dHat)

    Active when d < dHat.
    """
    H = 0.0
    if d < dHat:
        H = 4.0 * kappa * (1.0 - d / dHat)
    return H


# ============================================================================
# Adaptive Kappa Versions (for cubic barrier)
# ============================================================================

@ti.func
def barrier_g_cubic_adaptive(d: ti.f32, dHat: ti.f32, kappa_local: ti.f32) -> ti.f32:
    """
    Cubic barrier gradient with local adaptive kappa.
    """
    g = 0.0
    if d < dHat:
        y = d - dHat
        g = -2.0 * kappa_local * (y * y) / dHat
    return g


@ti.func
def barrier_H_cubic_adaptive(d: ti.f32, dHat: ti.f32, kappa_local: ti.f32) -> ti.f32:
    """
    Cubic barrier Hessian with local adaptive kappa.
    """
    H = 0.0
    if d < dHat:
        H = 4.0 * kappa_local * (1.0 - d / dHat)
    return H


# ============================================================================
# Contact Hessian Structure Utilities
# ============================================================================

@ti.func
def compute_contact_subblock(
    para0: ti.f32,
    para: ti.f32,
    t: ti.template(),
    coeff: ti.f32
) -> ti.Matrix:
    """
    Compute 3x3 contact Hessian sub-block.

    H_ij = coeff * (para0 * t @ t^T + para * I_3x3)

    where:
        coeff = scale * cord[i] * cord[j]
        para = barrier_g / dist
        para0 = (barrier_H - para) / dist^2
        t = contact direction (3D vector)

    Returns:
        3x3 matrix representing the contact Hessian sub-block
    """
    H_ij = ti.Matrix.zero(ti.f32, 3, 3)
    for di in ti.static(range(3)):
        for dj in ti.static(range(3)):
            H_ij[di, dj] = coeff * para0 * t[di] * t[dj]
            if di == dj:
                H_ij[di, dj] += coeff * para
    return H_ij


@ti.func
def project_contact_spd(
    H_ij: ti.template(),
    para0: ti.f32,
    para: ti.f32,
    t: ti.template()
) -> ti.Matrix:
    """
    Project 3x3 contact sub-block to SPD (if needed).

    The contact Hessian structure is:
        H_ij = para0 * (t @ t^T) + para * I_3x3

    Eigenvalues of t @ t^T: [||t||^2, 0, 0]
    Eigenvalues of H_ij: [para0 * ||t||^2 + para, para, para]

    For SPD: clamp all eigenvalues to >= 0

    Note: In most cases, the barrier functions ensure para >= 0 and para0 is
    properly scaled, so explicit projection may not be needed. This function
    is provided for safety in edge cases.
    """
    t_norm_sq = t.norm_sqr()
    lambda_1 = para0 * t_norm_sq + para  # Principal eigenvalue
    lambda_23 = para  # Repeated eigenvalue

    # Clamp eigenvalues
    lambda_1_clamped = ti.max(lambda_1, 0.0)
    lambda_23_clamped = ti.max(lambda_23, 0.0)

    # Reconstruct: H_ij = lambda_23 * I + (lambda_1 - lambda_23) * (t @ t^T / ||t||^2)
    H_spd = ti.Matrix.zero(ti.f32, 3, 3)
    if t_norm_sq > 1e-12:
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                H_spd[di, dj] = (lambda_1_clamped - lambda_23_clamped) * t[di] * t[dj] / t_norm_sq
                if di == dj:
                    H_spd[di, dj] += lambda_23_clamped
    else:
        for di in ti.static(range(3)):
            H_spd[di, di] = lambda_23_clamped

    return H_spd


# ============================================================================
# Contact Direction Jacobians (from math_utils/matrix_util.py)
# ============================================================================

@ti.func
def compute_dtdx_t(t: ti.template(), cord: ti.template()) -> ti.Matrix:
    """
    Construct 12x1 vector: dtdx^T with barycentric weighting.

    For each vertex i in 0..3:
        row 3i:3i+3 = cord[i] * t

    Returns:
        12-vector: [cord[0]*t[0], cord[0]*t[1], cord[0]*t[2],
                    cord[1]*t[0], ..., cord[3]*t[2]]
    """
    dtdx_t = ti.Vector.zero(ti.f32, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            dtdx_t[i * 3 + j] = cord[i] * t[j]
    return dtdx_t


@ti.func
def compute_d_dtdx(d: ti.template(), cord: ti.template()) -> ti.Vector:
    """
    Compute distance gradient: sum(cord[i] * d[3i:3i+3])

    Sums the position displacement weighted by barycentric coords.

    Args:
        d: 12-vector of position displacements
        cord: 4-vector of barycentric coordinates

    Returns:
        3-vector: sum of weighted displacements
    """
    result = ti.Vector.zero(ti.f32, 3)
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            result[j] += cord[i] * d[i * 3 + j]
    return result
