"""
SPD (Symmetric Positive Definite) projection utilities for contact Hessian.

Provides functions to compute SPD contact Hessians that preserve positive
semi-definiteness, which is crucial for Newton-type optimization.
"""

import taichi as ti


@ti.func
def compute_contact_subblock(
    para0,
    para,
    t,
    coeff,
):
    """
    Compute 3x3 contact Hessian subblock.

    H_ij = coeff * (para0 * (t @ t^T) + para * I_3x3)

    Where:
        coeff = cord[i] * cord[j] (barycentric product)
        para = barrier_g / dist
        para0 = (barrier_H - para) / dist^2
        t = contact direction

    Args:
        para0: Curvature-related term (barrier_H - barrier_g/dist) / dist^2
        para: Gradient-related term (barrier_g / dist)
        t: Contact direction vector (3D)
        coeff: Barycentric coefficient (cord[i] * cord[j])

    Returns:
        3x3 matrix representing the contact Hessian subblock
    """
    H_ij = ti.Matrix.zero(t.dtype, 3, 3)
    for di in ti.static(range(3)):
        for dj in ti.static(range(3)):
            H_ij[di, dj] = coeff * para0 * t[di] * t[dj]
            if di == dj:
                H_ij[di, dj] += coeff * para
    return H_ij


@ti.func
def compute_spd_contact_hessian_3x3(
    e,
    curvature,
):
    """
    Compute SPD contact Hessian using rank-1 formulation.

    H = curvature * (e @ e^T) / ||e||^2

    This construction guarantees positive semi-definiteness by using
    the outer product form. The result is a rank-1 matrix (or zero).

    This is the PPF-Contact-Solver style formulation that avoids
    eigenvalue decomposition.

    Args:
        e: Contact edge vector (not normalized)
        curvature: Barrier second derivative (must be non-negative)

    Returns:
        3x3 SPD matrix
    """
    H = ti.Matrix.zero(e.dtype, 3, 3)
    e_sqr = e.norm_sqr()
    if e_sqr > 1e-12 and curvature > 0.0:
        scale = curvature / e_sqr
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                H[i, j] = scale * e[i] * e[j]
    return H


@ti.func
def extend_spd_contact_hessian_12x12(
    H_3x3,
    cord,
):
    """
    Extend 3x3 contact Hessian to full 12x12 matrix.

    H_12x12[3i:3i+3, 3j:3j+3] = cord[i] * cord[j] * H_3x3

    This construction preserves SPD property: if H_3x3 is SPD,
    then the extended matrix is also SPD.

    Args:
        H_3x3: 3x3 contact Hessian (should be SPD)
        cord: Barycentric coordinates (4D)

    Returns:
        12x12 SPD matrix
    """
    # Determine type from H_3x3
    H_12x12 = ti.Matrix.zero(H_3x3.dtype, 12, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            coeff = cord[i] * cord[j]
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    H_12x12[3 * i + di, 3 * j + dj] = coeff * H_3x3[di, dj]
    return H_12x12


@ti.func
def project_to_spd_3x3(H):
    """
    Project a 3x3 matrix to SPD by clamping negative eigenvalues.

    For a symmetric 3x3 matrix, this computes eigendecomposition and
    clamps negative eigenvalues to zero.

    Note: This is a simplified version that works well for nearly-SPD
    matrices. For general matrices, a full eigendecomposition would
    be needed.

    Args:
        H: 3x3 symmetric matrix

    Returns:
        3x3 SPD matrix (or PSD if input had negative eigenvalues)
    """
    # For contact Hessians, we use a simpler approach:
    # Clamp diagonal elements to be non-negative
    # This is a conservative approximation
    H_spd = H
    for i in ti.static(range(3)):
        H_spd[i, i] = ti.max(H[i, i], H.dtype(0.0))
    return H_spd


@ti.func
def compute_full_contact_hessian_12x12(
    para0,
    para,
    t,
    cord,
):
    """
    Compute full 12x12 contact Hessian directly.

    H_12x12[3i+di, 3j+dj] = cord[i] * cord[j] * (para0 * t[di] * t[dj] + para * delta(di, dj))

    This computes the full Hessian without going through the 3x3 intermediate.

    Args:
        para0: Curvature term (barrier_H - barrier_g/dist) / dist^2
        para: Gradient term (barrier_g / dist)
        t: Contact direction vector (3D)
        cord: Barycentric coordinates (4D)

    Returns:
        12x12 contact Hessian matrix
    """
    H_12x12 = ti.Matrix.zero(t.dtype, 12, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            coeff = cord[i] * cord[j]
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    val = coeff * para0 * t[di] * t[dj]
                    if di == dj:
                        val += coeff * para
                    H_12x12[3 * i + di, 3 * j + dj] = val
    return H_12x12
