"""
Contact Assembly Utilities for MAS Preconditioner.

This module provides barrier function implementations and utility functions
for contact Hessian assembly.

Barrier functions:
- Log barrier (default): -kappa * (d - dHat)^2 * log(d / dHat)
- Cubic barrier: -2*kappa/(3*dHat) * (d - dHat)^3

SPD Barrier Hessian (PPF-Contact-Solver style):
- Contact Hessian is guaranteed PSD by construction
- H = curvature * (e ⊗ e^T) / ||e||², where curvature >= 0 by design
- No eigenvalue decomposition needed for contact barrier

Friction Hessian:
- H = λ * P, where λ >= 0 and P = I - n⊗n^T is projection matrix
- P is PSD (eigenvalues: 1, 1, 0), so friction Hessian is PSD
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


# ============================================================================
# SPD Barrier Hessian (PPF-Contact-Solver style)
# ============================================================================
# Key insight: Instead of computing H = para0 * (t⊗t^T) + para * I and then
# projecting to SPD, we use a different formulation that is SPD by design:
#
# H = curvature * (e ⊗ e^T) / ||e||²
#
# where:
# - e is the contact edge vector (not normalized)
# - curvature = d²ψ/dg² is the barrier curvature (always non-negative)
#
# This is a rank-1 PSD matrix scaled by a non-negative curvature.
# ============================================================================

@ti.func
def barrier_curvature_cubic(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Cubic barrier curvature (second derivative): c = 4*kappa * (1 - d/dHat)

    This is always non-negative when d < dHat (contact zone).
    Reference: PPF-Contact-Solver cubic.hpp
    """
    c = 0.0
    if d < dHat and d > 0.0:
        c = 4.0 * kappa * (1.0 - d / dHat)
    return c


@ti.func
def barrier_curvature_log(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.f32:
    """
    Log barrier curvature (second derivative).

    For log barrier: E = -kappa * (d - dHat)^2 * log(d / dHat)
    Curvature: c = kappa * [(-2) * log(d/dHat) - 4 + 4*dHat/d + (d-dHat)^2 / d^2]

    Note: This can be negative in some regions. For SPD assembly, we clamp to >= 0.
    """
    c = 0.0
    if d < dHat and d > 1e-10:
        c = kappa * ((-2.0) * ti.log(d / dHat) - 4.0 + 4.0 * dHat / d + (d - dHat) ** 2 / (d * d))
        c = ti.max(c, 0.0)  # Clamp to ensure non-negative
    return c


@ti.func
def compute_spd_contact_hessian_3x3(
    e: ti.template(),
    curvature: ti.f32,
) -> ti.Matrix:
    """
    Compute SPD 3x3 contact Hessian using PPF-Contact-Solver formula.

    H = curvature * (e ⊗ e^T) / ||e||²

    This is guaranteed PSD:
    - Rank-1 matrix e⊗e^T has eigenvalues [||e||², 0, 0]
    - Divided by ||e||² gives eigenvalues [1, 0, 0]
    - Scaled by non-negative curvature gives [curvature, 0, 0]

    Args:
        e: Contact edge vector (3D)
        curvature: Barrier curvature (non-negative)

    Returns:
        3x3 PSD matrix
    """
    H = ti.Matrix.zero(ti.f32, 3, 3)
    e_sqr = e.norm_sqr()
    if e_sqr > 1e-12 and curvature > 0.0:
        scale = curvature / e_sqr
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                H[i, j] = scale * e[i] * e[j]
    return H


@ti.func
def compute_spd_edge_gradient(
    e: ti.template(),
    dHat: ti.f32,
    kappa: ti.f32,
    use_cubic: ti.template(),
) -> ti.Vector:
    """
    Compute contact edge gradient: g = n * gradient(||e||)

    where n = e / ||e|| is the unit normal.

    Args:
        e: Contact edge vector
        dHat: Distance threshold
        kappa: Barrier stiffness
        use_cubic: Whether to use cubic barrier (otherwise log)

    Returns:
        3D gradient vector
    """
    g = ti.Vector.zero(ti.f32, 3)
    d = e.norm()
    if d > 1e-10:
        n = e / d
        grad_scalar = 0.0
        if ti.static(use_cubic):
            grad_scalar = barrier_g_cubic(d, dHat, kappa)
        else:
            grad_scalar = barrier_g_log(d, dHat, kappa)
        g = n * grad_scalar
    return g


@ti.func
def compute_spd_edge_hessian(
    e: ti.template(),
    dHat: ti.f32,
    kappa: ti.f32,
    use_cubic: ti.template(),
) -> ti.Matrix:
    """
    Compute SPD contact edge Hessian.

    H = curvature * (e ⊗ e^T) / ||e||²

    This formulation guarantees PSD by construction.
    Reference: PPF-Contact-Solver barrier.cu compute_edge_hessian()

    Args:
        e: Contact edge vector
        dHat: Distance threshold
        kappa: Barrier stiffness
        use_cubic: Whether to use cubic barrier (otherwise log)

    Returns:
        3x3 PSD Hessian matrix
    """
    d = e.norm()
    c = 0.0
    if ti.static(use_cubic):
        c = barrier_curvature_cubic(d, dHat, kappa)
    else:
        c = barrier_curvature_log(d, dHat, kappa)
    return compute_spd_contact_hessian_3x3(e, c)


# ============================================================================
# Friction Hessian (PPF-Contact-Solver style)
# ============================================================================
# Friction contributes a PSD Hessian: H = λ * P
# where:
# - λ = μ * |f_contact| / max(ε, ||P*dx||) >= 0 is the friction multiplier
# - P = I - n⊗n^T is the tangent projection matrix
# - P is PSD with eigenvalues [1, 1, 0]
#
# Reference: PPF-Contact-Solver friction.hpp
# ============================================================================

@ti.func
def compute_friction_projection_matrix(n: ti.template()) -> ti.Matrix:
    """
    Compute tangent space projection matrix: P = I - n⊗n^T

    This projects vectors onto the tangent plane perpendicular to n.
    P is positive semi-definite with eigenvalues [1, 1, 0].

    Args:
        n: Unit normal vector (must be normalized)

    Returns:
        3x3 projection matrix
    """
    P = ti.Matrix.identity(ti.f32, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            P[i, j] -= n[i] * n[j]
    return P


@ti.func
def compute_friction_lambda(
    contact_force: ti.template(),
    dx: ti.template(),
    n: ti.template(),
    mu: ti.f32,
    min_dx: ti.f32,
) -> ti.f32:
    """
    Compute friction multiplier λ = μ * |f_n| / max(ε, ||P*dx||)

    where f_n = -n·f_contact is the normal contact force magnitude.

    Args:
        contact_force: Contact force vector (points into surface)
        dx: Relative displacement vector
        n: Unit contact normal
        mu: Friction coefficient
        min_dx: Minimum displacement for regularization

    Returns:
        Non-negative friction multiplier
    """
    lam = 0.0

    # Normal contact force magnitude: f_n = -n·f_contact
    # contact_force points outward (repulsive), so dot with n gives negative value
    # We want the magnitude of normal force
    f_n = -n.dot(contact_force)
    f_n = ti.max(f_n, 0.0)  # Ensure non-negative (only friction when in contact)

    if mu > 0.0 and f_n > 0.0:
        # Compute tangent displacement: P * dx
        P = compute_friction_projection_matrix(n)
        P_dx = P @ dx
        P_dx_norm = P_dx.norm()

        # λ = μ * f_n / max(ε, ||P*dx||)
        denom = ti.max(min_dx, P_dx_norm)
        lam = mu * f_n / denom

    return lam


@ti.func
def compute_friction_gradient(
    contact_force: ti.template(),
    dx: ti.template(),
    n: ti.template(),
    mu: ti.f32,
    min_dx: ti.f32,
) -> ti.Vector:
    """
    Compute friction gradient: g = λ * P * dx

    Args:
        contact_force: Contact force vector
        dx: Relative displacement vector
        n: Unit contact normal
        mu: Friction coefficient
        min_dx: Minimum displacement for regularization

    Returns:
        3D friction gradient vector
    """
    lam = compute_friction_lambda(contact_force, dx, n, mu, min_dx)
    P = compute_friction_projection_matrix(n)
    return lam * (P @ dx)


@ti.func
def compute_friction_hessian(
    contact_force: ti.template(),
    dx: ti.template(),
    n: ti.template(),
    mu: ti.f32,
    min_dx: ti.f32,
) -> ti.Matrix:
    """
    Compute friction Hessian: H = λ * P

    This is guaranteed PSD:
    - λ >= 0 (non-negative friction multiplier)
    - P = I - n⊗n^T is PSD (eigenvalues: 1, 1, 0)

    Args:
        contact_force: Contact force vector
        dx: Relative displacement vector
        n: Unit contact normal
        mu: Friction coefficient
        min_dx: Minimum displacement for regularization

    Returns:
        3x3 PSD friction Hessian
    """
    lam = compute_friction_lambda(contact_force, dx, n, mu, min_dx)
    P = compute_friction_projection_matrix(n)
    return lam * P


# ============================================================================
# Combined SPD Contact + Friction Hessian
# ============================================================================

@ti.func
def compute_spd_contact_friction_hessian(
    e: ti.template(),
    dx: ti.template(),
    dHat: ti.f32,
    kappa: ti.f32,
    mu: ti.f32,
    min_dx: ti.f32,
    use_cubic: ti.template(),
) -> ti.Matrix:
    """
    Compute combined SPD contact + friction Hessian.

    H_total = H_barrier + H_friction

    Both components are guaranteed PSD:
    - H_barrier = curvature * (e⊗e^T) / ||e||²
    - H_friction = λ * P

    Args:
        e: Contact edge vector
        dx: Relative displacement (for friction)
        dHat: Distance threshold
        kappa: Barrier stiffness
        mu: Friction coefficient (0 to disable friction)
        min_dx: Minimum displacement for friction regularization
        use_cubic: Whether to use cubic barrier

    Returns:
        3x3 PSD Hessian matrix
    """
    # Barrier Hessian
    H = compute_spd_edge_hessian(e, dHat, kappa, use_cubic)

    # Add friction if enabled
    if mu > 0.0:
        d = e.norm()
        if d > 1e-10:
            n = e / d
            # Contact force (gradient)
            f_contact = compute_spd_edge_gradient(e, dHat, kappa, use_cubic)
            H_friction = compute_friction_hessian(f_contact, dx, n, mu, min_dx)
            H = H + H_friction

    return H


@ti.func
def compute_spd_contact_friction_gradient(
    e: ti.template(),
    dx: ti.template(),
    dHat: ti.f32,
    kappa: ti.f32,
    mu: ti.f32,
    min_dx: ti.f32,
    use_cubic: ti.template(),
) -> ti.Vector:
    """
    Compute combined contact + friction gradient.

    g_total = g_barrier + g_friction

    Args:
        e: Contact edge vector
        dx: Relative displacement (for friction)
        dHat: Distance threshold
        kappa: Barrier stiffness
        mu: Friction coefficient (0 to disable friction)
        min_dx: Minimum displacement for friction regularization
        use_cubic: Whether to use cubic barrier

    Returns:
        3D gradient vector
    """
    # Barrier gradient
    g = compute_spd_edge_gradient(e, dHat, kappa, use_cubic)

    # Add friction if enabled
    if mu > 0.0:
        d = e.norm()
        if d > 1e-10:
            n = e / d
            g_friction = compute_friction_gradient(g, dx, n, mu, min_dx)
            g = g + g_friction

    return g


# ============================================================================
# Extended SPD Contact Hessian (12x12 for 4 vertices)
# ============================================================================

@ti.func
def extend_spd_contact_hessian_12x12(
    H_3x3: ti.template(),
    cord: ti.template(),
) -> ti.Matrix:
    """
    Extend 3x3 contact Hessian to 12x12 for 4 contact vertices.

    H_12x12[3i:3i+3, 3j:3j+3] = cord[i] * cord[j] * H_3x3

    This preserves PSD property: if H_3x3 is PSD, then H_12x12 is PSD.

    Args:
        H_3x3: 3x3 contact Hessian (PSD)
        cord: 4-vector of barycentric coordinates

    Returns:
        12x12 PSD Hessian matrix
    """
    H_12x12 = ti.Matrix.zero(ti.f32, 12, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            coeff = cord[i] * cord[j]
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    H_12x12[3*i + di, 3*j + dj] = coeff * H_3x3[di, dj]
    return H_12x12


@ti.func
def extend_spd_contact_gradient_12(
    g_3: ti.template(),
    cord: ti.template(),
) -> ti.Vector:
    """
    Extend 3D contact gradient to 12D for 4 contact vertices.

    g_12[3i:3i+3] = cord[i] * g_3

    Args:
        g_3: 3D contact gradient
        cord: 4-vector of barycentric coordinates

    Returns:
        12D gradient vector
    """
    g_12 = ti.Vector.zero(ti.f32, 12)
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            g_12[3*i + j] = cord[i] * g_3[j]
    return g_12
