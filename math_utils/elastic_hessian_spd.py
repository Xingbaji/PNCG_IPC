"""
Elastic Hessian SPD Projection using Taichi

Implementation of eigenanalysis-based SPD projection for 3D elastic energy Hessian.
Based on the SVD-space eigenvalue decomposition approach.

Reference: ppf-contact-solver/src/cpp/eigenanalysis/eigenanalysis.cu
"""

import taichi as ti
import taichi.math as tm


@ti.dataclass
class Svd3x3:
    """SVD decomposition result for 3x3 matrix: F = U @ diag(S) @ Vt"""
    U: ti.math.mat3
    S: ti.math.vec3  # Singular values (σ₁, σ₂, σ₃)
    Vt: ti.math.mat3


@ti.dataclass
class DiffTable3:
    """Energy derivatives with respect to singular values"""
    deda: ti.math.vec3      # ∂E/∂σᵢ (first derivatives)
    d2ed2a: ti.math.mat3    # ∂²E/∂σᵢ∂σⱼ (second derivatives)


@ti.func
def solve_symm_eigen3x3(A: ti.math.mat3) -> tuple[ti.math.vec3, ti.math.mat3]:
    """
    Compute eigenvalues and eigenvectors of a 3x3 symmetric matrix.
    Uses analytical solution based on Cardano's formula.

    Returns:
        eigenvalues: vec3 of eigenvalues (sorted)
        eigenvectors: mat3 where each column is an eigenvector
    """
    # For symmetric matrix, use iterative Jacobi method or analytical solution
    # Here we use a simplified approach with ti.sym_eig if available,
    # otherwise fall back to power iteration

    eigenvalues = ti.math.vec3(0.0)
    eigenvectors = ti.math.mat3(0.0)

    # Analytical eigenvalue computation for 3x3 symmetric matrix
    # Using Cardano's formula
    p1 = A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2]

    if p1 < 1e-12:
        # A is diagonal
        eigenvalues[0] = A[0, 0]
        eigenvalues[1] = A[1, 1]
        eigenvalues[2] = A[2, 2]
        eigenvectors = ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        q = (A[0, 0] + A[1, 1] + A[2, 2]) / 3.0
        p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2.0 * p1
        p = ti.sqrt(p2 / 6.0)

        B = (A - q * ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) / p

        # det(B) / 2
        det_B = (B[0, 0] * (B[1, 1] * B[2, 2] - B[1, 2] * B[2, 1])
                - B[0, 1] * (B[1, 0] * B[2, 2] - B[1, 2] * B[2, 0])
                + B[0, 2] * (B[1, 0] * B[2, 1] - B[1, 1] * B[2, 0]))
        r = det_B / 2.0

        # Clamp r to [-1, 1] for numerical stability
        r = ti.max(-1.0, ti.min(1.0, r))

        phi = ti.acos(r) / 3.0

        # Eigenvalues in decreasing order
        eigenvalues[0] = q + 2.0 * p * ti.cos(phi)
        eigenvalues[2] = q + 2.0 * p * ti.cos(phi + 2.0 * tm.pi / 3.0)
        eigenvalues[1] = 3.0 * q - eigenvalues[0] - eigenvalues[2]

        # Compute eigenvectors using inverse iteration
        for i in ti.static(range(3)):
            eigenvectors[:, i] = compute_eigenvector_3x3(A, eigenvalues[i])

    return eigenvalues, eigenvectors


@ti.func
def compute_eigenvector_3x3(A: ti.math.mat3, eigenvalue: ti.types.f32) -> ti.math.vec3:
    """Compute eigenvector for a given eigenvalue using cross product method"""
    # A - λI
    B = A - eigenvalue * ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Find two rows with largest norm and take their cross product
    row0 = ti.math.vec3(B[0, 0], B[0, 1], B[0, 2])
    row1 = ti.math.vec3(B[1, 0], B[1, 1], B[1, 2])
    row2 = ti.math.vec3(B[2, 0], B[2, 1], B[2, 2])

    n0 = row0.norm_sqr()
    n1 = row1.norm_sqr()
    n2 = row2.norm_sqr()

    v = ti.math.vec3(0.0)

    if n0 >= n1 and n0 >= n2:
        if n1 >= n2:
            v = row0.cross(row1)
        else:
            v = row0.cross(row2)
    elif n1 >= n0 and n1 >= n2:
        if n0 >= n2:
            v = row1.cross(row0)
        else:
            v = row1.cross(row2)
    else:
        if n0 >= n1:
            v = row2.cross(row0)
        else:
            v = row2.cross(row1)

    # Normalize
    norm = v.norm()
    if norm > 1e-10:
        v = v / norm
    else:
        # Fallback: return unit vector
        v = ti.math.vec3(1.0, 0.0, 0.0)

    return v


@ti.func
def svd3x3(F: ti.math.mat3) -> Svd3x3:
    """
    Compute SVD of 3x3 matrix: F = U @ diag(S) @ Vt
    Using polar decomposition and eigendecomposition.
    """
    # F^T F = V Σ² V^T
    FtF = F.transpose() @ F

    # Eigendecomposition of F^T F
    sigma_sq, V = solve_symm_eigen3x3(FtF)

    # Singular values (clamp to avoid sqrt of negative)
    # Note: eigenvalues should be sorted in decreasing order
    S = ti.math.vec3(
        ti.sqrt(ti.max(0.0, sigma_sq[0])),
        ti.sqrt(ti.max(0.0, sigma_sq[1])),
        ti.sqrt(ti.max(0.0, sigma_sq[2]))
    )

    # U = F V Σ^{-1}
    U = ti.math.mat3(0.0)
    eps = 1e-10
    for i in ti.static(range(3)):
        if S[i] > eps:
            col = (F @ V[:, i]) / S[i]
            # Normalize to ensure orthonormality
            col_norm = col.norm()
            if col_norm > eps:
                col = col / col_norm
            U[:, i] = col
        else:
            # Handle zero singular value - use Gram-Schmidt
            U[:, i] = ti.math.vec3(0.0)
            if i == 0:
                U[:, i] = ti.math.vec3(1.0, 0.0, 0.0)
            elif i == 1:
                # Orthogonalize against U[:,0]
                u0 = U[:, 0]
                u1 = ti.math.vec3(0.0, 1.0, 0.0)
                u1 = u1 - u0.dot(u1) * u0
                u1_norm = u1.norm()
                if u1_norm > eps:
                    u1 = u1 / u1_norm
                else:
                    u1 = ti.math.vec3(0.0, 0.0, 1.0)
                    u1 = u1 - u0.dot(u1) * u0
                    u1 = u1 / u1.norm()
                U[:, i] = u1
            else:
                # Cross product of first two columns
                U[:, i] = U[:, 0].cross(U[:, 1])

    # Ensure proper rotation (det(U) = det(V) = 1)
    det_U = U.determinant()
    if det_U < 0:
        U[:, 2] = -U[:, 2]
        S[2] = -S[2]

    det_V = V.determinant()
    if det_V < 0:
        V[:, 2] = -V[:, 2]
        S[2] = -S[2]

    return Svd3x3(U=U, S=S, Vt=V.transpose())


@ti.func
def get_Q_matrix(idx: ti.i32, eigvectors: ti.math.mat3) -> ti.math.mat3:
    """
    Get the idx-th basis matrix Q for Hessian eigenspace decomposition.

    Q[0-2]: Twist modes (antisymmetric)
    Q[3-5]: Flip modes (symmetric off-diagonal)
    Q[6-8]: Stretch modes (diagonal, from eigenvectors of d²E/d²σ)
    """
    inv_sqrt2 = 0.7071067811865475
    Q = ti.math.mat3(0.0)

    if idx == 0:
        # Twist mode 01
        Q[0, 1] = inv_sqrt2
        Q[1, 0] = -inv_sqrt2
    elif idx == 1:
        # Twist mode 02
        Q[0, 2] = inv_sqrt2
        Q[2, 0] = -inv_sqrt2
    elif idx == 2:
        # Twist mode 12
        Q[1, 2] = inv_sqrt2
        Q[2, 1] = -inv_sqrt2
    elif idx == 3:
        # Flip mode 01
        Q[0, 1] = inv_sqrt2
        Q[1, 0] = inv_sqrt2
    elif idx == 4:
        # Flip mode 02
        Q[0, 2] = inv_sqrt2
        Q[2, 0] = inv_sqrt2
    elif idx == 5:
        # Flip mode 12
        Q[1, 2] = inv_sqrt2
        Q[2, 1] = inv_sqrt2
    elif idx == 6:
        # Stretch mode 0
        for j in ti.static(range(3)):
            Q[j, j] = eigvectors[j, 0]
    elif idx == 7:
        # Stretch mode 1
        for j in ti.static(range(3)):
            Q[j, j] = eigvectors[j, 1]
    else:  # idx == 8
        # Stretch mode 2
        for j in ti.static(range(3)):
            Q[j, j] = eigvectors[j, 2]

    return Q


@ti.func
def compute_hessian_spd_3d(
    table: DiffTable3,
    svd: Svd3x3,
    eps: ti.types.f32
) -> ti.types.matrix(9, 9, ti.f32):
    """
    Compute SPD-projected Hessian of elastic energy in 3D.

    The Hessian is computed in the flattened F space (9x9 matrix).
    Eigenvalue clamping ensures the result is positive semi-definite.

    Args:
        table: Energy derivatives (dE/dσ and d²E/dσ²)
        svd: SVD decomposition of deformation gradient F
        eps: Small value for numerical stability

    Returns:
        9x9 SPD Hessian matrix
    """
    # Compute eigenvalues of d²E/d²σ
    eigvalues, eigvectors = solve_symm_eigen3x3(table.d2ed2a)

    # Singular values
    a = svd.S[0]
    b = svd.S[1]
    c = svd.S[2]

    denom_ab = a - b
    denom_ac = a - c
    denom_bc = b - c

    # Compute 9 eigenvalues with SPD clamping
    # Lambda storage
    lambda_vals = ti.Vector.zero(ti.f32, 9)

    # Twist modes: (∂E/∂σᵢ + ∂E/∂σⱼ) / (σᵢ + σⱼ)
    lambda_vals[0] = ti.max(0.0, (table.deda[0] + table.deda[1]) / (a + b + eps))
    lambda_vals[1] = ti.max(0.0, (table.deda[0] + table.deda[2]) / (a + c + eps))
    lambda_vals[2] = ti.max(0.0, (table.deda[1] + table.deda[2]) / (b + c + eps))

    # Flip modes: (∂E/∂σᵢ - ∂E/∂σⱼ) / (σᵢ - σⱼ)
    # With L'Hôpital rule for degenerate cases
    if ti.abs(denom_ab) > eps:
        lambda_vals[3] = ti.max(0.0, (table.deda[0] - table.deda[1]) / denom_ab)
    else:
        lambda_vals[3] = ti.max(0.0, 0.5 * (table.d2ed2a[0, 0] + table.d2ed2a[1, 1])
                                   - 0.5 * (table.d2ed2a[0, 1] + table.d2ed2a[1, 0]))

    if ti.abs(denom_ac) > eps:
        lambda_vals[4] = ti.max(0.0, (table.deda[0] - table.deda[2]) / denom_ac)
    else:
        lambda_vals[4] = ti.max(0.0, 0.5 * (table.d2ed2a[0, 0] + table.d2ed2a[2, 2])
                                   - 0.5 * (table.d2ed2a[0, 2] + table.d2ed2a[2, 0]))

    if ti.abs(denom_bc) > eps:
        lambda_vals[5] = ti.max(0.0, (table.deda[1] - table.deda[2]) / denom_bc)
    else:
        lambda_vals[5] = ti.max(0.0, 0.5 * (table.d2ed2a[1, 1] + table.d2ed2a[2, 2])
                                   - 0.5 * (table.d2ed2a[1, 2] + table.d2ed2a[2, 1]))

    # Stretch modes: eigenvalues of d²E/d²σ
    lambda_vals[6] = ti.max(0.0, eigvalues[0])
    lambda_vals[7] = ti.max(0.0, eigvalues[1])
    lambda_vals[8] = ti.max(0.0, eigvalues[2])

    # Build SPD Hessian via rank-1 updates: H = Σᵢ λᵢ qᵢ qᵢᵀ
    result = ti.Matrix.zero(ti.f32, 9, 9)

    for i in ti.static(range(9)):
        lam_i = lambda_vals[i]
        if lam_i > 1e-12:
            # Get Q[i] matrix
            Qi = get_Q_matrix(i, eigvectors)

            # Transform to F space: tmp = U @ Qi @ Vt
            tmp = svd.U @ Qi @ svd.Vt

            # Flatten to 9-vector (column-major)
            q = ti.Vector.zero(ti.f32, 9)
            for col in ti.static(range(3)):
                for row in ti.static(range(3)):
                    q[col * 3 + row] = tmp[row, col]

            # Rank-1 update: result += λᵢ q qᵀ
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    result[ii, jj] += lam_i * q[ii] * q[jj]

    # Symmetrize to eliminate numerical errors: H = (H + H^T) / 2
    for ii in ti.static(range(9)):
        for jj in ti.static(range(ii + 1, 9)):
            avg = 0.5 * (result[ii, jj] + result[jj, ii])
            result[ii, jj] = avg
            result[jj, ii] = avg

    return result


@ti.func
def compute_elastic_force_3d(table: DiffTable3, svd: Svd3x3) -> ti.math.mat3:
    """
    Compute elastic force (gradient of energy) in 3D.

    P = U @ diag(∂E/∂σ) @ Vt

    Args:
        table: Energy derivatives
        svd: SVD decomposition

    Returns:
        3x3 First Piola-Kirchhoff stress tensor
    """
    # dE/dF = U @ diag(dE/dσ) @ Vt
    deda_diag = ti.math.mat3([
        [table.deda[0], 0.0, 0.0],
        [0.0, table.deda[1], 0.0],
        [0.0, 0.0, table.deda[2]]
    ])

    P = svd.U @ deda_diag @ svd.Vt
    return P


# ============================================================================
# Material-specific DiffTable computation
# ============================================================================

@ti.func
def compute_difftable_stvk(
    S: ti.math.vec3,
    mu: ti.types.f32,
    lam: ti.types.f32
) -> DiffTable3:
    """
    Compute DiffTable for St. Venant-Kirchhoff material.

    Energy: E = μ ||E||² + (λ/2) tr(E)²
    where E = (FᵀF - I) / 2 is Green strain
    """
    # σᵢ² - 1
    s2 = ti.math.vec3(S[0]*S[0], S[1]*S[1], S[2]*S[2])
    strain = 0.5 * (s2 - ti.math.vec3(1.0, 1.0, 1.0))

    # tr(E) = (σ₁² + σ₂² + σ₃² - 3) / 2
    tr_E = strain[0] + strain[1] + strain[2]

    # ∂E/∂σᵢ = 2μ Eᵢ σᵢ + λ tr(E) σᵢ
    deda = ti.math.vec3(0.0)
    for i in ti.static(range(3)):
        deda[i] = 2.0 * mu * strain[i] * S[i] + lam * tr_E * S[i]

    # ∂²E/∂σᵢ∂σⱼ
    d2ed2a = ti.math.mat3(0.0)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            if i == j:
                # Diagonal: 2μ(3σᵢ² - 1) + λ(3σᵢ² - 1)
                d2ed2a[i, i] = 2.0 * mu * (3.0 * s2[i] - 1.0) + lam * (3.0 * s2[i] - 1.0)
            else:
                # Off-diagonal: λ σᵢ σⱼ
                d2ed2a[i, j] = lam * S[i] * S[j]

    return DiffTable3(deda=deda, d2ed2a=d2ed2a)


@ti.func
def compute_difftable_neohookean(
    S: ti.math.vec3,
    mu: ti.types.f32,
    lam: ti.types.f32
) -> DiffTable3:
    """
    Compute DiffTable for Neo-Hookean material.

    Energy: E = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2) ln²(J)
    where I₁ = σ₁² + σ₂² + σ₃², J = σ₁σ₂σ₃
    """
    J = S[0] * S[1] * S[2]
    log_J = ti.log(ti.max(J, 1e-10))

    # ∂E/∂σᵢ = μ σᵢ - μ/σᵢ + λ ln(J)/σᵢ
    deda = ti.math.vec3(0.0)
    for i in ti.static(range(3)):
        inv_si = 1.0 / ti.max(S[i], 1e-10)
        deda[i] = mu * S[i] - mu * inv_si + lam * log_J * inv_si

    # ∂²E/∂σᵢ∂σⱼ
    d2ed2a = ti.math.mat3(0.0)
    for i in ti.static(range(3)):
        inv_si = 1.0 / ti.max(S[i], 1e-10)
        inv_si2 = inv_si * inv_si

        # Diagonal
        d2ed2a[i, i] = mu + mu * inv_si2 + lam * (1.0 - log_J) * inv_si2

        # Off-diagonal
        for j in ti.static(range(3)):
            if i != j:
                inv_sj = 1.0 / ti.max(S[j], 1e-10)
                d2ed2a[i, j] = lam * inv_si * inv_sj

    return DiffTable3(deda=deda, d2ed2a=d2ed2a)


@ti.func
def compute_difftable_arap(
    S: ti.math.vec3,
    mu: ti.types.f32
) -> DiffTable3:
    """
    Compute DiffTable for ARAP (As-Rigid-As-Possible) material.

    Energy: E = μ ||F - R||² = μ Σᵢ(σᵢ - 1)²
    """
    # ∂E/∂σᵢ = 2μ (σᵢ - 1)
    deda = ti.math.vec3(0.0)
    for i in ti.static(range(3)):
        deda[i] = 2.0 * mu * (S[i] - 1.0)

    # ∂²E/∂σᵢ∂σⱼ = 2μ δᵢⱼ
    d2ed2a = ti.math.mat3([
        [2.0 * mu, 0.0, 0.0],
        [0.0, 2.0 * mu, 0.0],
        [0.0, 0.0, 2.0 * mu]
    ])

    return DiffTable3(deda=deda, d2ed2a=d2ed2a)


# ============================================================================
# High-level API
# ============================================================================

@ti.func
def compute_element_hessian_spd(
    F: ti.math.mat3,
    mu: ti.types.f32,
    lam: ti.types.f32,
    material: ti.i32,  # 0: StVK, 1: NeoHookean, 2: ARAP
    eps: ti.types.f32
) -> ti.types.matrix(9, 9, ti.f32):
    """
    Compute SPD-projected element Hessian for a 3D elastic element.

    Args:
        F: 3x3 deformation gradient
        mu: First Lame parameter (shear modulus)
        lam: Second Lame parameter
        material: Material type (0=StVK, 1=NeoHookean, 2=ARAP)
        eps: Numerical stability threshold

    Returns:
        9x9 SPD Hessian in flattened F space
    """
    # Compute SVD
    svd = svd3x3(F)

    # Compute material-specific DiffTable
    table = DiffTable3(deda=ti.math.vec3(0.0), d2ed2a=ti.math.mat3(0.0))

    if material == 0:
        table = compute_difftable_stvk(svd.S, mu, lam)
    elif material == 1:
        table = compute_difftable_neohookean(svd.S, mu, lam)
    else:
        table = compute_difftable_arap(svd.S, mu)

    # Compute SPD Hessian
    H = compute_hessian_spd_3d(table, svd, eps)

    return H


@ti.func
def compute_element_force(
    F: ti.math.mat3,
    mu: ti.types.f32,
    lam: ti.types.f32,
    material: ti.i32
) -> ti.math.mat3:
    """
    Compute element force (First Piola-Kirchhoff stress).

    Args:
        F: 3x3 deformation gradient
        mu: First Lame parameter
        lam: Second Lame parameter
        material: Material type

    Returns:
        3x3 stress tensor P
    """
    svd = svd3x3(F)

    table = DiffTable3(deda=ti.math.vec3(0.0), d2ed2a=ti.math.mat3(0.0))

    if material == 0:
        table = compute_difftable_stvk(svd.S, mu, lam)
    elif material == 1:
        table = compute_difftable_neohookean(svd.S, mu, lam)
    else:
        table = compute_difftable_arap(svd.S, mu)

    P = compute_elastic_force_3d(table, svd)
    return P


# ============================================================================
# Testing utilities
# ============================================================================

@ti.kernel
def test_hessian_spd_kernel(
    F: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mu: ti.f32,
    lam: ti.f32,
    material: ti.i32,
    H_out: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    """Test kernel to compute Hessian for a single element"""
    F_mat = ti.math.mat3([
        [F[0, 0], F[0, 1], F[0, 2]],
        [F[1, 0], F[1, 1], F[1, 2]],
        [F[2, 0], F[2, 1], F[2, 2]]
    ])

    H = compute_element_hessian_spd(F_mat, mu, lam, material, 1e-6)

    for i in range(9):
        for j in range(9):
            H_out[i, j] = H[i, j]


@ti.kernel
def test_force_kernel(
    F: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mu: ti.f32,
    lam: ti.f32,
    material: ti.i32,
    P_out: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    """Test kernel to compute force for a single element"""
    F_mat = ti.math.mat3([
        [F[0, 0], F[0, 1], F[0, 2]],
        [F[1, 0], F[1, 1], F[1, 2]],
        [F[2, 0], F[2, 1], F[2, 2]]
    ])

    P = compute_element_force(F_mat, mu, lam, material)

    for i in range(3):
        for j in range(3):
            P_out[i, j] = P[i, j]


def test_spd_projection():
    """Test that the projected Hessian is indeed SPD"""
    import numpy as np

    ti.init(arch=ti.cpu)

    # Test with a deformed configuration
    F = np.array([
        [1.2, 0.1, 0.05],
        [0.1, 0.9, 0.1],
        [0.05, 0.1, 1.1]
    ], dtype=np.float32)

    H_out = np.zeros((9, 9), dtype=np.float32)

    mu = 1e6
    lam = 1e6

    # Relative tolerance for SPD check (relative to max eigenvalue)
    rel_tol = 1e-6

    all_passed = True

    # Test StVK
    print("Testing StVK material...")
    test_hessian_spd_kernel(F, mu, lam, 0, H_out)

    # Check symmetry
    sym_error = np.max(np.abs(H_out - H_out.T))
    print(f"  Symmetry error: {sym_error:.2e}")

    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(H_out)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    spd_ok = min_eig >= -rel_tol * max_eig
    print(f"  Min eigenvalue: {min_eig:.2e}")
    print(f"  Max eigenvalue: {max_eig:.2e}")
    print(f"  SPD (relative): {spd_ok}")
    all_passed = all_passed and spd_ok

    # Test NeoHookean
    print("\nTesting NeoHookean material...")
    test_hessian_spd_kernel(F, mu, lam, 1, H_out)

    eigenvalues = np.linalg.eigvalsh(H_out)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    spd_ok = min_eig >= -rel_tol * max_eig
    print(f"  Min eigenvalue: {min_eig:.2e}")
    print(f"  Max eigenvalue: {max_eig:.2e}")
    print(f"  SPD (relative): {spd_ok}")
    all_passed = all_passed and spd_ok

    # Test ARAP
    print("\nTesting ARAP material...")
    test_hessian_spd_kernel(F, mu, lam, 2, H_out)

    eigenvalues = np.linalg.eigvalsh(H_out)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    spd_ok = min_eig >= -rel_tol * max_eig
    print(f"  Min eigenvalue: {min_eig:.2e}")
    print(f"  Max eigenvalue: {max_eig:.2e}")
    print(f"  SPD (relative): {spd_ok}")
    all_passed = all_passed and spd_ok

    # Test with more extreme deformation
    print("\n\nTesting with extreme deformation (compression)...")
    F_extreme = np.array([
        [0.5, 0.2, 0.1],
        [0.2, 0.3, 0.15],
        [0.1, 0.15, 0.4]
    ], dtype=np.float32)

    for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
        test_hessian_spd_kernel(F_extreme, mu, lam, material, H_out)
        eigenvalues = np.linalg.eigvalsh(H_out)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        spd_ok = min_eig >= -rel_tol * max_eig
        print(f"  {name}: min_eig={min_eig:.2e}, max_eig={max_eig:.2e}, SPD={spd_ok}")
        all_passed = all_passed and spd_ok

    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")


if __name__ == "__main__":
    test_spd_projection()
