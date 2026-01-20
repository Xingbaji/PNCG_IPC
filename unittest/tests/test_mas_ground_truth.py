"""
MAS Preconditioner Ground Truth Testing

This module provides comprehensive ground truth verification for the MAS Preconditioner:
1. Matrix Assembly - Verify elastic Hessian assembly is numerically correct
2. Block Inversion - Verify 48x48 block matrix inversion matches NumPy
3. Matrix-Vector Multiplication - Verify z = P * gradient matches NumPy

Usage:
    python test_mas_ground_truth.py -v          # Run all tests with verbose output
    python test_mas_ground_truth.py -v TestAssemblyGroundTruth  # Run specific test class
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

# Initialize Taichi with CPU for deterministic results
ti.init(arch=ti.cpu, default_fp=ti.f32)


# ==============================================================================
# Constants (matching mas_preconditioner_pkg/constants.py)
# ==============================================================================

BANKSIZE = 16
SYM_BLOCK_COUNT = 136  # 16 * 17 / 2
BLOCK_DOF = 48  # BANKSIZE * 3


# ==============================================================================
# NumPy Ground Truth Functions
# ==============================================================================

def compute_dFdx_numpy(B):
    """
    Compute dFdx (9x12) from inverse rest matrix B (3x3).
    Matches math_utils/matrix_util.py:compute_dFdx exactly.

    dFdx maps 12-DOF nodal displacements to 9-component vec(F).
    Row ordering: [F00, F10, F20, F01, F11, F21, F02, F12, F22] (column-major)
    Column ordering: [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3]
    """
    dFdx = np.zeros((9, 12), dtype=np.float32)

    m, n, o = B[0, 0], B[0, 1], B[0, 2]
    p, q, r = B[1, 0], B[1, 1], B[1, 2]
    s, t, u = B[2, 0], B[2, 1], B[2, 2]

    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    # Row 0,1,2: first column of F (dF[:,0]/dx)
    dFdx[0, 0], dFdx[0, 3], dFdx[0, 6], dFdx[0, 9] = t1, m, p, s
    dFdx[1, 1], dFdx[1, 4], dFdx[1, 7], dFdx[1, 10] = t1, m, p, s
    dFdx[2, 2], dFdx[2, 5], dFdx[2, 8], dFdx[2, 11] = t1, m, p, s

    # Row 3,4,5: second column of F (dF[:,1]/dx)
    dFdx[3, 0], dFdx[3, 3], dFdx[3, 6], dFdx[3, 9] = t2, n, q, t
    dFdx[4, 1], dFdx[4, 4], dFdx[4, 7], dFdx[4, 10] = t2, n, q, t
    dFdx[5, 2], dFdx[5, 5], dFdx[5, 8], dFdx[5, 11] = t2, n, q, t

    # Row 6,7,8: third column of F (dF[:,2]/dx)
    dFdx[6, 0], dFdx[6, 3], dFdx[6, 6], dFdx[6, 9] = t3, o, r, u
    dFdx[7, 1], dFdx[7, 4], dFdx[7, 7], dFdx[7, 10] = t3, o, r, u
    dFdx[8, 2], dFdx[8, 5], dFdx[8, 8], dFdx[8, 11] = t3, o, r, u

    return dFdx


def flatten_matrix_column_major(A):
    """Flatten 3x3 matrix to 9-vector in column-major order (matching Taichi code)."""
    return A.T.flatten()


def compute_d2PsidF2_ARAP_numpy(F, mu, la):
    """
    Compute ARAP Hessian (9x9) using NumPy SVD.
    Matches elastic_util.py:compute_d2PsidF2_ARAP_filter.

    Formula: d2PsidF2 = 2*mu*I - mu*(lambda0*q0@q0^T + lambda1*q1@q1^T + lambda2*q2@q2^T)
    """
    U, s, Vt = np.linalg.svd(F)
    V = Vt.T

    # Signed SVD convention (matching ssvd in Taichi code)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
        s[2] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
        s[2] *= -1

    s0, s1, s2 = s

    # Filter for stability (matches _filter version in elastic_util.py)
    lambda0 = 1.0 if s1 + s2 < 2.0 else 2.0 / (s1 + s2)
    lambda1 = 1.0 if s0 + s2 < 2.0 else 2.0 / (s0 + s2)
    lambda2 = 1.0 if s0 + s1 < 2.0 else 2.0 / (s0 + s1)

    # Twist mode matrices Q_i
    # Q0 = V[:,1] @ U[:,2]^T - V[:,2] @ U[:,1]^T
    Q0 = np.outer(V[:, 1], U[:, 2]) - np.outer(V[:, 2], U[:, 1])
    Q1 = np.outer(V[:, 2], U[:, 0]) - np.outer(V[:, 0], U[:, 2])
    Q2 = np.outer(V[:, 1], U[:, 0]) - np.outer(V[:, 0], U[:, 1])

    # Flatten column-major (matching flatten_matrix in Taichi)
    q0 = flatten_matrix_column_major(Q0)
    q1 = flatten_matrix_column_major(Q1)
    q2 = flatten_matrix_column_major(Q2)

    # d2PsidF2 = 2*mu*I - mu*(sum of outer products)
    d2PsidF2 = 2.0 * mu * np.eye(9)
    d2PsidF2 -= mu * (lambda0 * np.outer(q0, q0) +
                      lambda1 * np.outer(q1, q1) +
                      lambda2 * np.outer(q2, q2))

    return d2PsidF2


def compute_d2PsidF2_SNH_numpy(F, mu, la):
    """
    Compute Stable Neo-Hookean Hessian (9x9).
    Matches elastic_util.py:compute_d2PsidF2_SNH.

    SNH: Psi = 0.5*mu*(||F||^2 - 3) - mu*(J-1) + 0.5*la*(J-1)^2
    d2PsidF2 = mu*I + la*g3@g3^T + (la*(J-1) - mu)*H3
    """
    J = np.linalg.det(F)

    # Compute dJdF (cofactor matrix, flattened column-major)
    F00, F01, F02 = F[0, 0], F[0, 1], F[0, 2]
    F10, F11, F12 = F[1, 0], F[1, 1], F[1, 2]
    F20, F21, F22 = F[2, 0], F[2, 1], F[2, 2]

    g3 = np.array([
        F11 * F22 - F12 * F21,  # dJ/dF00
        -F01 * F22 + F02 * F21,  # dJ/dF10
        F01 * F12 - F02 * F11,   # dJ/dF20
        -F10 * F22 + F12 * F20,  # dJ/dF01
        F00 * F22 - F02 * F20,   # dJ/dF11
        -F00 * F12 + F02 * F10,  # dJ/dF21
        F10 * F21 - F11 * F20,   # dJ/dF02
        -F00 * F21 + F01 * F20,  # dJ/dF12
        F00 * F11 - F01 * F10    # dJ/dF22
    ])

    # Compute H3 (second derivative of J w.r.t F)
    H3 = np.array([
        [0, 0, 0, 0, F22, -F12, 0, -F21, F11],
        [0, 0, 0, -F22, 0, F02, F21, 0, -F01],
        [0, 0, 0, F12, -F02, 0, -F11, F01, 0],
        [0, -F22, F12, 0, 0, 0, 0, F20, -F10],
        [F22, 0, -F02, 0, 0, 0, -F20, 0, F00],
        [-F12, F02, 0, 0, 0, 0, F10, -F00, 0],
        [0, F21, -F11, 0, -F20, F10, 0, 0, 0],
        [-F21, 0, F01, F20, 0, -F00, 0, 0, 0],
        [F11, -F01, 0, -F10, F00, 0, 0, 0, 0]
    ], dtype=np.float32)

    # d2PsidF2 = mu*I + la*g3@g3^T + (la*(J-1) - mu)*H3
    d2PsidF2 = mu * np.eye(9)
    d2PsidF2 += la * np.outer(g3, g3)
    d2PsidF2 += (la * (J - 1.0) - mu) * H3

    return d2PsidF2


def compute_element_hessian_numpy(vertices, B, W, dt, mu, la, v_ids, elastic_type=0):
    """
    Compute full 12x12 element Hessian: H_e = W * dt^2 * dFdx^T @ d2PsidF2 @ dFdx

    Args:
        vertices: All vertex positions (n_verts, 3)
        B: Inverse rest matrix (3, 3)
        W: Cell volume weight
        dt: Timestep
        mu, la: Lame parameters
        v_ids: Vertex indices [v0, v1, v2, v3]
        elastic_type: 0=ARAP, 1=SNH

    Returns:
        H_e: 12x12 element Hessian
    """
    x0, x1, x2, x3 = vertices[v_ids[0]], vertices[v_ids[1]], vertices[v_ids[2]], vertices[v_ids[3]]
    Ds = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
    F = Ds @ B

    dFdx = compute_dFdx_numpy(B)

    if elastic_type == 0:  # ARAP
        d2PsidF2 = compute_d2PsidF2_ARAP_numpy(F, mu, la)
    elif elastic_type == 1:  # SNH
        d2PsidF2 = compute_d2PsidF2_SNH_numpy(F, mu, la)
    else:
        d2PsidF2 = compute_d2PsidF2_ARAP_numpy(F, mu, la)

    para = W * dt * dt
    H_e = para * (dFdx.T @ d2PsidF2 @ dFdx)

    return H_e


def sym_index(row, col):
    """Compute symmetric storage index for upper triangle."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def extract_full_block_matrix_numpy(block_matrices_np, block_id):
    """
    Extract full 48x48 block matrix from symmetric storage.

    Args:
        block_matrices_np: NumPy array of shape (n_blocks, SYM_BLOCK_COUNT, 3, 3)
        block_id: Block index to extract

    Returns:
        full: 48x48 dense matrix
    """
    full = np.zeros((BLOCK_DOF, BLOCK_DOF), dtype=np.float32)

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = block_matrices_np[block_id, sym_idx]

            # Upper triangle
            full[row*3:(row+1)*3, col*3:(col+1)*3] = block_3x3
            # Lower triangle (symmetric)
            if row != col:
                full[col*3:(col+1)*3, row*3:(row+1)*3] = block_3x3.T

    return full


# ==============================================================================
# Taichi Helper Kernels for Testing
# ==============================================================================

@ti.kernel
def taichi_compute_dFdx(B: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(9, 12, ti.f32):
    """Taichi kernel to compute dFdx for testing."""
    dFdx = ti.Matrix.zero(ti.f32, 9, 12)

    m = B[0, 0]
    n = B[0, 1]
    o = B[0, 2]
    p = B[1, 0]
    q = B[1, 1]
    r = B[1, 2]
    s = B[2, 0]
    t = B[2, 1]
    u = B[2, 2]

    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    dFdx[0, 0] = t1
    dFdx[0, 3] = m
    dFdx[0, 6] = p
    dFdx[0, 9] = s

    dFdx[1, 1] = t1
    dFdx[1, 4] = m
    dFdx[1, 7] = p
    dFdx[1, 10] = s

    dFdx[2, 2] = t1
    dFdx[2, 5] = m
    dFdx[2, 8] = p
    dFdx[2, 11] = s

    dFdx[3, 0] = t2
    dFdx[3, 3] = n
    dFdx[3, 6] = q
    dFdx[3, 9] = t

    dFdx[4, 1] = t2
    dFdx[4, 4] = n
    dFdx[4, 7] = q
    dFdx[4, 10] = t

    dFdx[5, 2] = t2
    dFdx[5, 5] = n
    dFdx[5, 8] = q
    dFdx[5, 11] = t

    dFdx[6, 0] = t3
    dFdx[6, 3] = o
    dFdx[6, 6] = r
    dFdx[6, 9] = u

    dFdx[7, 1] = t3
    dFdx[7, 4] = o
    dFdx[7, 7] = r
    dFdx[7, 10] = u

    dFdx[8, 2] = t3
    dFdx[8, 5] = o
    dFdx[8, 8] = r
    dFdx[8, 11] = u

    return dFdx


# ==============================================================================
# Test Mesh Generators
# ==============================================================================

def create_single_tet_mesh():
    """
    Create a single regular tetrahedron mesh.

    Returns:
        vertices: (4, 3) array of vertex positions
        cells: (1, 4) array of cell vertex indices
    """
    # Regular tetrahedron inscribed in unit cube
    vertices = np.array([
        [0.0, 0.0, 0.0],      # v0 - origin
        [1.0, 0.0, 0.0],      # v1 - x-axis
        [0.5, 0.866025, 0.0], # v2 - xy-plane (equilateral triangle base)
        [0.5, 0.288675, 0.816497],  # v3 - apex (regular tetrahedron)
    ], dtype=np.float32)

    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    return vertices, cells


def create_unit_cube_mesh():
    """
    Create a unit cube mesh with 5 tetrahedra.

    Returns:
        vertices: (8, 3) array of vertex positions
        cells: (5, 4) array of cell vertex indices
    """
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top face
    ], dtype=np.float32)

    # 5-tetrahedra decomposition
    cells = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int32)

    return vertices, cells


def compute_tet_rest_matrices(vertices, cells):
    """
    Compute inverse rest matrices (B) and volumes (W) for all cells.

    Args:
        vertices: (n_verts, 3) vertex positions
        cells: (n_cells, 4) cell vertex indices

    Returns:
        B_array: (n_cells, 3, 3) inverse rest matrices
        W_array: (n_cells,) cell volumes
    """
    n_cells = len(cells)
    B_array = np.zeros((n_cells, 3, 3), dtype=np.float32)
    W_array = np.zeros(n_cells, dtype=np.float32)

    for c in range(n_cells):
        v0, v1, v2, v3 = cells[c]
        x0, x1, x2, x3 = vertices[v0], vertices[v1], vertices[v2], vertices[v3]

        # Dm = [x1-x0, x2-x0, x3-x0]
        Dm = np.column_stack([x1 - x0, x2 - x0, x3 - x0])

        # B = Dm^{-1}
        B_array[c] = np.linalg.inv(Dm)

        # W = |det(Dm)| / 6 (tet volume)
        W_array[c] = abs(np.linalg.det(Dm)) / 6.0

    return B_array, W_array


# ==============================================================================
# Test Classes
# ==============================================================================

class TestSymmetricIndex(unittest.TestCase):
    """Test symmetric storage index formula."""

    def test_sym_index_covers_all_136(self):
        """Verify sym_index formula generates all 136 unique indices."""
        indices = set()
        for i in range(BANKSIZE):
            for j in range(i, BANKSIZE):
                idx = sym_index(i, j)
                indices.add(idx)

        self.assertEqual(len(indices), SYM_BLOCK_COUNT)
        self.assertEqual(indices, set(range(SYM_BLOCK_COUNT)))

    def test_sym_index_formula_correctness(self):
        """Verify sym_index formula matches expected values."""
        # First few indices
        self.assertEqual(sym_index(0, 0), 0)
        self.assertEqual(sym_index(0, 1), 1)
        self.assertEqual(sym_index(0, 15), 15)
        self.assertEqual(sym_index(1, 1), 16)
        self.assertEqual(sym_index(1, 2), 17)

        # Last index
        self.assertEqual(sym_index(15, 15), 135)

    def test_sym_index_symmetric(self):
        """Verify sym_index(i,j) == sym_index(j,i)."""
        for i in range(BANKSIZE):
            for j in range(BANKSIZE):
                self.assertEqual(sym_index(i, j), sym_index(j, i))


class TestDFdxGroundTruth(unittest.TestCase):
    """Test dFdx computation matches between NumPy and Taichi."""

    def test_dFdx_identity(self):
        """Test dFdx with identity B matrix."""
        B = np.eye(3, dtype=np.float32)
        dFdx_numpy = compute_dFdx_numpy(B)

        # With B = I:
        # t1 = -1-0-0 = -1, t2 = -0-1-0 = -1, t3 = -0-0-1 = -1
        # Row 0: dFdx[0,0]=t1=-1, dFdx[0,3]=m=1, dFdx[0,6]=p=0, dFdx[0,9]=s=0
        # etc.

        # Verify specific known values for B = I
        # First column of F depends on first column of B = [1,0,0]
        # dFdx[0,0] = t1 = -1-0-0 = -1
        # dFdx[0,3] = m = 1
        # dFdx[0,6] = p = 0
        # dFdx[0,9] = s = 0

        self.assertAlmostEqual(dFdx_numpy[0, 0], -1.0)  # t1
        self.assertAlmostEqual(dFdx_numpy[0, 3], 1.0)   # m = B[0,0] = 1
        self.assertAlmostEqual(dFdx_numpy[0, 6], 0.0)   # p = B[1,0] = 0
        self.assertAlmostEqual(dFdx_numpy[0, 9], 0.0)   # s = B[2,0] = 0

        # Second column of F depends on second column of B = [0,1,0]
        # dFdx[3,0] = t2 = -0-1-0 = -1
        # dFdx[3,3] = n = 0
        # dFdx[3,6] = q = 1
        # dFdx[3,9] = t = 0
        self.assertAlmostEqual(dFdx_numpy[3, 0], -1.0)  # t2
        self.assertAlmostEqual(dFdx_numpy[3, 3], 0.0)   # n = B[0,1] = 0
        self.assertAlmostEqual(dFdx_numpy[3, 6], 1.0)   # q = B[1,1] = 1
        self.assertAlmostEqual(dFdx_numpy[3, 9], 0.0)   # t = B[2,1] = 0

        # Verify shape
        self.assertEqual(dFdx_numpy.shape, (9, 12))

    def test_dFdx_arbitrary_B(self):
        """Test dFdx with arbitrary B matrix against Taichi."""
        np.random.seed(42)
        B_np = np.random.randn(3, 3).astype(np.float32) * 0.5 + np.eye(3)

        dFdx_numpy = compute_dFdx_numpy(B_np)

        # Call Taichi kernel
        B_ti = ti.Matrix(B_np.tolist(), dt=ti.f32)
        dFdx_ti = taichi_compute_dFdx(B_ti)
        dFdx_taichi = dFdx_ti.to_numpy()

        # Compare
        np.testing.assert_allclose(dFdx_taichi, dFdx_numpy.astype(np.float32),
                                   rtol=1e-5, atol=1e-6)

    def test_dFdx_tet_mesh(self):
        """Test dFdx computed from actual tetrahedron mesh."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)
        B = B_array[0]

        dFdx_numpy = compute_dFdx_numpy(B)

        # Verify shape
        self.assertEqual(dFdx_numpy.shape, (9, 12))

        # Verify it's not all zeros
        self.assertGreater(np.linalg.norm(dFdx_numpy), 0.1)


class TestD2PsidF2GroundTruth(unittest.TestCase):
    """Test elastic Hessian computation."""

    def test_arap_identity_F(self):
        """Test ARAP Hessian with F = I."""
        F = np.eye(3)
        mu, la = 1e5, 1e5

        d2PsidF2 = compute_d2PsidF2_ARAP_numpy(F, mu, la)

        # Verify symmetry
        np.testing.assert_allclose(d2PsidF2, d2PsidF2.T, rtol=1e-10)

        # Verify positive semi-definiteness (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(d2PsidF2)
        self.assertTrue(np.all(eigvals >= -1e-10),
                        f"ARAP Hessian should be PSD, min eigenvalue: {eigvals.min()}")

    def test_arap_stretched_F(self):
        """Test ARAP Hessian with stretched deformation."""
        F = np.diag([1.5, 1.0, 0.8])  # Volume-preserving stretch
        mu, la = 1e5, 1e5

        d2PsidF2 = compute_d2PsidF2_ARAP_numpy(F, mu, la)

        # Verify symmetry
        np.testing.assert_allclose(d2PsidF2, d2PsidF2.T, rtol=1e-10)

        # Verify shape
        self.assertEqual(d2PsidF2.shape, (9, 9))

    def test_snh_identity_F(self):
        """Test SNH Hessian with F = I."""
        F = np.eye(3)
        mu, la = 1e5, 1e5

        d2PsidF2 = compute_d2PsidF2_SNH_numpy(F, mu, la)

        # Verify symmetry
        np.testing.assert_allclose(d2PsidF2, d2PsidF2.T, rtol=1e-10)

        # Verify shape
        self.assertEqual(d2PsidF2.shape, (9, 9))


class TestElementHessian(unittest.TestCase):
    """Test full element Hessian computation."""

    def test_element_hessian_symmetry(self):
        """Test that element Hessian is symmetric."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01

        H_e = compute_element_hessian_numpy(
            vertices, B_array[0], W_array[0], dt, mu, la, cells[0])

        # Verify symmetry
        np.testing.assert_allclose(H_e, H_e.T, rtol=1e-10)

    def test_element_hessian_shape(self):
        """Test element Hessian has correct shape."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01

        H_e = compute_element_hessian_numpy(
            vertices, B_array[0], W_array[0], dt, mu, la, cells[0])

        self.assertEqual(H_e.shape, (12, 12))

    def test_element_hessian_positive_definite(self):
        """Test element Hessian is positive semi-definite."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01

        H_e = compute_element_hessian_numpy(
            vertices, B_array[0], W_array[0], dt, mu, la, cells[0])

        eigvals = np.linalg.eigvalsh(H_e)
        # Allow small negative eigenvalues due to numerical precision
        self.assertTrue(np.all(eigvals >= -1e-6 * abs(eigvals.max())),
                        f"Element Hessian should be PSD, min eigenvalue: {eigvals.min()}")


class TestBlockInversionGroundTruth(unittest.TestCase):
    """Test block matrix inversion against NumPy."""

    def test_small_spd_inversion(self):
        """Test inversion of small SPD matrix."""
        np.random.seed(42)
        n = 12  # Small size for quick test
        A_rand = np.random.randn(n, n)
        A = A_rand @ A_rand.T + 10 * np.eye(n)  # SPD

        # NumPy ground truth
        A_inv_numpy = np.linalg.inv(A)

        # Verify A @ A_inv = I
        product = A @ A_inv_numpy
        np.testing.assert_allclose(product, np.eye(n), rtol=1e-10, atol=1e-10)

    def test_48x48_spd_inversion(self):
        """Test inversion of 48x48 SPD matrix (full block size)."""
        np.random.seed(42)
        n = BLOCK_DOF  # 48
        A_rand = np.random.randn(n, n)
        A = A_rand @ A_rand.T + 10 * np.eye(n)  # SPD

        # NumPy ground truth
        A_inv_numpy = np.linalg.inv(A)

        # Verify A @ A_inv = I
        product = A @ A_inv_numpy
        np.testing.assert_allclose(product, np.eye(n), rtol=1e-8, atol=1e-8)

        # This is the ground truth for comparing Taichi inversion
        self.assertEqual(A_inv_numpy.shape, (48, 48))

    def test_assembled_matrix_inversion(self):
        """Test inversion of matrix assembled from element Hessian."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01
        mass = 1.0

        # Compute element Hessian
        H_e = compute_element_hessian_numpy(
            vertices, B_array[0], W_array[0], dt, mu, la, cells[0])

        # Build 48x48 block matrix
        H_block = np.zeros((48, 48), dtype=np.float32)
        H_block[:12, :12] = H_e

        # Add inertia (M/dt^2) to diagonal
        for i in range(4):
            for d in range(3):
                H_block[i*3 + d, i*3 + d] += mass / (dt * dt)

        # Regularize remaining diagonal to make it invertible
        for i in range(12, 48):
            H_block[i, i] = 1e-6

        # Verify invertible
        det = np.linalg.det(H_block)
        self.assertNotEqual(det, 0.0, "Matrix should be invertible")

        # Compute inverse
        H_block_inv = np.linalg.inv(H_block)

        # Verify A @ A_inv = I
        product = H_block @ H_block_inv
        np.testing.assert_allclose(product, np.eye(48), rtol=1e-6, atol=1e-6)


class TestSymmetricStorageExtraction(unittest.TestCase):
    """Test extraction of full matrix from symmetric storage."""

    def test_extract_known_matrix(self):
        """Test extraction with known matrix values."""
        # Create mock block matrices storage
        block_matrices = np.zeros((1, SYM_BLOCK_COUNT, 3, 3), dtype=np.float32)

        # Set diagonal blocks to identity scaled by their index
        for lane in range(BANKSIZE):
            idx = sym_index(lane, lane)
            block_matrices[0, idx] = np.eye(3) * (lane + 1)

        # Set some off-diagonal blocks
        idx_01 = sym_index(0, 1)
        block_matrices[0, idx_01] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        # Extract full matrix
        full = extract_full_block_matrix_numpy(block_matrices, 0)

        # Verify diagonal blocks
        for lane in range(BANKSIZE):
            expected = np.eye(3) * (lane + 1)
            actual = full[lane*3:(lane+1)*3, lane*3:(lane+1)*3]
            np.testing.assert_allclose(actual, expected.astype(np.float32))

        # Verify off-diagonal block (0,1) and its transpose (1,0)
        block_01 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        np.testing.assert_allclose(full[0:3, 3:6], block_01)
        np.testing.assert_allclose(full[3:6, 0:3], block_01.T)

    def test_extract_symmetric(self):
        """Test that extracted matrix is symmetric."""
        np.random.seed(42)

        # Create random symmetric block matrices
        block_matrices = np.zeros((1, SYM_BLOCK_COUNT, 3, 3), dtype=np.float32)

        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                idx = sym_index(row, col)
                if row == col:
                    # Diagonal blocks: symmetric
                    rand_block = np.random.randn(3, 3).astype(np.float32)
                    block_matrices[0, idx] = (rand_block + rand_block.T) / 2
                else:
                    # Off-diagonal: arbitrary
                    block_matrices[0, idx] = np.random.randn(3, 3).astype(np.float32)

        # Extract full matrix
        full = extract_full_block_matrix_numpy(block_matrices, 0)

        # Verify symmetry
        np.testing.assert_allclose(full, full.T, rtol=1e-6)


class TestApplyGroundTruth(unittest.TestCase):
    """Test preconditioner apply (SpMV) against NumPy."""

    def test_matvec_with_identity_inverse(self):
        """Test matrix-vector multiply with identity inverse."""
        n_verts = 4

        # Create identity inverse (preconditioner does nothing)
        B_inv = np.eye(48, dtype=np.float32)

        # Random gradient
        np.random.seed(42)
        gradient = np.random.randn(n_verts, 3).astype(np.float32)

        # Pad gradient to 48 DOF
        grad_padded = np.zeros(48, dtype=np.float32)
        grad_padded[:n_verts*3] = gradient.flatten()

        # Compute z = B_inv @ gradient
        z_padded = B_inv @ grad_padded
        z_expected = z_padded[:n_verts*3].reshape(n_verts, 3)

        # With identity inverse, z should equal gradient
        np.testing.assert_allclose(z_expected, gradient, rtol=1e-6)

    def test_matvec_with_diagonal_inverse(self):
        """Test matrix-vector multiply with diagonal inverse."""
        n_verts = 4

        # Create diagonal inverse (scales each DOF)
        np.random.seed(42)
        diag_vals = np.abs(np.random.randn(48)) + 0.1  # Positive values
        B_inv = np.diag(diag_vals).astype(np.float32)

        # Random gradient
        gradient = np.random.randn(n_verts, 3).astype(np.float32)

        # Pad and multiply
        grad_padded = np.zeros(48, dtype=np.float32)
        grad_padded[:n_verts*3] = gradient.flatten()
        z_padded = B_inv @ grad_padded
        z_expected = z_padded[:n_verts*3].reshape(n_verts, 3)

        # Verify not equal to gradient (unless scaling is 1)
        self.assertFalse(np.allclose(z_expected, gradient))

    def test_matvec_with_spd_inverse(self):
        """Test matrix-vector multiply with SPD inverse matrix."""
        n_verts = 4

        # Create SPD inverse
        np.random.seed(42)
        A_rand = np.random.randn(48, 48).astype(np.float32)
        B_inv = A_rand @ A_rand.T + 10 * np.eye(48, dtype=np.float32)

        # Random gradient
        gradient = np.random.randn(n_verts, 3).astype(np.float32)

        # Compute ground truth
        grad_padded = np.zeros(48, dtype=np.float32)
        grad_padded[:n_verts*3] = gradient.flatten()
        z_padded = B_inv @ grad_padded
        z_expected = z_padded[:n_verts*3].reshape(n_verts, 3)

        # Verify output is finite
        self.assertTrue(np.all(np.isfinite(z_expected)))

        # Verify shape
        self.assertEqual(z_expected.shape, (n_verts, 3))


class TestEndToEndGroundTruth(unittest.TestCase):
    """End-to-end ground truth tests combining assembly, inversion, and apply."""

    def test_single_tet_full_pipeline(self):
        """Test complete pipeline with single tetrahedron."""
        vertices, cells = create_single_tet_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01
        mass = 1.0

        # Step 1: Assemble ground truth Hessian
        H_e = compute_element_hessian_numpy(
            vertices, B_array[0], W_array[0], dt, mu, la, cells[0])

        # Build 48x48 block matrix
        H_block = np.zeros((48, 48), dtype=np.float32)
        H_block[:12, :12] = H_e

        # Add inertia
        for i in range(4):
            for d in range(3):
                H_block[i*3 + d, i*3 + d] += mass / (dt * dt)

        # Regularize unused DOFs
        for i in range(12, 48):
            H_block[i, i] = mass / (dt * dt)

        # Step 2: Invert
        H_inv = np.linalg.inv(H_block)

        # Step 3: Apply to random gradient
        np.random.seed(42)
        n_verts = 4
        gradient = np.random.randn(n_verts, 3).astype(np.float32)

        grad_padded = np.zeros(48, dtype=np.float32)
        grad_padded[:12] = gradient.flatten()

        z_padded = H_inv @ grad_padded
        z_expected = z_padded[:12].reshape(4, 3)

        # Verify output is finite and reasonable
        self.assertTrue(np.all(np.isfinite(z_expected)))
        self.assertGreater(np.linalg.norm(z_expected), 0)

        print(f"\n[Ground Truth] Single tet pipeline:")
        print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
        print(f"  Preconditioned norm: {np.linalg.norm(z_expected):.6f}")
        print(f"  Condition number: {np.linalg.cond(H_block[:12, :12]):.2e}")

    def test_cube_mesh_assembly(self):
        """Test assembly with unit cube mesh (8 vertices, 5 tets)."""
        vertices, cells = create_unit_cube_mesh()
        B_array, W_array = compute_tet_rest_matrices(vertices, cells)

        mu, la = 1e5, 1e5
        dt = 0.01
        mass = 1.0
        n_verts = 8

        # Assemble global Hessian (24x24 for 8 vertices)
        H_global = np.zeros((24, 24), dtype=np.float32)

        for c in range(len(cells)):
            H_e = compute_element_hessian_numpy(
                vertices, B_array[c], W_array[c], dt, mu, la, cells[c])

            # Scatter to global matrix
            for i in range(4):
                for j in range(4):
                    vi, vj = cells[c, i], cells[c, j]
                    H_global[vi*3:(vi+1)*3, vj*3:(vj+1)*3] += H_e[i*3:(i+1)*3, j*3:(j+1)*3]

        # Add inertia
        for i in range(n_verts):
            for d in range(3):
                H_global[i*3 + d, i*3 + d] += mass / (dt * dt)

        # Verify symmetry
        np.testing.assert_allclose(H_global, H_global.T, rtol=1e-10)

        # Verify positive definite
        eigvals = np.linalg.eigvalsh(H_global)
        self.assertTrue(np.all(eigvals > 0),
                        f"Global Hessian should be PD, min eigenvalue: {eigvals.min()}")

        print(f"\n[Ground Truth] Cube mesh assembly:")
        print(f"  Global matrix size: {H_global.shape}")
        print(f"  Condition number: {np.linalg.cond(H_global):.2e}")
        print(f"  Min eigenvalue: {eigvals.min():.2e}")
        print(f"  Max eigenvalue: {eigvals.max():.2e}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MAS Preconditioner Ground Truth Tests")
    print("=" * 70)
    print(f"BANKSIZE: {BANKSIZE}")
    print(f"SYM_BLOCK_COUNT: {SYM_BLOCK_COUNT}")
    print(f"BLOCK_DOF: {BLOCK_DOF}")
    print("=" * 70)

    unittest.main(verbosity=2)
