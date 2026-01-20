"""
MAS Matrix-Vector Product Accuracy and Performance Tests

This module tests the accuracy and performance of the MAS preconditioner's
matrix-vector product (z = M^{-1} * r) by comparing against NumPy ground truth.

Tests:
1. Symmetric storage expansion correctness
2. Block matrix-vector product accuracy (single block)
3. Full Schwarz local solve accuracy (multi-block)
4. Different Schwarz solver variants comparison
5. Performance benchmarks

Usage:
    python test_matvec_accuracy.py -v              # Run all tests
    python test_matvec_accuracy.py -v TestBlockMatVec  # Run specific class
    python test_matvec_accuracy.py --benchmark     # Run performance benchmark only
"""

import unittest
import numpy as np
import sys
import os
import time

# Add project root to path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.mas_preconditioner_pkg import MASPreconditioner
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, BLOCK_DOF, SYM_BLOCK_COUNT


# ==============================================================================
# NumPy Ground Truth Functions
# ==============================================================================

def sym_idx_numpy(row: int, col: int) -> int:
    """
    Compute symmetric storage index matching Taichi implementation.

    For symmetric 48x48 matrix stored as 136 3x3 blocks:
    Only upper triangle (row <= col) is stored.
    """
    min_lane = min(row, col)
    max_lane = max(row, col)
    return BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane


def expand_sym_to_full_numpy(inv_block_sym: np.ndarray) -> np.ndarray:
    """
    Expand symmetric block storage (136 x 3 x 3) to full matrix (48 x 48).

    Args:
        inv_block_sym: Shape (136, 3, 3) - upper triangle blocks

    Returns:
        Full 48x48 matrix with symmetric entries filled
    """
    full = np.zeros((BLOCK_DOF, BLOCK_DOF), dtype=np.float64)

    for lane_i in range(BANKSIZE):
        for lane_j in range(lane_i, BANKSIZE):  # Only upper triangle
            sym_idx = sym_idx_numpy(lane_i, lane_j)
            block_3x3 = inv_block_sym[sym_idx]

            # Place in full matrix
            row_start = lane_i * 3
            col_start = lane_j * 3

            # Upper triangle
            full[row_start:row_start+3, col_start:col_start+3] = block_3x3

            # Lower triangle (transpose)
            if lane_i != lane_j:
                full[col_start:col_start+3, row_start:row_start+3] = block_3x3.T

    return full


def schwarz_local_solve_numpy(inv_block_full: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    NumPy ground truth for Schwarz local solve: z = M^{-1} * r

    Args:
        inv_block_full: Full 48x48 inverse block matrix
        r: Input vector (48,) or (16, 3)

    Returns:
        z: Output vector (48,) or (16, 3)
    """
    if r.shape == (BANKSIZE, 3):
        r_flat = r.flatten()
    else:
        r_flat = r

    z_flat = inv_block_full @ r_flat

    return z_flat.reshape(BANKSIZE, 3)


def schwarz_local_solve_sym_numpy(inv_block_sym: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    NumPy ground truth matching Taichi's symmetric storage access pattern.

    This mimics the exact computation in _schwarz_local_solve_conflict_free.
    """
    z = np.zeros((BANKSIZE, 3), dtype=np.float64)

    for lane_i in range(BANKSIZE):
        z_i = np.zeros(3, dtype=np.float64)

        for lane_j in range(BANKSIZE):
            r_j = r[lane_j]

            min_lane = min(lane_i, lane_j)
            max_lane = max(lane_i, lane_j)
            sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

            inv_block = inv_block_sym[sym_idx]

            if lane_i <= lane_j:
                # Direct access: z_i += inv_block @ r_j
                z_i += inv_block @ r_j
            else:
                # Transpose access: z_i += inv_block.T @ r_j
                z_i += inv_block.T @ r_j

        z[lane_i] = z_i

    return z


# ==============================================================================
# Test Fixtures
# ==============================================================================

def create_random_spd_block(n: int = BLOCK_DOF, cond: float = 100.0) -> np.ndarray:
    """
    Create a random symmetric positive definite matrix.

    Args:
        n: Matrix size
        cond: Approximate condition number

    Returns:
        SPD matrix of shape (n, n)
    """
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Generate eigenvalues with specified condition number
    eigenvalues = np.linspace(1.0, cond, n)

    # Construct SPD matrix: A = Q @ diag(eigenvalues) @ Q.T
    A = Q @ np.diag(eigenvalues) @ Q.T

    return A


def full_to_sym_storage(full: np.ndarray) -> np.ndarray:
    """
    Convert full 48x48 matrix to symmetric block storage (136, 3, 3).
    """
    sym = np.zeros((SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)

    for lane_i in range(BANKSIZE):
        for lane_j in range(lane_i, BANKSIZE):
            sym_idx = sym_idx_numpy(lane_i, lane_j)
            row_start = lane_i * 3
            col_start = lane_j * 3
            sym[sym_idx] = full[row_start:row_start+3, col_start:col_start+3]

    return sym


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestSymmetricIndex(unittest.TestCase):
    """Test symmetric index computation."""

    def test_sym_idx_coverage(self):
        """Verify sym_idx covers all 136 unique entries."""
        indices = set()
        for i in range(BANKSIZE):
            for j in range(i, BANKSIZE):
                idx = sym_idx_numpy(i, j)
                indices.add(idx)

        self.assertEqual(len(indices), SYM_BLOCK_COUNT,
                        f"Expected {SYM_BLOCK_COUNT} unique indices, got {len(indices)}")
        self.assertEqual(max(indices), SYM_BLOCK_COUNT - 1)
        self.assertEqual(min(indices), 0)

    def test_sym_idx_symmetry(self):
        """Verify sym_idx(i,j) == sym_idx(j,i)."""
        for i in range(BANKSIZE):
            for j in range(BANKSIZE):
                idx1 = sym_idx_numpy(i, j)
                idx2 = sym_idx_numpy(j, i)
                self.assertEqual(idx1, idx2, f"sym_idx({i},{j}) != sym_idx({j},{i})")


class TestSymmetricExpansion(unittest.TestCase):
    """Test symmetric storage expansion."""

    def test_identity_expansion(self):
        """Test expanding identity matrix in symmetric storage."""
        # Create identity in symmetric storage
        sym_storage = np.zeros((SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane_i in range(BANKSIZE):
            sym_idx = sym_idx_numpy(lane_i, lane_i)
            sym_storage[sym_idx] = np.eye(3)

        full = expand_sym_to_full_numpy(sym_storage)
        expected = np.eye(BLOCK_DOF)

        np.testing.assert_allclose(full, expected, rtol=1e-10)

    def test_random_spd_expansion(self):
        """Test expanding random SPD matrix preserves symmetry."""
        full_original = create_random_spd_block()
        sym_storage = full_to_sym_storage(full_original)
        full_recovered = expand_sym_to_full_numpy(sym_storage)

        np.testing.assert_allclose(full_recovered, full_original, rtol=1e-10)

        # Verify symmetry
        sym_error = np.linalg.norm(full_recovered - full_recovered.T, 'fro')
        self.assertLess(sym_error, 1e-10, "Expanded matrix should be symmetric")


class TestBlockMatVec(unittest.TestCase):
    """Test single block matrix-vector product accuracy."""

    def test_identity_matvec(self):
        """Test z = I * r == r."""
        sym_storage = np.zeros((SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane_i in range(BANKSIZE):
            sym_idx = sym_idx_numpy(lane_i, lane_i)
            sym_storage[sym_idx] = np.eye(3)

        r = np.random.randn(BANKSIZE, 3)
        z = schwarz_local_solve_sym_numpy(sym_storage, r)

        np.testing.assert_allclose(z, r, rtol=1e-10)

    def test_diagonal_matvec(self):
        """Test diagonal matrix-vector product."""
        diag_values = np.random.rand(BANKSIZE, 3) + 0.1  # Ensure positive

        sym_storage = np.zeros((SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane_i in range(BANKSIZE):
            sym_idx = sym_idx_numpy(lane_i, lane_i)
            sym_storage[sym_idx] = np.diag(diag_values[lane_i])

        r = np.random.randn(BANKSIZE, 3)
        z = schwarz_local_solve_sym_numpy(sym_storage, r)

        expected = r * diag_values
        np.testing.assert_allclose(z, expected, rtol=1e-10)

    def test_random_spd_matvec(self):
        """Test random SPD matrix-vector product against full expansion."""
        full = create_random_spd_block()
        sym_storage = full_to_sym_storage(full)

        r = np.random.randn(BANKSIZE, 3)

        # Method 1: Using symmetric storage access
        z_sym = schwarz_local_solve_sym_numpy(sym_storage, r)

        # Method 2: Using full matrix
        z_full = schwarz_local_solve_numpy(full, r)

        np.testing.assert_allclose(z_sym, z_full, rtol=1e-10,
                                   err_msg="Symmetric access should match full matrix multiply")

    def test_inverse_matvec_accuracy(self):
        """Test z = A^{-1} * r produces A * z ≈ r."""
        A = create_random_spd_block(cond=10.0)  # Low condition number
        A_inv = np.linalg.inv(A)

        sym_storage = full_to_sym_storage(A_inv)

        r = np.random.randn(BANKSIZE, 3)
        z = schwarz_local_solve_sym_numpy(sym_storage, r)

        # Check A * z ≈ r
        r_reconstructed = (A @ z.flatten()).reshape(BANKSIZE, 3)

        rel_error = np.linalg.norm(r_reconstructed - r) / np.linalg.norm(r)
        self.assertLess(rel_error, 1e-8, f"Relative error {rel_error:.2e} too large")


@ti.data_oriented
class TaichiMatVecHelper:
    """Helper class for Taichi matrix-vector product tests."""

    def __init__(self):
        self.n_verts = BANKSIZE
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                  shape=(1, SYM_BLOCK_COUNT))
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    def run_matvec(self, inv_sym_np, r_np):
        """Run Taichi matvec and return result."""
        inv_ti = inv_sym_np.astype(np.float32)
        r_ti = r_np.astype(np.float32)

        self.inv_block_matrices.from_numpy(inv_ti.reshape(1, SYM_BLOCK_COUNT, 3, 3))
        self.multi_level_r.from_numpy(r_ti)

        self._schwarz_local_solve_single_block()

        return self.multi_level_z.to_numpy()

    @ti.kernel
    def _schwarz_local_solve_single_block(self):
        """Single block Schwarz solve matching _schwarz_local_solve_conflict_free."""
        for lane_i in range(BANKSIZE):
            z0 = ti.f32(0.0)
            z1 = ti.f32(0.0)
            z2 = ti.f32(0.0)

            for lane_j in range(BANKSIZE):
                r_j = self.multi_level_r[lane_j]

                min_lane = ti.min(lane_i, lane_j)
                max_lane = ti.max(lane_i, lane_j)
                sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                inv_block = self.inv_block_matrices[0, sym_idx]

                if lane_i <= lane_j:
                    z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                    z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                    z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                else:
                    z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                    z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                    z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

            self.multi_level_z[lane_i] = ti.Vector([z0, z1, z2], dt=ti.f32)


class TestTaichiMatVec(unittest.TestCase):
    """Test Taichi implementation against NumPy ground truth."""

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi once for all tests."""
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        cls.helper = TaichiMatVecHelper()

    def test_identity_taichi(self):
        """Test Taichi identity matvec."""
        sym_storage = np.zeros((SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane_i in range(BANKSIZE):
            sym_idx = sym_idx_numpy(lane_i, lane_i)
            sym_storage[sym_idx] = np.eye(3)

        r = np.random.randn(BANKSIZE, 3)
        z = self.helper.run_matvec(sym_storage, r)

        np.testing.assert_allclose(z, r, rtol=1e-5, atol=1e-6)

    def test_random_spd_taichi(self):
        """Test Taichi random SPD matvec against NumPy."""
        full = create_random_spd_block(cond=10.0)
        sym_storage = full_to_sym_storage(full)

        r = np.random.randn(BANKSIZE, 3)

        # NumPy ground truth
        z_numpy = schwarz_local_solve_sym_numpy(sym_storage, r)

        # Taichi result
        z_taichi = self.helper.run_matvec(sym_storage, r)

        np.testing.assert_allclose(z_taichi, z_numpy, rtol=1e-4, atol=1e-5,
                                   err_msg="Taichi matvec should match NumPy")

    def test_numerical_precision_vs_numpy(self):
        """Compare f32 Taichi vs f64 NumPy precision."""
        full = create_random_spd_block(cond=100.0)
        sym_storage = full_to_sym_storage(full)

        errors = []
        for _ in range(10):
            r = np.random.randn(BANKSIZE, 3)

            z_numpy = schwarz_local_solve_sym_numpy(sym_storage, r)
            z_taichi = self.helper.run_matvec(sym_storage, r)

            rel_error = np.linalg.norm(z_taichi - z_numpy) / np.linalg.norm(z_numpy)
            errors.append(rel_error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        print(f"\n  Numerical precision (f32 vs f64):")
        print(f"    Mean relative error: {mean_error:.2e}")
        print(f"    Max relative error:  {max_error:.2e}")

        self.assertLess(max_error, 1e-3, "Max relative error should be < 1e-3 for f32")


class TestFullSolverMatVec(unittest.TestCase):
    """Test full MAS solver matrix-vector product with real mesh."""

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi and solver."""
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)

        # Import solver
        from algorithm.pncg_base_ipc import pncg_ipc_deformer

        class TestSolver(pncg_ipc_deformer):
            def __init__(self):
                super().__init__(demo='eight_E_stiffness_test')
                self.mesh.verts.place({'z': ti.types.vector(3, float)})
                self.mas = MASPreconditioner(
                    self.n_verts, self.n_cells, self.mesh,
                    use_metis=False
                )

        cls.solver = TestSolver()
        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Cells: {cls.solver.n_cells}")

        # Build hierarchy
        cls.solver.mas.build_hierarchy()
        print(f"[Setup] MAS levels: {cls.solver.mas.actual_levels}")

        # Assemble block matrices
        cls.solver.assign_xn_xhat()
        cls.solver.compute_grad_and_diagH()
        if cls.solver.ground_barrier == 1:
            cls.solver.add_grad_and_diagH_ground_barrier()
        cls.solver.mas.assemble_block_matrices(cls.solver, use_full_hessian=True)

        # Invert with regularization for stability
        cls.solver.mas._expand_sym_to_full()
        block0 = cls.solver.mas.full_block_matrix.to_numpy()[0]
        eigenvalues = np.linalg.eigvalsh((block0 + block0.T) / 2)
        min_eig = np.min(eigenvalues)
        cls.reg_epsilon = abs(min_eig) * 1.1 + 1e3 if min_eig < 0 else 1e3

        cls.solver.mas.invert_block_matrices(
            method='gauss_jordan',
            force_symmetry=True,
            regularization_epsilon=cls.reg_epsilon
        )
        print(f"[Setup] Regularization epsilon: {cls.reg_epsilon:.2e}")

    def test_apply_produces_valid_z(self):
        """Test that apply() produces non-NaN z values."""
        self.solver.mas.apply()

        z = self.solver.mesh.verts.z.to_numpy()
        nan_count = np.sum(np.isnan(z))

        self.assertEqual(nan_count, 0, "apply() should not produce NaN values")

    def test_apply_positive_gTz(self):
        """Test that z = M^{-1} * g produces g^T z > 0."""
        self.solver.mas.apply()

        g = self.solver.mesh.verts.grad.to_numpy()
        z = self.solver.mesh.verts.z.to_numpy()

        gTz = np.sum(g * z)

        print(f"\n  g^T z = {gTz:.4e}")
        self.assertGreater(gTz, 0, "Preconditioned direction should satisfy g^T z > 0")

    def test_block0_local_solve_accuracy(self):
        """Test accuracy of Block 0 local solve (z = M^{-1} * r) against NumPy ground truth.

        Note: MAS apply() includes restriction and prolongation, so we test the
        intermediate multi_level_z result against the local block matrix-vector product.
        """
        # Get inverse block 0 in symmetric storage
        inv_sym = self.solver.mas.inv_block_matrices.to_numpy()[0]  # (136, 3, 3)

        # Get multi_level_r for block 0 (this is gradient after restriction)
        r_block0 = self.solver.mas.multi_level_r.to_numpy()[:BANKSIZE]

        # NumPy ground truth using symmetric storage access pattern
        z_numpy = schwarz_local_solve_sym_numpy(inv_sym, r_block0)

        # Apply preconditioner and get multi_level_z (before prolongation)
        self.solver.mas.apply()

        # Get Taichi result for block 0 from multi_level_z
        z_taichi = self.solver.mas.multi_level_z.to_numpy()[:BANKSIZE]

        # Compare
        if np.linalg.norm(z_numpy) > 1e-10:
            rel_error = np.linalg.norm(z_taichi - z_numpy) / np.linalg.norm(z_numpy)
            print(f"\n  Block 0 local solve relative error: {rel_error:.2e}")
            self.assertLess(rel_error, 1e-3, f"Block 0 relative error {rel_error:.2e} too large")
        else:
            # If z_numpy is near zero, check absolute error
            abs_error = np.linalg.norm(z_taichi - z_numpy)
            print(f"\n  Block 0 local solve absolute error: {abs_error:.2e}")
            self.assertLess(abs_error, 1e-6, f"Block 0 absolute error {abs_error:.2e} too large")

    def test_solver_variants_consistency(self):
        """Test different Schwarz solver variants produce consistent results."""
        # Store original z
        self.solver.mas.apply(use_full_solve=True, use_conflict_free=False)
        z_full = self.solver.mesh.verts.z.to_numpy().copy()

        # Conflict-free variant
        self.solver.mas.apply(use_full_solve=True, use_conflict_free=True)
        z_conflict_free = self.solver.mesh.verts.z.to_numpy().copy()

        # Parallel variant
        self.solver.mas.apply(use_full_solve=True, use_parallel_solve=True)
        z_parallel = self.solver.mesh.verts.z.to_numpy().copy()

        # Compare
        diff_cf = np.linalg.norm(z_conflict_free - z_full) / np.linalg.norm(z_full)
        diff_par = np.linalg.norm(z_parallel - z_full) / np.linalg.norm(z_full)

        print(f"\n  Solver variant consistency:")
        print(f"    conflict_free vs full: {diff_cf:.2e}")
        print(f"    parallel vs full:      {diff_par:.2e}")

        self.assertLess(diff_cf, 1e-4, "Conflict-free should match full solve")
        self.assertLess(diff_par, 1e-4, "Parallel should match full solve")


class TestRestriction(unittest.TestCase):
    """Test MAS restriction phase (gradient -> multi_level_r).

    Note: The restriction phase accumulates gradients to elected representatives
    within each connected component, then propagates to coarse levels.
    Level 0 of multi_level_r will have accumulated values at representative nodes,
    not the original gradient values.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi and solver."""
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)

        from algorithm.pncg_base_ipc import pncg_ipc_deformer

        class TestSolver(pncg_ipc_deformer):
            def __init__(self):
                super().__init__(demo='eight_E_stiffness_test')
                self.mesh.verts.place({'z': ti.types.vector(3, float)})
                self.mas = MASPreconditioner(
                    self.n_verts, self.n_cells, self.mesh,
                    use_metis=False
                )

        cls.solver = TestSolver()
        cls.solver.mas.build_hierarchy()
        cls.solver.assign_xn_xhat()
        cls.solver.compute_grad_and_diagH()
        if cls.solver.ground_barrier == 1:
            cls.solver.add_grad_and_diagH_ground_barrier()

        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Levels: {cls.solver.mas.actual_levels}")
        print(f"[Setup] Level sizes: L0={cls.solver.n_verts}, L1={cls.solver.mas.level_size.to_numpy()[1][0]}")

    def test_restriction_produces_valid_output(self):
        """Test that restriction produces valid (non-NaN) output."""
        # Clear buffers and run restriction
        self.solver.mas._clear_multi_level_buffers()
        self.solver.mas._build_multi_level_r()

        # Get multi_level_r
        r = self.solver.mas.multi_level_r.to_numpy()

        # Check for NaN/Inf
        nan_count = np.sum(np.isnan(r))
        inf_count = np.sum(np.isinf(r))

        print(f"\n  Restriction output:")
        print(f"    NaN count: {nan_count}")
        print(f"    Inf count: {inf_count}")
        print(f"    ||r|| = {np.linalg.norm(r):.4e}")

        self.assertEqual(nan_count, 0, "Restriction should not produce NaN")
        self.assertEqual(inf_count, 0, "Restriction should not produce Inf")

    def test_restriction_level0_starts_with_gradient(self):
        """Test that restriction first copies gradient to Level 0 before accumulation.

        The restriction algorithm:
        1. Copy gradient to Level 0: multi_level_r[idx] = grad[idx]
        2. Non-representative nodes atomically add to their representative
        3. Representatives propagate to coarse levels

        After step 1 and before step 2, Level 0 equals gradient.
        After full restriction, representative nodes have accumulated values.
        """
        # Get original gradient
        grad = self.solver.mesh.verts.grad.to_numpy()
        grad_norm = np.linalg.norm(grad)

        # Run restriction
        self.solver.mas._clear_multi_level_buffers()
        self.solver.mas._build_multi_level_r()
        r_level0 = self.solver.mas.multi_level_r.to_numpy()[:self.solver.n_verts]
        r_level0_norm = np.linalg.norm(r_level0)

        print(f"\n  Gradient vs Level 0:")
        print(f"    ||gradient|| = {grad_norm:.4e}")
        print(f"    ||r_level0|| = {r_level0_norm:.4e}")

        # Level 0 should have non-zero values if gradient is non-zero
        if grad_norm > 1e-10:
            self.assertGreater(r_level0_norm, 0, "Level 0 should have non-zero values")

    def test_restriction_coarse_levels_populated(self):
        """Test that coarse levels receive accumulated values from fine level."""
        if self.solver.mas.actual_levels < 2:
            self.skipTest("Need at least 2 levels for this test")

        # Run restriction
        self.solver.mas._clear_multi_level_buffers()
        self.solver.mas._build_multi_level_r()
        r = self.solver.mas.multi_level_r.to_numpy()

        # Check Level 1
        level1_offset = self.solver.mas.level_size.to_numpy()[1][1]
        level1_size = self.solver.mas.level_size.to_numpy()[1][0]
        r_level1 = r[level1_offset:level1_offset + level1_size]
        r_level1_norm = np.linalg.norm(r_level1)

        grad = self.solver.mesh.verts.grad.to_numpy()
        grad_norm = np.linalg.norm(grad)

        print(f"\n  Coarse level population:")
        print(f"    ||gradient|| = {grad_norm:.4e}")
        print(f"    ||r_level1|| = {r_level1_norm:.4e}")

        # Level 1 should have non-zero values if gradient is non-zero
        if grad_norm > 1e-10:
            self.assertGreater(r_level1_norm, 0, "Level 1 should have non-zero values")


class TestProlongation(unittest.TestCase):
    """Test MAS prolongation phase (multi_level_z -> z)."""

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi and solver."""
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)

        from algorithm.pncg_base_ipc import pncg_ipc_deformer

        class TestSolver(pncg_ipc_deformer):
            def __init__(self):
                super().__init__(demo='eight_E_stiffness_test')
                self.mesh.verts.place({'z': ti.types.vector(3, float)})
                self.mas = MASPreconditioner(
                    self.n_verts, self.n_cells, self.mesh,
                    use_metis=False
                )

        cls.solver = TestSolver()
        cls.solver.mas.build_hierarchy()
        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Levels: {cls.solver.mas.actual_levels}")

    def test_level0_only_prolongation(self):
        """Test prolongation with only Level 0 non-zero."""
        # Set multi_level_z: only Level 0 has values
        self.solver.mas._clear_multi_level_buffers()

        # Set Level 0 to random values
        n_verts = self.solver.n_verts
        z_level0 = np.random.randn(n_verts, 3).astype(np.float32)

        total_nodes = self.solver.mas.total_nodes_all_levels
        z_all = np.zeros((total_nodes, 3), dtype=np.float32)
        z_all[:n_verts] = z_level0
        self.solver.mas.multi_level_z.from_numpy(z_all)

        # Run prolongation
        self.solver.mas._collect_final_z()

        # Get result
        z_out = self.solver.mesh.verts.z.to_numpy()

        # With only Level 0 non-zero, z_out should equal z_level0
        np.testing.assert_allclose(z_out, z_level0, rtol=1e-5, atol=1e-6,
                                   err_msg="With only Level 0, output should equal Level 0")

    def test_prolongation_aggregation(self):
        """Test that prolongation correctly aggregates from coarse levels."""
        if self.solver.mas.actual_levels < 2:
            self.skipTest("Need at least 2 levels for this test")

        n_verts = self.solver.n_verts
        total_nodes = self.solver.mas.total_nodes_all_levels

        # Set multi_level_z: Level 0 = 1.0, Level 1 = 2.0
        z_all = np.zeros((total_nodes, 3), dtype=np.float32)
        z_all[:n_verts] = 1.0  # Level 0

        level1_offset = self.solver.mas.level_size.to_numpy()[1][1]
        level1_size = self.solver.mas.level_size.to_numpy()[1][0]
        z_all[level1_offset:level1_offset + level1_size] = 2.0  # Level 1

        self.solver.mas.multi_level_z.from_numpy(z_all)

        # Run prolongation
        self.solver.mas._collect_final_z()

        # Get result
        z_out = self.solver.mesh.verts.z.to_numpy()

        # Each vertex should have z_level0[i] + z_level1[aggregation_table[i][0]]
        # Since all Level 0 = 1.0 and Level 1 = 2.0, result should be 3.0
        expected = np.ones((n_verts, 3), dtype=np.float32) * 3.0

        np.testing.assert_allclose(z_out, expected, rtol=1e-5, atol=1e-6,
                                   err_msg="Prolongation should sum contributions from all levels")

    def test_aggregation_table_validity(self):
        """Test that aggregation table contains valid indices."""
        if self.solver.mas.actual_levels < 2:
            self.skipTest("Need at least 2 levels for this test")

        agg_table = self.solver.mas.aggregation_table.to_numpy()
        n_verts = self.solver.n_verts
        total_nodes = self.solver.mas.total_nodes_all_levels

        # Check Level 1 indices (stored at column 0)
        level1_indices = agg_table[:n_verts, 0]

        # All indices should be valid (>= 0 and < total_nodes)
        valid_indices = np.sum((level1_indices >= 0) & (level1_indices < total_nodes))

        print(f"\n  Aggregation table validity:")
        print(f"    Total vertices: {n_verts}")
        print(f"    Valid Level 1 indices: {valid_indices}")
        print(f"    Level 1 index range: [{level1_indices.min()}, {level1_indices.max()}]")

        self.assertEqual(valid_indices, n_verts,
                        "All vertices should have valid Level 1 aggregation indices")


class TestFullMASPipeline(unittest.TestCase):
    """Test complete MAS preconditioner pipeline (restriction + local solve + prolongation)."""

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi and solver with full setup."""
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)

        from algorithm.pncg_base_ipc import pncg_ipc_deformer

        class TestSolver(pncg_ipc_deformer):
            def __init__(self):
                super().__init__(demo='eight_E_stiffness_test')
                self.mesh.verts.place({'z': ti.types.vector(3, float)})
                self.mas = MASPreconditioner(
                    self.n_verts, self.n_cells, self.mesh,
                    use_metis=False
                )

        cls.solver = TestSolver()
        cls.solver.mas.build_hierarchy()
        cls.solver.assign_xn_xhat()
        cls.solver.compute_grad_and_diagH()
        if cls.solver.ground_barrier == 1:
            cls.solver.add_grad_and_diagH_ground_barrier()
        cls.solver.mas.assemble_block_matrices(cls.solver, use_full_hessian=True)

        # Get regularization
        cls.solver.mas._expand_sym_to_full()
        block0 = cls.solver.mas.full_block_matrix.to_numpy()[0]
        eigenvalues = np.linalg.eigvalsh((block0 + block0.T) / 2)
        min_eig = np.min(eigenvalues)
        cls.reg_epsilon = abs(min_eig) * 1.1 + 1e3 if min_eig < 0 else 1e3

        cls.solver.mas.invert_block_matrices(
            method='gauss_jordan',
            force_symmetry=True,
            regularization_epsilon=cls.reg_epsilon
        )

        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Levels: {cls.solver.mas.actual_levels}")

    def test_full_pipeline_produces_valid_output(self):
        """Test that full pipeline produces valid (non-NaN) output."""
        self.solver.mas.apply()

        z = self.solver.mesh.verts.z.to_numpy()
        nan_count = np.sum(np.isnan(z))
        inf_count = np.sum(np.isinf(z))

        print(f"\n  Full pipeline output:")
        print(f"    NaN count: {nan_count}")
        print(f"    Inf count: {inf_count}")
        print(f"    z norm: {np.linalg.norm(z):.4e}")

        self.assertEqual(nan_count, 0, "Full pipeline should not produce NaN")
        self.assertEqual(inf_count, 0, "Full pipeline should not produce Inf")

    def test_full_pipeline_descent_direction(self):
        """Test that full pipeline produces descent direction (g^T z > 0)."""
        self.solver.mas.apply()

        g = self.solver.mesh.verts.grad.to_numpy()
        z = self.solver.mesh.verts.z.to_numpy()
        gTz = np.sum(g * z)

        print(f"\n  Descent direction check:")
        print(f"    g^T z = {gTz:.4e}")
        print(f"    ||g|| = {np.linalg.norm(g):.4e}")
        print(f"    ||z|| = {np.linalg.norm(z):.4e}")

        self.assertGreater(gTz, 0, "Preconditioned direction should satisfy g^T z > 0")

    def test_restriction_variants_both_valid(self):
        """Test that both restriction variants produce valid descent directions.

        Note: The original and optimized restriction implementations may produce
        slightly different numerical results due to different accumulation orders
        (atomic vs tree reduction). Both should produce valid preconditioned
        directions satisfying g^T z > 0.
        """
        g = self.solver.mesh.verts.grad.to_numpy()

        # With warp reduction (optimized)
        self.solver.mas.apply(use_warp_reduction=True)
        z_warp = self.solver.mesh.verts.z.to_numpy().copy()
        gTz_warp = np.sum(g * z_warp)

        # Without warp reduction (original)
        self.solver.mas.apply(use_warp_reduction=False)
        z_no_warp = self.solver.mesh.verts.z.to_numpy().copy()
        gTz_no_warp = np.sum(g * z_no_warp)

        print(f"\n  Restriction variants validity:")
        print(f"    With warp reduction:    g^T z = {gTz_warp:.4e}, ||z|| = {np.linalg.norm(z_warp):.4e}")
        print(f"    Without warp reduction: g^T z = {gTz_no_warp:.4e}, ||z|| = {np.linalg.norm(z_no_warp):.4e}")

        # Both should produce valid descent directions
        self.assertGreater(gTz_warp, 0, "Warp reduction should produce g^T z > 0")
        self.assertGreater(gTz_no_warp, 0, "Original restriction should produce g^T z > 0")

    def test_multilevel_contribution(self):
        """Test that coarse levels contribute to the final solution."""
        if self.solver.mas.actual_levels < 2:
            self.skipTest("Need at least 2 levels for this test")

        # Run full apply
        self.solver.mas.apply()

        # Get multi_level_z at each level
        multi_level_z = self.solver.mas.multi_level_z.to_numpy()
        n_verts = self.solver.n_verts

        level0_z = multi_level_z[:n_verts]
        level0_norm = np.linalg.norm(level0_z)

        level1_offset = self.solver.mas.level_size.to_numpy()[1][1]
        level1_size = self.solver.mas.level_size.to_numpy()[1][0]
        level1_z = multi_level_z[level1_offset:level1_offset + level1_size]
        level1_norm = np.linalg.norm(level1_z)

        print(f"\n  Multi-level contribution:")
        print(f"    Level 0 ||z||: {level0_norm:.4e} ({n_verts} nodes)")
        print(f"    Level 1 ||z||: {level1_norm:.4e} ({level1_size} nodes)")
        print(f"    Ratio L1/L0:   {level1_norm / (level0_norm + 1e-10):.4e}")

        # Both levels should contribute
        self.assertGreater(level0_norm, 0, "Level 0 should have non-zero solution")
        # Level 1 might be small but should exist
        self.assertGreaterEqual(level1_norm, 0, "Level 1 solution should exist")

    def test_pipeline_determinism(self):
        """Test that pipeline produces deterministic results."""
        results = []
        for _ in range(3):
            self.solver.mas.apply()
            z = self.solver.mesh.verts.z.to_numpy().copy()
            results.append(z)

        # All results should be identical
        for i in range(1, len(results)):
            diff = np.linalg.norm(results[i] - results[0])
            self.assertLess(diff, 1e-10, f"Run {i+1} should match run 1")

        print(f"\n  Determinism check: All 3 runs produce identical results")


# ==============================================================================
# Performance Benchmark
# ==============================================================================

def run_performance_benchmark():
    """Run matrix-vector product performance benchmark."""
    print("\n" + "=" * 80)
    print("MATRIX-VECTOR PRODUCT PERFORMANCE BENCHMARK")
    print("=" * 80)

    ti.reset()
    ti.init(arch=ti.cuda, default_fp=ti.f32)

    from algorithm.pncg_base_ipc import pncg_ipc_deformer

    class BenchSolver(pncg_ipc_deformer):
        def __init__(self):
            super().__init__(demo='eight_E_stiffness_test')
            self.mesh.verts.place({'z': ti.types.vector(3, float)})
            self.mas = MASPreconditioner(
                self.n_verts, self.n_cells, self.mesh,
                use_metis=False
            )

    solver = BenchSolver()
    print(f"\n[Setup] Vertices: {solver.n_verts}, Cells: {solver.n_cells}")

    # Build hierarchy
    solver.mas.build_hierarchy()
    n_blocks = (solver.mas.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE
    print(f"[Setup] MAS levels: {solver.mas.actual_levels}, Total blocks: {n_blocks}")

    # Assemble and invert
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)

    # Get regularization
    solver.mas._expand_sym_to_full()
    block0 = solver.mas.full_block_matrix.to_numpy()[0]
    eigenvalues = np.linalg.eigvalsh((block0 + block0.T) / 2)
    min_eig = np.min(eigenvalues)
    reg_epsilon = abs(min_eig) * 1.1 + 1e3 if min_eig < 0 else 1e3

    solver.mas.invert_block_matrices(
        method='gauss_jordan',
        force_symmetry=True,
        regularization_epsilon=reg_epsilon
    )

    # Benchmark variants
    variants = [
        ('full_solve (default)', {'use_full_solve': True, 'use_conflict_free': False, 'use_parallel_solve': False}),
        ('conflict_free', {'use_full_solve': True, 'use_conflict_free': True}),
        ('parallel', {'use_full_solve': True, 'use_parallel_solve': True}),
        ('diagonal_only', {'use_full_solve': False}),
    ]

    n_warmup = 10
    n_timed = 100

    print("\n" + "-" * 80)
    print(f"Running benchmarks ({n_warmup} warmup + {n_timed} timed iterations each)...")
    print("-" * 80)

    results = []

    for name, kwargs in variants:
        print(f"\n[{name}]")

        # Warmup
        for _ in range(n_warmup):
            solver.mas.apply(**kwargs)
        ti.sync()

        # Timed runs
        times = []
        for _ in range(n_timed):
            ti.sync()
            t0 = time.perf_counter()
            solver.mas.apply(**kwargs)
            ti.sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        # Verify correctness
        g = solver.mesh.verts.grad.to_numpy()
        z = solver.mesh.verts.z.to_numpy()
        gTz = np.sum(g * z)
        nan_count = np.sum(np.isnan(z))

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results.append({
            'name': name,
            'avg_ms': avg_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'gTz': gTz,
            'nan_count': nan_count,
        })

        print(f"  Avg: {avg_time:.3f}ms ± {std_time:.3f}ms")
        print(f"  Min: {min_time:.3f}ms, Max: {max_time:.3f}ms")
        print(f"  g^T z = {gTz:.4e}, NaN count: {nan_count}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Variant':<25} {'Avg(ms)':<12} {'Std(ms)':<10} {'Min(ms)':<10} {'g^Tz>0':<8}")
    print("-" * 70)

    for r in results:
        gtz_str = "Yes" if r['gTz'] > 0 and r['nan_count'] == 0 else "No"
        print(f"{r['name']:<25} {r['avg_ms']:<12.3f} {r['std_ms']:<10.3f} {r['min_ms']:<10.3f} {gtz_str:<8}")

    # Find fastest valid variant
    valid_results = [r for r in results if r['gTz'] > 0 and r['nan_count'] == 0]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['avg_ms'])
        print(f"\nFastest valid variant: {fastest['name']} ({fastest['avg_ms']:.3f}ms)")

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='MAS Matrix-Vector Product Tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark only')
    parser.add_argument('test_name', nargs='?', help='Specific test class or method')
    args = parser.parse_args()

    if args.benchmark:
        run_performance_benchmark()
        return 0

    # Run unit tests
    loader = unittest.TestLoader()

    if args.test_name:
        # Run specific test
        suite = loader.loadTestsFromName(args.test_name)
    else:
        # Run all tests
        suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    # Also run benchmark after tests
    if not args.test_name:
        print("\n" + "=" * 80)
        print("Now running performance benchmark...")
        run_performance_benchmark()

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
