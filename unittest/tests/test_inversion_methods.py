"""
Inversion Methods Comprehensive Benchmark Test

Compare different block matrix inversion methods:
- gauss_jordan: Full Gauss-Jordan elimination (most robust)
- oneway_gj: One-way Gauss-Jordan (faster, optimized for SPD)
- cholesky: Cholesky decomposition (requires SPD)
- blocked_cholesky: Blocked Cholesky (better GPU parallelism)
- incomplete: Incomplete Cholesky IC(0) (approximate, fastest)
- diagonal_only: Diagonal blocks only (simplest approximation)

Tests accuracy, numerical stability, and performance.
"""

import sys
import os
import time
import numpy as np
import unittest

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, BLOCK_DOF, SYM_BLOCK_COUNT


# Available inversion methods
# Format: (key, name, kwargs_dict)
INVERSION_METHODS = [
    ('gauss_jordan', 'Gauss-Jordan', {
        'use_full_inversion': True,
        'use_cholesky': False,
        'use_incomplete': False,
        'use_oneway_gj': False,
        'use_blocked': False,
    }),
    ('oneway_gj', 'One-way GJ', {
        'use_full_inversion': True,
        'use_cholesky': False,
        'use_incomplete': False,
        'use_oneway_gj': True,
        'use_blocked': False,
    }),
    ('cholesky', 'Cholesky', {
        'use_full_inversion': True,
        'use_cholesky': True,
        'use_incomplete': False,
        'use_oneway_gj': False,
        'use_blocked': False,
    }),
    ('blocked_cholesky', 'Blocked Cholesky', {
        'use_full_inversion': True,
        'use_cholesky': True,
        'use_incomplete': False,
        'use_oneway_gj': False,
        'use_blocked': True,
    }),
    ('incomplete', 'Incomplete Cholesky IC(0)', {
        'use_full_inversion': True,
        'use_cholesky': True,
        'use_incomplete': True,
        'use_oneway_gj': False,
        'use_blocked': False,
    }),
    ('diagonal_only', 'Diagonal Only', {
        'use_full_inversion': False,
        'use_cholesky': False,
        'use_incomplete': False,
        'use_oneway_gj': False,
        'use_blocked': False,
    }),
]


@ti.data_oriented
class InversionTestSolver(pncg_ipc_deformer):
    """Solver for testing inversion methods."""

    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)

        # Add z field for preconditioned gradient
        self.mesh.verts.place({'z': ti.types.vector(3, float)})

        # Create MAS without METIS
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient."""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def compute_z_norm(self) -> float:
        """Compute L2 norm of z."""
        z_sum = 0.0
        for vert in self.mesh.verts:
            z_sum += vert.z.norm_sqr()
        return ti.sqrt(z_sum)

    @ti.kernel
    def check_z_nan_count(self) -> int:
        """Count NaN values in z."""
        count = 0
        for vert in self.mesh.verts:
            if ti.math.isnan(vert.z[0]) or ti.math.isnan(vert.z[1]) or ti.math.isnan(vert.z[2]):
                count += 1
        return count

    @ti.kernel
    def compute_gTz(self) -> float:
        """Compute g^T * z (should be positive for valid preconditioner)."""
        gTz = 0.0
        for vert in self.mesh.verts:
            gTz += vert.grad.dot(vert.z)
        return gTz

    @ti.kernel
    def compute_init_p_mas(self):
        """p = -z"""
        for vert in self.mesh.verts:
            vert.p = -vert.z


def get_block_matrix_as_numpy(mas, block_id: int) -> np.ndarray:
    """Extract a single full block matrix from MAS as numpy array."""
    mas._expand_sym_to_full()
    full_matrix = mas.full_block_matrix.to_numpy()
    return full_matrix[block_id]


def get_inverse_matrix_as_numpy(mas, block_id: int) -> np.ndarray:
    """Extract a single inverse block matrix from MAS as numpy array."""
    inv_matrix = mas.full_block_inverse.to_numpy()
    return inv_matrix[block_id]


def compute_inversion_error(A: np.ndarray, A_inv: np.ndarray) -> dict:
    """
    Compute various error metrics for matrix inversion.

    Returns:
        dict with keys:
        - frobenius_error: ||A * A_inv - I||_F
        - relative_error: ||A * A_inv - I||_F / ||I||_F
        - max_error: max(|A * A_inv - I|)
        - condition_number: cond(A)
    """
    n = A.shape[0]
    I = np.eye(n)
    AA_inv = A @ A_inv

    diff = AA_inv - I

    frobenius_error = np.linalg.norm(diff, 'fro')
    relative_error = frobenius_error / np.sqrt(n)  # ||I||_F = sqrt(n)
    max_error = np.max(np.abs(diff))

    try:
        condition_number = np.linalg.cond(A)
    except:
        condition_number = float('inf')

    return {
        'frobenius_error': frobenius_error,
        'relative_error': relative_error,
        'max_error': max_error,
        'condition_number': condition_number,
    }


def compute_symmetry_error(A: np.ndarray) -> dict:
    """
    Compute symmetry error of a matrix.

    Returns:
        dict with keys:
        - absolute_error: ||A - A^T||_F
        - relative_error: ||A - A^T||_F / ||A||_F
    """
    diff = A - A.T
    frobenius_norm = np.linalg.norm(A, 'fro')
    absolute_error = np.linalg.norm(diff, 'fro')
    relative_error = absolute_error / frobenius_norm if frobenius_norm > 0 else 0

    return {
        'absolute_error': absolute_error,
        'relative_error': relative_error,
    }


def compute_eigenvalue_stats(A: np.ndarray) -> dict:
    """
    Compute eigenvalue statistics of a matrix.

    Returns:
        dict with keys:
        - min_eigenvalue, max_eigenvalue
        - n_negative: count of negative eigenvalues
        - n_zero: count of near-zero eigenvalues
    """
    # Symmetrize for eigenvalue computation
    A_sym = (A + A.T) / 2

    try:
        eigenvalues = np.linalg.eigvalsh(A_sym)
        return {
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'n_negative': np.sum(eigenvalues < -1e-6),
            'n_zero': np.sum(np.abs(eigenvalues) < 1e-6),
            'eigenvalues': eigenvalues,
        }
    except:
        return {
            'min_eigenvalue': float('nan'),
            'max_eigenvalue': float('nan'),
            'n_negative': -1,
            'n_zero': -1,
            'eigenvalues': None,
        }


class TestInversionMethods(unittest.TestCase):
    """Unit tests for inversion methods."""

    @classmethod
    def setUpClass(cls):
        """Initialize Taichi and solver once."""
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        cls.solver = InversionTestSolver(demo='eight_E_stiffness_test')
        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Cells: {cls.solver.n_cells}")

        # Build hierarchy once
        cls.solver.mas.build_hierarchy()
        print(f"[Setup] MAS levels: {cls.solver.mas.actual_levels}")

        # Assemble matrices once
        cls.solver.assign_xn_xhat()
        cls.solver.compute_grad_and_diagH()
        if cls.solver.ground_barrier == 1:
            cls.solver.add_grad_and_diagH_ground_barrier()
        cls.solver.mas.assemble_block_matrices(cls.solver, use_full_hessian=True)
        print("[Setup] Block matrices assembled")

        # Store original block matrices for testing
        cls.solver.mas._expand_sym_to_full()
        cls.original_matrices = cls.solver.mas.full_block_matrix.to_numpy().copy()

        # Compute eigenvalue info for Block 0
        block0 = cls.original_matrices[0]
        cls.block0_eigen = compute_eigenvalue_stats(block0)
        print(f"[Setup] Block 0 eigenvalues: min={cls.block0_eigen['min_eigenvalue']:.2e}, "
              f"max={cls.block0_eigen['max_eigenvalue']:.2e}, "
              f"n_negative={cls.block0_eigen['n_negative']}")

    def test_01_block_matrix_symmetry(self):
        """Test that assembled block matrices are symmetric (after fix)."""
        print("\n[Test] Block matrix symmetry...")

        n_blocks = min(10, self.solver.mas.full_block_matrix.shape[0])
        max_rel_error = 0.0

        for block_id in range(n_blocks):
            A = self.original_matrices[block_id]
            sym_error = compute_symmetry_error(A)
            max_rel_error = max(max_rel_error, sym_error['relative_error'])

        print(f"  Max relative symmetry error: {max_rel_error:.2e}")
        self.assertLess(max_rel_error, 1e-5, "Block matrices should be nearly symmetric")

    def test_02_gauss_jordan_accuracy(self):
        """Test Gauss-Jordan inversion accuracy."""
        print("\n[Test] Gauss-Jordan accuracy...")

        self._run_inversion_accuracy_test('gauss_jordan', 'Gauss-Jordan',
                                          force_symmetry=True, regularization=0.0,
                                          expected_rel_error=1e-3)

    def test_03_oneway_gj_validity(self):
        """Test One-way Gauss-Jordan produces valid preconditioned direction.

        Note: One-way GJ is optimized for SPD matrices. For non-SPD matrices,
        it may have poor inversion accuracy but can still produce a valid
        preconditioned direction (g^T z > 0). This test verifies validity
        rather than strict accuracy.
        """
        print("\n[Test] One-way GJ validity...")

        method_info = next((m for m in INVERSION_METHODS if m[0] == 'oneway_gj'), None)
        self.assertIsNotNone(method_info)

        _, method_name, kwargs = method_info

        # Reassemble and invert (no regularization - test robustness)
        self.solver.mas.assemble_block_matrices(self.solver, use_full_hessian=True)
        self.solver.mas.invert_block_matrices(**kwargs, force_symmetry=True,
                                               regularization_epsilon=0.0)

        # Apply preconditioner
        self.solver.mas.apply()

        # Check results - focus on validity not accuracy
        nan_count = self.solver.check_z_nan_count()
        gTz = self.solver.compute_gTz()

        print(f"  g^T z = {gTz:.4e}, NaN count = {nan_count}")

        self.assertEqual(nan_count, 0, "One-way GJ should not produce NaN")
        self.assertGreater(gTz, 0, "One-way GJ should produce positive g^T z")

    def test_04_cholesky_with_regularization(self):
        """Test Cholesky with regularization (for non-SPD matrices)."""
        print("\n[Test] Cholesky with regularization...")

        # Need regularization since block matrices are not SPD
        reg_epsilon = abs(self.block0_eigen['min_eigenvalue']) * 1.1 + 1e3

        self._run_inversion_accuracy_test('cholesky', 'Cholesky',
                                          force_symmetry=True, regularization=reg_epsilon,
                                          expected_rel_error=1e-2)

    def test_05_incomplete_cholesky_with_regularization(self):
        """Test Incomplete Cholesky with regularization."""
        print("\n[Test] Incomplete Cholesky with regularization...")

        reg_epsilon = abs(self.block0_eigen['min_eigenvalue']) * 1.1 + 1e3

        self._run_inversion_accuracy_test('incomplete', 'Incomplete Cholesky',
                                          force_symmetry=True, regularization=reg_epsilon,
                                          expected_rel_error=0.1)  # IC(0) is approximate

    def test_06_preconditioner_validity(self):
        """Test that z = M^{-1} g produces valid preconditioned direction (g^T z > 0)."""
        print("\n[Test] Preconditioner validity (g^T z > 0)...")

        methods_to_test = [
            ('gauss_jordan', 0.0),
            ('oneway_gj', 0.0),
        ]

        # Add methods requiring regularization
        reg_epsilon = abs(self.block0_eigen['min_eigenvalue']) * 1.1 + 1e3
        methods_to_test.extend([
            ('cholesky', reg_epsilon),
            ('incomplete', reg_epsilon),
        ])

        for method_key, reg in methods_to_test:
            method_info = next((m for m in INVERSION_METHODS if m[0] == method_key), None)
            if method_info is None:
                continue

            _, method_name, kwargs = method_info

            # Reassemble matrices (inversion modifies them)
            self.solver.mas.assemble_block_matrices(self.solver, use_full_hessian=True)

            # Invert
            self.solver.mas.invert_block_matrices(
                **kwargs,
                force_symmetry=True,
                regularization_epsilon=reg
            )

            # Apply preconditioner
            self.solver.mas.apply()

            # Check results
            nan_count = self.solver.check_z_nan_count()
            gTz = self.solver.compute_gTz()

            print(f"  {method_name}: g^T z = {gTz:.4e}, NaN count = {nan_count}")

            self.assertEqual(nan_count, 0, f"{method_name} should not produce NaN")
            self.assertGreater(gTz, 0, f"{method_name} should produce positive g^T z")

    def _run_inversion_accuracy_test(self, method_key: str, method_name: str,
                                      force_symmetry: bool, regularization: float,
                                      expected_rel_error: float):
        """Helper to run inversion accuracy test for a single method."""
        method_info = next((m for m in INVERSION_METHODS if m[0] == method_key), None)
        self.assertIsNotNone(method_info, f"Method {method_key} not found")

        _, _, kwargs = method_info

        # Reassemble matrices (inversion modifies them)
        self.solver.mas.assemble_block_matrices(self.solver, use_full_hessian=True)

        # Apply regularization to original for comparison
        self.solver.mas._expand_sym_to_full()
        A_original = self.solver.mas.full_block_matrix.to_numpy()[0].copy()
        if regularization > 0:
            A_original += np.eye(BLOCK_DOF) * regularization

        # Invert
        self.solver.mas.invert_block_matrices(
            **kwargs,
            force_symmetry=force_symmetry,
            regularization_epsilon=regularization
        )

        # Get inverse
        A_inv = get_inverse_matrix_as_numpy(self.solver.mas, 0)

        # Check for NaN
        nan_count = np.sum(np.isnan(A_inv))
        self.assertEqual(nan_count, 0, f"{method_name} inverse should not contain NaN")

        # Compute error
        error = compute_inversion_error(A_original, A_inv)

        print(f"  Relative error: {error['relative_error']:.4e}")
        print(f"  Max element error: {error['max_error']:.4e}")
        print(f"  Condition number: {error['condition_number']:.2e}")

        self.assertLess(error['relative_error'], expected_rel_error,
                        f"{method_name} relative error should be < {expected_rel_error}")


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Reinitialize
    ti.reset()
    ti.init(arch=ti.cuda, default_fp=ti.f32)

    solver = InversionTestSolver(demo='eight_E_stiffness_test')
    print(f"\n[Setup] Vertices: {solver.n_verts}, Cells: {solver.n_cells}")

    # Build hierarchy
    solver.mas.build_hierarchy()
    n_blocks = (solver.mas.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE
    print(f"[Setup] MAS levels: {solver.mas.actual_levels}, Total blocks: {n_blocks}")

    # Get eigenvalue info for regularization
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)

    solver.mas._expand_sym_to_full()
    block0 = solver.mas.full_block_matrix.to_numpy()[0]
    eigen_stats = compute_eigenvalue_stats(block0)
    reg_epsilon = abs(eigen_stats['min_eigenvalue']) * 1.1 + 1e3
    print(f"[Setup] Using regularization epsilon: {reg_epsilon:.2e}")

    results = []

    print("\n" + "-" * 80)
    print("Running benchmarks (3 warmup + 10 timed iterations each)...")
    print("-" * 80)

    for method_key, method_name, kwargs in INVERSION_METHODS:
        print(f"\n[{method_name}]")

        # Determine if method needs regularization
        needs_reg = method_key in ['cholesky', 'blocked_cholesky', 'incomplete']
        reg = reg_epsilon if needs_reg else 0.0

        # Warmup
        for _ in range(3):
            solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
            try:
                solver.mas.invert_block_matrices(**kwargs, force_symmetry=True,
                                                  regularization_epsilon=reg)
            except Exception as e:
                print(f"  [ERROR] {e}")
                break

        # Timed runs
        assemble_times = []
        invert_times = []
        apply_times = []
        success = True

        for _ in range(10):
            # Assemble
            ti.sync()
            t0 = time.perf_counter()
            solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
            ti.sync()
            t1 = time.perf_counter()
            assemble_times.append((t1 - t0) * 1000)

            # Invert
            ti.sync()
            t2 = time.perf_counter()
            try:
                solver.mas.invert_block_matrices(**kwargs, force_symmetry=True,
                                                  regularization_epsilon=reg)
            except Exception as e:
                print(f"  [ERROR] Inversion failed: {e}")
                success = False
                break
            ti.sync()
            t3 = time.perf_counter()
            invert_times.append((t3 - t2) * 1000)

            # Apply
            ti.sync()
            t4 = time.perf_counter()
            solver.mas.apply()
            ti.sync()
            t5 = time.perf_counter()
            apply_times.append((t5 - t4) * 1000)

        if not success:
            results.append({
                'method': method_name,
                'success': False,
            })
            continue

        # Check validity
        nan_count = solver.check_z_nan_count()
        gTz = solver.compute_gTz()

        # Compute accuracy for Block 0
        solver.mas._expand_sym_to_full()
        A = solver.mas.full_block_matrix.to_numpy()[0]
        if reg > 0:
            A = block0 + np.eye(BLOCK_DOF) * reg
        else:
            A = block0.copy()
        A_inv = solver.mas.full_block_inverse.to_numpy()[0]
        error = compute_inversion_error(A, A_inv)

        result = {
            'method': method_name,
            'success': True,
            'nan_count': nan_count,
            'gTz': gTz,
            'gTz_positive': gTz > 0,
            'avg_assemble_ms': np.mean(assemble_times),
            'avg_invert_ms': np.mean(invert_times),
            'avg_apply_ms': np.mean(apply_times),
            'std_invert_ms': np.std(invert_times),
            'relative_error': error['relative_error'],
            'max_error': error['max_error'],
            'regularization': reg,
        }
        results.append(result)

        print(f"  Assemble: {result['avg_assemble_ms']:.2f}ms")
        print(f"  Invert:   {result['avg_invert_ms']:.2f}ms (±{result['std_invert_ms']:.2f})")
        print(f"  Apply:    {result['avg_apply_ms']:.2f}ms")
        print(f"  g^T z:    {result['gTz']:.4e} ({'OK' if result['gTz_positive'] else 'BAD'})")
        print(f"  Rel. err: {result['relative_error']:.4e}")
        print(f"  NaN:      {result['nan_count']}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<25} {'Status':<8} {'Invert(ms)':<12} {'Rel.Err':<12} {'g^Tz>0':<8} {'Reg.ε':<12}")
    print("-" * 80)

    for r in results:
        if r['success']:
            status = "OK"
            invert_str = f"{r['avg_invert_ms']:.2f}±{r['std_invert_ms']:.1f}"
            err_str = f"{r['relative_error']:.2e}"
            gtz_str = "Yes" if r['gTz_positive'] else "No"
            reg_str = f"{r['regularization']:.1e}" if r['regularization'] > 0 else "0"
        else:
            status = "FAILED"
            invert_str = "N/A"
            err_str = "N/A"
            gtz_str = "N/A"
            reg_str = "N/A"

        print(f"{r['method']:<25} {status:<8} {invert_str:<12} {err_str:<12} {gtz_str:<8} {reg_str:<12}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)

    successful = [r for r in results if r['success'] and r['gTz_positive']]

    if successful:
        fastest = min(successful, key=lambda x: x['avg_invert_ms'])
        most_accurate = min(successful, key=lambda x: x['relative_error'])

        print(f"\n  Fastest inversion:      {fastest['method']} ({fastest['avg_invert_ms']:.2f}ms)")
        print(f"  Most accurate:          {most_accurate['method']} (rel.err={most_accurate['relative_error']:.2e})")

        # Best balance
        # Score = speed_rank + accuracy_rank
        sorted_by_speed = sorted(successful, key=lambda x: x['avg_invert_ms'])
        sorted_by_acc = sorted(successful, key=lambda x: x['relative_error'])

        scores = {}
        for i, r in enumerate(sorted_by_speed):
            scores[r['method']] = i
        for i, r in enumerate(sorted_by_acc):
            scores[r['method']] += i

        best_balance = min(scores.items(), key=lambda x: x[1])[0]
        print(f"  Best balance:           {best_balance}")

        print("\n  Recommended for production:")
        print(f"    - General use: Gauss-Jordan (most robust)")
        print(f"    - Speed critical: {fastest['method']}")
        print(f"    - SPD matrices only: Cholesky or Incomplete Cholesky")
    else:
        print("\n  WARNING: No methods produced valid results!")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='MAS Inversion Methods Benchmark')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--benchmark-only', action='store_true', help='Only run performance benchmark')
    parser.add_argument('--test-only', action='store_true', help='Only run unit tests')
    args = parser.parse_args()

    if args.benchmark_only:
        run_performance_benchmark()
        return

    if args.test_only:
        # Run unit tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestInversionMethods)
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    # Run both
    print("=" * 80)
    print("MAS INVERSION METHODS COMPREHENSIVE TEST")
    print("=" * 80)

    # Unit tests first
    print("\n" + "=" * 80)
    print("UNIT TESTS")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInversionMethods)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    # Then benchmark
    benchmark_results = run_performance_benchmark()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nUnit Tests: {test_result.testsRun} run, "
          f"{len(test_result.failures)} failures, "
          f"{len(test_result.errors)} errors")

    if benchmark_results:
        n_successful = sum(1 for r in benchmark_results if r['success'])
        print(f"Benchmark: {n_successful}/{len(benchmark_results)} methods successful")


if __name__ == '__main__':
    main()
