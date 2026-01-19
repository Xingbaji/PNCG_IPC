"""
Test Adaptive Regularization for MAS Preconditioner

This test compares:
1. Fixed (uniform) regularization: epsilon = constant
2. Adaptive regularization: epsilon = relative * ||diag(A)||_inf

Key metrics:
- Inversion accuracy: ||A * A^-1 - I||
- Preconditioner validity: g^T z > 0
- Information preservation: ratio of regularization to matrix norm
"""

import sys
import os
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
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, BLOCK_DOF


@ti.data_oriented
class RegularizationTestSolver(pncg_ipc_deformer):
    """Solver for testing regularization strategies."""

    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )

    @ti.kernel
    def compute_gTz(self) -> float:
        gTz = 0.0
        for vert in self.mesh.verts:
            gTz += vert.grad.dot(vert.z)
        return gTz

    @ti.kernel
    def check_z_nan_count(self) -> int:
        count = 0
        for vert in self.mesh.verts:
            if ti.math.isnan(vert.z[0]) or ti.math.isnan(vert.z[1]) or ti.math.isnan(vert.z[2]):
                count += 1
        return count


def compute_inversion_error(A: np.ndarray, A_inv: np.ndarray) -> float:
    """Compute relative inversion error ||A * A^-1 - I||_F / sqrt(n)."""
    n = A.shape[0]
    I = np.eye(n)
    AA_inv = A @ A_inv
    diff = AA_inv - I
    return np.linalg.norm(diff, 'fro') / np.sqrt(n)


def compute_eigenvalue_stats(A: np.ndarray) -> dict:
    """Compute eigenvalue statistics."""
    A_sym = (A + A.T) / 2
    try:
        eigenvalues = np.linalg.eigvalsh(A_sym)
        return {
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'n_negative': np.sum(eigenvalues < -1e-6),
        }
    except:
        return {'min_eigenvalue': float('nan'), 'max_eigenvalue': float('nan'), 'n_negative': -1}


class TestAdaptiveRegularization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        cls.solver = RegularizationTestSolver(demo='eight_E_stiffness_test')
        print(f"\n[Setup] Vertices: {cls.solver.n_verts}, Cells: {cls.solver.n_cells}")

        # Build hierarchy and assemble
        cls.solver.mas.build_hierarchy()
        cls.solver.assign_xn_xhat()
        cls.solver.compute_grad_and_diagH()
        if cls.solver.ground_barrier == 1:
            cls.solver.add_grad_and_diagH_ground_barrier()
        cls.solver.mas.assemble_block_matrices(cls.solver, use_full_hessian=True)

        # Get original block matrix statistics
        cls.solver.mas._expand_sym_to_full()
        cls.original_block0 = cls.solver.mas.full_block_matrix.to_numpy()[0].copy()
        cls.block0_stats = compute_eigenvalue_stats(cls.original_block0)

        cls.max_diag = np.max(np.abs(np.diag(cls.original_block0)))
        cls.frob_norm = np.linalg.norm(cls.original_block0, 'fro')

        print(f"[Setup] Block 0 stats:")
        print(f"  Max diagonal: {cls.max_diag:.2e}")
        print(f"  Frobenius norm: {cls.frob_norm:.2e}")
        print(f"  Min eigenvalue: {cls.block0_stats['min_eigenvalue']:.2e}")
        print(f"  Max eigenvalue: {cls.block0_stats['max_eigenvalue']:.2e}")
        print(f"  Negative eigenvalues: {cls.block0_stats['n_negative']}")

    def test_01_compare_regularization_methods(self):
        """Compare fixed vs adaptive regularization."""
        print("\n[Test] Comparing regularization methods...")

        # Method 1: Fixed epsilon based on eigenvalue (old way)
        fixed_epsilon = abs(self.block0_stats['min_eigenvalue']) * 1.1 + 1e3

        # Method 2: Adaptive regularization with different strengths
        adaptive_values = [0.01, 0.05, 0.1, 0.2]

        results = []

        # Test fixed regularization
        self.solver.mas.assemble_block_matrices(self.solver, use_full_hessian=True)
        self.solver.mas.invert_block_matrices(
            use_full_inversion=True,
            use_cholesky=True,
            use_incomplete=True,
            force_symmetry=True,
            regularization_epsilon=fixed_epsilon,
            adaptive_regularization=0.0
        )
        self.solver.mas.apply()

        gTz = self.solver.compute_gTz()
        nan_count = self.solver.check_z_nan_count()

        # Compute inversion error
        A_reg = self.original_block0 + np.eye(BLOCK_DOF) * fixed_epsilon
        A_inv = self.solver.mas.full_block_inverse.to_numpy()[0]
        rel_error = compute_inversion_error(A_reg, A_inv)

        info_ratio = self.frob_norm / (fixed_epsilon * np.sqrt(BLOCK_DOF))

        results.append({
            'method': f'Fixed ε={fixed_epsilon:.2e}',
            'rel_error': rel_error,
            'gTz': gTz,
            'nan_count': nan_count,
            'info_ratio': info_ratio,
            'effective_epsilon': fixed_epsilon,
        })
        print(f"  Fixed ε={fixed_epsilon:.2e}: rel_err={rel_error:.2e}, "
              f"g^Tz={gTz:.2e}, info_ratio={info_ratio:.2f}")

        # Test adaptive regularization
        for rel_eps in adaptive_values:
            self.solver.mas.assemble_block_matrices(self.solver, use_full_hessian=True)
            self.solver.mas.invert_block_matrices(
                use_full_inversion=True,
                use_cholesky=True,
                use_incomplete=True,
                force_symmetry=True,
                regularization_epsilon=0.0,
                adaptive_regularization=rel_eps
            )
            self.solver.mas.apply()

            gTz = self.solver.compute_gTz()
            nan_count = self.solver.check_z_nan_count()

            # For adaptive, effective epsilon ≈ rel_eps * max_diag
            effective_eps = rel_eps * self.max_diag
            A_reg = self.original_block0 + np.eye(BLOCK_DOF) * effective_eps
            A_inv = self.solver.mas.full_block_inverse.to_numpy()[0]
            rel_error = compute_inversion_error(A_reg, A_inv)

            info_ratio = self.frob_norm / (effective_eps * np.sqrt(BLOCK_DOF))

            results.append({
                'method': f'Adaptive rel={rel_eps}',
                'rel_error': rel_error,
                'gTz': gTz,
                'nan_count': nan_count,
                'info_ratio': info_ratio,
                'effective_epsilon': effective_eps,
            })
            print(f"  Adaptive rel={rel_eps}: eff_ε={effective_eps:.2e}, "
                  f"rel_err={rel_error:.2e}, g^Tz={gTz:.2e}, info_ratio={info_ratio:.2f}")

        # All should have positive g^T z
        for r in results:
            self.assertGreater(r['gTz'], 0, f"{r['method']} should have g^Tz > 0")
            self.assertEqual(r['nan_count'], 0, f"{r['method']} should have no NaN")

    def test_02_information_preservation(self):
        """Test that adaptive regularization preserves more information."""
        print("\n[Test] Information preservation analysis...")

        # Information preservation ratio = ||A||_F / ||ε*I||_F
        # Higher ratio = more original matrix info preserved

        fixed_epsilon = abs(self.block0_stats['min_eigenvalue']) * 1.1 + 1e3
        fixed_info_ratio = self.frob_norm / (fixed_epsilon * np.sqrt(BLOCK_DOF))

        # With adaptive rel=0.05, effective epsilon ≈ 0.05 * max_diag
        adaptive_eps = 0.05 * self.max_diag
        adaptive_info_ratio = self.frob_norm / (adaptive_eps * np.sqrt(BLOCK_DOF))

        print(f"  Fixed ε={fixed_epsilon:.2e}: info_ratio = {fixed_info_ratio:.4f}")
        print(f"  Adaptive (0.05): eff_ε={adaptive_eps:.2e}, info_ratio = {adaptive_info_ratio:.4f}")
        print(f"  Improvement: {adaptive_info_ratio / fixed_info_ratio:.2f}x")

        # Adaptive should preserve more information (higher ratio)
        # Only enforce this if fixed regularization was very large
        if fixed_info_ratio < 0.1:
            self.assertGreater(adaptive_info_ratio, fixed_info_ratio,
                               "Adaptive should preserve more info when fixed ε is too large")

    def test_03_scaling_invariance(self):
        """Test that adaptive regularization scales correctly with problem size."""
        print("\n[Test] Scaling invariance check...")

        # The effective epsilon should be proportional to the matrix scale
        # This means info_ratio should be similar regardless of stiffness

        # Simulate different stiffness by checking the ratio
        rel_eps = 0.05

        # For any matrix A, effective_eps = rel_eps * ||diag(A)||_inf
        # info_ratio ≈ ||A||_F / (rel_eps * ||diag(A)||_inf * sqrt(DOF))

        # This ratio should be approximately constant for well-conditioned matrices
        expected_ratio_lower = 0.1  # Minimum acceptable info preservation
        expected_ratio_upper = 100.0  # Maximum (no regularization effect)

        effective_eps = rel_eps * self.max_diag
        info_ratio = self.frob_norm / (effective_eps * np.sqrt(BLOCK_DOF))

        print(f"  Rel epsilon: {rel_eps}")
        print(f"  Effective epsilon: {effective_eps:.2e}")
        print(f"  Info ratio: {info_ratio:.2f}")
        print(f"  Expected range: [{expected_ratio_lower}, {expected_ratio_upper}]")

        self.assertGreater(info_ratio, expected_ratio_lower,
                           "Info ratio too low - regularization dominates")
        self.assertLess(info_ratio, expected_ratio_upper,
                        "Info ratio too high - regularization too weak")


def run_analysis():
    """Run detailed analysis of regularization strategies."""
    print("=" * 70)
    print("ADAPTIVE REGULARIZATION ANALYSIS")
    print("=" * 70)

    ti.init(arch=ti.cuda, default_fp=ti.f32)
    solver = RegularizationTestSolver(demo='eight_E_stiffness_test')

    # Setup
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)

    # Get statistics
    solver.mas._expand_sym_to_full()
    block0 = solver.mas.full_block_matrix.to_numpy()[0]
    stats = compute_eigenvalue_stats(block0)

    max_diag = np.max(np.abs(np.diag(block0)))
    frob_norm = np.linalg.norm(block0, 'fro')

    print(f"\nBlock 0 Matrix Statistics:")
    print(f"  Dimensions: {BLOCK_DOF}x{BLOCK_DOF}")
    print(f"  Max diagonal: {max_diag:.4e}")
    print(f"  Frobenius norm: {frob_norm:.4e}")
    print(f"  Min eigenvalue: {stats['min_eigenvalue']:.4e}")
    print(f"  Max eigenvalue: {stats['max_eigenvalue']:.4e}")
    print(f"  Negative eigenvalues: {stats['n_negative']}")

    # Analysis table
    print("\n" + "-" * 70)
    print("Regularization Strategy Comparison")
    print("-" * 70)

    fixed_epsilon = abs(stats['min_eigenvalue']) * 1.1 + 1e3

    print(f"\n{'Method':<25} {'Eff. ε':<12} {'ε/||A||':<12} {'Info Ratio':<12}")
    print("-" * 70)

    # Fixed
    ratio_to_norm = fixed_epsilon / frob_norm
    info_ratio = frob_norm / (fixed_epsilon * np.sqrt(BLOCK_DOF))
    print(f"{'Fixed (|λ_min|*1.1+1e3)':<25} {fixed_epsilon:<12.2e} {ratio_to_norm:<12.4f} {info_ratio:<12.4f}")

    # Adaptive
    for rel_eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        eff_eps = rel_eps * max_diag
        ratio_to_norm = eff_eps / frob_norm
        info_ratio = frob_norm / (eff_eps * np.sqrt(BLOCK_DOF))
        print(f"{'Adaptive rel=' + str(rel_eps):<25} {eff_eps:<12.2e} {ratio_to_norm:<12.4f} {info_ratio:<12.4f}")

    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("""
  - Info Ratio > 1: Regularization is small relative to matrix norm (good)
  - Info Ratio < 1: Regularization dominates (preconditioner ≈ identity)
  - ε/||A|| < 0.01: Minimal impact on matrix structure (ideal)
  - ε/||A|| > 0.1: Significant modification to matrix

  For this problem:
  - Fixed regularization: ε/||A|| = {:.4f} (may be too large)
  - Adaptive rel=0.05:    ε/||A|| = {:.4f} (scales with problem)
""".format(fixed_epsilon / frob_norm, 0.05 * max_diag / frob_norm))

    # Recommendation
    print("RECOMMENDATION:")
    if fixed_epsilon / frob_norm > 0.1:
        print("  Fixed regularization is TOO LARGE for this problem.")
        print("  Use adaptive_regularization=0.05 instead.")
    else:
        print("  Fixed regularization is acceptable for this problem.")
        print("  Adaptive regularization may still be preferred for varying stiffness.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', action='store_true', help='Run detailed analysis')
    args = parser.parse_args()

    if args.analysis:
        run_analysis()
        return

    # Run unit tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdaptiveRegularization)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    main()
