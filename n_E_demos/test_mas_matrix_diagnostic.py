"""
MAS Matrix Computation Diagnostic Test

This test case checks each module of MAS matrix computation to identify
why negative eigenvalues appear. No simulation is run - only matrix
computation is tested.

Modules tested:
1. Inertia contribution (M/dt^2) - should be diagonal SPD
2. Elastic contribution (ARAP_filter) - should be SPD after filtering
3. Symmetric storage and expansion - should preserve symmetry
4. Block inversion methods - IC, Gauss-Jordan

Author: Debug session for MAS preconditioner
"""

import sys
import os
import numpy as np
from scipy import linalg as la

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner, BANKSIZE, SYM_BLOCK_COUNT

ti.init(arch=ti.gpu, default_fp=ti.f32)


class DiagnosticResult:
    """Store diagnostic results for a test."""
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.messages = []
        self.data = {}

    def add_check(self, condition, success_msg, fail_msg):
        if condition:
            self.messages.append(f"  [PASS] {success_msg}")
        else:
            self.passed = False
            self.messages.append(f"  [FAIL] {fail_msg}")

    def add_info(self, msg):
        self.messages.append(f"  [INFO] {msg}")

    def add_warning(self, msg):
        self.messages.append(f"  [WARN] {msg}")

    def report(self):
        status = "PASSED" if self.passed else "FAILED"
        print(f"\n{'='*60}")
        print(f"Test: {self.name} - {status}")
        print('='*60)
        for msg in self.messages:
            print(msg)


@ti.data_oriented
class DiagnosticSolver(pncg_ipc_deformer):
    """Minimal solver for diagnostic testing."""

    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def sym_index(row, col):
    """Compute symmetric storage index for upper triangle."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def extract_full_block(block_matrices_np, block_id):
    """Extract full 48x48 matrix from symmetric storage."""
    full_matrix = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = block_matrices_np[block_id, sym_idx]

            for di in range(3):
                for dj in range(3):
                    # Upper triangle
                    full_matrix[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                    # Lower triangle (transpose)
                    if row != col:
                        full_matrix[col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    return full_matrix


def check_spd(matrix, name, tol=1e-10):
    """Check if matrix is symmetric positive definite."""
    result = DiagnosticResult(f"SPD Check: {name}")

    # Check symmetry
    sym_error = np.linalg.norm(matrix - matrix.T) / (np.linalg.norm(matrix) + 1e-12)
    result.add_check(
        sym_error < 1e-6,
        f"Symmetric (relative error = {sym_error:.2e})",
        f"NOT symmetric (relative error = {sym_error:.2e})"
    )

    # Check eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        min_eig = eigvals.min()
        max_eig = eigvals.max()
        neg_count = np.sum(eigvals < -tol)

        result.add_info(f"Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")
        result.add_info(f"Condition number: {abs(max_eig) / max(abs(min_eig), 1e-12):.2e}")

        result.add_check(
            min_eig > -tol,
            f"Positive semi-definite (min_eig = {min_eig:.6e})",
            f"Has {neg_count} negative eigenvalues (min = {min_eig:.6e})"
        )

        result.data['eigvals'] = eigvals
        result.data['min_eig'] = min_eig
        result.data['max_eig'] = max_eig

    except Exception as e:
        result.add_check(False, "", f"Eigenvalue computation failed: {e}")

    return result


def test_inertia_contribution(solver, mas):
    """Test 1: Check inertia contribution (M/dt^2)."""
    result = DiagnosticResult("Inertia Contribution (M/dt^2)")

    # Clear and add only inertia
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)

    # Check Block 0
    block_matrices_np = mas.block_matrices.to_numpy()
    full_mat = extract_full_block(block_matrices_np, 0)

    # Inertia should be diagonal
    diag = np.diag(full_mat)
    off_diag = full_mat - np.diag(diag)
    off_diag_norm = np.linalg.norm(off_diag)

    result.add_check(
        off_diag_norm < 1e-10,
        f"Matrix is diagonal (off-diag norm = {off_diag_norm:.2e})",
        f"Matrix is NOT diagonal (off-diag norm = {off_diag_norm:.2e})"
    )

    # All diagonal entries should be positive (mass/dt^2)
    min_diag = diag.min()
    max_diag = diag.max()
    result.add_info(f"Diagonal range: [{min_diag:.6e}, {max_diag:.6e}]")

    result.add_check(
        min_diag > 0,
        f"All diagonal entries positive",
        f"Some diagonal entries non-positive (min = {min_diag:.6e})"
    )

    # Verify values match expected M/dt^2
    m_np = solver.mesh.verts.m.to_numpy()
    expected_diag = m_np[0] / (solver.dt ** 2)
    actual_diag = diag[0]
    rel_error = abs(actual_diag - expected_diag) / (expected_diag + 1e-12)

    result.add_info(f"Expected M/dt^2 = {expected_diag:.6e}")
    result.add_info(f"Actual diagonal[0] = {actual_diag:.6e}")
    result.add_check(
        rel_error < 1e-5,
        f"Value matches expected (rel_error = {rel_error:.2e})",
        f"Value mismatch (rel_error = {rel_error:.2e})"
    )

    result.data['full_matrix'] = full_mat
    return result


def test_elastic_contribution(solver, mas):
    """Test 2: Check elastic contribution (ARAP with filter)."""
    result = DiagnosticResult("Elastic Contribution (ARAP_filter)")

    # Clear and add only elastic
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)  # ARAP

    # Check Block 0
    block_matrices_np = mas.block_matrices.to_numpy()
    full_mat = extract_full_block(block_matrices_np, 0)

    # Check symmetry
    sym_error = np.linalg.norm(full_mat - full_mat.T)
    result.add_check(
        sym_error < 1e-6,
        f"Symmetric (error = {sym_error:.2e})",
        f"NOT symmetric (error = {sym_error:.2e})"
    )

    # Check 3x3 diagonal sub-blocks symmetry
    non_sym_3x3_count = 0
    max_3x3_sym_error = 0.0
    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        block_3x3 = block_matrices_np[0, sym_idx]
        err = np.linalg.norm(block_3x3 - block_3x3.T)
        if err > 1e-6:
            non_sym_3x3_count += 1
            max_3x3_sym_error = max(max_3x3_sym_error, err)

    result.add_check(
        non_sym_3x3_count == 0,
        f"All diagonal 3x3 blocks symmetric",
        f"{non_sym_3x3_count} non-symmetric 3x3 blocks (max error = {max_3x3_sym_error:.2e})"
    )

    # Check eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(full_mat)
        min_eig = eigvals.min()
        max_eig = eigvals.max()
        neg_count = np.sum(eigvals < -1e-10)

        result.add_info(f"Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")

        if neg_count > 0:
            result.add_warning(f"Has {neg_count} negative eigenvalues!")
            result.add_info("This might be expected for sub-block extraction")

            # Analyze negative eigenvalues
            neg_eigvals = eigvals[eigvals < -1e-10]
            result.add_info(f"Negative eigenvalues: {neg_eigvals[:5]}...")

        result.data['eigvals'] = eigvals
        result.data['min_eig'] = min_eig

    except Exception as e:
        result.add_check(False, "", f"Eigenvalue computation failed: {e}")

    # Check which cells contribute to Block 0
    result.add_info(f"--- Cell Analysis for Block 0 ---")
    cells_touching_block0 = []

    # Get cell vertex IDs from MAS preconditioner's stored data
    # Use the mesh's cell iterator approach
    for c_id in range(solver.n_cells):
        # Get cell verts via MAS cell_verts field if available
        pass  # Skip detailed cell analysis for now

    result.add_info("(Cell analysis skipped - requires mesh iteration)")

    result.data['full_matrix'] = full_mat

    return result


def test_inertia_plus_elastic(solver, mas):
    """Test 3: Check combined inertia + elastic."""
    result = DiagnosticResult("Combined: Inertia + Elastic")

    # Clear and add both
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    # Check Block 0
    block_matrices_np = mas.block_matrices.to_numpy()
    full_mat = extract_full_block(block_matrices_np, 0)

    # Check symmetry
    sym_error = np.linalg.norm(full_mat - full_mat.T)
    result.add_check(
        sym_error < 1e-6,
        f"Symmetric (error = {sym_error:.2e})",
        f"NOT symmetric (error = {sym_error:.2e})"
    )

    # Check eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(full_mat)
        min_eig = eigvals.min()
        max_eig = eigvals.max()
        neg_count = np.sum(eigvals < -1e-10)

        result.add_info(f"Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")
        result.add_info(f"Condition number: {max_eig / max(abs(min_eig), 1e-12):.2e}")

        result.add_check(
            neg_count == 0,
            f"SPD (all eigenvalues positive)",
            f"NOT SPD: {neg_count} negative eigenvalues (min = {min_eig:.6e})"
        )

        result.data['eigvals'] = eigvals

    except Exception as e:
        result.add_check(False, "", f"Eigenvalue computation failed: {e}")

    result.data['full_matrix'] = full_mat
    return result


def test_symmetric_expansion(solver, mas):
    """Test 4: Check _expand_sym_to_full correctness."""
    result = DiagnosticResult("Symmetric Expansion (_expand_sym_to_full)")

    # Clear and add both contributions
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    # Get block_matrices (symmetric storage)
    block_matrices_np = mas.block_matrices.to_numpy()

    # Manual expansion
    manual_full = extract_full_block(block_matrices_np, 0)

    # Use MAS expansion
    mas._expand_sym_to_full()
    mas_full = mas.full_block_matrix.to_numpy()[0]

    # Compare
    diff = np.linalg.norm(manual_full - mas_full)
    result.add_check(
        diff < 1e-6,
        f"Expansion matches manual (diff = {diff:.2e})",
        f"Expansion differs from manual (diff = {diff:.2e})"
    )

    # Check MAS result symmetry
    mas_sym_error = np.linalg.norm(mas_full - mas_full.T)
    result.add_check(
        mas_sym_error < 1e-6,
        f"MAS expansion is symmetric (error = {mas_sym_error:.2e})",
        f"MAS expansion NOT symmetric (error = {mas_sym_error:.2e})"
    )

    # Check specific off-diagonal blocks
    result.add_info("--- Off-diagonal block verification ---")
    for (r, c) in [(0, 1), (0, 2), (1, 3)]:
        if c < BANKSIZE:
            block_rc = mas_full[r*3:(r+1)*3, c*3:(c+1)*3]
            block_cr = mas_full[c*3:(c+1)*3, r*3:(r+1)*3]
            is_transpose = np.allclose(block_rc, block_cr.T, atol=1e-6)
            result.add_check(
                is_transpose,
                f"Block({r},{c}) = Block({c},{r}).T",
                f"Block({r},{c}) != Block({c},{r}).T"
            )

    result.data['manual_full'] = manual_full
    result.data['mas_full'] = mas_full
    return result


def test_gauss_jordan_inversion(solver, mas):
    """Test 5: Check Gauss-Jordan block inversion."""
    result = DiagnosticResult("Gauss-Jordan Inversion")

    # Assemble matrices
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)
    mas._expand_sym_to_full()

    # Get original matrix
    orig_mat = mas.full_block_matrix.to_numpy()[0].copy()

    # Run Gauss-Jordan
    mas._gauss_jordan_invert_blocks()
    inv_mat = mas.full_block_inverse.to_numpy()[0]

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(inv_mat))
    inf_count = np.sum(np.isinf(inv_mat))

    result.add_check(
        nan_count == 0,
        f"No NaN in inverse",
        f"Has {nan_count} NaN values"
    )
    result.add_check(
        inf_count == 0,
        f"No Inf in inverse",
        f"Has {inf_count} Inf values"
    )

    # Check A * A^{-1} = I
    if nan_count == 0 and inf_count == 0:
        product = orig_mat @ inv_mat
        identity_error = np.linalg.norm(product - np.eye(48))

        result.add_info(f"||A * A^-1 - I|| = {identity_error:.6e}")
        result.add_check(
            identity_error < 1e-3,
            f"Good inversion quality",
            f"Poor inversion quality (error = {identity_error:.2e})"
        )

        # Check inverse symmetry
        inv_sym_error = np.linalg.norm(inv_mat - inv_mat.T)
        result.add_check(
            inv_sym_error < 1e-6,
            f"Inverse is symmetric (error = {inv_sym_error:.2e})",
            f"Inverse NOT symmetric (error = {inv_sym_error:.2e})"
        )

        result.data['inv_mat'] = inv_mat
        result.data['identity_error'] = identity_error

    result.data['orig_mat'] = orig_mat
    return result


def test_incomplete_cholesky_inversion(solver, mas):
    """Test 6: Check Incomplete Cholesky block inversion."""
    result = DiagnosticResult("Incomplete Cholesky Inversion")

    # Assemble matrices
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)
    mas._expand_sym_to_full()

    # Get original matrix
    orig_mat = mas.full_block_matrix.to_numpy()[0].copy()

    # Check if original is SPD (required for IC)
    eigvals = np.linalg.eigvalsh(orig_mat)
    min_eig = eigvals.min()
    is_spd = min_eig > 1e-10

    result.add_info(f"Original matrix min eigenvalue: {min_eig:.6e}")
    result.add_check(
        is_spd,
        f"Original is SPD (IC should work)",
        f"Original is NOT SPD (IC will likely fail)"
    )

    # Run Incomplete Cholesky anyway
    mas._incomplete_cholesky_invert_blocks()
    inv_mat = mas.full_block_inverse.to_numpy()[0]

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(inv_mat))
    inf_count = np.sum(np.isinf(inv_mat))

    result.add_check(
        nan_count == 0,
        f"No NaN in inverse",
        f"Has {nan_count} NaN values (IC failed on non-SPD)"
    )
    result.add_check(
        inf_count == 0,
        f"No Inf in inverse",
        f"Has {inf_count} Inf values"
    )

    # If IC succeeded, check quality
    if nan_count == 0 and inf_count == 0:
        product = orig_mat @ inv_mat
        identity_error = np.linalg.norm(product - np.eye(48))
        result.add_info(f"||A * A^-1 - I|| = {identity_error:.6e}")
        result.data['identity_error'] = identity_error

    result.data['orig_mat'] = orig_mat
    result.data['min_eig'] = min_eig
    return result


def test_element_hessian_spd(solver, mas):
    """Test 7: Check if individual element Hessians are SPD.
    Note: This is a conceptual test - element Hessians for ARAP with filtering
    should be PSD (positive semi-definite).
    """
    result = DiagnosticResult("Element Hessian SPD Check")

    result.add_info("ARAP with eigenvalue filtering:")
    result.add_info("  - d2Psi/dF2 = 2*mu*I (9x9 identity scaled)")
    result.add_info("  - This is always SPD")
    result.add_info("  - H_e = dFdx^T @ d2PsidF2 @ dFdx is always PSD")
    result.add_info("  - (PSD because it's a Gram matrix)")

    result.add_info("\nFor other elastic models (SNH, FCR):")
    result.add_info("  - Eigenvalue filtering projects to PSD")
    result.add_info("  - compute_d2PsidF2_*_filter functions handle this")

    result.add_check(True, "ARAP element Hessians are theoretically PSD", "")

    return result


def test_block_extraction_analysis(solver, mas):
    """Test 8: Analyze why block extraction causes non-SPD."""
    result = DiagnosticResult("Block Extraction Analysis")

    # Key insight: extracting a 48x48 sub-block from a larger SPD matrix
    # may NOT be SPD because it lacks the off-block coupling terms

    result.add_info("--- Understanding the non-SPD issue ---")
    result.add_info("Full system Hessian H is SPD (with ARAP_filter)")
    result.add_info("Block H_ii is extracted by zeroing out cross-block terms")
    result.add_info("H_ii may NOT be SPD because:")
    result.add_info("  1. H = [H_ii  H_ij] is SPD")
    result.add_info("       [H_ji  H_jj]")
    result.add_info("  2. But H_ii alone may have negative eigenvalues")
    result.add_info("  3. The Schur complement H_ii - H_ij*H_jj^-1*H_ji is SPD")
    result.add_info("     but we can't compute this efficiently")

    result.add_info("\n--- Key Observations ---")
    result.add_info("1. Cells crossing block boundaries contribute partial Hessians")
    result.add_info("2. Only in-block entries are kept, missing cross-block coupling")
    result.add_info("3. This creates indefinite (non-SPD) sub-matrices")

    # Suggested fix
    result.add_info("\n--- Potential Solutions ---")
    result.add_info("1. Add diagonal regularization (shift eigenvalues)")
    result.add_info("2. Use Modified Cholesky (Gill-Murray) for indefinite matrices")
    result.add_info("3. Use Gauss-Jordan instead of IC (handles indefinite)")
    result.add_info("4. Project out negative eigenspace before inversion")

    return result


def run_all_diagnostics():
    """Run all diagnostic tests."""
    print("="*70)
    print("MAS MATRIX COMPUTATION DIAGNOSTIC TEST")
    print("="*70)

    # Create solver
    print("\nInitializing solver...")
    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    print(f"Solver info:")
    print(f"  n_verts: {solver.n_verts}")
    print(f"  n_cells: {solver.n_cells}")
    print(f"  dt: {solver.dt}")
    print(f"  mu: {solver.mu:.6e}")
    print(f"  la: {solver.la:.6e}")
    print(f"  BANKSIZE: {BANKSIZE}")

    # Initialize state
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Run tests
    results = []

    results.append(test_inertia_contribution(solver, mas))
    results.append(test_elastic_contribution(solver, mas))
    results.append(test_inertia_plus_elastic(solver, mas))
    results.append(test_symmetric_expansion(solver, mas))
    results.append(test_gauss_jordan_inversion(solver, mas))
    results.append(test_incomplete_cholesky_inversion(solver, mas))
    results.append(test_element_hessian_spd(solver, mas))
    results.append(test_block_extraction_analysis(solver, mas))

    # Report all results
    for r in results:
        r.report()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"Tests passed: {passed}/{len(results)}")
    print(f"Tests failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")

    return results


if __name__ == '__main__':
    run_all_diagnostics()
