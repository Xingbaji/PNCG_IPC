"""
Debug MAS Preconditioner NaN issue

Investigate why Incomplete Cholesky produces NaN.
Check block matrices before and after inversion.
"""

import sys
import os
import numpy as np

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


@ti.data_oriented
class DebugMASTestSolver(pncg_ipc_deformer):
    """Debug solver for MAS NaN investigation."""

    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)

        # Initialize MAS preconditioner
        print(f"[Debug] Initializing MAS preconditioner...")
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )
        print(f"[Debug] MAS initialized with {self.mas.level_num} levels")


def check_block_matrices(mas, label=""):
    """Check block matrices for NaN, Inf, and eigenvalue issues."""
    print(f"\n{'='*60}")
    print(f"Checking block matrices: {label}")
    print(f"{'='*60}")

    n_blocks = (mas.n_verts + BANKSIZE - 1) // BANKSIZE
    print(f"Number of level-0 blocks: {n_blocks}")

    # Sample a few blocks
    block_matrices_np = mas.block_matrices.to_numpy()

    nan_count = 0
    inf_count = 0
    zero_diag_count = 0
    negative_diag_count = 0
    small_diag_count = 0

    diag_values = []
    off_diag_norms = []

    for block_id in range(min(n_blocks, 10)):  # Check first 10 blocks
        # Extract diagonal blocks
        for lane_id in range(BANKSIZE):
            sym_idx = lane_id * (lane_id + 1) // 2 + lane_id  # Diagonal in symmetric storage
            diag_block = block_matrices_np[block_id, sym_idx]

            # Check for NaN/Inf
            if np.any(np.isnan(diag_block)):
                nan_count += 1
            if np.any(np.isinf(diag_block)):
                inf_count += 1

            # Check diagonal entries of the 3x3 block
            for d in range(3):
                val = diag_block[d, d]
                diag_values.append(val)
                if np.abs(val) < 1e-12:
                    zero_diag_count += 1
                if val < 0:
                    negative_diag_count += 1
                if val < 1e-6 and val > 0:
                    small_diag_count += 1

        # Check off-diagonal blocks
        for row in range(BANKSIZE):
            for col in range(row + 1, BANKSIZE):
                sym_idx = row * (row + 1) // 2 + col  # This is wrong, let me fix
                # Correct: sym_idx = BANKSIZE * row - row * (row + 1) // 2 + col
                sym_idx = BANKSIZE * row - row * (row + 1) // 2 + col
                off_diag_block = block_matrices_np[block_id, sym_idx]
                off_diag_norms.append(np.linalg.norm(off_diag_block))

    diag_values = np.array(diag_values)
    off_diag_norms = np.array(off_diag_norms)

    print(f"\nDiagonal entries statistics (first 10 blocks):")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  Zero diag (< 1e-12): {zero_diag_count}")
    print(f"  Negative diag: {negative_diag_count}")
    print(f"  Small diag (< 1e-6): {small_diag_count}")
    print(f"  Diag min: {diag_values.min():.6e}")
    print(f"  Diag max: {diag_values.max():.6e}")
    print(f"  Diag mean: {diag_values.mean():.6e}")

    print(f"\nOff-diagonal norms (first 10 blocks):")
    print(f"  Off-diag norm min: {off_diag_norms.min():.6e}")
    print(f"  Off-diag norm max: {off_diag_norms.max():.6e}")
    print(f"  Off-diag norm mean: {off_diag_norms.mean():.6e}")

    return nan_count, inf_count, negative_diag_count


def check_full_block_matrix(mas, block_id=0, label=""):
    """Check a specific full 48x48 block matrix."""
    print(f"\n{'='*60}")
    print(f"Checking full block matrix {block_id}: {label}")
    print(f"{'='*60}")

    full_matrix = mas.full_block_matrix.to_numpy()[block_id]

    print(f"Shape: {full_matrix.shape}")
    print(f"NaN count: {np.sum(np.isnan(full_matrix))}")
    print(f"Inf count: {np.sum(np.isinf(full_matrix))}")

    # Check symmetry
    sym_error = np.linalg.norm(full_matrix - full_matrix.T)
    print(f"Symmetry error (Frobenius): {sym_error:.6e}")

    # Check diagonal
    diag = np.diag(full_matrix)
    print(f"Diagonal min: {diag.min():.6e}")
    print(f"Diagonal max: {diag.max():.6e}")
    print(f"Negative diagonal count: {np.sum(diag < 0)}")
    print(f"Zero diagonal count: {np.sum(np.abs(diag) < 1e-12)}")

    # Check eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(full_matrix)
        print(f"Eigenvalue min: {eigvals.min():.6e}")
        print(f"Eigenvalue max: {eigvals.max():.6e}")
        print(f"Negative eigenvalue count: {np.sum(eigvals < 0)}")
        print(f"Condition number: {eigvals.max() / max(eigvals.min(), 1e-12):.6e}")

        if eigvals.min() < 0:
            print(f"  WARNING: Matrix is NOT SPD!")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")

    return full_matrix


def check_inverse_matrix(mas, block_id=0, label=""):
    """Check a specific inverted block matrix."""
    print(f"\n{'='*60}")
    print(f"Checking inverse block matrix {block_id}: {label}")
    print(f"{'='*60}")

    inv_matrix = mas.full_block_inverse.to_numpy()[block_id]

    print(f"Shape: {inv_matrix.shape}")
    print(f"NaN count: {np.sum(np.isnan(inv_matrix))}")
    print(f"Inf count: {np.sum(np.isinf(inv_matrix))}")

    # Check symmetry
    sym_error = np.linalg.norm(inv_matrix - inv_matrix.T)
    print(f"Symmetry error (Frobenius): {sym_error:.6e}")

    # Check diagonal
    diag = np.diag(inv_matrix)
    print(f"Diagonal min: {diag.min():.6e}")
    print(f"Diagonal max: {diag.max():.6e}")
    print(f"Negative diagonal count: {np.sum(diag < 0)}")

    return inv_matrix


def main():
    print("="*60)
    print("Debug MAS Preconditioner NaN Issue")
    print("="*60)

    # Create solver
    solver = DebugMASTestSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    print(f"\nSolver info:")
    print(f"  n_verts: {solver.n_verts}")
    print(f"  n_cells: {solver.n_cells}")
    print(f"  dt: {solver.dt}")
    print(f"  mu: {solver.mu}")
    print(f"  la: {solver.la}")

    # Step 1: Initialize
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()

    # Step 2: Build hierarchy
    print("\n[Step 1] Building hierarchy...")
    mas.build_hierarchy()

    # Step 3: Assemble block matrices
    print("\n[Step 2] Assembling block matrices...")
    mas.elastic_type = getattr(solver, 'elastic_type', 0)
    mas.assemble_block_matrices(solver, use_full_hessian=True)

    # Check assembled matrices
    check_block_matrices(mas, "After assembly (before inversion)")

    # Step 4: Expand to full matrix
    print("\n[Step 3] Expanding to full 48x48 matrices...")
    mas._expand_sym_to_full()

    # Check full matrix
    full_mat_0 = check_full_block_matrix(mas, block_id=0, label="After expansion")
    full_mat_1 = check_full_block_matrix(mas, block_id=1, label="Block 1")

    # Step 5: Try Incomplete Cholesky
    print("\n[Step 4] Running Incomplete Cholesky inversion...")
    mas._incomplete_cholesky_invert_blocks()

    # Check inverse matrix
    inv_mat_0 = check_inverse_matrix(mas, block_id=0, label="After IC inversion")
    inv_mat_1 = check_inverse_matrix(mas, block_id=1, label="Block 1")

    # Step 6: Verify inversion quality
    print("\n[Step 5] Verifying inversion quality...")

    # Re-read original matrix (since IC overwrites it)
    mas._expand_sym_to_full()
    orig_mat_0 = mas.full_block_matrix.to_numpy()[0].copy()

    # Check A * A^{-1} ~= I
    product = orig_mat_0 @ inv_mat_0
    identity_error = np.linalg.norm(product - np.eye(48))
    print(f"||A * A^-1 - I|| for block 0: {identity_error:.6e}")

    # Step 7: Compare with Gauss-Jordan
    print("\n[Step 6] Comparing with Gauss-Jordan inversion...")
    mas._expand_sym_to_full()
    mas._gauss_jordan_invert_blocks()
    gj_inv_0 = check_inverse_matrix(mas, block_id=0, label="Gauss-Jordan inverse")

    # Verify GJ inversion
    mas._expand_sym_to_full()
    orig_mat_0 = mas.full_block_matrix.to_numpy()[0].copy()
    product_gj = orig_mat_0 @ gj_inv_0
    identity_error_gj = np.linalg.norm(product_gj - np.eye(48))
    print(f"||A * A^-1 - I|| for block 0 (GJ): {identity_error_gj:.6e}")

    # Step 8: Check if problem is in apply phase
    print("\n[Step 7] Testing apply with GJ inverse...")

    # Use GJ inverse
    mas._copy_inverse_to_sym()
    mas.apply()

    # Check z field
    z_field = solver.mesh.verts.z.to_numpy()
    print(f"z field NaN count: {np.sum(np.isnan(z_field))}")
    print(f"z field Inf count: {np.sum(np.isinf(z_field))}")
    print(f"z field norm: {np.linalg.norm(z_field):.6e}")

    print("\n" + "="*60)
    print("Debug complete")
    print("="*60)


if __name__ == '__main__':
    main()
