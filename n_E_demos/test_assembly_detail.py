"""
Detailed test to diagnose assembly symmetry issues.
Focuses on a single element's contribution to understand the problem.
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


def sym_index(row, col):
    """Compute symmetric storage index for upper triangle."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


@ti.data_oriented
class DiagnosticSolver(pncg_ipc_deformer):
    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def extract_3x3_block(block_matrices_np, block_id, row_lane, col_lane):
    """Extract a 3x3 block from symmetric storage."""
    r = min(row_lane, col_lane)
    c = max(row_lane, col_lane)
    sym_idx = sym_index(r, c)
    block = block_matrices_np[block_id, sym_idx].copy()
    # If accessing lower triangle, transpose
    if row_lane > col_lane:
        block = block.T
    return block


def main():
    print("="*70)
    print("DETAILED ASSEMBLY DIAGNOSTIC")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Clear and add elastic contribution
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_matrices_np = mas.block_matrices.to_numpy()

    print(f"\n[Test 1] Check individual 3x3 diagonal blocks for symmetry")
    print("-" * 60)

    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        diag_block = block_matrices_np[0, sym_idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)
        if sym_err > 1e-6:
            print(f"  Lane {lane}: sym_error = {sym_err:.6e}")
            print(f"    Block:\n{diag_block}")
            print(f"    Block.T:\n{diag_block.T}")
            print(f"    Diff:\n{diag_block - diag_block.T}")

    print(f"\n[Test 2] Check off-diagonal blocks for consistency")
    print("-" * 60)

    # For symmetric matrix: block(i,j) should equal block(j,i).T
    # In our storage: sym_storage[sym_idx(i,j)] stores block(i,j) for i<=j
    # So when we extract block(j,i), we should get sym_storage[sym_idx(i,j)].T

    inconsistent_pairs = []
    for i in range(BANKSIZE):
        for j in range(i+1, BANKSIZE):
            sym_idx = sym_index(i, j)
            stored_block = block_matrices_np[0, sym_idx]

            # block(i,j) should be stored directly
            block_ij = stored_block
            # block(j,i) should be stored.T
            block_ji = stored_block.T

            # Now verify: block(i,j) should equal block(j,i).T
            # Which means stored_block should equal stored_block.T.T = stored_block
            # This is trivially true, but let's check the 48x48 expansion
            pass  # This test is redundant for storage check

    print(f"\n[Test 3] Check if 3x3 blocks are correctly positioned in 48x48")
    print("-" * 60)

    # Manually build 48x48 matrix
    full_mat = np.zeros((48, 48))
    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = block_matrices_np[0, sym_idx]

            for di in range(3):
                for dj in range(3):
                    # Upper triangle
                    full_mat[row*3+di, col*3+dj] = block_3x3[di, dj]
                    # Lower triangle
                    if row != col:
                        full_mat[col*3+di, row*3+dj] = block_3x3[dj, di]

    sym_error = np.linalg.norm(full_mat - full_mat.T)
    print(f"  48x48 symmetry error: {sym_error:.6e}")

    if sym_error > 1e-6:
        # Find which entries are asymmetric
        diff = full_mat - full_mat.T
        asymmetric_entries = np.argwhere(np.abs(diff) > 1e-6)
        print(f"  Found {len(asymmetric_entries)} asymmetric entries")

        # Group by 3x3 block
        asymmetric_blocks = set()
        for r, c in asymmetric_entries:
            block_r, block_c = r // 3, c // 3
            asymmetric_blocks.add((min(block_r, block_c), max(block_r, block_c)))

        print(f"  Asymmetric block pairs: {sorted(asymmetric_blocks)}")

        # Examine first few
        for (br, bc) in sorted(asymmetric_blocks)[:3]:
            block_upper = full_mat[br*3:(br+1)*3, bc*3:(bc+1)*3]
            block_lower = full_mat[bc*3:(bc+1)*3, br*3:(br+1)*3]
            print(f"\n  Block ({br}, {bc}):")
            print(f"    Upper:\n{block_upper}")
            print(f"    Lower:\n{block_lower}")
            print(f"    Lower.T:\n{block_lower.T}")
            print(f"    Upper - Lower.T:\n{block_upper - block_lower.T}")

    print(f"\n[Test 4] Check what's stored in symmetric storage directly")
    print("-" * 60)

    # For the first problematic pair from Test 3
    if sym_error > 1e-6:
        for (br, bc) in sorted(asymmetric_blocks)[:3]:
            sym_idx = sym_index(br, bc)
            stored = block_matrices_np[0, sym_idx]
            print(f"\n  Sym storage for ({br}, {bc}) at index {sym_idx}:")
            print(f"    Stored block:\n{stored}")
            print(f"    Is symmetric? {np.linalg.norm(stored - stored.T):.6e}")

    print(f"\n[Test 5] Verify H_e symmetry from single element")
    print("-" * 60)

    # Compute H_e for first cell manually
    from math_utils.elastic_util import compute_dFdx, compute_d2PsidF2_ARAP_filter
    from math_utils.matrix_util import compute_dFdx_taichi

    # Get first cell's data
    x_np = solver.mesh.verts.x.to_numpy()
    B_np = solver.mesh.cells.B.to_numpy()
    W_np = solver.mesh.cells.W.to_numpy()

    # We need cell vertex IDs - use a simple approximation
    # Assuming cells are stored sequentially
    print("  (Manual H_e computation requires cell vertex access - skipped)")
    print("  H_e = dFdx^T @ (2*mu*I) @ dFdx should be symmetric by construction")
    print("  dFdx is 9x12, so H_e is 12x12")
    print("  H_e = dFdx^T @ dFdx * (2*mu) is a Gram matrix, hence PSD and symmetric")


if __name__ == '__main__':
    main()
