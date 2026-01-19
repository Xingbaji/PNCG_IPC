"""
Verify symmetric storage expansion is correct.
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
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.mas_preconditioner_pkg import MASPreconditioner
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, BLOCK_DOF, SYM_BLOCK_COUNT


class DebugSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def sym_idx(row, col):
    """Compute symmetric storage index."""
    return BANKSIZE * row - row * (row + 1) // 2 + col


def expand_sym_to_full_numpy(sym_storage):
    """NumPy implementation of symmetric expansion."""
    full = np.zeros((BLOCK_DOF, BLOCK_DOF))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            idx = sym_idx(row, col)
            block_3x3 = sym_storage[idx]

            # Upper triangle
            for di in range(3):
                for dj in range(3):
                    full[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                    # Lower triangle (transpose)
                    if row != col:
                        full[col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    return full


def main():
    print("=" * 70)
    print("Symmetric Expansion Verification")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    # Assemble elastic only
    solver.mas._clear_block_matrices()
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)

    # Get symmetric storage for block 0
    sym_storage = solver.mas.block_matrices.to_numpy()[0]  # (136, 3, 3)

    print(f"\nSymmetric storage shape: {sym_storage.shape}")
    print(f"Expected: ({SYM_BLOCK_COUNT}, 3, 3)")

    # Expand using Taichi
    solver.mas._expand_sym_to_full()
    full_taichi = solver.mas.full_block_matrix.to_numpy()[0]

    # Expand using NumPy
    full_numpy = expand_sym_to_full_numpy(sym_storage)

    # Compare
    diff = np.linalg.norm(full_taichi - full_numpy)
    print(f"\nExpansion difference (Taichi vs NumPy): {diff:.4e}")

    # Check symmetry
    asym_taichi = np.linalg.norm(full_taichi - full_taichi.T)
    asym_numpy = np.linalg.norm(full_numpy - full_numpy.T)
    print(f"Asymmetry (Taichi): {asym_taichi:.4e}")
    print(f"Asymmetry (NumPy): {asym_numpy:.4e}")

    # Check diagonal entries
    print(f"\n--- Diagonal analysis ---")
    diag_taichi = np.diag(full_taichi)
    diag_numpy = np.diag(full_numpy)

    print(f"Diagonal comparison (first 6 entries):")
    for i in range(6):
        print(f"  [{i}] Taichi: {diag_taichi[i]:.4e}, NumPy: {diag_numpy[i]:.4e}")

    # Check the (0,0) 3x3 block directly
    print(f"\n--- Block (0,0) analysis ---")
    idx_00 = sym_idx(0, 0)
    print(f"sym_idx(0,0) = {idx_00}")
    block_00 = sym_storage[idx_00]
    print(f"sym_storage[{idx_00}]:")
    print(block_00)

    print(f"\nfull_numpy[0:3, 0:3]:")
    print(full_numpy[0:3, 0:3])

    print(f"\nfull_taichi[0:3, 0:3]:")
    print(full_taichi[0:3, 0:3])

    # Are they equal?
    if np.allclose(block_00, full_numpy[0:3, 0:3]):
        print("✓ Block (0,0) correctly expanded")
    else:
        print("⚠️ Block (0,0) expansion mismatch!")

    # Check eigenvalues of block 0 elastic contribution
    print(f"\n--- Eigenvalue analysis ---")
    full_sym = (full_numpy + full_numpy.T) / 2
    eigvals = np.linalg.eigvalsh(full_sym)
    print(f"Eigenvalues: min={np.min(eigvals):.4e}, max={np.max(eigvals):.4e}")
    print(f"Negative count: {np.sum(eigvals < 0)}")

    # Check if the symmetric storage itself has negative diagonals
    print(f"\n--- Symmetric storage diagonal check ---")
    for i in range(min(5, BANKSIZE)):
        idx = sym_idx(i, i)
        diag_block = sym_storage[idx]
        print(f"  Block ({i},{i}) diagonal: [{diag_block[0,0]:.4e}, {diag_block[1,1]:.4e}, {diag_block[2,2]:.4e}]")


if __name__ == '__main__':
    main()
