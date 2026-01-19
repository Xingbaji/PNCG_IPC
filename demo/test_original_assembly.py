"""
Test original assembly function to verify it produces correct results.
"""

import sys
import os

# Setup path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.mas_preconditioner_pkg import MASPreconditioner
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, SYM_BLOCK_COUNT


class DebugSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def sym_idx(row, col):
    return BANKSIZE * row - row * (row + 1) // 2 + col


def main():
    print("=" * 60)
    print("Test Original Assembly Function")
    print("=" * 60)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nn_verts = {solver.n_verts}")
    print(f"mu = {solver.mu:.4e}, la = {solver.la:.4e}, dt = {solver.dt:.4e}")

    # Test with the ORIGINAL assembly function
    print("\n--- Testing _add_elastic_contribution_full ---")
    solver.mas._clear_block_matrices()
    solver.mas._add_inertia_contribution(solver.dt)

    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_inertia = sym_storage[sym_idx(0, 0)].diagonal()
    print(f"After inertia: [{diag_inertia[0]:.4e}, {diag_inertia[1]:.4e}, {diag_inertia[2]:.4e}]")

    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)

    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_full = sym_storage[sym_idx(0, 0)].diagonal()
    print(f"After elastic: [{diag_full[0]:.4e}, {diag_full[1]:.4e}, {diag_full[2]:.4e}]")

    elastic_contrib = diag_full - diag_inertia
    print(f"Elastic contribution: [{elastic_contrib[0]:.4e}, {elastic_contrib[1]:.4e}, {elastic_contrib[2]:.4e}]")

    if np.any(diag_full < 0):
        print("\n*** NEGATIVE DIAGONAL DETECTED ***")
    else:
        print("\n*** All diagonals positive ***")

    # Check eigenvalues
    print("\n--- Eigenvalue analysis ---")
    solver.mas._expand_sym_to_full()
    full_block0 = solver.mas.full_block_matrix.to_numpy()[0]
    full_sym = (full_block0 + full_block0.T) / 2
    eigvals = np.linalg.eigvalsh(full_sym)
    print(f"Block 0 eigenvalues: min={np.min(eigvals):.4e}, max={np.max(eigvals):.4e}")
    print(f"Negative eigenvalues: {np.sum(eigvals < 0)}")


if __name__ == '__main__':
    main()
