"""
Compare optimized and non-optimized assembly versions.
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
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE


class DebugSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def main():
    print("=" * 70)
    print("Compare Assembly Versions")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")

    # Version 1: Non-optimized
    print("\n--- Non-optimized version ---")
    solver.mas._clear_block_matrices()
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    solver.mas._expand_sym_to_full()
    block0_nonopt = solver.mas.full_block_matrix.to_numpy()[0].copy()

    diag_nonopt = np.array([block0_nonopt[0, 0], block0_nonopt[1, 1], block0_nonopt[2, 2]])
    print(f"Vertex 0 diagonal: [{diag_nonopt[0]:.4e}, {diag_nonopt[1]:.4e}, {diag_nonopt[2]:.4e}]")

    block0_nonopt_sym = (block0_nonopt + block0_nonopt.T) / 2
    eigvals_nonopt = np.linalg.eigvalsh(block0_nonopt_sym)
    print(f"Eigenvalues: min={np.min(eigvals_nonopt):.4e}, max={np.max(eigvals_nonopt):.4e}")
    print(f"Negative count: {np.sum(eigvals_nonopt < 0)}")

    # Version 2: Optimized
    print("\n--- Optimized version ---")
    solver.mas._clear_block_matrices()
    solver.mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)
    solver.mas._expand_sym_to_full()
    block0_opt = solver.mas.full_block_matrix.to_numpy()[0].copy()

    diag_opt = np.array([block0_opt[0, 0], block0_opt[1, 1], block0_opt[2, 2]])
    print(f"Vertex 0 diagonal: [{diag_opt[0]:.4e}, {diag_opt[1]:.4e}, {diag_opt[2]:.4e}]")

    block0_opt_sym = (block0_opt + block0_opt.T) / 2
    eigvals_opt = np.linalg.eigvalsh(block0_opt_sym)
    print(f"Eigenvalues: min={np.min(eigvals_opt):.4e}, max={np.max(eigvals_opt):.4e}")
    print(f"Negative count: {np.sum(eigvals_opt < 0)}")

    # Compare
    print("\n--- Comparison ---")
    diff = np.linalg.norm(block0_nonopt - block0_opt)
    print(f"||nonopt - opt|| = {diff:.4e}")

    diag_diff = diag_nonopt - diag_opt
    print(f"Diagonal diff: [{diag_diff[0]:.4e}, {diag_diff[1]:.4e}, {diag_diff[2]:.4e}]")

    if diff > 1e-3:
        print("⚠️ SIGNIFICANT DIFFERENCE between versions!")
    else:
        print("✓ Versions produce similar results")


if __name__ == '__main__':
    main()
