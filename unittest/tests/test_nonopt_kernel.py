"""
Test using non-optimized kernel to compare.
"""

import sys
import os
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner, BANKSIZE

ti.init(arch=ti.gpu, default_fp=ti.f32)


@ti.data_oriented
class DiagnosticSolver(pncg_ipc_deformer):
    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def sym_index(row, col):
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def check_lane0_symmetry(mas, label):
    block_np = mas.block_matrices.to_numpy()
    sym_idx = sym_index(0, 0)
    diag_block = block_np[0, sym_idx]
    sym_err = np.linalg.norm(diag_block - diag_block.T)
    print(f"  {label}: Lane 0 sym_error = {sym_err:.6e}")
    if sym_err > 1e-3:
        print(f"    Block:\n{diag_block}")


def main():
    print("="*70)
    print("OPTIMIZED vs NON-OPTIMIZED KERNEL TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Test 1: Optimized kernel (CUDA)
    print("\n[Test 1] Optimized kernel (_add_elastic_contribution_full_optimized):")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)
    check_lane0_symmetry(mas, "Optimized")

    # Test 2: Non-optimized kernel
    print("\n[Test 2] Non-optimized kernel (_add_elastic_contribution_full):")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    check_lane0_symmetry(mas, "Non-optimized")


if __name__ == '__main__':
    main()
