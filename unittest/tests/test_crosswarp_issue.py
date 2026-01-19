"""
Test to identify the cross-warp assembly issue.
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


def main():
    print("="*70)
    print("CROSS-WARP ASSEMBLY ISSUE TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Test 1: Only inertia (should be diagonal and symmetric)
    print("\n[Test 1] Inertia only:")
    print("-" * 60)
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)

    block_np = mas.block_matrices.to_numpy()
    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        diag_block = block_np[0, sym_idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)
        if sym_err > 1e-6:
            print(f"  Lane {lane}: sym_error = {sym_err:.6e}")
    print("  (Should be all zeros - inertia is diagonal)")

    # Test 2: Elastic only, single element contribution tracking
    print("\n[Test 2] Elastic only:")
    print("-" * 60)
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()
    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        diag_block = block_np[0, sym_idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)
        if sym_err > 1e-6:
            print(f"  Lane {lane}: sym_error = {sym_err:.6e}")
            print(f"    Block:\n{diag_block}")
            print(f"    Block - Block.T:\n{diag_block - diag_block.T}")

    # Check if the issue is specific to Lane 0
    print("\n[Analysis] Why only Lane 0?")
    print("-" * 60)
    print("  Lane 0 corresponds to vertex 0")
    print("  Vertex 0 is special because it's the first vertex of many elements")
    print("  In tetrahedral elements, vertex 0 appears in dFdx with special coefficients")

    # Check off-diagonal blocks involving Lane 0
    print("\n[Test 3] Off-diagonal blocks involving Lane 0:")
    print("-" * 60)
    for col in range(1, 5):
        sym_idx = sym_index(0, col)
        block = block_np[0, sym_idx]
        norm = np.linalg.norm(block)
        if norm > 1e-6:
            print(f"  Block(0,{col}): norm = {norm:.6e}")
            # Check if this block is getting contaminated
            print(f"    Block:\n{block}")

    # Test 4: Check if level 1 (coarse) also has issues
    level_size_np = mas.level_size.to_numpy()
    if mas.actual_levels > 1:
        level1_offset = level_size_np[1][1]
        level1_block0_idx = level1_offset // BANKSIZE

        print(f"\n[Test 4] Level 1 Block 0 (at index {level1_block0_idx}):")
        print("-" * 60)

        for lane in range(min(BANKSIZE, level_size_np[1][0])):
            sym_idx = sym_index(lane, lane)
            diag_block = block_np[level1_block0_idx, sym_idx]
            norm = np.linalg.norm(diag_block)
            sym_err = np.linalg.norm(diag_block - diag_block.T)
            if norm > 1e-6:
                print(f"  Lane {lane}: norm={norm:.6e}, sym_error={sym_err:.6e}")
                if sym_err > 1e-3:
                    print(f"    Block:\n{diag_block}")


if __name__ == '__main__':
    main()
