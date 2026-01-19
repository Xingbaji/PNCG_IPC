"""
Test hierarchy mapping to understand what's going on.
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


def main():
    print("="*70)
    print("HIERARCHY MAPPING TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    print(f"\nHierarchy info:")
    print(f"  n_verts: {solver.n_verts}")
    print(f"  actual_levels: {mas.actual_levels}")

    # Check going_next mapping
    going_next_np = mas.going_next.to_numpy()

    print(f"\n[Test 1] going_next mapping for first 32 vertices:")
    print("-" * 60)
    for v in range(32):
        coarse = going_next_np[v]
        print(f"  v{v} -> coarse v{coarse}")

    # Check level sizes
    level_size_np = mas.level_size.to_numpy()
    print(f"\n[Test 2] Level sizes:")
    print("-" * 60)
    for level in range(mas.actual_levels):
        size, offset = level_size_np[level]
        print(f"  Level {level}: size={size}, offset={offset}")

    # Check what blocks are involved at level 0
    print(f"\n[Test 3] Block structure at Level 0:")
    print("-" * 60)
    n_blocks_level0 = (solver.n_verts + BANKSIZE - 1) // BANKSIZE
    print(f"  Number of blocks at Level 0: {n_blocks_level0}")

    # Now run assembly and check
    print(f"\n[Test 4] Assembling block matrices...")
    print("-" * 60)

    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()

    # Check Block 0 at level 0
    print(f"\n[Test 5] Block 0 diagonal entries symmetry:")
    print("-" * 60)

    def sym_index(row, col):
        r = min(row, col)
        c = max(row, col)
        return BANKSIZE * r - r * (r + 1) // 2 + c

    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        diag_block = block_np[0, sym_idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)
        if sym_err > 1e-6:
            print(f"  Lane {lane}: sym_error = {sym_err:.6e}")

    # Check if coarse level blocks have contributions
    if mas.actual_levels > 1:
        level1_offset = level_size_np[1][1]
        n_blocks_level1 = (level_size_np[1][0] + BANKSIZE - 1) // BANKSIZE
        print(f"\n[Test 6] Level 1 blocks (offset={level1_offset}):")
        print("-" * 60)
        print(f"  Number of blocks at Level 1: {n_blocks_level1}")

        # Check first block at level 1
        level1_block0_idx = level1_offset // BANKSIZE
        print(f"  Level 1 Block 0 is at global index {level1_block0_idx}")

        for lane in range(min(BANKSIZE, level_size_np[1][0])):
            sym_idx = sym_index(lane, lane)
            diag_block = block_np[level1_block0_idx, sym_idx]
            diag_norm = np.linalg.norm(diag_block)
            if diag_norm > 1e-6:
                sym_err = np.linalg.norm(diag_block - diag_block.T)
                print(f"  Lane {lane}: norm={diag_norm:.6e}, sym_error={sym_err:.6e}")


if __name__ == '__main__':
    main()
