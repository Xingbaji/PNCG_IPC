"""
Step-by-step debug of assembly process to find the source of negative diagonals.
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


def main():
    print("=" * 70)
    print("Step-by-Step Assembly Debug")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")
    print(f"Hierarchy levels: {solver.mas.actual_levels}")
    print(f"Total blocks: {solver.mas.total_blocks}")

    # Step 1: Clear matrices
    print("\n--- Step 1: Clear matrices ---")
    solver.mas._clear_block_matrices()
    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_00 = sym_storage[sym_idx(0, 0)]
    print(f"Block 0, (0,0) diagonal after clear: [{diag_00[0,0]:.4e}, {diag_00[1,1]:.4e}, {diag_00[2,2]:.4e}]")

    # Step 2: Add inertia contribution
    print("\n--- Step 2: Add inertia contribution ---")
    solver.mas._add_inertia_contribution(solver.dt)
    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_00 = sym_storage[sym_idx(0, 0)]
    print(f"Block 0, (0,0) diagonal after inertia: [{diag_00[0,0]:.4e}, {diag_00[1,1]:.4e}, {diag_00[2,2]:.4e}]")

    # Check vertex 0 mass
    mass_v0 = solver.mesh.verts.m[0]
    expected_inertia = mass_v0 / (solver.dt ** 2)
    print(f"Expected inertia contribution: {expected_inertia:.4e} (mass={mass_v0:.4e}, dt={solver.dt:.4e})")

    # Store before elastic
    diag_before_elastic = diag_00.copy()

    # Step 3: Add elastic contribution
    print("\n--- Step 3: Add elastic contribution ---")
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_00 = sym_storage[sym_idx(0, 0)]
    print(f"Block 0, (0,0) diagonal after elastic: [{diag_00[0,0]:.4e}, {diag_00[1,1]:.4e}, {diag_00[2,2]:.4e}]")

    # Compute elastic contribution = total - before
    elastic_contrib = diag_00 - diag_before_elastic
    print(f"Elastic contribution to (0,0): [{elastic_contrib[0,0]:.4e}, {elastic_contrib[1,1]:.4e}, {elastic_contrib[2,2]:.4e}]")

    # Check hierarchy info
    print("\n--- Hierarchy info ---")
    print(f"actual_levels: {solver.mas.actual_levels}")
    print(f"hierarchy_built: {solver.mas.hierarchy_built}")

    if solver.mas.actual_levels > 1:
        level_size_np = solver.mas.level_size.to_numpy()
        for lvl in range(solver.mas.actual_levels):
            print(f"  Level {lvl}: size={level_size_np[lvl][0]}, offset={level_size_np[lvl][1]}")

        going_next_np = solver.mas.going_next.to_numpy()
        print(f"\ngoing_next chain for vertex 0:")
        v = 0
        for lvl in range(solver.mas.actual_levels):
            next_v = going_next_np[v]
            if next_v < 0:
                print(f"  Level {lvl}: vertex {v} -> NONE")
                break
            print(f"  Level {lvl}: vertex {v} -> next={next_v}, warp={next_v // BANKSIZE}, lane={next_v % BANKSIZE}")
            v = next_v

    # Step 4: Check if aggregation changes anything
    if solver.mas.hierarchy_built and solver.mas.actual_levels > 1:
        print("\n--- Step 4: Fine-to-coarse aggregation ---")
        # Save before
        diag_before = sym_storage[sym_idx(0, 0)].copy()

        solver.mas._aggregate_fine_to_coarse(solver.mas.actual_levels)

        sym_storage_after = solver.mas.block_matrices.to_numpy()[0]
        diag_after = sym_storage_after[sym_idx(0, 0)]

        print(f"Block 0, (0,0) diagonal before aggregation: [{diag_before[0,0]:.4e}, {diag_before[1,1]:.4e}, {diag_before[2,2]:.4e}]")
        print(f"Block 0, (0,0) diagonal after aggregation: [{diag_after[0,0]:.4e}, {diag_after[1,1]:.4e}, {diag_after[2,2]:.4e}]")
        print(f"Change: [{(diag_after[0,0] - diag_before[0,0]):.4e}, {(diag_after[1,1] - diag_before[1,1]):.4e}, {(diag_after[2,2] - diag_before[2,2]):.4e}]")

    # Check eigenvalues of the final block 0
    print("\n--- Eigenvalue analysis ---")
    solver.mas._expand_sym_to_full()
    full_block0 = solver.mas.full_block_matrix.to_numpy()[0]

    # Make symmetric
    full_block0_sym = (full_block0 + full_block0.T) / 2
    eigvals = np.linalg.eigvalsh(full_block0_sym)
    print(f"Block 0 eigenvalues: min={np.min(eigvals):.4e}, max={np.max(eigvals):.4e}")
    print(f"Negative eigenvalues: {np.sum(eigvals < 0)}")

    # Check other blocks too
    print("\n--- Other blocks ---")
    for block_id in range(min(5, solver.mas.total_blocks)):
        sym_storage_b = solver.mas.block_matrices.to_numpy()[block_id]
        diag = sym_storage_b[sym_idx(0, 0)]
        print(f"Block {block_id}, lane 0 diagonal: [{diag[0,0]:.4e}, {diag[1,1]:.4e}, {diag[2,2]:.4e}]")


if __name__ == '__main__':
    main()
