"""
Detailed debug of Block 0 to understand why it has negative eigenvalues.
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
class DebugSolver(pncg_ipc_deformer):
    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(self.n_verts, self.n_cells, self.mesh, use_metis=False)


def main():
    solver = DebugSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    # Initialize
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Get vertex positions for Block 0
    x_np = solver.mesh.verts.x.to_numpy()
    print(f"Block 0 vertex positions (vertices 0-15):")
    for i in range(BANKSIZE):
        print(f"  v{i}: {x_np[i]}")

    # Get mass for Block 0
    m_np = solver.mesh.verts.m.to_numpy()
    print(f"\nBlock 0 vertex masses:")
    for i in range(BANKSIZE):
        print(f"  v{i}: {m_np[i]:.6f}")

    # Assemble only inertia first
    print("\n[Test 1] Inertia contribution only:")
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._expand_sym_to_full()

    full_mat = mas.full_block_matrix.to_numpy()[0]
    eigvals = np.linalg.eigvalsh(full_mat)
    print(f"  Eigenvalue min: {eigvals.min():.6e}")
    print(f"  Eigenvalue max: {eigvals.max():.6e}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")
    print(f"  Symmetry error: {np.linalg.norm(full_mat - full_mat.T):.6e}")

    # Assemble elastic only
    print("\n[Test 2] Elastic contribution only:")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)  # ARAP
    mas._expand_sym_to_full()

    full_mat = mas.full_block_matrix.to_numpy()[0]
    eigvals = np.linalg.eigvalsh(full_mat)
    print(f"  Eigenvalue min: {eigvals.min():.6e}")
    print(f"  Eigenvalue max: {eigvals.max():.6e}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")
    print(f"  Symmetry error: {np.linalg.norm(full_mat - full_mat.T):.6e}")

    # Check if the 3x3 sub-blocks are symmetric
    print("\n  Checking 3x3 sub-block symmetry:")
    block_mat = mas.block_matrices.to_numpy()[0]
    non_sym_count = 0
    max_sym_err = 0.0
    for lane in range(BANKSIZE):
        sym_idx = BANKSIZE * lane - lane * (lane + 1) // 2 + lane
        block_3x3 = block_mat[sym_idx]
        sym_err = np.linalg.norm(block_3x3 - block_3x3.T)
        if sym_err > 1e-6:
            non_sym_count += 1
            max_sym_err = max(max_sym_err, sym_err)
    print(f"    Non-symmetric diagonal 3x3 blocks: {non_sym_count}")
    print(f"    Max symmetry error: {max_sym_err:.6e}")

    # Assemble inertia + elastic
    print("\n[Test 3] Inertia + Elastic:")
    mas._clear_block_matrices()
    mas._add_inertia_contribution(solver.dt)
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)
    mas._expand_sym_to_full()

    full_mat = mas.full_block_matrix.to_numpy()[0]
    eigvals = np.linalg.eigvalsh(full_mat)
    print(f"  Eigenvalue min: {eigvals.min():.6e}")
    print(f"  Eigenvalue max: {eigvals.max():.6e}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")
    print(f"  Symmetry error: {np.linalg.norm(full_mat - full_mat.T):.6e}")

    # Add regularization
    print("\n[Test 4] Inertia + Elastic + Regularization:")
    mas._add_regularization(1e-6)
    mas._expand_sym_to_full()

    full_mat = mas.full_block_matrix.to_numpy()[0]
    eigvals = np.linalg.eigvalsh(full_mat)
    print(f"  Eigenvalue min: {eigvals.min():.6e}")
    print(f"  Eigenvalue max: {eigvals.max():.6e}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")
    print(f"  Symmetry error: {np.linalg.norm(full_mat - full_mat.T):.6e}")

    # Check which blocks have negative eigenvalues
    print("\n[Test 5] Scanning all blocks for negative eigenvalues:")
    n_blocks = (solver.n_verts + BANKSIZE - 1) // BANKSIZE
    neg_eig_blocks = []
    for block_id in range(min(n_blocks, 100)):  # Check first 100 blocks
        full_mat = mas.full_block_matrix.to_numpy()[block_id]
        try:
            eigvals = np.linalg.eigvalsh(full_mat)
            if eigvals.min() < -1e-6:
                neg_eig_blocks.append((block_id, eigvals.min()))
        except:
            pass

    print(f"  Blocks with negative eigenvalues (first 100): {len(neg_eig_blocks)}")
    for bid, eig_min in neg_eig_blocks[:10]:
        print(f"    Block {bid}: min eigenvalue = {eig_min:.6e}")

    # Detailed look at Block 0 matrix structure
    print("\n[Test 6] Block 0 matrix structure:")
    full_mat = mas.full_block_matrix.to_numpy()[0]

    print(f"  Diagonal values:")
    diag = np.diag(full_mat)
    print(f"    Min: {diag.min():.6e}")
    print(f"    Max: {diag.max():.6e}")
    print(f"    Negative count: {np.sum(diag < 0)}")

    print(f"\n  Off-diagonal structure (sample):")
    print(f"    [0,3:6] (v0-v1 coupling): {full_mat[0, 3:6]}")
    print(f"    [3,0:3] (v1-v0 coupling): {full_mat[3, 0:3]}")

    # Check if it's the mesh connectivity causing issues
    print("\n[Test 7] Check mesh connectivity for Block 0 vertices:")
    neighbor_start = mas.neighbor_start.to_numpy()
    neighbor_list = mas.neighbor_list.to_numpy()

    for v in range(BANKSIZE):
        start = neighbor_start[v]
        end = neighbor_start[v+1]
        neighbors = neighbor_list[start:end]
        in_block = sum(1 for n in neighbors if n < BANKSIZE)
        cross_block = len(neighbors) - in_block
        print(f"  v{v}: {len(neighbors)} neighbors ({in_block} in-block, {cross_block} cross-block)")


if __name__ == '__main__':
    main()
