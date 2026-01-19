"""
Trace the full assembly process for Block 0 to understand the negative diagonal.
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
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter
from math_utils.matrix_util import compute_dFdx


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
    print("Full Assembly Trace for Block 0")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")
    print(f"mu = {solver.mu:.4e}, la = {solver.la:.4e}, dt = {solver.dt:.4e}")

    # Manual assembly for Block 0 using NumPy
    print("\n--- Manual assembly (NumPy) ---")

    # Initialize block 0 matrix
    block0_manual = np.zeros((SYM_BLOCK_COUNT, 3, 3))

    # Count cells contributing to Block 0
    cells_in_block0 = 0

    @ti.kernel
    def get_cell_data(cell_id: ti.i32) -> ti.types.vector(4, ti.i32):
        c = solver.mesh.cells[cell_id]
        return ti.Vector([c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id])

    @ti.kernel
    def get_cell_W(cell_id: ti.i32) -> ti.f32:
        return solver.mesh.cells[cell_id].W

    @ti.kernel
    def get_cell_B(cell_id: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
        return solver.mesh.cells[cell_id].B

    @ti.kernel
    def get_cell_F(cell_id: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
        c = solver.mesh.cells[cell_id]
        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        return Ds @ c.B

    # Iterate over cells
    for cell_id in range(solver.n_cells):
        v_ids = get_cell_data(cell_id).to_numpy()
        v0, v1, v2, v3 = v_ids

        # Check if any two vertices are in Block 0
        in_block0 = [v // BANKSIZE == 0 for v in v_ids]

        if not any(in_block0):
            continue

        # Get cell data
        W = get_cell_W(cell_id)
        para = W * solver.dt ** 2
        B = get_cell_B(cell_id).to_numpy()
        F = get_cell_F(cell_id).to_numpy()

        # Compute d2PsidF2 using NumPy (ARAP_filter logic)
        U, s, Vh = np.linalg.svd(F)
        V = Vh.T
        s0, s1, s2 = s

        # Lambda values with filter
        lambda0 = 1.0 if s1 + s2 < 2.0 else 2.0 / (s1 + s2)
        lambda1 = 1.0 if s0 + s2 < 2.0 else 2.0 / (s0 + s2)
        lambda2 = 1.0 if s0 + s1 < 2.0 else 2.0 / (s0 + s1)

        # Twist mode vectors
        U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
        V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]

        Q0 = np.outer(V1, U2) - np.outer(V2, U1)
        Q1 = np.outer(V2, U0) - np.outer(V0, U2)
        Q2 = np.outer(V1, U0) - np.outer(V0, U1)

        q0 = Q0.flatten()
        q1 = Q1.flatten()
        q2 = Q2.flatten()

        # d2PsidF2 = 2μI - μ * Σ λᵢ (qᵢ ⊗ qᵢ)
        d2PsidF2 = 2.0 * solver.mu * np.eye(9)
        d2PsidF2 -= solver.mu * lambda0 * np.outer(q0, q0)
        d2PsidF2 -= solver.mu * lambda1 * np.outer(q1, q1)
        d2PsidF2 -= solver.mu * lambda2 * np.outer(q2, q2)

        # dFdx (9x12)
        def compute_dFdx_numpy(B):
            dFdx = np.zeros((9, 12))
            m, n, o = B[0, :]
            p, q, r = B[1, :]
            s, t, u = B[2, :]
            t1, t2, t3 = -m-p-s, -n-q-t, -o-r-u

            dFdx[0, 0], dFdx[0, 3], dFdx[0, 6], dFdx[0, 9] = t1, m, p, s
            dFdx[1, 1], dFdx[1, 4], dFdx[1, 7], dFdx[1, 10] = t1, m, p, s
            dFdx[2, 2], dFdx[2, 5], dFdx[2, 8], dFdx[2, 11] = t1, m, p, s
            dFdx[3, 0], dFdx[3, 3], dFdx[3, 6], dFdx[3, 9] = t2, n, q, t
            dFdx[4, 1], dFdx[4, 4], dFdx[4, 7], dFdx[4, 10] = t2, n, q, t
            dFdx[5, 2], dFdx[5, 5], dFdx[5, 8], dFdx[5, 11] = t2, n, q, t
            dFdx[6, 0], dFdx[6, 3], dFdx[6, 6], dFdx[6, 9] = t3, o, r, u
            dFdx[7, 1], dFdx[7, 4], dFdx[7, 7], dFdx[7, 10] = t3, o, r, u
            dFdx[8, 2], dFdx[8, 5], dFdx[8, 8], dFdx[8, 11] = t3, o, r, u
            return dFdx

        dFdx = compute_dFdx_numpy(B)

        # H_e = para * dFdx^T @ d2PsidF2 @ dFdx
        H_e = para * (dFdx.T @ d2PsidF2 @ dFdx)

        # Add to block matrix (only same-block pairs)
        for i in range(4):
            for j in range(i, 4):
                vi, vj = v_ids[i], v_ids[j]
                warp_i, warp_j = vi // BANKSIZE, vj // BANKSIZE

                if warp_i == 0 and warp_j == 0:
                    lane_i, lane_j = vi % BANKSIZE, vj % BANKSIZE

                    sub_block = H_e[i*3:(i+1)*3, j*3:(j+1)*3]

                    if lane_i <= lane_j:
                        idx = sym_idx(lane_i, lane_j)
                        block0_manual[idx] += sub_block
                    else:
                        idx = sym_idx(lane_j, lane_i)
                        block0_manual[idx] += sub_block.T

        cells_in_block0 += 1

    print(f"Cells contributing to Block 0: {cells_in_block0}")

    # Check manual block 0 diagonal
    diag_manual = np.array([block0_manual[sym_idx(0,0)][0,0],
                            block0_manual[sym_idx(0,0)][1,1],
                            block0_manual[sym_idx(0,0)][2,2]])
    print(f"Manual Block 0, Vertex 0 diagonal: [{diag_manual[0]:.4e}, {diag_manual[1]:.4e}, {diag_manual[2]:.4e}]")

    # Compare with Taichi
    solver.mas._clear_block_matrices()
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    sym_storage = solver.mas.block_matrices.to_numpy()[0]

    diag_taichi = np.array([sym_storage[sym_idx(0,0)][0,0],
                           sym_storage[sym_idx(0,0)][1,1],
                           sym_storage[sym_idx(0,0)][2,2]])
    print(f"Taichi Block 0, Vertex 0 diagonal: [{diag_taichi[0]:.4e}, {diag_taichi[1]:.4e}, {diag_taichi[2]:.4e}]")

    # Difference
    diff = diag_taichi - diag_manual
    print(f"Difference (Taichi - Manual): [{diff[0]:.4e}, {diff[1]:.4e}, {diff[2]:.4e}]")

    if np.linalg.norm(diff) > 1e-3:
        print("\n⚠️ SIGNIFICANT DIFFERENCE between Taichi and manual assembly!")
        print("This suggests a bug in the Taichi assembly code.")
    else:
        print("\n✓ Manual and Taichi assembly match")


if __name__ == '__main__':
    main()
