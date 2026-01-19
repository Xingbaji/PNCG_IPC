"""
Direct test: Check block 0 assembly step by step.
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

from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter
from math_utils.matrix_util import compute_dFdx


# Debug: Track contributions to vertex 0's diagonal
debug_contrib_count = ti.field(dtype=ti.i32, shape=())
debug_contrib_values = ti.field(dtype=ti.f32, shape=(1000, 3))


class DebugSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


@ti.kernel
def debug_elastic_assembly(mesh: ti.template(), block_matrices: ti.template(),
                           mu: ti.f32, la: ti.f32, dt: ti.f32):
    """
    Custom assembly that tracks contributions to vertex 0's diagonal.
    """
    debug_contrib_count[None] = 0

    for c in mesh.cells:
        W = c.W
        para = W * dt * dt

        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
        v_ids = ti.Vector([v0, v1, v2, v3])

        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        B = c.B
        F = Ds @ B

        dFdx = compute_dFdx(B)
        d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

        temp = d2PsidF2 @ dFdx
        H_e = dFdx.transpose() @ temp
        H_e = para * H_e

        # Process all 16 vertex pairs (i, j)
        for i in ti.static(range(4)):
            for j in ti.static(range(i, 4)):  # upper triangle only
                vi = v_ids[i]
                vj = v_ids[j]
                warp_i = vi // BANKSIZE
                warp_j = vj // BANKSIZE

                if warp_i == warp_j:
                    lane_i = vi % BANKSIZE
                    lane_j = vj % BANKSIZE

                    # Extract sub-block
                    sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                    # Check if this contributes to vertex 0's diagonal (warp 0, lane 0)
                    if warp_i == 0 and lane_i == 0 and lane_j == 0:
                        # This is a diagonal contribution to vertex 0
                        idx = ti.atomic_add(debug_contrib_count[None], 1)
                        if idx < 1000:
                            debug_contrib_values[idx, 0] = sub_block[0, 0]
                            debug_contrib_values[idx, 1] = sub_block[1, 1]
                            debug_contrib_values[idx, 2] = sub_block[2, 2]

                    # Add to block matrix (same as original assembly)
                    if lane_i <= lane_j:
                        sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[di, dj])
                    else:
                        sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[dj, di])


def sym_idx(row, col):
    return BANKSIZE * row - row * (row + 1) // 2 + col


def main():
    print("=" * 70)
    print("Direct Block 0 Assembly Test")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")
    print(f"mu = {solver.mu:.4e}, la = {solver.la:.4e}, dt = {solver.dt:.4e}")

    # Clear matrices
    solver.mas._clear_block_matrices()

    # Add inertia
    solver.mas._add_inertia_contribution(solver.dt)
    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_after_inertia = sym_storage[sym_idx(0, 0)].diagonal()
    print(f"\nAfter inertia: diagonal = [{diag_after_inertia[0]:.4e}, {diag_after_inertia[1]:.4e}, {diag_after_inertia[2]:.4e}]")

    # Run custom debug assembly
    debug_elastic_assembly(solver.mesh, solver.mas.block_matrices,
                           solver.mu, solver.la, solver.dt)

    # Get results
    n_contrib = debug_contrib_count[None]
    print(f"\nContributions to vertex 0 diagonal: {n_contrib}")

    values = debug_contrib_values.to_numpy()[:n_contrib]
    total = np.sum(values, axis=0)
    print(f"Total elastic contribution: [{total[0]:.4e}, {total[1]:.4e}, {total[2]:.4e}]")

    # Check individual values
    print("\nIndividual contributions:")
    for i in range(min(10, n_contrib)):
        v = values[i]
        sign = "+" if v[0] >= 0 else "-"
        print(f"  [{i}] {sign} [{v[0]:.4e}, {v[1]:.4e}, {v[2]:.4e}]")

    # Check negative contributions
    neg_count = np.sum(values[:, 0] < 0)
    print(f"\nNegative contributions: {neg_count}")

    # Get final block matrix value
    sym_storage = solver.mas.block_matrices.to_numpy()[0]
    diag_final = sym_storage[sym_idx(0, 0)].diagonal()
    print(f"\nFinal diagonal: [{diag_final[0]:.4e}, {diag_final[1]:.4e}, {diag_final[2]:.4e}]")

    # Expected = inertia + elastic
    expected = diag_after_inertia + total
    print(f"Expected (inertia + elastic): [{expected[0]:.4e}, {expected[1]:.4e}, {expected[2]:.4e}]")

    # Difference
    diff = diag_final - expected
    print(f"Difference: [{diff[0]:.4e}, {diff[1]:.4e}, {diff[2]:.4e}]")

    if np.max(np.abs(diff)) > 1e-3:
        print("\n*** SIGNIFICANT DIFFERENCE - There's extra contribution from somewhere! ***")
    else:
        print("\n*** Values match - Assembly is correct ***")


if __name__ == '__main__':
    main()
