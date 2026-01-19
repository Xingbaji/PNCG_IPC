"""
Debug which cells contribute to Vertex 0 and their element Hessians.
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


# Store cell data
cell_v_ids = ti.field(dtype=ti.i32, shape=(100000, 4))
cell_count = ti.field(dtype=ti.i32, shape=())
cell_H_diag = ti.field(dtype=ti.f32, shape=(100000, 12))  # Store diagonal of H_e


@ti.kernel
def find_cells_with_vertex0(mesh: ti.template()):
    """Find all cells that contain vertex 0."""
    cell_count[None] = 0
    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
        if v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0:
            idx = ti.atomic_add(cell_count[None], 1)
            cell_v_ids[idx, 0] = v0
            cell_v_ids[idx, 1] = v1
            cell_v_ids[idx, 2] = v2
            cell_v_ids[idx, 3] = v3


@ti.kernel
def compute_cell_hessians(mesh: ti.template(), mu: ti.f32, la: ti.f32, dt: ti.f32):
    """Compute element Hessians for cells containing vertex 0."""
    for cell_idx in range(cell_count[None]):
        # Find the cell
        v0_target = cell_v_ids[cell_idx, 0]
        v1_target = cell_v_ids[cell_idx, 1]
        v2_target = cell_v_ids[cell_idx, 2]
        v3_target = cell_v_ids[cell_idx, 3]

        for c in mesh.cells:
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            if v0 == v0_target and v1 == v1_target and v2 == v2_target and v3 == v3_target:
                # Compute element Hessian
                W = c.W
                para = W * dt * dt

                Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
                B = c.B
                F = Ds @ B

                dFdx = compute_dFdx(B)
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

                temp = d2PsidF2 @ dFdx
                H_e = dFdx.transpose() @ temp
                H_e = para * H_e

                # Store diagonal
                for i in ti.static(range(12)):
                    cell_H_diag[cell_idx, i] = H_e[i, i]


def main():
    print("=" * 70)
    print("Debug: Cells Contributing to Vertex 0")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    # Find cells with vertex 0
    find_cells_with_vertex0(solver.mesh)
    n_cells_with_v0 = cell_count[None]

    print(f"\nNumber of cells containing vertex 0: {n_cells_with_v0}")

    # Get cell vertex IDs
    v_ids = cell_v_ids.to_numpy()[:n_cells_with_v0]

    # For each cell, find which local index is vertex 0
    for i, vids in enumerate(v_ids):
        local_idx = np.where(vids == 0)[0][0]
        print(f"  Cell {i}: vertices {vids}, vertex 0 is at local index {local_idx}")

    # Compute element Hessians
    compute_cell_hessians(solver.mesh, solver.mu, solver.la, solver.dt)
    H_diags = cell_H_diag.to_numpy()[:n_cells_with_v0]

    print(f"\n" + "-" * 70)
    print("Element Hessian diagonal analysis")
    print("-" * 70)

    # Accumulate contribution to vertex 0's diagonal
    v0_diag_contrib = np.zeros(3)

    for i, (vids, h_diag) in enumerate(zip(v_ids, H_diags)):
        local_idx = np.where(vids == 0)[0][0]
        v0_contrib = h_diag[local_idx*3:(local_idx+1)*3]
        v0_diag_contrib += v0_contrib

        print(f"\nCell {i}: vertices {vids}")
        print(f"  Local index for vertex 0: {local_idx}")
        print(f"  H_e diagonal for vertex 0: [{v0_contrib[0]:.4e}, {v0_contrib[1]:.4e}, {v0_contrib[2]:.4e}]")

        # Is this contribution positive?
        if np.any(v0_contrib < 0):
            print(f"  ⚠️ NEGATIVE diagonal contribution!")
        else:
            print(f"  ✓ All positive")

    print(f"\n" + "-" * 70)
    print("Accumulated diagonal for vertex 0")
    print("-" * 70)
    print(f"Total: [{v0_diag_contrib[0]:.4e}, {v0_diag_contrib[1]:.4e}, {v0_diag_contrib[2]:.4e}]")

    # The block matrix should have this accumulated value
    # But we saw ~-4e5 on the diagonal, which is very different!

    # Check if vertex 0 is in block 0
    print(f"\nVertex 0 is in block {0 // BANKSIZE}, lane {0 % BANKSIZE}")

    # Check F for these cells
    print(f"\n" + "-" * 70)
    print("Deformation gradient analysis")
    print("-" * 70)

    @ti.kernel
    def get_F_for_cell_idx(mesh: ti.template(), idx: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
        result = ti.Matrix.zero(ti.f32, 3, 3)
        v0_target = cell_v_ids[idx, 0]
        v1_target = cell_v_ids[idx, 1]
        v2_target = cell_v_ids[idx, 2]
        v3_target = cell_v_ids[idx, 3]

        for c in mesh.cells:
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            if v0 == v0_target and v1 == v1_target and v2 == v2_target and v3 == v3_target:
                Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
                B = c.B
                result = Ds @ B
        return result

    for i in range(min(3, n_cells_with_v0)):
        F = get_F_for_cell_idx(solver.mesh, i).to_numpy()
        U, s, Vh = np.linalg.svd(F)
        print(f"\nCell {i}:")
        print(f"  F =\n{F}")
        print(f"  Singular values: {s}")
        print(f"  det(F) = {np.linalg.det(F):.6f}")


if __name__ == '__main__':
    main()
