"""
Trace the assembly of Vertex 0's diagonal in the block matrix.
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


# Store cell data for vertex 0
max_cells = 100
cell_local_idx = ti.field(dtype=ti.i32, shape=max_cells)  # Which local index (0-3) is vertex 0
cell_he_diag = ti.Vector.field(3, dtype=ti.f32, shape=max_cells)  # H_e diagonal for that vertex
cell_count = ti.field(dtype=ti.i32, shape=())
cell_all_in_block0 = ti.field(dtype=ti.i32, shape=max_cells)  # 1 if all 4 vertices in block 0


@ti.kernel
def trace_vertex0_assembly(mesh: ti.template(), mu: ti.f32, la: ti.f32, dt: ti.f32):
    """Trace the elastic contribution to vertex 0's diagonal."""
    cell_count[None] = 0

    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
        v_ids = ti.Vector([v0, v1, v2, v3])

        # Check if any vertex is vertex 0
        has_v0 = v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0

        if has_v0:
            # Find which local index
            local_idx = -1
            if v0 == 0:
                local_idx = 0
            elif v1 == 0:
                local_idx = 1
            elif v2 == 0:
                local_idx = 2
            elif v3 == 0:
                local_idx = 3

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

            # Extract diagonal for vertex 0
            diag = ti.Vector([H_e[local_idx * 3, local_idx * 3],
                              H_e[local_idx * 3 + 1, local_idx * 3 + 1],
                              H_e[local_idx * 3 + 2, local_idx * 3 + 2]])

            # Check if all vertices are in block 0 (warp 0)
            all_in_block0 = 1
            if v0 // BANKSIZE != 0 or v1 // BANKSIZE != 0 or v2 // BANKSIZE != 0 or v3 // BANKSIZE != 0:
                all_in_block0 = 0

            idx = ti.atomic_add(cell_count[None], 1)
            if idx < max_cells:
                cell_local_idx[idx] = local_idx
                cell_he_diag[idx] = diag
                cell_all_in_block0[idx] = all_in_block0


def main():
    print("=" * 70)
    print("Trace Vertex 0 Assembly")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")
    print(f"Vertex 0 is in block {0 // BANKSIZE}, lane {0 % BANKSIZE}")
    print(f"mu = {solver.mu:.4e}, dt = {solver.dt:.4e}")

    # Trace assembly
    trace_vertex0_assembly(solver.mesh, solver.mu, solver.la, solver.dt)

    n_cells = cell_count[None]
    print(f"\nNumber of cells containing vertex 0: {n_cells}")

    # Get data
    local_idx = cell_local_idx.to_numpy()[:n_cells]
    he_diag = cell_he_diag.to_numpy()[:n_cells]
    all_in_block0 = cell_all_in_block0.to_numpy()[:n_cells]

    # Accumulate
    total_diag = np.zeros(3)
    total_diag_block0_only = np.zeros(3)
    positive_count = 0
    negative_count = 0

    print(f"\n--- Individual cell contributions ---")
    for i in range(min(n_cells, 20)):  # Show first 20
        diag = he_diag[i]
        is_positive = np.all(diag >= 0)
        status = "✓" if is_positive else "⚠️"
        in_block0 = "block0" if all_in_block0[i] else "cross-block"

        total_diag += diag

        if all_in_block0[i]:
            total_diag_block0_only += diag

        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        if i < 10 or not is_positive:
            print(f"  Cell {i}: local_idx={local_idx[i]}, diag=[{diag[0]:.4e}, {diag[1]:.4e}, {diag[2]:.4e}], {in_block0} {status}")

    print(f"\n--- Summary ---")
    print(f"Total cells: {n_cells}")
    print(f"Positive diagonal cells: {positive_count}")
    print(f"Negative diagonal cells: {negative_count}")

    print(f"\n--- Accumulated diagonal ---")
    print(f"Total (all cells): [{total_diag[0]:.4e}, {total_diag[1]:.4e}, {total_diag[2]:.4e}]")
    print(f"Block0 only cells: [{total_diag_block0_only[0]:.4e}, {total_diag_block0_only[1]:.4e}, {total_diag_block0_only[2]:.4e}]")

    # Compare with actual block matrix
    solver.mas._clear_block_matrices()
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    solver.mas._expand_sym_to_full()
    block0 = solver.mas.full_block_matrix.to_numpy()[0]

    print(f"\n--- Actual block matrix diagonal for vertex 0 ---")
    actual_diag = np.array([block0[0, 0], block0[1, 1], block0[2, 2]])
    print(f"Actual: [{actual_diag[0]:.4e}, {actual_diag[1]:.4e}, {actual_diag[2]:.4e}]")

    # Difference
    print(f"\n--- Difference ---")
    diff = actual_diag - total_diag
    print(f"Actual - Accumulated: [{diff[0]:.4e}, {diff[1]:.4e}, {diff[2]:.4e}]")

    # The difference might be due to cross-block contributions being added to coarse levels
    # But wait, vertex 0 is in block 0, so its diagonal should only come from same-block cells


if __name__ == '__main__':
    main()
