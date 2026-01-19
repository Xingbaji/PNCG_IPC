"""
Test to verify diagonal block contributions.

Key insight: Block(0,0) receives H_e[i,i] where i is the local index of global vertex 0
in each element. Different elements can have global vertex 0 at different local indices.
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
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter, compute_dFdx

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


# Storage for H_e diagonal blocks
H_e_diag_storage = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(100,))
H_e_diag_info = ti.field(dtype=ti.i32, shape=(100, 3))  # (cell_id, local_idx, warp_id)
H_e_diag_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def extract_diagonal_contributions_for_v0(
    mesh: ti.template(),
    mu: ti.f32, la: ti.f32, dt: ti.f32
):
    """Extract H_e[i,i] contributions that go to Block(0,0)."""
    H_e_diag_count[None] = 0

    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id

        # Check if global vertex 0 is in this element
        local_idx = -1
        for k in ti.static(range(4)):
            v = c.verts[k].id
            if v == 0:
                local_idx = k

        if local_idx < 0:
            continue

        # Compute the element's warp id for this vertex
        warp_id = c.verts[local_idx].id // BANKSIZE

        # Only care about warp 0 (Level 0 assembly)
        if warp_id != 0:
            continue

        # Compute H_e for this element
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

        # Extract H_e[local_idx, local_idx] - the diagonal 3x3 block
        diag_block = ti.Matrix.zero(ti.f32, 3, 3)
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                diag_block[di, dj] = H_e[local_idx * 3 + di, local_idx * 3 + dj]

        # Store it
        idx = ti.atomic_add(H_e_diag_count[None], 1)
        if idx < 100:
            H_e_diag_storage[idx] = diag_block
            H_e_diag_info[idx, 0] = c.id
            H_e_diag_info[idx, 1] = local_idx
            H_e_diag_info[idx, 2] = warp_id


def main():
    print("="*70)
    print("DIAGONAL CONTRIBUTION TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Extract diagonal contributions
    print("\n[Step 1] Extracting H_e[i,i] contributions to Block(0,0)...")
    extract_diagonal_contributions_for_v0(solver.mesh, solver.mu, solver.la, solver.dt)

    n_diag = H_e_diag_count[None]
    print(f"  Found {n_diag} diagonal contributions to Block(0,0)")

    diag_storage_np = H_e_diag_storage.to_numpy()
    diag_info_np = H_e_diag_info.to_numpy()

    # Analyze each contribution
    print("\n[Step 2] Analyzing individual contributions:")
    total_diag = np.zeros((3, 3))

    for idx in range(n_diag):
        cell_id = diag_info_np[idx, 0]
        local_idx = diag_info_np[idx, 1]
        warp_id = diag_info_np[idx, 2]

        diag_block = diag_storage_np[idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)

        print(f"\n  Contribution {idx} (cell {cell_id}, local_idx {local_idx}):")
        print(f"    H_e[{local_idx},{local_idx}] symmetry error: {sym_err:.6e}")

        if sym_err > 1e-6:
            print(f"    WARNING: Non-symmetric diagonal block!")
            print(f"    [[{diag_block[0,0]:12.4f}, {diag_block[0,1]:12.4f}, {diag_block[0,2]:12.4f}]")
            print(f"     [{diag_block[1,0]:12.4f}, {diag_block[1,1]:12.4f}, {diag_block[1,2]:12.4f}]")
            print(f"     [{diag_block[2,0]:12.4f}, {diag_block[2,1]:12.4f}, {diag_block[2,2]:12.4f}]]")

        total_diag += diag_block

    # Check total
    print(f"\n[Step 3] Sum of all diagonal contributions:")
    sym_err_total = np.linalg.norm(total_diag - total_diag.T)
    print(f"  Symmetry error of sum: {sym_err_total:.6e}")

    if sym_err_total > 1e-6:
        print(f"\n  Sum matrix:")
        print(f"  [[{total_diag[0,0]:12.4f}, {total_diag[0,1]:12.4f}, {total_diag[0,2]:12.4f}]")
        print(f"   [{total_diag[1,0]:12.4f}, {total_diag[1,1]:12.4f}, {total_diag[1,2]:12.4f}]")
        print(f"   [{total_diag[2,0]:12.4f}, {total_diag[2,1]:12.4f}, {total_diag[2,2]:12.4f}]]")

        print(f"\n  Sum - Sum.T:")
        diff = total_diag - total_diag.T
        print(f"  [[{diff[0,0]:12.4f}, {diff[0,1]:12.4f}, {diff[0,2]:12.4f}]")
        print(f"   [{diff[1,0]:12.4f}, {diff[1,1]:12.4f}, {diff[1,2]:12.4f}]")
        print(f"   [{diff[2,0]:12.4f}, {diff[2,1]:12.4f}, {diff[2,2]:12.4f}]]")

    # Now run actual assembly
    print(f"\n[Step 4] Running actual assembly and comparing:")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_np[0, sym_idx_00]

    print(f"  Actual Block(0,0):")
    print(f"  [[{block_00[0,0]:12.4f}, {block_00[0,1]:12.4f}, {block_00[0,2]:12.4f}]")
    print(f"   [{block_00[1,0]:12.4f}, {block_00[1,1]:12.4f}, {block_00[1,2]:12.4f}]")
    print(f"   [{block_00[2,0]:12.4f}, {block_00[2,1]:12.4f}, {block_00[2,2]:12.4f}]]")

    print(f"\n  Difference (actual - expected from diag sum):")
    diff_actual = block_00 - total_diag
    print(f"  [[{diff_actual[0,0]:12.4f}, {diff_actual[0,1]:12.4f}, {diff_actual[0,2]:12.4f}]")
    print(f"   [{diff_actual[1,0]:12.4f}, {diff_actual[1,1]:12.4f}, {diff_actual[1,2]:12.4f}]")
    print(f"   [{diff_actual[2,0]:12.4f}, {diff_actual[2,1]:12.4f}, {diff_actual[2,2]:12.4f}]]")

    print(f"\n  Norm of difference: {np.linalg.norm(diff_actual):.6e}")

    # THE KEY QUESTION:
    # Is actual Block(0,0) != sum of H_e[i,i] contributions?
    # If yes, then there's something ELSE being added to Block(0,0)


if __name__ == '__main__':
    main()
