"""
Simple diagonal contribution test without fancy features.
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


# Storage for individual H_e computations
H_e_blocks = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(10, 4, 4))  # Up to 10 elements, 4x4 blocks
cell_indices = ti.field(dtype=ti.i32, shape=(10,))
cell_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def compute_He_for_elements_with_v0(
    mesh: ti.template(),
    mu: ti.f32, la: ti.f32, dt: ti.f32
):
    """Compute full H_e for elements containing global vertex 0."""
    cell_count[None] = 0

    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id

        has_v0 = 0
        if v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0:
            has_v0 = 1

        if has_v0 == 0:
            continue

        # Store this element
        idx = ti.atomic_add(cell_count[None], 1)
        if idx >= 10:
            continue

        cell_indices[idx] = c.id

        # Compute H_e
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

        # Store all 4x4 = 16 3x3 blocks
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                block = ti.Matrix.zero(ti.f32, 3, 3)
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        block[di, dj] = H_e[i * 3 + di, j * 3 + dj]
                H_e_blocks[idx, i, j] = block


def main():
    print("="*70)
    print("DIAGONAL CONTRIBUTION SIMPLE TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Compute H_e for elements with v0
    print("\n[Step 1] Computing H_e for elements with vertex 0...")
    compute_He_for_elements_with_v0(solver.mesh, solver.mu, solver.la, solver.dt)

    n_cells = cell_count[None]
    print(f"  Found {n_cells} elements")

    H_e_np = H_e_blocks.to_numpy()
    cell_idx_np = cell_indices.to_numpy()

    # Get element info from previous test
    # Element structures (from test_debug_simple.py):
    # Element 0 (cell 500): [299, 0, 300, 3], v0 at local 1
    # Element 1 (cell 501): [0, 299, 4, 3], v0 at local 0
    # Element 2 (cell 0):   [0, 1, 2, 3], v0 at local 0
    # Element 3 (cell 1):   [0, 4, 1, 3], v0 at local 0
    # Element 4 (cell 62):  [55, 0, 2, 3], v0 at local 1
    # Element 5 (cell 642): [300, 0, 55, 3], v0 at local 1

    element_info = {
        500: ([299, 0, 300, 3], 1),
        501: ([0, 299, 4, 3], 0),
        0: ([0, 1, 2, 3], 0),
        1: ([0, 4, 1, 3], 0),
        62: ([55, 0, 2, 3], 1),
        642: ([300, 0, 55, 3], 1),
    }

    print("\n[Step 2] Analyzing H_e diagonal blocks:")
    total_diag = np.zeros((3, 3))

    for i in range(n_cells):
        cell_id = cell_idx_np[i]
        if cell_id not in element_info:
            print(f"  Unknown cell {cell_id}")
            continue

        verts, local_idx = element_info[cell_id]
        warps = [v // BANKSIZE for v in verts]
        lanes = [v % BANKSIZE for v in verts]

        # Check if this is same-warp
        all_same_warp = all(w == warps[local_idx] for w in warps)
        warp0 = warps[local_idx] == 0

        # Get the H_e[local_idx, local_idx] block
        diag_block = H_e_np[i, local_idx, local_idx]
        sym_err = np.linalg.norm(diag_block - diag_block.T)

        print(f"\n  Cell {cell_id}: verts={verts}, v0 at local {local_idx}")
        print(f"    warps={warps}, lanes={lanes}")
        print(f"    H_e[{local_idx},{local_idx}] symmetry error: {sym_err:.6e}")

        # This element contributes to Block(lane_0, lane_0) = Block(0,0) only if:
        # - warp = 0 (same warp for all verts in the pair, which is just v0 to v0)
        # Actually, diagonal i==j always has warp_i == warp_j, so the condition
        # is just warp_ids[local_idx] == 0

        if warps[local_idx] == 0:
            print(f"    -> Contributes to Block(0,0)")
            total_diag += diag_block
        else:
            print(f"    -> Does NOT contribute to Block(0,0) (different warp)")

    print(f"\n[Step 3] Sum of diagonal contributions:")
    print(f"  Sum symmetry error: {np.linalg.norm(total_diag - total_diag.T):.6e}")
    print(f"\n  Sum matrix:")
    print(f"  [[{total_diag[0,0]:12.4f}, {total_diag[0,1]:12.4f}, {total_diag[0,2]:12.4f}]")
    print(f"   [{total_diag[1,0]:12.4f}, {total_diag[1,1]:12.4f}, {total_diag[1,2]:12.4f}]")
    print(f"   [{total_diag[2,0]:12.4f}, {total_diag[2,1]:12.4f}, {total_diag[2,2]:12.4f}]]")

    # Now run actual assembly
    print(f"\n[Step 4] Running actual assembly:")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_np[0, sym_idx_00]

    print(f"  Actual Block(0,0):")
    print(f"  [[{block_00[0,0]:12.4f}, {block_00[0,1]:12.4f}, {block_00[0,2]:12.4f}]")
    print(f"   [{block_00[1,0]:12.4f}, {block_00[1,1]:12.4f}, {block_00[1,2]:12.4f}]")
    print(f"   [{block_00[2,0]:12.4f}, {block_00[2,1]:12.4f}, {block_00[2,2]:12.4f}]]")

    print(f"\n  Actual Block(0,0) symmetry error: {np.linalg.norm(block_00 - block_00.T):.6e}")

    diff = block_00 - total_diag
    print(f"\n[Step 5] Difference (actual - expected from diagonal sum):")
    print(f"  [[{diff[0,0]:12.4f}, {diff[0,1]:12.4f}, {diff[0,2]:12.4f}]")
    print(f"   [{diff[1,0]:12.4f}, {diff[1,1]:12.4f}, {diff[1,2]:12.4f}]")
    print(f"   [{diff[2,0]:12.4f}, {diff[2,1]:12.4f}, {diff[2,2]:12.4f}]]")

    print(f"\n  Difference norm: {np.linalg.norm(diff):.6e}")

    if np.linalg.norm(diff) > 1e-3:
        print("\n  *** THERE ARE EXTRA CONTRIBUTIONS TO Block(0,0) ***")
        print("  This suggests off-diagonal H_e[i,j] (i != j) is being added")
        print("  to the diagonal position somehow.")

        # Check the difference pattern
        diff_sym_err = np.linalg.norm(diff - diff.T)
        print(f"\n  Difference symmetry error: {diff_sym_err:.6e}")

        if diff_sym_err > 1e-3:
            print("  The difference is NOT symmetric, which means asymmetric")
            print("  contributions are being added to Block(0,0)")


if __name__ == '__main__':
    main()
