"""
Test Level 0 assembly ONLY (no coarse propagation) to isolate the issue.
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


SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2


def sym_index(row, col):
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


# Storage for tracking contributions
contribution_count = ti.field(dtype=ti.i32, shape=(BANKSIZE, BANKSIZE))
contribution_source = ti.field(dtype=ti.i32, shape=(BANKSIZE, BANKSIZE, 100))  # (lane_i, lane_j, which_cell)


@ti.kernel
def test_level0_assembly_only(
    mesh: ti.template(),
    block_matrices: ti.template(),
    mu: ti.f32, la: ti.f32, dt: ti.f32
):
    """Assembly that ONLY does Level 0 (same-warp) contributions."""
    for c in mesh.cells:
        W = c.W
        para = W * dt * dt

        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
        v_ids = ti.Vector([v0, v1, v2, v3])

        warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                              v2 // BANKSIZE, v3 // BANKSIZE])
        lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                              v2 % BANKSIZE, v3 % BANKSIZE])

        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        B = c.B
        F = Ds @ B

        dFdx = compute_dFdx(B)
        d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

        temp = d2PsidF2 @ dFdx
        H_e = dFdx.transpose() @ temp
        H_e = para * H_e

        # ONLY process same-warp, Level 0 contributions
        # Skip cross-warp (coarse level) entirely
        for i in ti.static(range(4)):
            for j in ti.static(range(i, 4)):  # j >= i
                warp_i = warp_ids[i]
                warp_j = warp_ids[j]
                lane_i = lane_ids[i]
                lane_j = lane_ids[j]

                if warp_i == warp_j and warp_i == 0:  # Only warp 0 for analysis
                    sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                    if lane_i <= lane_j:
                        sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[di, dj])
                    else:
                        # lane_i > lane_j: transpose
                        sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[dj, di])


@ti.kernel
def count_contributions_to_lane0(
    mesh: ti.template(),
    mu: ti.f32, la: ti.f32, dt: ti.f32
) -> ti.i32:
    """Count how many elements contribute to Lane 0's diagonal."""
    count = 0
    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id

        # Check if any vertex is global vertex 0
        if v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0:
            ti.atomic_add(count, 1)

    return count


@ti.kernel
def test_single_element_at_vertex0(
    mesh: ti.template(),
    mu: ti.f32, la: ti.f32, dt: ti.f32
) -> ti.i32:
    """Find and process only elements containing vertex 0, count them."""
    count = 0

    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id

        # Skip if vertex 0 is not in this element
        has_v0 = 0
        if v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0:
            has_v0 = 1

        if has_v0 == 0:
            continue

        ti.atomic_add(count, 1)

        v_ids = ti.Vector([v0, v1, v2, v3])
        warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                              v2 // BANKSIZE, v3 // BANKSIZE])
        lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                              v2 % BANKSIZE, v3 % BANKSIZE])

        # Find which local index corresponds to global vertex 0
        local_idx_for_v0 = -1
        for i in ti.static(range(4)):
            if v_ids[i] == 0:
                local_idx_for_v0 = i

        # Print debug info for first few elements
        if count <= 5:
            print(f"  Element {count}: v_ids=[{v0},{v1},{v2},{v3}], local_idx_for_v0={local_idx_for_v0}")
            print(f"    warp_ids=[{warp_ids[0]},{warp_ids[1]},{warp_ids[2]},{warp_ids[3]}]")
            print(f"    lane_ids=[{lane_ids[0]},{lane_ids[1]},{lane_ids[2]},{lane_ids[3]}]")

            # Which pairs (i,j) with j>=i will contribute to Lane 0?
            # Diagonal: when i==j and lane_ids[i]==0 and warp_ids[i]==0
            # Off-diagonal: when i<j and lane_ids[i]==lane_ids[j]==0 and same warp
            #              This is IMPOSSIBLE since v_ids[i] != v_ids[j]

    return count


def main():
    print("="*70)
    print("LEVEL 0 ONLY ASSEMBLY TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Count elements touching vertex 0
    print("\n[Step 1] Counting elements containing vertex 0...")
    n_touching = count_contributions_to_lane0(solver.mesh, solver.mu, solver.la, solver.dt)
    print(f"  Elements containing vertex 0: {n_touching}")

    # Analyze those elements
    print("\n[Step 2] Analyzing elements containing vertex 0...")
    test_single_element_at_vertex0(solver.mesh, solver.mu, solver.la, solver.dt)

    # Run Level 0 only assembly
    print("\n[Step 3] Running Level 0 ONLY assembly...")
    mas._clear_block_matrices()
    test_level0_assembly_only(solver.mesh, mas.block_matrices, solver.mu, solver.la, solver.dt)

    block_np = mas.block_matrices.to_numpy()

    # Check Block(0,0) symmetry
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_np[0, sym_idx_00]
    sym_err_00 = np.linalg.norm(block_00 - block_00.T)

    print(f"\n[Result] Block(0,0) after Level 0 only assembly:")
    print(f"  Symmetry error: {sym_err_00:.6e}")

    if sym_err_00 > 1e-6:
        print(f"\n  Block(0,0):")
        print(f"  [[{block_00[0,0]:12.4f}, {block_00[0,1]:12.4f}, {block_00[0,2]:12.4f}]")
        print(f"   [{block_00[1,0]:12.4f}, {block_00[1,1]:12.4f}, {block_00[1,2]:12.4f}]")
        print(f"   [{block_00[2,0]:12.4f}, {block_00[2,1]:12.4f}, {block_00[2,2]:12.4f}]]")

        print(f"\n  Block(0,0) - Block(0,0)^T:")
        diff = block_00 - block_00.T
        print(f"  [[{diff[0,0]:12.4f}, {diff[0,1]:12.4f}, {diff[0,2]:12.4f}]")
        print(f"   [{diff[1,0]:12.4f}, {diff[1,1]:12.4f}, {diff[1,2]:12.4f}]")
        print(f"   [{diff[2,0]:12.4f}, {diff[2,1]:12.4f}, {diff[2,2]:12.4f}]]")

    # Check other lanes
    print(f"\n[Step 4] Checking all Lane diagonal blocks in Warp 0:")
    for lane in range(BANKSIZE):
        sym_idx = sym_index(lane, lane)
        block = block_np[0, sym_idx]
        sym_err = np.linalg.norm(block - block.T)
        norm = np.linalg.norm(block)
        if sym_err > 1e-6:
            print(f"  Lane {lane}: norm={norm:.2e}, sym_error={sym_err:.2e}")

    # Now run full assembly and compare
    print(f"\n[Step 5] Running FULL assembly (with cross-warp)...")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np_full = mas.block_matrices.to_numpy()

    sym_idx_00 = sym_index(0, 0)
    block_00_full = block_np_full[0, sym_idx_00]
    sym_err_00_full = np.linalg.norm(block_00_full - block_00_full.T)

    print(f"  Block(0,0) after FULL assembly:")
    print(f"  Symmetry error: {sym_err_00_full:.6e}")

    # Compare Level 0 only vs Full
    print(f"\n[Comparison] Level 0 only vs Full assembly:")
    diff_level0_vs_full = np.linalg.norm(block_00 - block_00_full)
    print(f"  Difference in Block(0,0): {diff_level0_vs_full:.6e}")


if __name__ == '__main__':
    main()
