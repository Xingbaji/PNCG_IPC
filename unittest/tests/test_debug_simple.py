"""
Simple debug test to understand the assembly issue.
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


def sym_index(row, col):
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


# Field to track elements touching vertex 0
element_info = ti.field(dtype=ti.i32, shape=(100, 5))  # (elem_idx, v0, v1, v2, v3)
element_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def find_elements_with_v0(mesh: ti.template()):
    """Find elements that contain global vertex 0."""
    element_count[None] = 0

    for c in mesh.cells:
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id

        if v0 == 0 or v1 == 0 or v2 == 0 or v3 == 0:
            idx = ti.atomic_add(element_count[None], 1)
            if idx < 100:
                element_info[idx, 0] = c.id
                element_info[idx, 1] = v0
                element_info[idx, 2] = v1
                element_info[idx, 3] = v2
                element_info[idx, 4] = v3


def main():
    print("="*70)
    print("DEBUG SIMPLE TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Find elements containing vertex 0
    print("\n[Step 1] Finding elements containing vertex 0...")
    find_elements_with_v0(solver.mesh)

    n_elems = element_count[None]
    print(f"  Found {n_elems} elements containing vertex 0")

    elem_info_np = element_info.to_numpy()

    print("\n[Step 2] Analyzing element structure:")
    for i in range(min(n_elems, 20)):
        cell_id = elem_info_np[i, 0]
        v0, v1, v2, v3 = elem_info_np[i, 1:5]

        # Find which local index is vertex 0
        local_idx = -1
        for j, v in enumerate([v0, v1, v2, v3]):
            if v == 0:
                local_idx = j

        # Compute warp and lane for each vertex
        warps = [v // BANKSIZE for v in [v0, v1, v2, v3]]
        lanes = [v % BANKSIZE for v in [v0, v1, v2, v3]]

        # Check if all vertices are in warp 0
        all_in_warp0 = all(w == 0 for w in warps)

        print(f"\n  Element {i} (cell {cell_id}):")
        print(f"    vertices: [{v0}, {v1}, {v2}, {v3}]")
        print(f"    warps:    {warps}")
        print(f"    lanes:    {lanes}")
        print(f"    vertex 0 is at local index {local_idx}")
        print(f"    all in warp 0: {all_in_warp0}")

        # For this element, what pairs (i,j) with j>=i will contribute to Lane 0?
        # Answer: only when i==j and v_ids[i]==0 (lane=0, warp=0)
        # Because H_e[i,i] is symmetric

        # BUT WAIT - let me check if there's an off-diagonal contribution
        # When i < j and both lane_i == 0 and lane_j == 0?
        # This would require v_ids[i] = 0 and v_ids[j] = 0, which is impossible
        # since the element has distinct vertices

        # However, there's another case: when i < j and lane_i == lane_j == 0
        # but they are in DIFFERENT warps. In that case, cross-warp propagation
        # goes to coarse level, not Level 0.

        # So for Level 0, only diagonal contributions to Lane 0's Block(0,0)

    # Run assembly and check
    print("\n[Step 3] Running assembly...")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()

    # Get Block(0,0)
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_np[0, sym_idx_00]

    print(f"\n[Result] Block(0,0):")
    print(f"  [[{block_00[0,0]:12.4f}, {block_00[0,1]:12.4f}, {block_00[0,2]:12.4f}]")
    print(f"   [{block_00[1,0]:12.4f}, {block_00[1,1]:12.4f}, {block_00[1,2]:12.4f}]")
    print(f"   [{block_00[2,0]:12.4f}, {block_00[2,1]:12.4f}, {block_00[2,2]:12.4f}]]")

    print(f"\n[Result] Block(0,0) - Block(0,0)^T:")
    diff = block_00 - block_00.T
    print(f"  [[{diff[0,0]:12.4f}, {diff[0,1]:12.4f}, {diff[0,2]:12.4f}]")
    print(f"   [{diff[1,0]:12.4f}, {diff[1,1]:12.4f}, {diff[1,2]:12.4f}]")
    print(f"   [{diff[2,0]:12.4f}, {diff[2,1]:12.4f}, {diff[2,2]:12.4f}]]")

    sym_err = np.linalg.norm(diff)
    print(f"\n  Symmetry error: {sym_err:.6e}")

    # Key analysis: The off-diagonal terms in diff suggest that
    # something is adding asymmetric contributions to the diagonal block

    print("\n" + "="*70)
    print("HYPOTHESIS TESTING")
    print("="*70)

    # The difference pattern shows:
    # diff[0,1] = -10170, diff[1,0] = +10170
    # This means block_00[0,1] and block_00[1,0] differ by 2*10170 = 20340

    # For a diagonal block to have this pattern, we need:
    # - Some code path adding value A to [0,1]
    # - Some code path adding value A to [1,0] (but with wrong transpose)

    # In the current code, diagonal contributions come from H_e[i,i] which is symmetric
    # So diagonal blocks SHOULD be symmetric

    # UNLESS... there's a bug in extracting H_e[i,i] or in the atomic_add

    print("\nChecking off-diagonal blocks involving Lane 0:")
    for lane_j in range(1, 10):
        sym_idx = sym_index(0, lane_j)
        block = block_np[0, sym_idx]
        norm = np.linalg.norm(block)
        if norm > 1e-6:
            print(f"  Block(0,{lane_j}): norm={norm:.2e}")
            # Check if this block's transpose appears elsewhere
            # In symmetric storage, Block(0, lane_j) is the coupling from lane 0 to lane_j


if __name__ == '__main__':
    main()
