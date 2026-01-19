"""
Test to identify why only processing upper triangle (j >= i) causes issues.

Key insight: We're processing element vertex pairs (i,j) with j >= i, but the
global storage is in terms of (lane_i, lane_j). When lane_i > lane_j even though i < j,
we need to think carefully about what we're storing.

The symmetric block storage stores Block(min_lane, max_lane) which represents
the UPPER triangle in the global matrix.

When reconstructing full matrix from symmetric storage:
  - Block(l1, l2) for l1 < l2 is stored as-is (upper triangle)
  - Block(l1, l2) for l1 > l2 is obtained by Block(l2, l1)^T

So for the stored Block(min, max):
  - It IS the upper triangle entry M[min*3:(min+1)*3, max*3:(max+1)*3]
  - The lower triangle entry is its transpose: M[max*3, min*3] = Block(min, max)^T

When we process element (i, j) with i < j:
  - H_e[i,j] is the coupling from element vertex i to j
  - This goes to global position M[v_i*3, v_j*3] where v_i, v_j are global vertex ids
  - v_i = lane_i (if same warp), v_j = lane_j

Case 1: lane_i < lane_j
  - M[lane_i, lane_j] is upper triangle, store H_e[i,j] directly at sym_idx(lane_i, lane_j)
  - This is correct

Case 2: lane_i > lane_j
  - M[lane_i, lane_j] is lower triangle
  - In symmetric storage, we store UPPER triangle only
  - So we store at sym_idx(lane_j, lane_i) which is Block(lane_j, lane_i) = M[lane_j, lane_i]
  - M[lane_j, lane_i] = M[lane_i, lane_j]^T = H_e[i,j]^T (by symmetry)
  - So we should store H_e[i,j]^T at sym_idx(lane_j, lane_i) ✓
  - Current code: sub_block[dj, di] which is H_e[i,j]^T ✓

So the off-diagonal logic seems correct. Let's focus on why the DIAGONAL has issues.

For diagonal blocks (lane_i == lane_j):
  - Only happens when global vertices are same, i.e., i == j (same element vertex)
  - Because j >= i, we get i == j cases: (0,0), (1,1), (2,2), (3,3)
  - These add H_e[i,i] which is symmetric
  - Should be fine...

WAIT! The issue might be when lane_i == lane_j but i != j!

This can happen when two different element vertices map to the same global vertex.
But that's impossible in a valid mesh - each element has distinct vertices.

Let me re-read the error pattern:
- Lane 0 has error 1.649e+04
- Other lanes have ~1e-06 (float precision)

What's special about Lane 0?
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


def main():
    print("="*70)
    print("UPPER TRIANGLE ASSEMBLY BUG ANALYSIS")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    mas = solver.mas

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    mas.build_hierarchy()

    # Run assembly with current code
    print("\n[Step 1] Running elastic assembly (j >= i)...")
    mas._clear_block_matrices()
    mas._add_elastic_contribution_full_optimized(solver.mu, solver.la, solver.dt, 0)

    block_np = mas.block_matrices.to_numpy()

    # Get Lane 0's diagonal block
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_np[0, sym_idx_00]

    print(f"\nBlock(0,0) after assembly:")
    print(f"  [[{block_00[0,0]:12.4f}, {block_00[0,1]:12.4f}, {block_00[0,2]:12.4f}]")
    print(f"   [{block_00[1,0]:12.4f}, {block_00[1,1]:12.4f}, {block_00[1,2]:12.4f}]")
    print(f"   [{block_00[2,0]:12.4f}, {block_00[2,1]:12.4f}, {block_00[2,2]:12.4f}]]")

    print(f"\nBlock(0,0) - Block(0,0)^T:")
    diff = block_00 - block_00.T
    print(f"  [[{diff[0,0]:12.4f}, {diff[0,1]:12.4f}, {diff[0,2]:12.4f}]")
    print(f"   [{diff[1,0]:12.4f}, {diff[1,1]:12.4f}, {diff[1,2]:12.4f}]")
    print(f"   [{diff[2,0]:12.4f}, {diff[2,1]:12.4f}, {diff[2,2]:12.4f}]]")

    sym_err = np.linalg.norm(diff)
    print(f"\nSymmetry error: {sym_err:.6e}")

    # Key observation: the difference matrix is anti-symmetric
    # diff[i,j] = -diff[j,i]
    # This means there's something adding a[i,j] to position (i,j) and a[i,j] to position (j,i)
    # instead of a[i,j] to (i,j) and a[j,i]=a[i,j]^T to (j,i)

    print("\n" + "="*70)
    print("HYPOTHESIS: Off-diagonal contributions to Block(0,0)")
    print("="*70)

    # When can an off-diagonal (i < j) contribute to Block(0,0)?
    # This happens when lane_i == lane_j == 0
    # i.e., element vertices i and j both map to global lane 0

    # For this to happen: v_i % BANKSIZE == 0 AND v_j % BANKSIZE == 0
    # Both global vertices are at lane 0 of their respective warps

    # But wait, in our assembly we check warp_i == warp_j first
    # So both vertices must be in the same warp AND at lane 0
    # This means v_i = 0 and v_j = 0 which is impossible (i != j means v_i != v_j)

    # Unless... let me check if there's another code path

    print("\nAnalyzing which elements touch global vertex 0...")

    # Count elements containing vertex 0
    # Can't easily iterate mesh from Python
    print("(Need Taichi kernel to analyze elements)")

    # Let me think about this differently...
    # The anti-symmetric difference:
    # diff = [[0, -10170, 1931], [10170, 0, -5370], [-1931, 5370, 0]]
    #
    # This is exactly the form of H_e[i,j] - H_e[j,i] for some i != j
    # Since H_e is symmetric, H_e[i,j] = H_e[j,i]^T, so H_e[i,j] - H_e[j,i]
    # is anti-symmetric (as expected).
    #
    # But in our code, we only process j >= i, and for diagonal blocks
    # we only add H_e[i,i] which is symmetric.
    #
    # UNLESS... there's a bug where off-diagonal H_e[i,j] is being added
    # to diagonal Block(0,0) somehow.

    print("\n" + "="*70)
    print("CHECKING OFF-DIAGONAL BLOCKS")
    print("="*70)

    # Check Block(0,1), Block(0,2), etc. to see if they have contributions
    for lane_j in range(1, 16):
        sym_idx = sym_index(0, lane_j)
        block = block_np[0, sym_idx]
        norm = np.linalg.norm(block)
        if norm > 1e-6:
            print(f"Block(0,{lane_j}): norm={norm:.2e}")

    # Now let's look at the actual H_e structure
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)

    print("""
The difference matrix at Block(0,0) is:
  [[     0, -10170,  1931]
   [ 10170,      0, -5370]
   [ -1931,  5370,     0]]

This is an anti-symmetric matrix. For the diagonal block to become
non-symmetric, we need something like:

  Block(0,0) += A (some contribution)
  Block(0,0) += A^T (mirrored contribution) -- THIS IS MISSING

Looking at the code:
  for i in range(4):
    for j in range(i, 4):  # j >= i

When i == j:
  - We add H_e[i,i] which is symmetric ✓

When i < j and lane_i == lane_j (same lane!):
  - This can happen if two element vertices map to the same global vertex
  - But that's impossible in valid mesh

When i < j and lane_i < lane_j:
  - We add H_e[i,j] to Block(lane_i, lane_j) -- OFF-diagonal, no issue for Block(0,0)

When i < j and lane_i > lane_j:
  - We add H_e[i,j]^T to Block(lane_j, lane_i) -- OFF-diagonal, no issue for Block(0,0)

So Block(0,0) should ONLY get H_e[i,i] contributions...

WAIT! Let me check if there's cross-warp propagation affecting Lane 0.
The coarse level assembly might have issues!
""")

    # Check level 1 (coarse)
    level_size_np = mas.level_size.to_numpy()
    if mas.actual_levels > 1:
        level1_offset = level_size_np[1][1]
        level1_block0_idx = level1_offset // BANKSIZE

        print(f"\n[Level 1 Analysis]")
        print(f"  Level 1 offset: {level1_offset}")
        print(f"  Level 1 Block 0 index: {level1_block0_idx}")

        # Get going_next to see how vertices map
        going_next_np = mas.going_next.to_numpy()

        print(f"\n  going_next mapping for first few vertices:")
        for v in range(min(32, solver.n_verts)):
            coarse_v = going_next_np[v]
            if coarse_v >= 0:
                coarse_lane = coarse_v % BANKSIZE
                coarse_warp = coarse_v // BANKSIZE
                if coarse_lane == 0:
                    print(f"    v{v} -> coarse v{coarse_v} (warp {coarse_warp}, lane 0)")

        # Check Block 0 at level 1
        for lane in range(BANKSIZE):
            sym_idx = sym_index(lane, lane)
            block = block_np[level1_block0_idx, sym_idx]
            sym_err = np.linalg.norm(block - block.T)
            if sym_err > 1e-6:
                print(f"\n  Level 1 Lane {lane}: sym_error = {sym_err:.6e}")


if __name__ == '__main__':
    main()
