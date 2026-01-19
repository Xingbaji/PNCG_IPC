"""
Debug assembly logic to understand the double-counting issue.
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

ti.init(arch=ti.gpu, default_fp=ti.f32)

BANKSIZE = 16

class DebugSolver(pncg_ipc_deformer):
    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)


def main():
    solver = DebugSolver(demo='eight_E_stiffness_test')

    # Get cell vertex IDs
    print("Checking cell-vertex mapping for cells touching Block 0...")

    # Find cells that have at least one vertex in Block 0 (vertices 0-15)
    cells_in_block0 = []

    for c_id in range(solver.n_cells):
        cell_verts = solver.mesh.cells.verts.to_numpy()[c_id]
        # Check if any vertex is in block 0
        if any(v < BANKSIZE for v in cell_verts):
            cells_in_block0.append(c_id)

    print(f"Found {len(cells_in_block0)} cells touching Block 0")

    # Show first few cells
    print("\nFirst 10 cells touching Block 0:")
    for c_id in cells_in_block0[:10]:
        cell_verts = solver.mesh.cells.verts.to_numpy()[c_id]
        warps = [v // BANKSIZE for v in cell_verts]
        lanes = [v % BANKSIZE for v in cell_verts]
        print(f"  Cell {c_id}: verts={cell_verts}, warps={warps}, lanes={lanes}")

        # Check which vertex pairs are in the same warp
        same_warp_pairs = []
        for i in range(4):
            for j in range(i, 4):  # Only upper triangle
                if warps[i] == warps[j]:
                    same_warp_pairs.append((i, j, cell_verts[i], cell_verts[j], lanes[i], lanes[j]))
        print(f"    Same-warp pairs: {same_warp_pairs}")

    # Analyze the issue: when cell has v0 in warp 0, v1 in warp 0
    # and cell's vertex order causes i < j but lane_i > lane_j
    print("\nAnalyzing ordering issues...")
    order_issues = []
    for c_id in cells_in_block0:
        cell_verts = solver.mesh.cells.verts.to_numpy()[c_id]
        warps = [v // BANKSIZE for v in cell_verts]
        lanes = [v % BANKSIZE for v in cell_verts]

        for i in range(4):
            for j in range(i+1, 4):  # Only i < j
                if warps[i] == warps[j]:
                    if lanes[i] > lanes[j]:
                        # This is a case where element index order differs from lane order
                        order_issues.append((c_id, i, j, cell_verts[i], cell_verts[j], lanes[i], lanes[j]))

    print(f"Found {len(order_issues)} (i,j) pairs where i < j but lane_i > lane_j")
    for issue in order_issues[:10]:
        c_id, i, j, vi, vj, li, lj = issue
        print(f"  Cell {c_id}: elem_pair=({i},{j}), verts=({vi},{vj}), lanes=({li},{lj})")

    # Now let's manually check what happens with a simple example
    print("\n" + "="*60)
    print("Manual example: symmetric Hessian storage")
    print("="*60)

    # Suppose H_e is a 12x12 symmetric matrix
    # H_e[i*3:(i+1)*3, j*3:(j+1)*3] is the (i,j)-th 3x3 block

    # For a cell with verts [0, 3, 8, 11] (all in warp 0)
    # lanes = [0, 3, 8, 11]

    # When assembling (i=0, j=1), we get H_e[0:3, 3:6] and store to sym_index(lane_0, lane_1) = sym_index(0, 3)
    # When assembling (i=1, j=0), we get H_e[3:6, 0:3] and check if lane_1 <= lane_0
    #   lane_1 = 3, lane_0 = 0, so lane_1 > lane_0
    #   We store H_e[3:6, 0:3].T to sym_index(0, 3)

    # But wait - the ORIGINAL code iterates ALL pairs (i,j), not just upper triangle!
    # So both (0,1) and (1,0) are processed.
    # For (0,1): lane_0=0 <= lane_1=3, store H_e[0:3, 3:6] to sym_index(0,3)
    # For (1,0): lane_1=3 > lane_0=0, store H_e[3:6, 0:3].T to sym_index(0,3)

    # Since H_e is symmetric, H_e[0:3, 3:6] = H_e[3:6, 0:3].T
    # So we're adding the same block TWICE to sym_index(0,3)!

    print("The original code iterates ALL (i,j) pairs:")
    print("  For (0,1): lane_0 <= lane_1, store H_e[0:3,3:6] to sym_idx")
    print("  For (1,0): lane_1 > lane_0, store H_e[3:6,0:3].T to same sym_idx")
    print("  Since H_e is symmetric, these are the SAME!")
    print("  -> Off-diagonal blocks are doubled!")

    print("\nThe FIXED code should only iterate upper triangle (i <= j):")
    print("  For (0,1): store H_e[0:3,3:6] to sym_idx")
    print("  -> No doubling!")

    print("\nBUT WAIT - if lanes are reordered, we might still have issues...")
    print("Consider cell with verts [3, 0, 8, 11] (note: v0=3, v1=0)")
    print("  lanes = [3, 0, 8, 11]")
    print("  For elem (i=0, j=1), we have lane_0=3, lane_1=0")
    print("  lane_0 > lane_1, so we store H_e[0:3,3:6].T to sym_index(0,3)")
    print("  This should be H_e[3:6,0:3] at position (0,3), which is correct!")


if __name__ == '__main__':
    main()
