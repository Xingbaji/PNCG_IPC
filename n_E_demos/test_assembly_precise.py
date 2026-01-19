"""
Precise test to find the assembly bug.
Check what exactly is being stored to sym_idx(0,0).
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

ti.init(arch=ti.gpu, default_fp=ti.f32)

BANKSIZE = 16
SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2  # 136


def sym_index(row, col):
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


# Create test fields
block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(100, SYM_BLOCK_COUNT))
debug_contributions = ti.field(dtype=ti.i32, shape=(100, SYM_BLOCK_COUNT))


@ti.kernel
def clear():
    for i, j in block_matrices:
        block_matrices[i, j] = ti.Matrix.zero(ti.f32, 3, 3)
        debug_contributions[i, j] = 0


@ti.kernel
def test_assembly_single_element():
    """Simulate assembly for a single element with known vertex IDs."""
    # Simulate an element with vertices that might cause issues
    # Test case 1: vertices all in same warp
    v_ids = ti.Vector([0, 1, 2, 3])

    # Create a non-symmetric 12x12 test matrix to detect ordering issues
    H_e = ti.Matrix.zero(ti.f32, 12, 12)
    for row in range(12):
        for col in range(12):
            # Make it identifiable: H_e[i,j] = 100*i + j
            H_e[row, col] = float(100 * row + col)

    # Process upper triangle (i <= j)
    for i in ti.static(range(4)):
        for j in ti.static(range(i, 4)):
            vi = v_ids[i]
            vj = v_ids[j]
            warp_i = vi // BANKSIZE
            warp_j = vj // BANKSIZE
            lane_i = vi % BANKSIZE
            lane_j = vj % BANKSIZE

            if warp_i == warp_j:
                sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                if lane_i <= lane_j:
                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[di, dj])
                    ti.atomic_add(debug_contributions[warp_i, sym_idx], 1)
                else:
                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[dj, di])
                    ti.atomic_add(debug_contributions[warp_i, sym_idx], 1)


@ti.kernel
def test_assembly_problematic_element():
    """Test with vertices where lane collision might occur."""
    # Vertices where different element indices map to same lane
    # This shouldn't happen in same warp, but let's verify
    v_ids = ti.Vector([0, 5, 10, 15])  # All different lanes in warp 0

    H_e = ti.Matrix.zero(ti.f32, 12, 12)
    for row in range(12):
        for col in range(12):
            H_e[row, col] = float(1000 * row + col)

    for i in ti.static(range(4)):
        for j in ti.static(range(i, 4)):
            vi = v_ids[i]
            vj = v_ids[j]
            warp_i = vi // BANKSIZE
            warp_j = vj // BANKSIZE
            lane_i = vi % BANKSIZE
            lane_j = vj % BANKSIZE

            if warp_i == warp_j:
                sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                if lane_i <= lane_j:
                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[di, dj])
                    ti.atomic_add(debug_contributions[warp_i, sym_idx], 1)
                else:
                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(block_matrices[warp_i, sym_idx][di, dj], sub_block[dj, di])
                    ti.atomic_add(debug_contributions[warp_i, sym_idx], 1)


def main():
    print("="*70)
    print("PRECISE ASSEMBLY TEST")
    print("="*70)

    # Test 1: Simple element with sequential vertices
    print("\n[Test 1] Element with v_ids = [0, 1, 2, 3]")
    print("-" * 60)
    clear()
    test_assembly_single_element()

    block_np = block_matrices.to_numpy()
    contrib_np = debug_contributions.to_numpy()

    # Check what's at sym_idx(0,0) - diagonal for lane 0
    sym_00 = sym_index(0, 0)
    print(f"  sym_idx(0,0) = {sym_00}")
    print(f"  Contributions to (warp=0, sym_idx={sym_00}): {contrib_np[0, sym_00]}")
    print(f"  Block at sym_idx(0,0):\n{block_np[0, sym_00]}")

    # For H_e[i,j] = 100*i + j, diagonal block H_e[0,0] should be:
    # [[0, 1, 2], [100, 101, 102], [200, 201, 202]]
    expected_diag = np.array([[0, 1, 2], [100, 101, 102], [200, 201, 202]], dtype=np.float32)
    print(f"  Expected (H_e[0:3, 0:3]):\n{expected_diag}")

    is_match = np.allclose(block_np[0, sym_00], expected_diag)
    print(f"  Match: {is_match}")

    # Check symmetry of the diagonal block
    diag_block = block_np[0, sym_00]
    sym_err = np.linalg.norm(diag_block - diag_block.T)
    print(f"  Symmetry error: {sym_err:.6e}")

    # Check off-diagonal blocks
    print(f"\n  Off-diagonal blocks:")
    for lane_i in range(4):
        for lane_j in range(lane_i+1, 4):
            sym_idx = sym_index(lane_i, lane_j)
            block = block_np[0, sym_idx]
            print(f"    sym_idx({lane_i},{lane_j})={sym_idx}: contributions={contrib_np[0, sym_idx]}")

    # Test 2: Element with different lanes
    print("\n[Test 2] Element with v_ids = [0, 5, 10, 15]")
    print("-" * 60)
    clear()
    test_assembly_problematic_element()

    block_np = block_matrices.to_numpy()
    contrib_np = debug_contributions.to_numpy()

    # Check diagonal for lane 0
    sym_00 = sym_index(0, 0)
    print(f"  sym_idx(0,0) = {sym_00}")
    print(f"  Contributions: {contrib_np[0, sym_00]}")
    diag_block = block_np[0, sym_00]
    print(f"  Block:\n{diag_block}")
    print(f"  Symmetry error: {np.linalg.norm(diag_block - diag_block.T):.6e}")

    # For this element:
    # - (i=0, j=0): lane_i=0, lane_j=0 → sym_idx(0,0), store H_e[0:3, 0:3]
    # Only i=0, j=0 contributes to lane 0 diagonal
    expected = np.array([[0, 1, 2], [1000, 1001, 1002], [2000, 2001, 2002]], dtype=np.float32)
    print(f"  Expected (H_e[0:3, 0:3]):\n{expected}")

    # Test 3: Check all diagonal contributions
    print("\n[Test 3] Check all diagonal blocks for Test 2")
    print("-" * 60)
    for lane in [0, 5, 10, 15]:
        sym_idx = sym_index(lane, lane)
        block = block_np[0, sym_idx]
        contrib = contrib_np[0, sym_idx]
        sym_err = np.linalg.norm(block - block.T)
        print(f"  Lane {lane}: sym_idx={sym_idx}, contrib={contrib}, sym_err={sym_err:.6e}")


if __name__ == '__main__':
    main()
