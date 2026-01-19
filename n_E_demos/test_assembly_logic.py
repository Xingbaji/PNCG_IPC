"""
Test the assembly logic to identify the symmetry issue.
Key insight: When i < j and lane_i > lane_j, we need to handle both contributions.
"""

import numpy as np

# Simulated BANKSIZE
BANKSIZE = 16

def sym_index(row, col):
    """Convert (row, col) to symmetric storage index."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def test_assembly_logic():
    """Test the assembly logic with a simple example."""
    print("="*70)
    print("ASSEMBLY LOGIC TEST")
    print("="*70)

    # Simulate a single element with 4 vertices
    # Vertices are arranged so that:
    # - v0 is at lane 0 (warp 0)
    # - v1 is at lane 1 (warp 0)
    # - etc.

    # Create a random but symmetric H_e
    np.random.seed(42)
    H_e = np.random.randn(12, 12)
    H_e = (H_e + H_e.T) / 2  # Make symmetric

    print(f"\nH_e is symmetric: ||H_e - H_e^T|| = {np.linalg.norm(H_e - H_e.T):.6e}")

    # Simulate block storage (only 1 warp for simplicity)
    # Storage size: (n_blocks, sym_size, 3, 3)
    sym_size = BANKSIZE * (BANKSIZE + 1) // 2
    block_matrices = np.zeros((1, sym_size, 3, 3))

    # Vertex IDs and their warp/lane assignments
    # Case 1: Normal ordering (lane_i <= lane_j for i < j)
    print("\n" + "-"*60)
    print("Case 1: Normal ordering (v0->lane0, v1->lane1, ...)")
    print("-"*60)

    v_ids = [0, 1, 2, 3]
    warp_ids = [0, 0, 0, 0]
    lane_ids = [0, 1, 2, 3]

    block_matrices_case1 = np.zeros((1, sym_size, 3, 3))

    # Current implementation: j >= i (upper triangle of element)
    for i in range(4):
        for j in range(i, 4):  # j >= i
            warp_i = warp_ids[i]
            warp_j = warp_ids[j]
            lane_i = lane_ids[i]
            lane_j = lane_ids[j]

            if warp_i == warp_j:
                sub_block = H_e[i*3:(i+1)*3, j*3:(j+1)*3]

                if lane_i <= lane_j:
                    sym_idx = sym_index(lane_i, lane_j)
                    block_matrices_case1[warp_i, sym_idx] += sub_block
                else:
                    # lane_i > lane_j: transpose
                    sym_idx = sym_index(lane_j, lane_i)
                    block_matrices_case1[warp_i, sym_idx] += sub_block.T

    # Check symmetry of Block(0,0)
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_matrices_case1[0, sym_idx_00]
    sym_err_00 = np.linalg.norm(block_00 - block_00.T)
    print(f"Block(0,0) symmetry error: {sym_err_00:.6e}")

    # Check if Block(0,0) equals H_e[0:3, 0:3]
    expected_00 = H_e[0:3, 0:3]
    print(f"Block(0,0) matches H_e[0,0]: {np.allclose(block_00, expected_00)}")

    # Case 2: Reversed ordering (lane_i > lane_j for some i < j)
    print("\n" + "-"*60)
    print("Case 2: Reversed ordering (v0->lane3, v1->lane2, v2->lane1, v3->lane0)")
    print("-"*60)

    # This simulates a case where vertex 0 of the element has a higher lane
    # than vertex 1, but we only process i < j
    v_ids = [0, 1, 2, 3]
    warp_ids = [0, 0, 0, 0]
    lane_ids = [3, 2, 1, 0]  # Reversed!

    block_matrices_case2 = np.zeros((1, sym_size, 3, 3))

    for i in range(4):
        for j in range(i, 4):  # j >= i
            warp_i = warp_ids[i]
            warp_j = warp_ids[j]
            lane_i = lane_ids[i]
            lane_j = lane_ids[j]

            if warp_i == warp_j:
                sub_block = H_e[i*3:(i+1)*3, j*3:(j+1)*3]

                if lane_i <= lane_j:
                    sym_idx = sym_index(lane_i, lane_j)
                    block_matrices_case2[warp_i, sym_idx] += sub_block
                else:
                    # lane_i > lane_j: transpose
                    sym_idx = sym_index(lane_j, lane_i)
                    block_matrices_case2[warp_i, sym_idx] += sub_block.T

    # Check symmetry of Block(0,0) - this is now lane 3's diagonal
    # In the reversed case:
    # - Element vertex 0 -> global vertex 0 -> lane 3
    # - Element vertex 3 -> global vertex 3 -> lane 0
    # So Block(3,3) gets H_e[0,0], Block(0,0) gets H_e[3,3]

    for lane in range(4):
        sym_idx = sym_index(lane, lane)
        block = block_matrices_case2[0, sym_idx]
        sym_err = np.linalg.norm(block - block.T)
        print(f"Block({lane},{lane}) symmetry error: {sym_err:.6e}")

    # Check off-diagonal blocks
    print("\nOff-diagonal blocks:")
    for lane_i in range(4):
        for lane_j in range(lane_i + 1, 4):
            sym_idx = sym_index(lane_i, lane_j)
            block = block_matrices_case2[0, sym_idx]
            print(f"Block({lane_i},{lane_j}): norm={np.linalg.norm(block):.6e}")

    # THE KEY ISSUE:
    print("\n" + "="*70)
    print("KEY ANALYSIS")
    print("="*70)

    # Let's trace what happens with i=0, j=1 when lane_ids=[3,2,1,0]
    print("\nFor element pair (i=0, j=1):")
    print(f"  lane_i = {lane_ids[0]}, lane_j = {lane_ids[1]}")
    print(f"  Since lane_i (3) > lane_j (2), we store transposed at sym_idx(2,3)")
    print(f"  We store H_e[0,1]^T at Block(2,3)")
    print(f"  But Block(2,3) should store the coupling between global vertices at lane 2 and 3")

    print("\nFor element pair (i=2, j=3):")
    print(f"  lane_i = {lane_ids[2]}, lane_j = {lane_ids[3]}")
    print(f"  Since lane_i (1) > lane_j (0), we store transposed at sym_idx(0,1)")
    print(f"  We store H_e[2,3]^T at Block(0,1)")

    # The issue is that H_e[i,j] with i < j corresponds to coupling between
    # element vertex i and j, NOT global lane i and j

    print("\n" + "="*70)
    print("THE BUG")
    print("="*70)
    print("""
When lane_i > lane_j (element vertex order != lane order):

Current code stores: H_e[i,j]^T at Block(lane_j, lane_i)
But symmetric storage means Block(lane_j, lane_i) = Block(lane_i, lane_j)^T

So we're storing H_e[i,j]^T, but we should be storing the coupling
between the global vertices at those lanes.

For off-diagonal blocks, this works because:
- Block(lane_j, lane_i) should store M[v_j, v_i] where v_j, v_i are global vertices
- M[v_j, v_i] = M[v_i, v_j]^T (matrix symmetry)
- H_e[i,j] is exactly M[v_i, v_j] for element vertices i,j
- So H_e[i,j]^T = M[v_i, v_j]^T = M[v_j, v_i] ✓

But for diagonal blocks (i == j, so lane_i can != lane_j... wait no, i==j means same vertex):
Actually for diagonal blocks, i == j always, so lane_i == lane_j always.
Diagonal blocks should be fine.

THE REAL ISSUE: When i < j but lane_i > lane_j...
- We extract sub_block = H_e[i,j]
- We want to add to Block(lane_i, lane_j) which is stored at sym_idx(lane_j, lane_i)
- Block(lane_i, lane_j) in symmetric storage means we store the UPPER triangle
- So Block(lane_j, lane_i)_stored represents Block(lane_j, lane_i) in actual matrix
- We want to add H_e[i,j] which represents coupling(v_i, v_j) = coupling(lane_i, lane_j)
- But coupling(lane_j, lane_i) = coupling(lane_i, lane_j)^T
- So Block(lane_j, lane_i)_stored should be coupling(lane_j, lane_i) = H_e[i,j]^T ✓

Wait, this seems correct... Let me check the diagonal case more carefully.
""")

    # Check diagonal case specifically
    print("\n" + "="*70)
    print("DIAGONAL BLOCK ANALYSIS")
    print("="*70)

    # For diagonal block Block(lane, lane), what contributes?
    # Only when lane_i == lane_j, which happens when element vertex i maps to lane
    # AND element vertex j maps to the same lane
    # This only happens when i == j (same element vertex)

    # So diagonal blocks only get contributions from i == j terms: H_e[i,i]
    # These are already symmetric!

    print("Diagonal blocks only receive H_e[i,i] contributions (i == j)")
    print("H_e[i,i] is symmetric by construction")
    print("So diagonal blocks should be symmetric...")

    # Let's check what's happening with our test
    print("\n" + "-"*60)
    print("Checking Case 2 diagonal contributions:")
    print("-"*60)

    for lane in range(4):
        # Find which element vertex maps to this lane
        for i, l in enumerate(lane_ids):
            if l == lane:
                expected = H_e[i*3:(i+1)*3, i*3:(i+1)*3]
                sym_idx = sym_index(lane, lane)
                actual = block_matrices_case2[0, sym_idx]
                print(f"Lane {lane}: element vertex {i}")
                print(f"  Expected H_e[{i},{i}], got Block({lane},{lane})")
                print(f"  Match: {np.allclose(expected, actual)}")
                print(f"  Difference norm: {np.linalg.norm(expected - actual):.6e}")


def test_multi_element():
    """Test with multiple elements sharing vertices."""
    print("\n" + "="*70)
    print("MULTI-ELEMENT TEST")
    print("="*70)

    # Two elements sharing vertex 0
    # Element 1: vertices 0, 1, 2, 3 -> lanes 0, 1, 2, 3
    # Element 2: vertices 0, 4, 5, 6 -> lanes 0, 4, 5, 6

    np.random.seed(42)
    H_e1 = np.random.randn(12, 12)
    H_e1 = (H_e1 + H_e1.T) / 2

    np.random.seed(123)
    H_e2 = np.random.randn(12, 12)
    H_e2 = (H_e2 + H_e2.T) / 2

    sym_size = BANKSIZE * (BANKSIZE + 1) // 2
    block_matrices = np.zeros((1, sym_size, 3, 3))

    # Process element 1: vertices 0,1,2,3 -> lanes 0,1,2,3
    lane_ids_1 = [0, 1, 2, 3]
    for i in range(4):
        for j in range(i, 4):
            lane_i, lane_j = lane_ids_1[i], lane_ids_1[j]
            sub_block = H_e1[i*3:(i+1)*3, j*3:(j+1)*3]
            if lane_i <= lane_j:
                sym_idx = sym_index(lane_i, lane_j)
                block_matrices[0, sym_idx] += sub_block
            else:
                sym_idx = sym_index(lane_j, lane_i)
                block_matrices[0, sym_idx] += sub_block.T

    # Process element 2: vertices 0,4,5,6 -> lanes 0,4,5,6
    lane_ids_2 = [0, 4, 5, 6]
    for i in range(4):
        for j in range(i, 4):
            lane_i, lane_j = lane_ids_2[i], lane_ids_2[j]
            sub_block = H_e2[i*3:(i+1)*3, j*3:(j+1)*3]
            if lane_i <= lane_j:
                sym_idx = sym_index(lane_i, lane_j)
                block_matrices[0, sym_idx] += sub_block
            else:
                sym_idx = sym_index(lane_j, lane_i)
                block_matrices[0, sym_idx] += sub_block.T

    # Check Block(0,0)
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_matrices[0, sym_idx_00]
    expected_00 = H_e1[0:3, 0:3] + H_e2[0:3, 0:3]

    print(f"Block(0,0) expected: H_e1[0,0] + H_e2[0,0]")
    print(f"Block(0,0) match: {np.allclose(block_00, expected_00)}")
    print(f"Block(0,0) symmetry error: {np.linalg.norm(block_00 - block_00.T):.6e}")

    # Everything should be symmetric because:
    # - Each H_e[i,i] is symmetric
    # - Sum of symmetric matrices is symmetric

    print("\nThis case is fine because lane order matches element vertex order")


def test_problematic_case():
    """Test the problematic case: multiple elements with different orderings."""
    print("\n" + "="*70)
    print("PROBLEMATIC CASE TEST")
    print("="*70)
    print("Two elements where vertex 0 has different local indices:")
    print("  Element 1: global vertices [0, 1, 2, 3] -> element verts [0, 1, 2, 3]")
    print("  Element 2: global vertices [4, 0, 5, 6] -> element verts [0, 1, 2, 3]")
    print("  (Global vertex 0 is element vertex 1 in element 2!)")

    np.random.seed(42)
    H_e1 = np.random.randn(12, 12)
    H_e1 = (H_e1 + H_e1.T) / 2

    np.random.seed(123)
    H_e2 = np.random.randn(12, 12)
    H_e2 = (H_e2 + H_e2.T) / 2

    sym_size = BANKSIZE * (BANKSIZE + 1) // 2
    block_matrices = np.zeros((1, sym_size, 3, 3))

    # Process element 1: global verts [0,1,2,3], element verts [0,1,2,3]
    global_verts_1 = [0, 1, 2, 3]
    lane_ids_1 = [0, 1, 2, 3]
    for i in range(4):
        for j in range(i, 4):
            lane_i, lane_j = lane_ids_1[i], lane_ids_1[j]
            sub_block = H_e1[i*3:(i+1)*3, j*3:(j+1)*3]
            if lane_i <= lane_j:
                sym_idx = sym_index(lane_i, lane_j)
                block_matrices[0, sym_idx] += sub_block
            else:
                sym_idx = sym_index(lane_j, lane_i)
                block_matrices[0, sym_idx] += sub_block.T

    # Process element 2: global verts [4,0,5,6], element verts [0,1,2,3]
    # So: element vert 0 -> global 4 -> lane 4
    #     element vert 1 -> global 0 -> lane 0
    #     element vert 2 -> global 5 -> lane 5
    #     element vert 3 -> global 6 -> lane 6
    global_verts_2 = [4, 0, 5, 6]
    lane_ids_2 = [4, 0, 5, 6]

    for i in range(4):
        for j in range(i, 4):
            lane_i, lane_j = lane_ids_2[i], lane_ids_2[j]
            sub_block = H_e2[i*3:(i+1)*3, j*3:(j+1)*3]
            if lane_i <= lane_j:
                sym_idx = sym_index(lane_i, lane_j)
                block_matrices[0, sym_idx] += sub_block
            else:
                sym_idx = sym_index(lane_j, lane_i)
                block_matrices[0, sym_idx] += sub_block.T

    # Check Block(0,0)
    # From element 1: H_e1[0,0] (element vert 0 = global 0)
    # From element 2: H_e2[1,1] (element vert 1 = global 0)
    sym_idx_00 = sym_index(0, 0)
    block_00 = block_matrices[0, sym_idx_00]
    expected_00 = H_e1[0:3, 0:3] + H_e2[3:6, 3:6]  # H_e1[0,0] + H_e2[1,1]

    print(f"\nBlock(0,0) (global vertex 0):")
    print(f"  From element 1: H_e1[0,0] (element vert 0)")
    print(f"  From element 2: H_e2[1,1] (element vert 1)")
    print(f"  Expected sum symmetry error: {np.linalg.norm(expected_00 - expected_00.T):.6e}")
    print(f"  Actual Block(0,0) symmetry error: {np.linalg.norm(block_00 - block_00.T):.6e}")
    print(f"  Match expected: {np.allclose(block_00, expected_00)}")

    # This should still be symmetric because we're adding symmetric blocks

    # Now let's look at off-diagonal blocks involving lane 0
    print(f"\nOff-diagonal blocks involving lane 0:")

    # Block(0,4) should have contributions from element 2's (0,1) pair
    # Element 2: i=0 (lane 4), j=1 (lane 0)
    # Since i < j and lane_i > lane_j: store H_e2[0,1]^T at sym_idx(0,4)
    sym_idx_04 = sym_index(0, 4)
    block_04 = block_matrices[0, sym_idx_04]

    # What does Block(0,4) represent?
    # It's the coupling between global vertex 0 (lane 0) and global vertex 4 (lane 4)
    # In element 2: global 4 is element vert 0, global 0 is element vert 1
    # So the coupling is H_e2[0,1] (coupling between element verts 0 and 1)
    # But since lane 4 > lane 0, we need Block(0,4)_upper_storage = Block(4,0)^T
    # And Block(4,0) = H_e2[0,1] (coupling from lane 4 to lane 0)

    # When we store, we have i=0, j=1, lane_i=4, lane_j=0
    # Since lane_i > lane_j, we store H_e2[0,1]^T at sym_idx(0,4)
    # This means Block(0,4)_stored = H_e2[0,1]^T

    # But wait, in symmetric storage, Block(0,4)_stored represents the upper triangle
    # i.e., the (row 0, col 4) block
    # And Block(0,4) = M[0:3, 12:15] for global matrix M
    # M[0:3, 12:15] = coupling from global vert 0 to global vert 4
    # = H_e2[element_vert_for_global_0, element_vert_for_global_4]
    # = H_e2[1, 0]

    # But H_e2[1,0] = H_e2[0,1]^T (symmetry of H_e)
    # So we expect Block(0,4) = H_e2[0,1]^T = H_e2[1,0]

    expected_04 = H_e2[3:6, 0:3]  # H_e2[1,0] = H_e2[0,1]^T
    print(f"\nBlock(0,4):")
    print(f"  Expected H_e2[1,0] (element verts 1->0)")
    print(f"  Actually stored: H_e2[0,1]^T (since lane order reversed)")
    print(f"  These should be equal (H_e2 symmetry)")
    print(f"  Match: {np.allclose(block_04, expected_04)}")
    print(f"  Difference norm: {np.linalg.norm(block_04 - expected_04):.6e}")

    # Actually let me verify what was stored
    print(f"\nDebug: What was actually stored at Block(0,4)?")
    print(f"  Processing i=0, j=1 with lanes [4, 0, 5, 6]")
    print(f"  lane_i = 4, lane_j = 0")
    print(f"  Since lane_i > lane_j, store H_e2[0,1]^T at sym_idx(0,4)")
    stored_04 = H_e2[0:3, 3:6].T  # H_e2[0,1]^T
    print(f"  Stored value = H_e2[0,1]^T")
    print(f"  Actual block_04 matches stored: {np.allclose(block_04, stored_04)}")


if __name__ == '__main__':
    test_assembly_logic()
    test_multi_element()
    test_problematic_case()
