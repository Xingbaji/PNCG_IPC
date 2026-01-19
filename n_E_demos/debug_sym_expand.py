"""
Debug symmetric storage expansion issue.
"""

import numpy as np

BANKSIZE = 16

def sym_index(row, col):
    """Compute symmetric storage index."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def test_expand_original():
    """Test original expansion logic."""
    print("Testing ORIGINAL expansion logic:")

    # Create a simple test case
    # Store a known pattern in symmetric storage
    sym_storage = np.zeros((136, 3, 3))  # 136 = 16*17/2

    # Fill with test pattern: block_ij[a,b] = 100*i + 10*j + a + 0.1*b
    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            for di in range(3):
                for dj in range(3):
                    sym_storage[sym_idx, di, dj] = 100*row + 10*col + di + 0.1*dj

    # Expand using ORIGINAL logic
    full_matrix = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = sym_storage[sym_idx]

            for di in range(3):
                for dj in range(3):
                    # Upper triangle
                    full_matrix[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                    # Lower triangle (symmetric) - ORIGINAL CODE
                    if row != col:
                        full_matrix[col * 3 + dj, row * 3 + di] = block_3x3[dj, di]

    # Check symmetry
    sym_error = np.linalg.norm(full_matrix - full_matrix.T)
    print(f"  Symmetry error: {sym_error:.6e}")

    # Check a specific block
    print(f"\n  Block (0,1) stored at sym_idx={sym_index(0,1)}:")
    print(f"    Original: {sym_storage[sym_index(0,1)]}")
    print(f"    At (0,1): {full_matrix[0:3, 3:6]}")
    print(f"    At (1,0): {full_matrix[3:6, 0:3]}")
    print(f"    Expected at (1,0): {sym_storage[sym_index(0,1)].T}")

    return full_matrix


def test_expand_fixed():
    """Test fixed expansion logic."""
    print("\nTesting FIXED expansion logic:")

    # Create same test case
    sym_storage = np.zeros((136, 3, 3))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            for di in range(3):
                for dj in range(3):
                    sym_storage[sym_idx, di, dj] = 100*row + 10*col + di + 0.1*dj

    # Expand using FIXED logic
    full_matrix = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = sym_storage[sym_idx]

            for di in range(3):
                for dj in range(3):
                    # Upper triangle
                    full_matrix[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                    # Lower triangle (symmetric) - FIXED CODE
                    if row != col:
                        # For symmetric matrix: A[col*3+di, row*3+dj] = A[row*3+dj, col*3+di] = block_3x3[dj, di]
                        # But we need: A[col*3+di, row*3+dj] = block_3x3.T[di, dj] = block_3x3[dj, di]
                        # So: A[col*3+di, row*3+dj] = block_3x3[di, dj] is wrong
                        # Correct: A[col*3+di, row*3+dj] should equal A[row*3+dj, col*3+di]
                        full_matrix[col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    # Check symmetry
    sym_error = np.linalg.norm(full_matrix - full_matrix.T)
    print(f"  Symmetry error: {sym_error:.6e}")

    # Check a specific block
    print(f"\n  Block (0,1) stored at sym_idx={sym_index(0,1)}:")
    print(f"    Original: {sym_storage[sym_index(0,1)]}")
    print(f"    At (0,1): {full_matrix[0:3, 3:6]}")
    print(f"    At (1,0): {full_matrix[3:6, 0:3]}")
    print(f"    Expected at (1,0): {sym_storage[sym_index(0,1)].T}")

    return full_matrix


def test_expand_correct():
    """Test correct expansion logic."""
    print("\nTesting CORRECT expansion logic:")

    # Create same test case
    sym_storage = np.zeros((136, 3, 3))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            for di in range(3):
                for dj in range(3):
                    sym_storage[sym_idx, di, dj] = 100*row + 10*col + di + 0.1*dj

    # Expand using CORRECT logic
    full_matrix = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = sym_storage[sym_idx]

            for di in range(3):
                for dj in range(3):
                    # Upper triangle: (row*3 + di, col*3 + dj)
                    full_matrix[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                    # Lower triangle: (col*3 + di, row*3 + dj) = transpose of upper
                    if row != col:
                        full_matrix[col * 3 + di, row * 3 + dj] = block_3x3[di, dj]

    # Check symmetry
    sym_error = np.linalg.norm(full_matrix - full_matrix.T)
    print(f"  Symmetry error: {sym_error:.6e}")

    # Check a specific block
    print(f"\n  Block (0,1) stored at sym_idx={sym_index(0,1)}:")
    print(f"    Original: {sym_storage[sym_index(0,1)]}")
    print(f"    At (0,1): {full_matrix[0:3, 3:6]}")
    print(f"    At (1,0): {full_matrix[3:6, 0:3]}")
    print(f"    Expected at (1,0): {sym_storage[sym_index(0,1)].T}")

    # Hmm this is still wrong. Let's think more carefully...
    return full_matrix


def test_expand_really_correct():
    """Test really correct expansion logic."""
    print("\nTesting REALLY CORRECT expansion logic:")
    print("  Note: For symmetric block matrix, if we store upper triangle")
    print("        then block(row,col) for row<=col is stored")
    print("        and block(col,row) = block(row,col).T")

    # Create same test case - but this time make the stored 3x3 blocks symmetric
    # to test if the problem is in the 3x3 block or the 48x48 layout
    sym_storage = np.zeros((136, 3, 3))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            # Make each 3x3 block NOT symmetric to test properly
            for di in range(3):
                for dj in range(3):
                    sym_storage[sym_idx, di, dj] = 100*row + 10*col + di + 0.1*dj

    # Expand correctly
    full_matrix = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = sym_storage[sym_idx]

            # Upper triangle block: rows [row*3, row*3+3), cols [col*3, col*3+3)
            for di in range(3):
                for dj in range(3):
                    full_matrix[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]

            # Lower triangle block: rows [col*3, col*3+3), cols [row*3, row*3+3)
            # This should be the TRANSPOSE of the upper triangle block
            if row != col:
                for di in range(3):
                    for dj in range(3):
                        # block(col, row)[di, dj] = block(row, col)[dj, di]
                        # So: full_matrix[col*3+di, row*3+dj] = block_3x3[dj, di]
                        full_matrix[col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    # Check symmetry
    sym_error = np.linalg.norm(full_matrix - full_matrix.T)
    print(f"  Symmetry error: {sym_error:.6e}")

    # Check a specific block
    print(f"\n  Block (0,1) stored at sym_idx={sym_index(0,1)}:")
    print(f"    Original block_3x3:")
    print(f"      {sym_storage[sym_index(0,1)]}")
    print(f"    At (0,1) in full matrix:")
    print(f"      {full_matrix[0:3, 3:6]}")
    print(f"    At (1,0) in full matrix:")
    print(f"      {full_matrix[3:6, 0:3]}")
    print(f"    Expected at (1,0) = block_3x3.T:")
    print(f"      {sym_storage[sym_index(0,1)].T}")

    # Verify
    block_01 = full_matrix[0:3, 3:6]
    block_10 = full_matrix[3:6, 0:3]
    print(f"\n  Is block_10 == block_01.T? {np.allclose(block_10, block_01.T)}")

    return full_matrix


if __name__ == '__main__':
    print("="*60)
    print("Debug Symmetric Storage Expansion")
    print("="*60)

    test_expand_original()
    test_expand_fixed()
    test_expand_correct()
    test_expand_really_correct()
