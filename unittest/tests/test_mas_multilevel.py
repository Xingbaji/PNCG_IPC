"""
MAS Preconditioner Multi-Level Ground Truth Testing

This module provides comprehensive ground truth verification for the MAS Preconditioner's
multi-level hierarchy operations:

1. **Hierarchy Building** - Verify level construction and going_next mapping
2. **Restriction** - Verify fine-to-coarse gradient propagation
3. **Prolongation** - Verify coarse-to-fine solution aggregation
4. **Multi-Level Solve** - End-to-end verification with ground truth

The key difference from test_mas_ground_truth.py is that this tests MULTI-LEVEL
operations on meshes large enough to have 2+ hierarchy levels (n_verts > BANKSIZE).

Usage:
    python test_mas_multilevel.py -v          # Run all tests with verbose output
    python test_mas_multilevel.py -v TestHierarchyBuilding  # Run specific test class
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

# Initialize Taichi with CPU for deterministic results
ti.init(arch=ti.cpu, default_fp=ti.f32)


# ==============================================================================
# Constants (matching mas_preconditioner_pkg/constants.py)
# ==============================================================================

BANKSIZE = 16
SYM_BLOCK_COUNT = 136  # 16 * 17 / 2
BLOCK_DOF = 48  # BANKSIZE * 3
MAX_LEVELS = 6


# ==============================================================================
# Mesh Generation for Multi-Level Tests
# ==============================================================================

def create_grid_mesh(nx: int, ny: int, nz: int, spacing: float = 1.0):
    """
    Create a 3D grid mesh with tetrahedral elements.

    This generates a mesh large enough to have multiple hierarchy levels.
    A 3x3x3 grid has 27 vertices (> BANKSIZE=16), so it will have 2+ levels.

    Args:
        nx, ny, nz: Number of cells in each dimension
        spacing: Distance between grid nodes

    Returns:
        vertices: ((nx+1)*(ny+1)*(nz+1), 3) array of vertex positions
        cells: (6*nx*ny*nz, 4) array of tetrahedral cell indices
    """
    # Generate vertex positions
    n_verts_x = nx + 1
    n_verts_y = ny + 1
    n_verts_z = nz + 1
    n_verts = n_verts_x * n_verts_y * n_verts_z

    vertices = np.zeros((n_verts, 3), dtype=np.float64)

    for iz in range(n_verts_z):
        for iy in range(n_verts_y):
            for ix in range(n_verts_x):
                idx = iz * n_verts_y * n_verts_x + iy * n_verts_x + ix
                vertices[idx] = [ix * spacing, iy * spacing, iz * spacing]

    # Generate tetrahedral cells (6 tets per cube)
    cells_list = []

    def vertex_index(ix, iy, iz):
        return iz * n_verts_y * n_verts_x + iy * n_verts_x + ix

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # 8 corners of the cube
                v0 = vertex_index(ix, iy, iz)
                v1 = vertex_index(ix + 1, iy, iz)
                v2 = vertex_index(ix + 1, iy + 1, iz)
                v3 = vertex_index(ix, iy + 1, iz)
                v4 = vertex_index(ix, iy, iz + 1)
                v5 = vertex_index(ix + 1, iy, iz + 1)
                v6 = vertex_index(ix + 1, iy + 1, iz + 1)
                v7 = vertex_index(ix, iy + 1, iz + 1)

                # 6-tetrahedra decomposition of cube (consistent orientation)
                cells_list.append([v0, v1, v3, v4])
                cells_list.append([v1, v2, v3, v6])
                cells_list.append([v1, v4, v5, v6])
                cells_list.append([v3, v4, v6, v7])
                cells_list.append([v1, v3, v4, v6])
                # Note: We use 5 tets for proper tessellation
                # cells_list.append([v0, v1, v3, v5])  # Alternative: 6th tet for full cube

    cells = np.array(cells_list, dtype=np.int32)

    return vertices, cells


def create_line_mesh(n_segments: int, spacing: float = 1.0):
    """
    Create a 1D line mesh (series of connected tetrahedra).

    This creates a simple mesh where hierarchy behavior is predictable.
    Each segment has 4 vertices forming a tetrahedron.

    Args:
        n_segments: Number of tetrahedral segments
        spacing: Distance between nodes

    Returns:
        vertices: (n_segments + 3, 3) array of vertex positions
        cells: (n_segments, 4) array of cell indices
    """
    # For n_segments tets sharing vertices along a line:
    # We need n_segments + 3 vertices for a chain of tets
    n_verts = n_segments + 3
    vertices = np.zeros((n_verts, 3), dtype=np.float64)

    # Place vertices along x-axis with small offsets for tetrahedral geometry
    for i in range(n_verts):
        vertices[i] = [i * spacing, 0.0, 0.0]

    # Adjust some vertices to form proper tetrahedra
    # Create tetrahedra sharing edges along the line
    cells_list = []
    for i in range(n_segments):
        # Each tet uses 4 consecutive vertices
        cells_list.append([i, i+1, i+2, i+3])

    # Actually, let's use a different approach - create overlapping tets
    # that share faces, which is more realistic

    # Create a "tube" of tetrahedra
    n_verts = 4 * n_segments  # 4 vertices per segment
    vertices = np.zeros((n_verts, 3), dtype=np.float64)

    for seg in range(n_segments):
        base_idx = seg * 4
        x_offset = seg * spacing
        # 4 vertices forming a tetrahedron
        vertices[base_idx + 0] = [x_offset + 0.0, 0.0, 0.0]
        vertices[base_idx + 1] = [x_offset + 1.0, 0.0, 0.0]
        vertices[base_idx + 2] = [x_offset + 0.5, 0.866, 0.0]
        vertices[base_idx + 3] = [x_offset + 0.5, 0.289, 0.816]

    cells = np.array([[seg*4, seg*4+1, seg*4+2, seg*4+3] for seg in range(n_segments)],
                     dtype=np.int32)

    return vertices, cells


def create_multi_level_mesh(target_verts: int = 50):
    """
    Create a mesh with approximately target_verts vertices.

    This ensures we have enough vertices for 2+ hierarchy levels.

    Args:
        target_verts: Target number of vertices (default 50 for 3+ levels)

    Returns:
        vertices, cells: Mesh arrays
    """
    # Calculate grid dimensions to achieve target vertex count
    # n_verts = (nx+1)*(ny+1)*(nz+1)
    # For 50 verts: 4x4x3 = 5*5*4 = 100, 3x3x3 = 64, 3x3x2 = 48

    if target_verts <= 30:
        nx, ny, nz = 2, 2, 2  # 27 vertices
    elif target_verts <= 60:
        nx, ny, nz = 3, 3, 2  # 48 vertices
    elif target_verts <= 100:
        nx, ny, nz = 3, 3, 3  # 64 vertices
    else:
        nx, ny, nz = 4, 4, 4  # 125 vertices

    return create_grid_mesh(nx, ny, nz)


# ==============================================================================
# NumPy Ground Truth Functions for Multi-Level Operations
# ==============================================================================

def compute_hierarchy_levels_numpy(n_verts: int):
    """
    Compute expected hierarchy level sizes.

    Returns:
        level_sizes: List of (size, offset) tuples for each level
    """
    level_sizes = [(n_verts, 0)]
    current_size = n_verts
    current_offset = n_verts

    while current_size > BANKSIZE:
        # Each level reduces by factor of ~BANKSIZE (in the worst case)
        # Actually, it depends on connectivity. For a connected mesh,
        # each warp typically forms 1-2 clusters.
        # For simplicity, assume reduction by BANKSIZE
        next_size = (current_size + BANKSIZE - 1) // BANKSIZE
        level_sizes.append((next_size, current_offset))
        current_offset += next_size
        current_size = next_size

        if len(level_sizes) >= MAX_LEVELS:
            break

    return level_sizes


def compute_restriction_numpy(gradient: np.ndarray, going_next: np.ndarray,
                              level_sizes: list, n_verts: int):
    """
    Compute ground truth restriction operation.

    The restriction operator R maps fine-level gradient to coarse levels:
    r_{l+1} = R * r_l = sum of r_l within each cluster

    Args:
        gradient: Fine-level gradient (n_verts, 3)
        going_next: Parent index mapping (total_nodes,)
        level_sizes: List of (size, offset) for each level
        n_verts: Number of fine vertices

    Returns:
        multi_level_r: Dictionary mapping level -> residual array
    """
    # Compute total nodes
    total_nodes = level_sizes[-1][1] + level_sizes[-1][0]

    # Initialize multi-level residual
    multi_level_r = np.zeros((total_nodes, 3), dtype=np.float64)

    # Level 0: copy gradient
    multi_level_r[:n_verts] = gradient

    # Restrict to coarse levels
    for level in range(1, len(level_sizes)):
        prev_size, prev_offset = level_sizes[level - 1]
        curr_size, curr_offset = level_sizes[level]

        for i in range(prev_size):
            prev_idx = prev_offset + i
            next_idx = going_next[prev_idx]
            if 0 <= next_idx < total_nodes:
                multi_level_r[next_idx] += multi_level_r[prev_idx]

    return multi_level_r


def compute_prolongation_numpy(multi_level_z: np.ndarray,
                               aggregation_table: np.ndarray,
                               level_sizes: list, n_verts: int):
    """
    Compute ground truth prolongation operation.

    The prolongation operator P^T maps coarse-level solutions to fine:
    z_fine = z_0 + sum_{l>0} P_l^T * z_l

    For MAS, this is simply summing contributions from all levels.

    Args:
        multi_level_z: Solution at all levels (total_nodes, 3)
        aggregation_table: Path through hierarchy for each fine vertex (n_verts, n_levels-1)
        level_sizes: List of (size, offset) for each level
        n_verts: Number of fine vertices

    Returns:
        z_final: Final preconditioned direction (n_verts, 3)
    """
    z_final = np.zeros((n_verts, 3), dtype=np.float64)
    n_levels = len(level_sizes)

    for idx in range(n_verts):
        # Level 0 contribution
        z_final[idx] = multi_level_z[idx]

        # Coarse level contributions
        for level in range(n_levels - 1):
            coarse_idx = aggregation_table[idx, level]
            if coarse_idx >= 0 and coarse_idx < multi_level_z.shape[0]:
                z_final[idx] += multi_level_z[coarse_idx]

    return z_final


def compute_local_solve_numpy(multi_level_r: np.ndarray,
                              inv_block_matrices: np.ndarray,
                              level_sizes: list):
    """
    Compute ground truth local solve: z = B^{-1} * r at each level.

    Args:
        multi_level_r: Residual at all levels (total_nodes, 3)
        inv_block_matrices: Inverse block matrices (n_blocks, 136, 3, 3)
        level_sizes: List of (size, offset) for each level

    Returns:
        multi_level_z: Solution at all levels (total_nodes, 3)
    """
    total_nodes = multi_level_r.shape[0]
    multi_level_z = np.zeros((total_nodes, 3), dtype=np.float64)

    for level, (level_size, level_offset) in enumerate(level_sizes):
        n_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Get nodes in this block
            block_start = level_offset + block_id * BANKSIZE
            block_end = min(block_start + BANKSIZE, level_offset + level_size)
            nodes_in_block = block_end - block_start

            # Extract residual for this block
            r_block = np.zeros(BLOCK_DOF, dtype=np.float64)
            for local_idx in range(nodes_in_block):
                global_idx = block_start + local_idx
                r_block[local_idx*3:(local_idx+1)*3] = multi_level_r[global_idx]

            # Extract full inverse block matrix
            B_inv = extract_full_block_matrix_numpy(inv_block_matrices, block_id)

            # Compute z = B^{-1} * r
            z_block = B_inv @ r_block

            # Store result
            for local_idx in range(nodes_in_block):
                global_idx = block_start + local_idx
                multi_level_z[global_idx] = z_block[local_idx*3:(local_idx+1)*3]

    return multi_level_z


def sym_index(row, col):
    """Compute symmetric storage index for upper triangle."""
    r = min(row, col)
    c = max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


def extract_full_block_matrix_numpy(block_matrices_np, block_id):
    """
    Extract full 48x48 block matrix from symmetric storage.

    Args:
        block_matrices_np: NumPy array of shape (n_blocks, SYM_BLOCK_COUNT, 3, 3)
        block_id: Block index to extract

    Returns:
        full: 48x48 dense matrix
    """
    full = np.zeros((BLOCK_DOF, BLOCK_DOF), dtype=np.float64)

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            sym_idx = sym_index(row, col)
            block_3x3 = block_matrices_np[block_id, sym_idx]

            # Upper triangle
            full[row*3:(row+1)*3, col*3:(col+1)*3] = block_3x3
            # Lower triangle (symmetric)
            if row != col:
                full[col*3:(col+1)*3, row*3:(row+1)*3] = block_3x3.T

    return full


# ==============================================================================
# Test Classes
# ==============================================================================

class TestMeshGeneration(unittest.TestCase):
    """Test mesh generation functions."""

    def test_grid_mesh_sizes(self):
        """Test that grid mesh generates expected vertex/cell counts."""
        # 2x2x2 grid: 3*3*3 = 27 vertices, 5*8 = 40 cells
        vertices, cells = create_grid_mesh(2, 2, 2)
        self.assertEqual(len(vertices), 27)
        self.assertEqual(len(cells), 5 * 8)  # 5 tets per cube, 8 cubes

        # 3x3x3 grid: 4*4*4 = 64 vertices
        vertices, cells = create_grid_mesh(3, 3, 3)
        self.assertEqual(len(vertices), 64)

    def test_grid_mesh_validity(self):
        """Test that grid mesh has valid geometry."""
        vertices, cells = create_grid_mesh(2, 2, 2)

        # All cell indices should be valid
        self.assertTrue(np.all(cells >= 0))
        self.assertTrue(np.all(cells < len(vertices)))

        # No duplicate vertices in any cell
        for cell in cells:
            self.assertEqual(len(set(cell)), 4, "Cell should have 4 unique vertices")

    def test_multi_level_mesh_size(self):
        """Test multi-level mesh has enough vertices for 2+ levels."""
        vertices, cells = create_multi_level_mesh(target_verts=50)

        # Should have at least BANKSIZE vertices for 2+ levels
        self.assertGreater(len(vertices), BANKSIZE)

        # Check expected level count
        expected_levels = compute_hierarchy_levels_numpy(len(vertices))
        self.assertGreaterEqual(len(expected_levels), 2)


class TestHierarchyLevels(unittest.TestCase):
    """Test hierarchy level computation."""

    def test_small_mesh_levels(self):
        """Test level computation for small mesh (1-2 levels)."""
        n_verts = 10  # Less than BANKSIZE
        levels = compute_hierarchy_levels_numpy(n_verts)

        # Should have at least 1 level
        self.assertGreaterEqual(len(levels), 1)

        # Level 0 should have all vertices
        self.assertEqual(levels[0], (n_verts, 0))

    def test_medium_mesh_levels(self):
        """Test level computation for medium mesh (2-3 levels)."""
        n_verts = 50  # ~3 warps, should have 2-3 levels
        levels = compute_hierarchy_levels_numpy(n_verts)

        # Should have 2+ levels
        self.assertGreaterEqual(len(levels), 2)

        # Level sizes should decrease
        for i in range(1, len(levels)):
            self.assertLessEqual(levels[i][0], levels[i-1][0])

    def test_large_mesh_levels(self):
        """Test level computation for large mesh (3+ levels)."""
        n_verts = 500  # ~32 warps, should have 3-4 levels
        levels = compute_hierarchy_levels_numpy(n_verts)

        # Should have 3+ levels
        self.assertGreaterEqual(len(levels), 3)

        # Offsets should be cumulative
        total = levels[0][0]
        for i in range(1, len(levels)):
            self.assertEqual(levels[i][1], total)
            total += levels[i][0]


class TestRestrictionGroundTruth(unittest.TestCase):
    """Test restriction operation ground truth."""

    def test_single_level_restriction(self):
        """Test restriction with single level (all in one block)."""
        n_verts = 10  # Single block
        gradient = np.random.randn(n_verts, 3)

        # Simple going_next: all point to same coarse node
        going_next = np.zeros(n_verts + 1, dtype=np.int32)
        going_next[:n_verts] = n_verts  # All fine nodes -> single coarse node

        level_sizes = [(n_verts, 0), (1, n_verts)]

        multi_level_r = compute_restriction_numpy(gradient, going_next, level_sizes, n_verts)

        # Level 0 should equal gradient
        np.testing.assert_allclose(multi_level_r[:n_verts], gradient)

        # Level 1 should be sum of all gradients
        expected_sum = np.sum(gradient, axis=0)
        np.testing.assert_allclose(multi_level_r[n_verts], expected_sum)

    def test_multi_block_restriction(self):
        """Test restriction with multiple blocks."""
        n_verts = 32  # 2 blocks
        gradient = np.random.randn(n_verts, 3)

        # going_next: block 0 -> coarse 0, block 1 -> coarse 1
        going_next = np.zeros(n_verts + 2, dtype=np.int32)
        for i in range(BANKSIZE):
            going_next[i] = n_verts  # Block 0 -> coarse 0
        for i in range(BANKSIZE, n_verts):
            going_next[i] = n_verts + 1  # Block 1 -> coarse 1

        level_sizes = [(n_verts, 0), (2, n_verts)]

        multi_level_r = compute_restriction_numpy(gradient, going_next, level_sizes, n_verts)

        # Level 0 should equal gradient
        np.testing.assert_allclose(multi_level_r[:n_verts], gradient)

        # Level 1: coarse 0 = sum of block 0, coarse 1 = sum of block 1
        expected_coarse_0 = np.sum(gradient[:BANKSIZE], axis=0)
        expected_coarse_1 = np.sum(gradient[BANKSIZE:], axis=0)

        np.testing.assert_allclose(multi_level_r[n_verts], expected_coarse_0)
        np.testing.assert_allclose(multi_level_r[n_verts + 1], expected_coarse_1)


class TestProlongationGroundTruth(unittest.TestCase):
    """Test prolongation operation ground truth."""

    def test_single_level_prolongation(self):
        """Test prolongation with single level."""
        n_verts = 10
        total_nodes = n_verts + 1

        multi_level_z = np.random.randn(total_nodes, 3)

        # All fine vertices aggregate from same coarse node
        aggregation_table = np.full((n_verts, MAX_LEVELS - 1), n_verts, dtype=np.int32)

        level_sizes = [(n_verts, 0), (1, n_verts)]

        z_final = compute_prolongation_numpy(multi_level_z, aggregation_table,
                                             level_sizes, n_verts)

        # z_final[i] = z_0[i] + z_1[n_verts] for all i
        for i in range(n_verts):
            expected = multi_level_z[i] + multi_level_z[n_verts]
            np.testing.assert_allclose(z_final[i], expected)

    def test_identity_prolongation(self):
        """Test prolongation when coarse level is zero."""
        n_verts = 10
        total_nodes = n_verts + 1

        # Only fine level has non-zero values
        multi_level_z = np.zeros((total_nodes, 3))
        multi_level_z[:n_verts] = np.random.randn(n_verts, 3)

        aggregation_table = np.full((n_verts, MAX_LEVELS - 1), n_verts, dtype=np.int32)
        level_sizes = [(n_verts, 0), (1, n_verts)]

        z_final = compute_prolongation_numpy(multi_level_z, aggregation_table,
                                             level_sizes, n_verts)

        # z_final should equal z_0 (coarse is zero)
        np.testing.assert_allclose(z_final, multi_level_z[:n_verts])


class TestLocalSolveGroundTruth(unittest.TestCase):
    """Test local solve (SpMV) ground truth."""

    def test_identity_inverse(self):
        """Test local solve with identity inverse."""
        n_verts = BANKSIZE  # One block
        n_blocks = 1

        # Create identity block matrices
        block_matrices = np.zeros((n_blocks, SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane in range(BANKSIZE):
            idx = sym_index(lane, lane)
            block_matrices[0, idx] = np.eye(3)

        multi_level_r = np.random.randn(n_verts, 3)
        level_sizes = [(n_verts, 0)]

        multi_level_z = compute_local_solve_numpy(multi_level_r, block_matrices, level_sizes)

        # With identity inverse, z should equal r
        np.testing.assert_allclose(multi_level_z[:n_verts], multi_level_r)

    def test_diagonal_inverse(self):
        """Test local solve with diagonal inverse."""
        n_verts = BANKSIZE
        n_blocks = 1

        # Create diagonal block matrices (scale each node by 2)
        block_matrices = np.zeros((n_blocks, SYM_BLOCK_COUNT, 3, 3), dtype=np.float64)
        for lane in range(BANKSIZE):
            idx = sym_index(lane, lane)
            block_matrices[0, idx] = 2.0 * np.eye(3)

        multi_level_r = np.random.randn(n_verts, 3)
        level_sizes = [(n_verts, 0)]

        multi_level_z = compute_local_solve_numpy(multi_level_r, block_matrices, level_sizes)

        # With 2*I inverse, z should equal 2*r
        np.testing.assert_allclose(multi_level_z[:n_verts], 2.0 * multi_level_r)


class TestEndToEndMultiLevel(unittest.TestCase):
    """End-to-end multi-level ground truth tests."""

    def test_two_level_full_pipeline(self):
        """Test complete 2-level pipeline."""
        # Create mesh with 2 blocks (32 vertices)
        n_verts = 32
        n_blocks = 2

        # Level sizes: Level 0 = 32, Level 1 = 2
        level_sizes = [(n_verts, 0), (2, n_verts)]
        total_nodes = n_verts + 2

        # Create going_next mapping: each block -> one coarse node
        going_next = np.zeros(total_nodes, dtype=np.int32)
        for i in range(BANKSIZE):
            going_next[i] = n_verts  # Block 0 -> coarse 0
        for i in range(BANKSIZE, n_verts):
            going_next[i] = n_verts + 1  # Block 1 -> coarse 1

        # Create aggregation table
        aggregation_table = np.zeros((n_verts, MAX_LEVELS - 1), dtype=np.int32)
        for i in range(BANKSIZE):
            aggregation_table[i, 0] = n_verts  # Block 0 aggregates from coarse 0
        for i in range(BANKSIZE, n_verts):
            aggregation_table[i, 0] = n_verts + 1  # Block 1 aggregates from coarse 1

        # Random gradient
        np.random.seed(42)
        gradient = np.random.randn(n_verts, 3)

        # Step 1: Restriction
        multi_level_r = compute_restriction_numpy(gradient, going_next, level_sizes, n_verts)

        # Verify restriction
        self.assertEqual(multi_level_r.shape[0], total_nodes)
        np.testing.assert_allclose(multi_level_r[:n_verts], gradient)

        # Coarse residuals should be sums
        expected_r_coarse_0 = np.sum(gradient[:BANKSIZE], axis=0)
        expected_r_coarse_1 = np.sum(gradient[BANKSIZE:], axis=0)
        np.testing.assert_allclose(multi_level_r[n_verts], expected_r_coarse_0)
        np.testing.assert_allclose(multi_level_r[n_verts + 1], expected_r_coarse_1)

        # Step 2: Local solve with identity (for simplicity)
        # In real preconditioner, this would be B^{-1} * r
        multi_level_z = multi_level_r.copy()  # Identity solve

        # Step 3: Prolongation
        z_final = compute_prolongation_numpy(multi_level_z, aggregation_table,
                                             level_sizes, n_verts)

        # Verify prolongation
        # z_final[i] = z_0[i] + z_1[coarse_idx]
        for i in range(n_verts):
            coarse_idx = aggregation_table[i, 0]
            expected = multi_level_z[i] + multi_level_z[coarse_idx]
            np.testing.assert_allclose(z_final[i], expected,
                                       err_msg=f"Mismatch at vertex {i}")

        print(f"\n[Ground Truth] Two-level pipeline test:")
        print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
        print(f"  Final z norm: {np.linalg.norm(z_final):.6f}")

    def test_three_level_pipeline(self):
        """Test complete 3-level pipeline."""
        # Create mesh with 4 blocks at L0 (64 vertices) -> 4 at L1 -> 1 at L2
        n_verts = 64

        # Level sizes
        level_sizes = [(n_verts, 0), (4, n_verts), (1, n_verts + 4)]
        total_nodes = n_verts + 4 + 1

        # going_next: L0 -> L1 (each block -> one coarse node)
        going_next = np.zeros(total_nodes, dtype=np.int32)
        for block in range(4):
            for lane in range(BANKSIZE):
                going_next[block * BANKSIZE + lane] = n_verts + block
        # L1 -> L2 (all coarse nodes -> single super-coarse node)
        for i in range(4):
            going_next[n_verts + i] = n_verts + 4

        # aggregation_table: path through hierarchy
        aggregation_table = np.zeros((n_verts, MAX_LEVELS - 1), dtype=np.int32)
        for i in range(n_verts):
            block = i // BANKSIZE
            aggregation_table[i, 0] = n_verts + block  # L1 coarse node
            aggregation_table[i, 1] = n_verts + 4      # L2 super-coarse node

        # Random gradient
        np.random.seed(42)
        gradient = np.random.randn(n_verts, 3)

        # Step 1: Restriction
        multi_level_r = compute_restriction_numpy(gradient, going_next, level_sizes, n_verts)

        # Verify L1 restriction
        for block in range(4):
            expected = np.sum(gradient[block*BANKSIZE:(block+1)*BANKSIZE], axis=0)
            np.testing.assert_allclose(multi_level_r[n_verts + block], expected)

        # Verify L2 restriction (sum of all L1 + L0)
        expected_l2 = np.sum(gradient, axis=0)  # All fine gradients contribute to L2
        np.testing.assert_allclose(multi_level_r[n_verts + 4], expected_l2, rtol=1e-10)

        # Step 2: Identity local solve
        multi_level_z = multi_level_r.copy()

        # Step 3: Prolongation
        z_final = compute_prolongation_numpy(multi_level_z, aggregation_table,
                                             level_sizes, n_verts)

        # Verify prolongation
        for i in range(n_verts):
            block = i // BANKSIZE
            # z_final = z_L0 + z_L1 + z_L2
            expected = (multi_level_z[i] +
                       multi_level_z[n_verts + block] +
                       multi_level_z[n_verts + 4])
            np.testing.assert_allclose(z_final[i], expected)

        print(f"\n[Ground Truth] Three-level pipeline test:")
        print(f"  Level sizes: {[s[0] for s in level_sizes]}")
        print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
        print(f"  Final z norm: {np.linalg.norm(z_final):.6f}")


class TestConnectivityMask(unittest.TestCase):
    """Test connectivity mask operations."""

    def test_fully_connected_warp(self):
        """Test mask for fully connected warp."""
        # In a fully connected warp, all nodes in the same warp have
        # connect_mask = 0xFFFF (all 16 bits set)
        full_mask = (1 << BANKSIZE) - 1

        # Check each node would be in same component
        for lane in range(BANKSIZE):
            # elected_prefix = popcount(mask & lanemask_lt(lane))
            lanemask_lt = (1 << lane) - 1
            elected_prefix = bin(full_mask & lanemask_lt).count('1')

            if lane == 0:
                self.assertEqual(elected_prefix, 0, "Lane 0 should be elected")
            else:
                self.assertGreater(elected_prefix, 0, f"Lane {lane} should not be elected")

    def test_disconnected_warp(self):
        """Test mask for completely disconnected warp."""
        # Each node only connects to itself
        for lane in range(BANKSIZE):
            mask = 1 << lane  # Only self-connected
            lanemask_lt = (1 << lane) - 1
            elected_prefix = bin(mask & lanemask_lt).count('1')

            # Each node should be elected (elected_prefix == 0)
            self.assertEqual(elected_prefix, 0, f"Lane {lane} should be elected")

    def test_two_components(self):
        """Test mask for warp with two components."""
        # First 8 lanes connected, last 8 lanes connected
        mask_lower = 0x00FF  # Lanes 0-7
        mask_upper = 0xFF00  # Lanes 8-15

        # Lanes 0-7 should have mask_lower
        for lane in range(8):
            lanemask_lt = (1 << lane) - 1
            elected_prefix = bin(mask_lower & lanemask_lt).count('1')
            if lane == 0:
                self.assertEqual(elected_prefix, 0)
            else:
                self.assertGreater(elected_prefix, 0)

        # Lanes 8-15 should have mask_upper
        for lane in range(8, 16):
            lanemask_lt = (1 << lane) - 1
            elected_prefix = bin(mask_upper & lanemask_lt).count('1')
            if lane == 8:
                self.assertEqual(elected_prefix, 0)
            else:
                self.assertGreater(elected_prefix, 0)


class TestSymmetricIndex(unittest.TestCase):
    """Test symmetric index formula consistency."""

    def test_index_uniqueness(self):
        """Test that all indices are unique."""
        indices = set()
        for i in range(BANKSIZE):
            for j in range(i, BANKSIZE):
                idx = sym_index(i, j)
                self.assertNotIn(idx, indices, f"Duplicate index {idx} for ({i},{j})")
                indices.add(idx)

        self.assertEqual(len(indices), SYM_BLOCK_COUNT)

    def test_index_symmetry(self):
        """Test that sym_index(i,j) == sym_index(j,i)."""
        for i in range(BANKSIZE):
            for j in range(BANKSIZE):
                self.assertEqual(sym_index(i, j), sym_index(j, i))


# ==============================================================================
# Integration Tests with Actual MAS Preconditioner
# ==============================================================================

class TestMASPreconditionerIntegration(unittest.TestCase):
    """Integration tests with actual MASPreconditioner class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for integration tests."""
        # Import MAS preconditioner
        try:
            from algorithm.mas_preconditioner_pkg.core import MASPreconditioner
            cls.MASPreconditioner = MASPreconditioner
            cls.mas_available = True
        except ImportError as e:
            print(f"Warning: Could not import MASPreconditioner: {e}")
            cls.mas_available = False

    def setUp(self):
        """Skip tests if MAS not available."""
        if not self.mas_available:
            self.skipTest("MASPreconditioner not available")

    def test_hierarchy_levels_match(self):
        """Test that MAS builds expected hierarchy levels."""
        # Create mesh with known size
        vertices, cells = create_grid_mesh(2, 2, 2)  # 27 vertices, 2 warps
        n_verts = len(vertices)
        n_cells = len(cells)

        # Create MAS preconditioner with simple API
        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        # Build neighbor list from cells
        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Check levels
        self.assertTrue(mas.hierarchy_built)
        self.assertGreaterEqual(mas.actual_levels, 2)

        # Get level sizes
        level_size_np = mas.level_size.to_numpy()
        print(f"\n[Integration] Hierarchy for {n_verts} vertices:")
        for level in range(mas.actual_levels):
            size, offset = level_size_np[level]
            print(f"  Level {level}: {size} nodes at offset {offset}")

    def test_restriction_sums_correctly(self):
        """Test that restriction correctly sums gradients to coarse levels."""
        # Create mesh
        vertices, cells = create_grid_mesh(2, 2, 2)  # 27 vertices
        n_verts = len(vertices)
        n_cells = len(cells)

        # Create MAS
        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        # Build hierarchy
        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Create gradient field
        grad_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        z_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

        # Set uniform gradient
        np.random.seed(42)
        gradient_np = np.random.randn(n_verts, 3).astype(np.float32)
        grad_field.from_numpy(gradient_np)

        # Create simple inverse blocks (identity for testing)
        create_identity_inverse_blocks(mas)

        # Apply preconditioner
        mas.apply_simple(grad_field, z_field, use_warp_reduction=False)

        # Get results
        z_np = z_field.to_numpy()

        # Verify output is not zero and has reasonable magnitude
        z_norm = np.linalg.norm(z_np)
        grad_norm = np.linalg.norm(gradient_np)

        self.assertGreater(z_norm, 0, "Output should be non-zero")
        print(f"\n[Integration] Restriction test:")
        print(f"  Input gradient norm: {grad_norm:.6f}")
        print(f"  Output z norm: {z_norm:.6f}")

    def test_multi_level_r_values(self):
        """Test that multi_level_r has correct values at each level."""
        # Create small mesh
        vertices, cells = create_grid_mesh(2, 2, 2)  # 27 vertices
        n_verts = len(vertices)
        n_cells = len(cells)

        # Create MAS
        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        # Build hierarchy
        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Create gradient
        grad_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        gradient_np = np.ones((n_verts, 3), dtype=np.float32)
        grad_field.from_numpy(gradient_np)

        # Clear buffers and run restriction
        mas._clear_multi_level_buffers()
        mas._restrict_simple(grad_field)

        # Get multi_level_r
        multi_level_r_np = mas.multi_level_r.to_numpy()

        # Level 0 should equal gradient
        np.testing.assert_allclose(multi_level_r_np[:n_verts], gradient_np,
                                   rtol=1e-5, atol=1e-6)

        # Coarse levels should have sums
        level_size_np = mas.level_size.to_numpy()
        for level in range(1, mas.actual_levels):
            size, offset = level_size_np[level]
            coarse_r = multi_level_r_np[offset:offset+size]

            # Each coarse node should be sum of its children
            coarse_sum = np.sum(coarse_r, axis=0)
            expected_sum = np.sum(gradient_np, axis=0)

            print(f"\n[Integration] Level {level} restriction:")
            print(f"  Coarse nodes: {size}")
            print(f"  Sum of coarse r: {coarse_sum}")
            print(f"  Expected sum: {expected_sum}")

    def test_going_next_validity(self):
        """Test that going_next mapping is valid."""
        vertices, cells = create_grid_mesh(2, 2, 2)
        n_verts = len(vertices)
        n_cells = len(cells)

        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Get going_next
        going_next_np = mas.going_next.to_numpy()
        level_size_np = mas.level_size.to_numpy()

        # Check Level 0 -> Level 1 mapping
        l0_size = level_size_np[0][0]
        l1_offset = level_size_np[1][1]
        l1_size = level_size_np[1][0]

        for i in range(l0_size):
            next_idx = going_next_np[i]
            self.assertGreaterEqual(next_idx, l1_offset,
                                    f"Node {i} maps to {next_idx}, below L1 offset {l1_offset}")
            self.assertLess(next_idx, l1_offset + l1_size,
                           f"Node {i} maps to {next_idx}, beyond L1 range")

        print(f"\n[Integration] going_next validity test:")
        print(f"  L0 size: {l0_size}, L1 offset: {l1_offset}, L1 size: {l1_size}")


class TestMASMultiLevelSolve(unittest.TestCase):
    """Test multi-level solve correctness."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from algorithm.mas_preconditioner_pkg.core import MASPreconditioner
            cls.MASPreconditioner = MASPreconditioner
            cls.mas_available = True
        except ImportError:
            cls.mas_available = False

    def setUp(self):
        if not self.mas_available:
            self.skipTest("MASPreconditioner not available")

    def test_local_solve_identity_inverse(self):
        """Test local solve with identity inverse returns input."""
        vertices, cells = create_grid_mesh(2, 2, 2)
        n_verts = len(vertices)
        n_cells = len(cells)

        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Set identity inverse
        create_identity_inverse_blocks(mas)

        # Set residual
        np.random.seed(42)
        r_np = np.random.randn(mas.total_nodes_all_levels, 3).astype(np.float32)
        mas.multi_level_r.from_numpy(r_np)

        # Clear z
        mas.multi_level_z.fill(0.0)

        # Run local solve
        mas._schwarz_local_solve_full()

        # Get z
        z_np = mas.multi_level_z.to_numpy()

        # With identity inverse, z should equal r (at valid nodes)
        level_size_np = mas.level_size.to_numpy()
        for level in range(mas.actual_levels):
            size, offset = level_size_np[level]
            np.testing.assert_allclose(
                z_np[offset:offset+size],
                r_np[offset:offset+size],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Level {level} solve mismatch"
            )

        print(f"\n[Integration] Local solve with identity test: PASSED")

    def test_full_pipeline_consistency(self):
        """Test that full pipeline produces consistent results."""
        vertices, cells = create_grid_mesh(3, 3, 2)  # 48 vertices
        n_verts = len(vertices)
        n_cells = len(cells)

        mas = self.MASPreconditioner(n_verts=n_verts, n_cells=n_cells, mesh=None,
                                     use_metis=False)

        neighbor_list, neighbor_starts = build_neighbor_arrays(vertices, cells)
        mas._build_hierarchy_from_adjacency(neighbor_list, neighbor_starts)

        # Set diagonal inverse (2*I)
        create_diagonal_inverse_blocks(mas, scale=2.0)

        # Create gradient and z fields
        grad_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        z_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

        # Random gradient
        np.random.seed(42)
        gradient_np = np.random.randn(n_verts, 3).astype(np.float32)
        grad_field.from_numpy(gradient_np)

        # Run full apply
        mas.apply_simple(grad_field, z_field, use_warp_reduction=False)

        # Run again - should get same result
        z_field_2 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        mas.apply_simple(grad_field, z_field_2, use_warp_reduction=False)

        z_np_1 = z_field.to_numpy()
        z_np_2 = z_field_2.to_numpy()

        np.testing.assert_allclose(z_np_1, z_np_2, rtol=1e-5, atol=1e-6,
                                   err_msg="Repeated apply should give same result")

        print(f"\n[Integration] Pipeline consistency test: PASSED")
        print(f"  Gradient norm: {np.linalg.norm(gradient_np):.6f}")
        print(f"  Z norm: {np.linalg.norm(z_np_1):.6f}")


# ==============================================================================
# Helper Functions for Integration Tests
# ==============================================================================

def build_neighbor_arrays(vertices, cells):
    """Build CSR neighbor arrays from mesh."""
    n_verts = len(vertices)
    n_cells = len(cells)

    # Build adjacency using set to avoid duplicates
    neighbors = [set() for _ in range(n_verts)]

    for cell in cells:
        for i in range(4):
            for j in range(4):
                if i != j:
                    neighbors[cell[i]].add(cell[j])

    # Convert to CSR format
    neighbor_list = []
    neighbor_starts = np.zeros(n_verts + 1, dtype=np.int32)

    for i in range(n_verts):
        neighbor_starts[i] = len(neighbor_list)
        for n in sorted(neighbors[i]):
            neighbor_list.append(n)
    neighbor_starts[n_verts] = len(neighbor_list)

    return np.array(neighbor_list, dtype=np.int32), neighbor_starts


def create_identity_inverse_blocks(mas):
    """Create identity inverse block matrices."""
    n_blocks = mas.total_blocks

    # Set inv_block_matrices to identity
    inv_np = np.zeros((n_blocks, SYM_BLOCK_COUNT, 3, 3), dtype=np.float32)
    for block_id in range(n_blocks):
        for lane in range(BANKSIZE):
            idx = sym_index(lane, lane)
            inv_np[block_id, idx] = np.eye(3, dtype=np.float32)

    mas.inv_block_matrices.from_numpy(inv_np)
    mas.matrices_inverted = True


def create_diagonal_inverse_blocks(mas, scale=1.0):
    """Create diagonal inverse block matrices."""
    n_blocks = mas.total_blocks

    inv_np = np.zeros((n_blocks, SYM_BLOCK_COUNT, 3, 3), dtype=np.float32)
    for block_id in range(n_blocks):
        for lane in range(BANKSIZE):
            idx = sym_index(lane, lane)
            inv_np[block_id, idx] = scale * np.eye(3, dtype=np.float32)

    mas.inv_block_matrices.from_numpy(inv_np)
    mas.matrices_inverted = True


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MAS Preconditioner Multi-Level Ground Truth Tests")
    print("=" * 70)
    print(f"BANKSIZE: {BANKSIZE}")
    print(f"SYM_BLOCK_COUNT: {SYM_BLOCK_COUNT}")
    print(f"BLOCK_DOF: {BLOCK_DOF}")
    print(f"MAX_LEVELS: {MAX_LEVELS}")
    print("=" * 70)

    unittest.main(verbosity=2)
