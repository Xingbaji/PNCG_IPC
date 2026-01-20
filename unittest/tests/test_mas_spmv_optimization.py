"""
Test MAS Preconditioner SPMV Optimization.

Tests the sorted triplet SPMV optimization for cross-block contributions.
"""
import sys
sys.path.insert(0, '/root/PNCG_IPC')

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True)

import numpy as np
import meshtaichi_patcher as Patcher
from algorithm.mas_preconditioner_small.core import MASPreconditionerSmall


def create_test_mesh():
    """Create a test mesh with 9 vertices and 12 tetrahedra."""
    # 8-corner cube + center = 9 vertices
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [0.0, 1.0, 1.0],  # 6
        [1.0, 1.0, 1.0],  # 7
        [0.5, 0.5, 0.5],  # 8 (center)
    ], dtype=np.float32)

    # Tetrahedra from cube corners to center
    cells = np.array([
        [0, 1, 3, 8],
        [0, 3, 2, 8],
        [0, 4, 5, 8],
        [0, 4, 6, 8],
        [0, 5, 1, 8],
        [0, 2, 6, 8],
        [7, 3, 1, 8],
        [7, 3, 2, 8],
        [7, 5, 1, 8],
        [7, 6, 2, 8],
        [7, 4, 5, 8],
        [7, 4, 6, 8],
    ], dtype=np.int32)

    return verts, cells


def create_large_test_mesh():
    """Create a larger test mesh that spans multiple blocks (>16 vertices)."""
    # Create a 3x3x3 grid of vertices = 27 vertices
    # This ensures we have at least 2 blocks (BANKSIZE=16)
    n = 3
    verts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                verts.append([float(i), float(j), float(k)])
    verts = np.array(verts, dtype=np.float32)

    # Create tetrahedra for each cube in the grid
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                # 8 vertices of this cube
                def idx(di, dj, dk):
                    return (i + di) * n * n + (j + dj) * n + (k + dk)

                v0 = idx(0, 0, 0)
                v1 = idx(1, 0, 0)
                v2 = idx(0, 1, 0)
                v3 = idx(1, 1, 0)
                v4 = idx(0, 0, 1)
                v5 = idx(1, 0, 1)
                v6 = idx(0, 1, 1)
                v7 = idx(1, 1, 1)

                # 5-tet decomposition of cube
                cells.append([v0, v1, v3, v5])
                cells.append([v0, v3, v2, v6])
                cells.append([v0, v5, v4, v6])
                cells.append([v3, v5, v6, v7])
                cells.append([v0, v3, v5, v6])

    cells = np.array(cells, dtype=np.int32)
    return verts, cells


@ti.kernel
def init_mesh_kernel(mesh: ti.template(), v_arr: ti.types.ndarray()):
    """Initialize mesh vertex positions and cell data."""
    for v in mesh.verts:
        v.x = ti.Vector([v_arr[v.id, 0], v_arr[v.id, 1], v_arr[v.id, 2]])
        v.m = 1.0
    for c in mesh.cells:
        c.B = ti.Matrix.identity(ti.f32, 3)
        c.W = 1.0


class MockSolver:
    """Mock solver with required parameters."""
    dt = 0.01
    mu = 1e5
    la = 1e5


def test_sorted_unsorted_equivalence():
    """Test that sorted and unsorted SPMV produce equivalent results."""
    print("=" * 60)
    print("Test: Sorted vs Unsorted SPMV Equivalence")
    print("=" * 60)

    # Create larger mesh that spans multiple blocks
    verts, cells = create_large_test_mesh()
    mesh = Patcher.load_mesh([{0: verts, 3: cells}], relations=['CV'])
    mesh.verts.place({'x': ti.math.vec3, 'm': ti.f32, 'z': ti.math.vec3, 'grad': ti.math.vec3})
    mesh.cells.place({'W': ti.f32, 'B': ti.math.mat3})

    init_mesh_kernel(mesh, verts)
    n_verts = len(mesh.verts)
    n_cells = len(mesh.cells)
    print(f"Mesh: {n_verts} vertices, {n_cells} cells (should have cross-block entries)")

    # Create preconditioner
    precond = MASPreconditionerSmall(mesh)
    precond.build_hierarchy()
    precond.assemble_block_matrices(MockSolver())
    precond.invert_block_matrices()

    triplet_count = int(precond.cross_block_count[None])
    print(f"Cross-block triplets: {triplet_count}")

    # Create test vectors
    v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    result_unsorted = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    result_sorted = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    np.random.seed(42)
    v_np = np.random.randn(n_verts, 3).astype(np.float32)
    v.from_numpy(v_np)

    # Test unsorted version
    result_unsorted.fill(0)
    precond.hessian_matvec_exact(v, result_unsorted, use_sorted=False)
    result_unsorted_np = result_unsorted.to_numpy()
    unsorted_norm = np.linalg.norm(result_unsorted_np)
    print(f"Unsorted result norm: {unsorted_norm:.2e}")

    if triplet_count > 0:
        # Sort triplets
        print("Sorting triplets...")
        precond.sort_cross_block_triplets()
        assert precond.triplets_sorted, "Triplets should be marked as sorted"

        # Test sorted version
        result_sorted.fill(0)
        precond.hessian_matvec_exact(v, result_sorted, use_sorted=True)
        result_sorted_np = result_sorted.to_numpy()
        sorted_norm = np.linalg.norm(result_sorted_np)
        print(f"Sorted result norm: {sorted_norm:.2e}")

        # Compare
        diff = np.abs(result_unsorted_np - result_sorted_np).max()
        rel_diff = diff / max(unsorted_norm, 1e-10)
        print(f"Max absolute diff: {diff:.2e}")
        print(f"Relative diff: {rel_diff:.2e}")

        assert diff < 1e-4, f"Results differ too much: {diff}"
        print("PASSED: Sorted and unsorted SPMV produce equivalent results!")
    else:
        print("SKIPPED: No cross-block triplets (mesh fits in single block)")

    print()


def test_sorted_spmv_methods():
    """Test that sorted SPMV methods exist."""
    print("=" * 60)
    print("Test: Sorted SPMV Methods")
    print("=" * 60)

    # Just verify the methods exist
    assert hasattr(MASPreconditionerSmall, 'sort_cross_block_triplets'), \
        "sort_cross_block_triplets should exist"
    assert hasattr(MASPreconditionerSmall, '_cross_block_spmv_sorted_row'), \
        "_cross_block_spmv_sorted_row should exist"
    print("Sorted SPMV methods exist")
    print("PASSED!")
    print()


def test_preconditioner_initialization():
    """Test that preconditioner initializes with new fields."""
    print("=" * 60)
    print("Test: Preconditioner Initialization")
    print("=" * 60)

    # Create a minimal mesh
    verts, cells = create_test_mesh()
    mesh = Patcher.load_mesh([{0: verts, 3: cells}], relations=['CV'])
    mesh.verts.place({'x': ti.math.vec3, 'm': ti.f32, 'z': ti.math.vec3, 'grad': ti.math.vec3})
    mesh.cells.place({'W': ti.f32, 'B': ti.math.mat3})
    init_mesh_kernel(mesh, verts)

    precond = MASPreconditionerSmall(mesh)

    # Check new fields exist
    assert hasattr(precond, 'sorted_triplet_row'), "sorted_triplet_row should exist"
    assert hasattr(precond, 'sorted_triplet_col'), "sorted_triplet_col should exist"
    assert hasattr(precond, 'sorted_triplet_val'), "sorted_triplet_val should exist"
    assert hasattr(precond, 'row_segment_start'), "row_segment_start should exist"
    assert hasattr(precond, 'triplets_sorted'), "triplets_sorted should exist"
    assert not precond.triplets_sorted, "triplets_sorted should be False initially"
    print("New fields initialized correctly")

    # Test that assembly works
    precond.build_hierarchy()
    precond.assemble_block_matrices(MockSolver())
    assert precond.matrices_assembled, "Matrices should be assembled"
    assert not precond.triplets_sorted, "triplets_sorted should be False after assembly"
    print("Assembly works correctly")
    print("PASSED!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MAS Preconditioner SPMV Optimization Tests")
    print("=" * 60 + "\n")

    test_sorted_spmv_methods()
    test_preconditioner_initialization()
    test_sorted_unsorted_equivalence()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
