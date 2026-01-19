"""
Test demo for HUP-MAS Preconditioner (Hierarchical Unified Partition).

This script tests the integration of MeshTaichi-style Patch partitioning
with MAS preconditioner.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np
import time

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f64)


def test_hierarchical_partition():
    """Test the hierarchical partition algorithm."""
    print("\n" + "="*60)
    print("Test 1: Hierarchical Partition Algorithm")
    print("="*60)

    from algorithm.hierarchical_partition import hierarchical_partition_mesh, PATCH_SIZE, BANKSIZE

    # Create a simple test mesh
    n_verts = 1000
    n_cells = 2000

    # Generate random cells (for testing)
    np.random.seed(42)
    cells = np.random.randint(0, n_verts, size=(n_cells, 4)).astype(np.int32)

    # Ensure cells have unique vertices
    for i in range(n_cells):
        while len(set(cells[i])) < 4:
            cells[i] = np.random.randint(0, n_verts, size=4).astype(np.int32)

    print(f"Test mesh: {n_verts} vertices, {n_cells} cells")

    # Test partition
    result = hierarchical_partition_mesh(n_verts, cells, patch_size=PATCH_SIZE)

    print(f"\nPartition Results:")
    print(f"  - Patches: {result['n_patches']}")
    print(f"  - Blocks: {result['n_blocks']}")
    print(f"  - Avg patch size: {result['stats']['avg_patch_size']:.1f}")
    print(f"  - Avg block size: {result['stats']['avg_block_size']:.1f}")
    print(f"  - Avg ribbon ratio: {result['stats']['avg_ribbon_ratio']:.2%}")

    # Verify sort_index
    sort_index = result['sort_index']
    assert len(sort_index) == n_verts
    assert len(set(sort_index)) == n_verts  # All unique

    print("\nTest 1 PASSED: Hierarchical partition works correctly")
    return True


def test_hup_mas_preconditioner():
    """Test the HUP-MAS preconditioner."""
    print("\n" + "="*60)
    print("Test 2: HUP-MAS Preconditioner")
    print("="*60)

    from algorithm.hup_mas_preconditioner import HUPMASPreconditioner

    # Load a small mesh for testing
    mesh_path = "../model/mesh/cube/cube.mesh"
    if not os.path.exists(mesh_path):
        mesh_path = "model/mesh/cube/cube.mesh"
    if not os.path.exists(mesh_path):
        print("Skipping test: cube mesh not found")
        return True

    # Build mesh using MeshTaichi
    mesh_builder = ti.TetMesh()
    mesh_builder.verts.place({'x': ti.math.vec3,
                              'v': ti.math.vec3,
                              'm': float,
                              'grad': ti.math.vec3,
                              'z': ti.math.vec3})
    mesh_builder.cells.place({'B': ti.math.mat3, 'W': float})

    mesh = mesh_builder.build(mesh_path)

    n_verts = len(mesh.verts)
    n_cells = len(mesh.cells)

    print(f"Loaded mesh: {n_verts} vertices, {n_cells} cells")

    # Extract cells
    cells_list = []
    for c_idx in range(n_cells):
        cell = mesh.cells[c_idx]
        try:
            v0 = cell.verts[0].id
            v1 = cell.verts[1].id
            v2 = cell.verts[2].id
            v3 = cell.verts[3].id
            cells_list.append([v0, v1, v2, v3])
        except:
            continue

    cells_np = np.array(cells_list, dtype=np.int32)
    print(f"Extracted {len(cells_np)} cells")

    # Initialize HUP-MAS preconditioner
    print("\nInitializing HUP-MAS preconditioner...")
    precond = HUPMASPreconditioner(n_verts, n_cells, mesh, cells_np)

    # Build hierarchy
    print("\nBuilding hierarchy...")
    precond.build_hierarchy()

    # Test assembly
    print("\nTesting matrix assembly...")

    # Create a mock solver with basic attributes
    class MockSolver:
        def __init__(self):
            self.dt = 0.01
            self.elastic_type = 0

    solver = MockSolver()
    precond.assemble_block_matrices(solver)

    # Test inversion
    print("\nTesting matrix inversion...")
    precond.invert_block_matrices()

    # Initialize vertex data
    mesh.verts.x.fill([0.0, 0.0, 0.0])
    mesh.verts.grad.fill([1.0, 0.5, 0.25])
    mesh.verts.z.fill([0.0, 0.0, 0.0])
    mesh.verts.m.fill(1.0)

    # Test apply
    print("\nTesting preconditioner apply...")
    precond.apply()

    # Check that z was modified
    z_np = mesh.verts.z.to_numpy()
    z_norm = np.linalg.norm(z_np)

    print(f"  z norm after apply: {z_norm:.6f}")
    assert z_norm > 0, "z should be non-zero after apply"

    # Get stats
    stats = precond.get_stats()
    print(f"\nPreconditioner stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    print("\nTest 2 PASSED: HUP-MAS preconditioner works correctly")
    return True


def test_prologue_epilogue():
    """Test the Prologue-Epilogue data transfer."""
    print("\n" + "="*60)
    print("Test 3: Prologue-Epilogue Data Transfer")
    print("="*60)

    from algorithm.hierarchical_partition import HierarchicalPartitionGPU

    n_verts = 500
    n_cells = 1000
    patch_size = 256  # Smaller for testing

    # Generate random cells
    np.random.seed(42)
    cells = np.random.randint(0, n_verts, size=(n_cells, 4)).astype(np.int32)

    for i in range(n_cells):
        while len(set(cells[i])) < 4:
            cells[i] = np.random.randint(0, n_verts, size=4).astype(np.int32)

    # Create HUP structure
    hup = HierarchicalPartitionGPU(n_verts, n_cells, patch_size)
    hup.build_hierarchical_partition(cells)

    # Create test data
    test_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
    test_grad = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
    test_z = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

    # Initialize with recognizable values
    x_np = np.random.randn(n_verts, 3)
    grad_np = np.random.randn(n_verts, 3)

    test_x.from_numpy(x_np)
    test_grad.from_numpy(grad_np)

    # Test prologue: original -> reordered
    @ti.kernel
    def prologue_test():
        for new_idx in range(n_verts):
            old_idx = hup.sort_index[new_idx]
            hup.reordered_x[new_idx] = test_x[old_idx]
            hup.reordered_grad[new_idx] = test_grad[old_idx]

    prologue_test()

    # Copy reordered_grad to reordered_z (simulating solve)
    @ti.kernel
    def copy_grad_to_z():
        for i in range(n_verts):
            hup.reordered_z[i] = hup.reordered_grad[i] * 2.0

    copy_grad_to_z()

    # Test epilogue: reordered -> original
    @ti.kernel
    def epilogue_test():
        for new_idx in range(n_verts):
            old_idx = hup.sort_index[new_idx]
            test_z[old_idx] = hup.reordered_z[new_idx]

    epilogue_test()

    # Verify: test_z[i] should equal test_grad[i] * 2.0
    z_np = test_z.to_numpy()
    expected = grad_np * 2.0

    error = np.max(np.abs(z_np - expected))
    print(f"Max error in Prologue-Epilogue: {error:.2e}")

    assert error < 1e-10, f"Prologue-Epilogue error too large: {error}"

    print("\nTest 3 PASSED: Prologue-Epilogue data transfer works correctly")
    return True


def test_performance():
    """Benchmark HUP vs standard partitioning."""
    print("\n" + "="*60)
    print("Test 4: Performance Benchmark")
    print("="*60)

    from algorithm.hierarchical_partition import HierarchicalPartitionGPU

    sizes = [1000, 5000, 10000]

    for n_verts in sizes:
        n_cells = n_verts * 2

        np.random.seed(42)
        cells = np.random.randint(0, n_verts, size=(n_cells, 4)).astype(np.int32)

        for i in range(n_cells):
            while len(set(cells[i])) < 4:
                cells[i] = np.random.randint(0, n_verts, size=4).astype(np.int32)

        print(f"\nMesh size: {n_verts} vertices, {n_cells} cells")

        # Time HUP partition
        t0 = time.time()
        hup = HierarchicalPartitionGPU(n_verts, n_cells)
        result = hup.build_hierarchical_partition(cells)
        t_partition = time.time() - t0

        print(f"  Partition time: {t_partition*1000:.1f} ms")
        print(f"  Patches: {result['n_patches']}")
        print(f"  Blocks: {result['n_blocks']}")

        # Time reordering operations
        test_data = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        test_data.from_numpy(np.random.randn(n_verts, 3))

        @ti.kernel
        def reorder_benchmark():
            for new_idx in range(n_verts):
                old_idx = hup.sort_index[new_idx]
                hup.reordered_x[new_idx] = test_data[old_idx]

        # Warmup
        reorder_benchmark()
        ti.sync()

        # Benchmark
        n_iter = 100
        t0 = time.time()
        for _ in range(n_iter):
            reorder_benchmark()
        ti.sync()
        t_reorder = (time.time() - t0) / n_iter

        print(f"  Reorder time (per call): {t_reorder*1000:.3f} ms")

    print("\nTest 4 PASSED: Performance benchmark complete")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("HUP-MAS Preconditioner Test Suite")
    print("="*60)

    # Run a single combined test to avoid Taichi re-initialization issues
    try:
        # Test 1: Basic partition
        print("\n" + "="*60)
        print("Test 1: Basic Partition")
        print("="*60)

        from algorithm.hierarchical_partition import HierarchicalPartitionGPU, BANKSIZE

        n_verts = 500
        n_cells = 1000
        patch_size = 256

        np.random.seed(42)
        cells = np.random.randint(0, n_verts, size=(n_cells, 4)).astype(np.int32)

        for i in range(n_cells):
            while len(set(cells[i])) < 4:
                cells[i] = np.random.randint(0, n_verts, size=4).astype(np.int32)

        print(f"Test mesh: {n_verts} vertices, {n_cells} cells")

        hup = HierarchicalPartitionGPU(n_verts, n_cells, patch_size)
        result = hup.build_hierarchical_partition(cells)

        print(f"Partition Results:")
        print(f"  - Patches: {result['n_patches']}")
        print(f"  - Blocks: {result['n_blocks']}")

        # Verify sort_index
        sort_index = result['sort_index']
        assert len(sort_index) == n_verts
        assert len(set(sort_index)) == n_verts

        print("Test 1 PASSED")

        # Test 2: Prologue-Epilogue
        print("\n" + "="*60)
        print("Test 2: Prologue-Epilogue")
        print("="*60)

        # Create test data
        test_data = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        test_z = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        x_np = np.random.randn(n_verts, 3)
        test_data.from_numpy(x_np)

        # Prologue
        @ti.kernel
        def prologue_test():
            for new_idx in range(n_verts):
                old_idx = hup.sort_index[new_idx]
                hup.reordered_x[new_idx] = test_data[old_idx]

        prologue_test()

        # Simulated transform
        @ti.kernel
        def transform():
            for i in range(n_verts):
                hup.reordered_z[i] = hup.reordered_x[i] * 2.0

        transform()

        # Epilogue
        @ti.kernel
        def epilogue_test():
            for new_idx in range(n_verts):
                old_idx = hup.sort_index[new_idx]
                test_z[old_idx] = hup.reordered_z[new_idx]

        epilogue_test()

        # Verify
        z_np = test_z.to_numpy()
        expected = x_np * 2.0
        error = np.max(np.abs(z_np - expected))
        print(f"Max error: {error:.2e}")

        assert error < 1e-10, f"Error too large: {error}"
        print("Test 2 PASSED")

        print("\n" + "="*60)
        print("All tests PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
