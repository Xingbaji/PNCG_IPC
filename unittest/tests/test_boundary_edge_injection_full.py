#!/usr/bin/env python3
"""
Full test of boundary edge injection with mesh-for loop comparison.

This test verifies:
1. load_mesh_with_boundary_edges() correctly injects boundary edges
2. mesh.edges contains only boundary edges
3. mesh-for loops work correctly
4. Performance comparison between mesh-for and range loops
"""

import sys
import os
sys.path.insert(0, '/root/PNCG_IPC')
os.chdir('/root/PNCG_IPC/demo')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import time
import numpy as np


def test_load_mesh_with_boundary_edges():
    """Test the new load_mesh_with_boundary_edges function."""
    print("\n" + "="*70)
    print("Test: load_mesh_with_boundary_edges()")
    print("="*70)

    from util.model_loading import model_loading, _merge_and_reorder_models
    import meshtaichi_patcher as Patcher

    # Manually set up models like load_demo_n_object does
    demo = 'eight_E_drop_demo_contact'
    demo_dict = {
        'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
        'epsilon': 1e-4, 'iter_max': 50, 'height': 0.5,
        'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'NH', 'ground_barrier': 1,
        'barrier_type': 'log',
        'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
        'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
        'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
        'translations': [[1.5 * j, 1.5*i, 0.0] for i in range(4) for j in range(2)],
    }

    # Create loader instance (will use legacy config loading)
    # We just need to borrow its add_object and load_mesh_with_boundary_edges methods
    class MinimalLoader:
        def __init__(self):
            self.dict = demo_dict

        def add_object(self, model_path, translation=[0., 0., 0.], rotation=[0., 0., 0.], scale=[1, 1, 1]):
            model = Patcher.load_mesh_rawdata(model_path)
            if scale != [1., 1., 1.]:
                from scipy.spatial.transform import Rotation
                S = np.array([
                    [scale[0], 0, 0],
                    [0, scale[1], 0],
                    [0, 0, scale[2]]
                ])
                model[0] = np.dot(model[0], S)
            if rotation != [0., 0., 0.]:
                from scipy.spatial.transform import Rotation
                rotation = np.asarray(rotation)
                rotation = rotation * np.pi / 180.0
                rotation_matrix = Rotation.from_rotvec(rotation).as_matrix()
                model[0] = np.dot(model[0], rotation_matrix)
            if translation != [0., 0., 0.]:
                model[0][:, 0] = model[0][:, 0] + translation[0]
                model[0][:, 1] = model[0][:, 1] + translation[1]
                model[0][:, 2] = model[0][:, 2] + translation[2]
            return model

    # Use the actual model_loading class with full functionality
    print("\nLoading demo with standard method for comparison...")
    loader_standard = model_loading(demo)

    n_verts_standard = len(loader_standard.mesh.verts)
    n_cells_standard = len(loader_standard.mesh.cells)
    n_boundary_edges_standard = loader_standard.boundary_edges.shape[0]

    print(f"\nStandard loading results:")
    print(f"  Vertices: {n_verts_standard}")
    print(f"  Cells: {n_cells_standard}")
    print(f"  Boundary edges: {n_boundary_edges_standard}")
    print(f"  mesh.edges available: {hasattr(loader_standard.mesh, 'edges')}")

    # Check if standard mesh has edge element
    try:
        n_edges_in_mesh = len(loader_standard.mesh.edges)
        print(f"  mesh.edges count: {n_edges_in_mesh}")
    except:
        print("  mesh.edges not available (CV-only loading)")

    # Now test with load_mesh_with_boundary_edges
    print("\n" + "-"*50)
    print("Testing load_mesh_with_boundary_edges()...")
    print("-"*50)

    # Prepare models
    models = []
    for i in range(len(demo_dict['model_paths'])):
        minimal = MinimalLoader()
        model = minimal.add_object(
            model_path=demo_dict['model_paths'][i],
            scale=demo_dict['scales'][i],
            translation=demo_dict['translations'][i],
            rotation=demo_dict['rotations'][i]
        )
        models.append(model)

    # Create a new loader and call load_mesh_with_boundary_edges
    # We need to create a fresh instance that won't auto-load
    class TestLoader(model_loading):
        def __init__(self):
            # Skip parent __init__ which auto-loads
            pass

    loader_new = TestLoader()
    loader_new.dict = demo_dict

    # Copy add_object method
    loader_new.add_object = MinimalLoader().add_object

    # Call the new method
    print("\nCalling load_mesh_with_boundary_edges...")
    loader_new.load_mesh_with_boundary_edges(demo, models)

    n_verts_new = len(loader_new.mesh.verts)
    n_cells_new = len(loader_new.mesh.cells)
    n_edges_new = len(loader_new.mesh.edges)
    n_boundary_edges_new = loader_new.boundary_edges.shape[0]

    print(f"\nNew loading results:")
    print(f"  Vertices: {n_verts_new}")
    print(f"  Cells: {n_cells_new}")
    print(f"  mesh.edges count: {n_edges_new}")
    print(f"  Boundary edges (field): {n_boundary_edges_new}")

    # Verify
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    if n_edges_new == n_boundary_edges_new:
        print(f"✓ mesh.edges count ({n_edges_new}) matches boundary edges ({n_boundary_edges_new})")
    else:
        print(f"✗ mesh.edges count ({n_edges_new}) != boundary edges ({n_boundary_edges_new})")

    if n_verts_new == n_verts_standard:
        print(f"✓ Vertex count matches ({n_verts_new})")
    else:
        print(f"✗ Vertex count mismatch: {n_verts_new} vs {n_verts_standard}")

    if n_cells_new == n_cells_standard:
        print(f"✓ Cell count matches ({n_cells_new})")
    else:
        print(f"✗ Cell count mismatch: {n_cells_new} vs {n_cells_standard}")

    return loader_standard, loader_new


def test_meshfor_performance(loader_standard, loader_new):
    """Compare mesh-for vs range loop performance."""
    print("\n" + "="*70)
    print("Performance Comparison: mesh-for vs range loop")
    print("="*70)

    mesh_new = loader_new.mesh
    n_verts = len(mesh_new.verts)
    n_edges = len(mesh_new.edges)

    # Place vertex positions for mesh-for
    mesh_new.verts.place({'x': ti.math.vec3})
    positions = mesh_new.get_position_as_numpy()
    mesh_new.verts.x.from_numpy(positions)

    # Setup for range loop
    verts_x = ti.Vector.field(3, ti.f32, shape=n_verts)
    verts_x.from_numpy(positions)

    boundary_edges_np = loader_new.boundary_edges.to_numpy()
    boundary_edges = ti.field(ti.i32, shape=(n_edges, 2))
    boundary_edges.from_numpy(boundary_edges_np)

    # Output fields
    total_length_meshfor = ti.field(ti.f32, shape=())
    total_length_range = ti.field(ti.f32, shape=())

    # Mesh-for kernel (without ti.mesh_local to avoid scalarize error)
    @ti.kernel
    def compute_total_meshfor():
        total = 0.0
        # Note: ti.mesh_local disabled due to Taichi scalarize issue
        # ti.mesh_local(mesh_new.verts.x)
        for e in mesh_new.edges:
            v0 = e.verts[0]
            v1 = e.verts[1]
            length = (v0.x - v1.x).norm()
            total += length
        total_length_meshfor[None] = total

    # Range loop kernel
    @ti.kernel
    def compute_total_range():
        total = 0.0
        for ei in range(n_edges):
            v0 = boundary_edges[ei, 0]
            v1 = boundary_edges[ei, 1]
            length = (verts_x[v0] - verts_x[v1]).norm()
            total += length
        total_length_range[None] = total

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        compute_total_meshfor()
        compute_total_range()
    ti.sync()

    # Verify results match
    len_meshfor = total_length_meshfor[None]
    len_range = total_length_range[None]
    print(f"\nTotal length (mesh-for): {len_meshfor:.4f}")
    print(f"Total length (range):    {len_range:.4f}")

    if abs(len_meshfor - len_range) < 0.01:
        print("✓ Results match!")
    else:
        print(f"✗ Results differ by {abs(len_meshfor - len_range):.4f}")

    # Benchmark
    n_iterations = 100

    print(f"\nBenchmarking ({n_iterations} iterations)...")

    # Range loop
    ti.sync()
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        compute_total_range()
    ti.sync()
    t_range = (time.perf_counter() - t_start) / n_iterations * 1000

    # Mesh-for loop
    ti.sync()
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        compute_total_meshfor()
    ti.sync()
    t_meshfor = (time.perf_counter() - t_start) / n_iterations * 1000

    # Calculate speedup
    speedup = t_range / t_meshfor if t_meshfor > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"Boundary edges: {n_edges}")
    print(f"Range loop:    {t_range:.3f} ms")
    print(f"Mesh-for loop: {t_meshfor:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    print("="*70)

    if speedup > 1.0:
        print("\n✓ Mesh-for is FASTER than range loop!")
    elif speedup < 1.0:
        print("\n○ Range loop is faster (mesh-for overhead may dominate)")
    else:
        print("\n○ Performance is similar")

    return {
        'n_edges': n_edges,
        'range_ms': t_range,
        'meshfor_ms': t_meshfor,
        'speedup': speedup
    }


if __name__ == '__main__':
    print("="*70)
    print("Full Boundary Edge Injection Test")
    print("="*70)

    # Test 1: Load mesh with boundary edges
    loader_standard, loader_new = test_load_mesh_with_boundary_edges()

    # Test 2: Performance comparison
    if loader_new:
        results = test_meshfor_performance(loader_standard, loader_new)

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
