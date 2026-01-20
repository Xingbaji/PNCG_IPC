#!/usr/bin/env python3
"""
Simple benchmark comparing mesh-for vs range loop iteration.

This test focuses purely on iteration performance, independent of collision detection.
"""

import sys
import os
sys.path.insert(0, '/root')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import time
import numpy as np

# Import meshtaichi_custom
from meshtaichi_custom.metadata_builder import build_surface_mesh_metadata
from taichi.lang.mesh import MeshMetadata, MeshBuilder


def create_test_surface_mesh(n_grid=50):
    """
    Create a test surface mesh (grid of triangles).

    Args:
        n_grid: Grid size (creates n_grid x n_grid vertices)

    Returns:
        Tuple of (positions, edges, triangles, mesh_instance)
    """
    print(f"Creating {n_grid}x{n_grid} grid mesh...")

    # Create grid vertices
    n_verts = n_grid * n_grid
    positions = np.zeros((n_verts, 3), dtype=np.float32)

    for i in range(n_grid):
        for j in range(n_grid):
            idx = i * n_grid + j
            positions[idx] = [i / n_grid, j / n_grid, 0.0]

    # Create triangles (2 per grid cell)
    triangles = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            v00 = i * n_grid + j
            v01 = i * n_grid + (j + 1)
            v10 = (i + 1) * n_grid + j
            v11 = (i + 1) * n_grid + (j + 1)

            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])

    triangles = np.array(triangles, dtype=np.int32)
    n_faces = triangles.shape[0]

    # Extract edges from triangles
    edge_set = set()
    for tri in triangles:
        edge_set.add(tuple(sorted([tri[0], tri[1]])))
        edge_set.add(tuple(sorted([tri[1], tri[2]])))
        edge_set.add(tuple(sorted([tri[2], tri[0]])))

    edges = np.array(list(edge_set), dtype=np.int32)
    n_edges = edges.shape[0]

    print(f"  Vertices: {n_verts}")
    print(f"  Edges: {n_edges}")
    print(f"  Triangles: {n_faces}")

    # Build mesh metadata
    print("Building mesh metadata...")
    metadata = build_surface_mesh_metadata(
        positions, edges, triangles,
        patch_size=256,
        use_single_patch=False
    )

    # Create mesh instance
    mesh_metadata = MeshMetadata(metadata)
    builder = MeshBuilder()
    mesh = builder.build(mesh_metadata)

    # Place fields
    mesh.verts.place({'x': ti.math.vec3})
    mesh.verts.x.from_numpy(positions)

    return positions, edges, triangles, mesh


def run_benchmark():
    """Run the benchmark comparing mesh-for vs range loop."""
    print("\n" + "="*70)
    print("MeshFor vs Range Loop Benchmark")
    print("="*70)

    # Create test mesh
    positions, edges, triangles, mesh = create_test_surface_mesh(n_grid=100)
    n_edges = edges.shape[0]

    # Create edge data for range loop
    edges_ti = ti.field(ti.i32, shape=(n_edges, 2))
    edges_ti.from_numpy(edges)

    # Create vertex positions for range loop
    verts_x = ti.Vector.field(3, ti.f32, shape=positions.shape[0])
    verts_x.from_numpy(positions)

    # Output field
    edge_lengths_meshfor = ti.field(ti.f32, shape=n_edges)
    edge_lengths_range = ti.field(ti.f32, shape=n_edges)
    total_length = ti.field(ti.f32, shape=())

    # Mesh-for kernel
    @ti.kernel
    def compute_lengths_meshfor():
        ti.mesh_local(mesh.verts.x)
        for e in mesh.edges:
            v0 = e.verts[0]
            v1 = e.verts[1]
            length = (v0.x - v1.x).norm()
            edge_lengths_meshfor[e.id] = length

    # Range loop kernel
    @ti.kernel
    def compute_lengths_range():
        for ei in range(n_edges):
            v0 = edges_ti[ei, 0]
            v1 = edges_ti[ei, 1]
            length = (verts_x[v0] - verts_x[v1]).norm()
            edge_lengths_range[ei] = length

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        compute_lengths_meshfor()
        compute_lengths_range()
        ti.sync()

    # Benchmark parameters
    n_iterations = 50

    # Benchmark range loop
    print(f"\nBenchmarking range loop ({n_iterations} iterations)...")
    ti.sync()
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        compute_lengths_range()
    ti.sync()
    t_range = (time.perf_counter() - t_start) / n_iterations * 1000

    # Benchmark mesh-for loop
    print(f"Benchmarking mesh-for loop ({n_iterations} iterations)...")
    ti.sync()
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        compute_lengths_meshfor()
    ti.sync()
    t_meshfor = (time.perf_counter() - t_start) / n_iterations * 1000

    # Verify correctness
    lengths_meshfor = edge_lengths_meshfor.to_numpy()
    lengths_range = edge_lengths_range.to_numpy()

    # Check that all lengths are valid
    assert np.all(lengths_meshfor >= 0), "Invalid mesh-for lengths"
    assert np.all(lengths_range >= 0), "Invalid range lengths"

    # Check results are similar (may not be identical due to reordering)
    mean_diff = np.abs(np.mean(lengths_meshfor) - np.mean(lengths_range))
    assert mean_diff < 0.01, f"Mean length difference too large: {mean_diff}"

    # Calculate speedup
    speedup = t_range / t_meshfor if t_meshfor > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"Edges: {n_edges}")
    print(f"Range loop:    {t_range:.3f} ms")
    print(f"Mesh-for loop: {t_meshfor:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    print("="*70)

    if speedup > 1.0:
        print("\nMesh-for is FASTER than range loop!")
    elif speedup < 1.0:
        print("\nRange loop is faster")
    else:
        print("\nPerformance is similar")

    return {
        'n_edges': n_edges,
        'range_ms': t_range,
        'meshfor_ms': t_meshfor,
        'speedup': speedup
    }


if __name__ == '__main__':
    results = run_benchmark()
