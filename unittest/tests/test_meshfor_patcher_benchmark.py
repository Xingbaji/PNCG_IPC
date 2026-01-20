#!/usr/bin/env python3
"""
Benchmark comparing mesh-for vs range loop using meshtaichi_patcher.

Uses standard meshtaichi_patcher to load mesh, ensuring correct metadata format.
"""

import sys
import os
sys.path.insert(0, '/root/PNCG_IPC')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import time
import numpy as np
import meshtaichi_patcher as Patcher


def run_benchmark():
    """Run the benchmark comparing mesh-for vs range loop."""
    print("\n" + "="*70)
    print("MeshFor vs Range Loop Benchmark (using meshtaichi_patcher)")
    print("="*70)

    # Load mesh with edges relation
    model_path = "/root/PNCG_IPC/model/cube.node"
    print(f"\nLoading mesh from: {model_path}")

    # Load with EV relation for edge iteration
    mesh = Patcher.load_mesh(model_path, relations=["EV", "CV"])

    # Place vertex positions
    mesh.verts.place({'x': ti.math.vec3})
    mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

    n_verts = len(mesh.verts)
    n_edges = len(mesh.edges)
    n_cells = len(mesh.cells)

    print(f"  Vertices: {n_verts}")
    print(f"  Edges: {n_edges}")
    print(f"  Cells: {n_cells}")

    # Create edge data for range loop (extract from mesh)
    # We need to extract edge connectivity
    edges_np = np.zeros((n_edges, 2), dtype=np.int32)

    @ti.kernel
    def extract_edges():
        for e in mesh.edges:
            edges_np_field[e.id, 0] = e.verts[0].id
            edges_np_field[e.id, 1] = e.verts[1].id

    edges_np_field = ti.field(ti.i32, shape=(n_edges, 2))
    extract_edges()
    ti.sync()
    edges_np = edges_np_field.to_numpy()

    # Create position field for range loop
    positions_np = mesh.get_position_as_numpy()
    verts_x = ti.Vector.field(3, ti.f32, shape=n_verts)
    verts_x.from_numpy(positions_np)

    edges_ti = ti.field(ti.i32, shape=(n_edges, 2))
    edges_ti.from_numpy(edges_np)

    # Output fields
    total_length_meshfor = ti.field(ti.f32, shape=())
    total_length_range = ti.field(ti.f32, shape=())

    # Mesh-for kernel
    @ti.kernel
    def compute_total_meshfor():
        total = 0.0
        ti.mesh_local(mesh.verts.x)
        for e in mesh.edges:
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
            v0 = edges_ti[ei, 0]
            v1 = edges_ti[ei, 1]
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

    if abs(len_meshfor - len_range) > 0.01:
        print("WARNING: Results don't match!")
    else:
        print("Results match: OK")

    # Benchmark parameters
    n_iterations = 100

    # Benchmark range loop
    print(f"\nBenchmarking range loop ({n_iterations} iterations)...")
    ti.sync()
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        compute_total_range()
    ti.sync()
    t_range = (time.perf_counter() - t_start) / n_iterations * 1000

    # Benchmark mesh-for loop
    print(f"Benchmarking mesh-for loop ({n_iterations} iterations)...")
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
    print(f"Edges: {n_edges}")
    print(f"Range loop:    {t_range:.3f} ms")
    print(f"Mesh-for loop: {t_meshfor:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    print("="*70)

    if speedup > 1.0:
        print("\nMesh-for is FASTER than range loop!")
    elif speedup < 1.0:
        print("\nRange loop is faster (mesh-for overhead may dominate for small meshes)")
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
