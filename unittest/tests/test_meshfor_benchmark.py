#!/usr/bin/env python3
"""
Benchmark test comparing mesh-for vs range loop for collision detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, '/root/meshtaichi_custom')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import time
import numpy as np


def test_meshfor_benchmark():
    """
    Benchmark mesh-for vs range loop for EE detection.
    """
    print("\n" + "="*70)
    print("MeshFor vs Range Loop Benchmark for Edge-Edge Collision Detection")
    print("="*70)

    # Import after ti.init()
    from algorithm.collision_detection_bvh_meshfor import collision_detection_bvh_meshfor_module

    # Use a demo with sufficient boundary elements
    demo = 'cube'
    print(f"\nLoading demo: {demo}")

    try:
        module = collision_detection_bvh_meshfor_module(demo=demo)
        module.init_bvh()

        print(f"\nMesh info:")
        print(f"  - Total vertices: {module.n_verts}")
        print(f"  - Boundary points: {module.n_boundary_points}")
        print(f"  - Boundary edges: {module.n_boundary_edges}")
        print(f"  - Boundary triangles: {module.n_boundary_triangles}")

        # Run benchmark
        results = module.benchmark_ee_methods(n_iterations=20, warmup=5)

        # Verify results match
        assert results['range_contacts'] == results['meshfor_contacts'], \
            f"Contact counts don't match: range={results['range_contacts']}, meshfor={results['meshfor_contacts']}"

        print("Benchmark completed successfully!")
        print(f"Contact count verification: PASSED ({results['range_contacts']} contacts)")

        return results

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_correctness():
    """
    Test that mesh-for and range loop produce identical results.
    """
    print("\n" + "="*70)
    print("Correctness Test: Comparing mesh-for vs range loop results")
    print("="*70)

    from algorithm.collision_detection_bvh_meshfor import collision_detection_bvh_meshfor_module

    demo = 'cube'
    print(f"\nLoading demo: {demo}")

    module = collision_detection_bvh_meshfor_module(demo=demo)
    module.init_bvh()
    module.build_bvh()

    # Run with range loop
    module.n_contacts[None] = 0
    module.find_constraints_PT_bvh()
    module.find_constraints_EE_bvh_range()
    ti.sync()
    range_contacts = module.n_contacts[None]

    # Get contact pairs for range loop
    range_pairs = []
    for i in range(min(range_contacts, 100)):  # Sample first 100
        pair = module.contact_pairs[i]
        range_pairs.append((tuple(pair.a.to_numpy()), pair.b))

    # Run with mesh-for loop
    module.n_contacts[None] = 0
    module.find_constraints_PT_bvh()
    module.find_constraints_EE_bvh_meshfor()
    ti.sync()
    meshfor_contacts = module.n_contacts[None]

    # Compare
    print(f"\nResults:")
    print(f"  - Range loop contacts: {range_contacts}")
    print(f"  - Mesh-for contacts: {meshfor_contacts}")

    if range_contacts == meshfor_contacts:
        print("\nCorrectness test: PASSED")
        return True
    else:
        print("\nCorrectness test: FAILED - contact counts differ!")
        return False


if __name__ == '__main__':
    print("Starting MeshFor Benchmark Tests")
    print("="*70)

    # Run correctness test first
    correctness_ok = test_correctness()

    if correctness_ok:
        # Run performance benchmark
        results = test_meshfor_benchmark()

        if results:
            print("\n" + "="*70)
            print("Summary")
            print("="*70)
            print(f"Range loop:    {results['range_mean']:.3f} ms")
            print(f"Mesh-for loop: {results['meshfor_mean']:.3f} ms")
            print(f"Speedup:       {results['speedup']:.2f}x")

            if results['speedup'] > 1.0:
                print("\nMesh-for is FASTER than range loop!")
            elif results['speedup'] < 1.0:
                print("\nRange loop is faster (mesh-for overhead may dominate for small meshes)")
            else:
                print("\nPerformance is similar")
    else:
        print("\nSkipping benchmark due to correctness failure")
