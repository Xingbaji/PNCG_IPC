"""
Test and benchmark for LBVH Final version.

Usage:
    python test_lbvh_final.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import taichi as ti
import numpy as np
import time


def test_correctness():
    """Test that LBVH Final produces correct results."""
    print("\n" + "=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    from algorithm.lbvh import LBVH_Triangles
    from algorithm.lbvh_final import LBVH_Triangles_Final

    n_triangles = 1000
    n_verts = 500

    # Create test data
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    vertices_np = np.random.rand(n_verts, 3).astype(np.float32) * 10.0
    vertices.from_numpy(vertices_np)

    triangles = ti.field(dtype=ti.i32, shape=(n_triangles, 3))
    triangles_np = np.random.randint(0, n_verts, size=(n_triangles, 3)).astype(np.int32)
    triangles.from_numpy(triangles_np)

    # Build with both versions
    bvh_orig = LBVH_Triangles(n_triangles)
    bvh_final = LBVH_Triangles_Final(n_triangles)

    bvh_orig.build(vertices, triangles, n_triangles)
    bvh_final.build(vertices, triangles, n_triangles)

    # Compare root AABBs (should cover same scene)
    # Note: exact AABB values may differ due to different tree structures
    # but root should cover all primitives

    orig_root_lower = np.array([bvh_orig.bv_lower[0][0], bvh_orig.bv_lower[0][1], bvh_orig.bv_lower[0][2]])
    orig_root_upper = np.array([bvh_orig.bv_upper[0][0], bvh_orig.bv_upper[0][1], bvh_orig.bv_upper[0][2]])

    final_root = bvh_final.aabb.to_numpy()[0]
    final_root_lower = final_root[:3]
    final_root_upper = final_root[3:]

    # Check that both roots cover the scene (only vertices used by triangles)
    # Get unique vertex indices used by triangles
    used_vertices = np.unique(triangles_np.flatten())
    used_verts_np = vertices_np[used_vertices]
    scene_lower = used_verts_np.min(axis=0)
    scene_upper = used_verts_np.max(axis=0)

    orig_covers = np.all(orig_root_lower <= scene_lower + 1e-5) and np.all(orig_root_upper >= scene_upper - 1e-5)
    final_covers = np.all(final_root_lower <= scene_lower + 1e-5) and np.all(final_root_upper >= scene_upper - 1e-5)

    print(f"Original root AABB covers scene: {orig_covers}")
    print(f"Final root AABB covers scene: {final_covers}")

    # Test refit
    vertices_np += 0.1
    vertices.from_numpy(vertices_np)

    bvh_orig.refit(vertices, triangles, n_triangles)
    bvh_final.refit(vertices, triangles, n_triangles)

    final_root_after = bvh_final.aabb.to_numpy()[0]
    used_verts_np_after = vertices_np[used_vertices]
    scene_lower_after = used_verts_np_after.min(axis=0)
    scene_upper_after = used_verts_np_after.max(axis=0)

    final_covers_after = (np.all(final_root_after[:3] <= scene_lower_after + 1e-5) and
                          np.all(final_root_after[3:] >= scene_upper_after - 1e-5))

    print(f"Final root AABB covers scene after refit: {final_covers_after}")

    success = orig_covers and final_covers and final_covers_after
    print(f"\nCorrectness test: {'PASSED' if success else 'FAILED'}")
    return success


def benchmark_final():
    """Benchmark LBVH Final vs Original."""
    print("\n" + "=" * 60)
    print("Benchmarking LBVH Final vs Original")
    print("=" * 60)

    from algorithm.lbvh import LBVH_Triangles
    from algorithm.lbvh_final import LBVH_Triangles_Final

    scales = [1000, 10000, 50000, 100000, 200000]
    n_warmup = 2
    n_iterations = 5

    print(f"\n{'N':>10} | {'Original':>12} | {'Final':>12} | {'Speedup':>10}")
    print("-" * 52)

    for n_triangles in scales:
        n_verts = max(n_triangles // 2, 100)

        vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        vertices_np = np.random.rand(n_verts, 3).astype(np.float32) * 10.0
        vertices.from_numpy(vertices_np)

        triangles = ti.field(dtype=ti.i32, shape=(n_triangles, 3))
        triangles_np = np.random.randint(0, n_verts, size=(n_triangles, 3)).astype(np.int32)
        triangles.from_numpy(triangles_np)

        bvh_orig = LBVH_Triangles(n_triangles)
        bvh_final = LBVH_Triangles_Final(n_triangles)

        # Warmup
        for _ in range(n_warmup):
            bvh_orig.build(vertices, triangles, n_triangles)
            bvh_final.build(vertices, triangles, n_triangles)
        ti.sync()

        # Benchmark Original
        times_orig = []
        for _ in range(n_iterations):
            ti.sync()
            t0 = time.perf_counter()
            bvh_orig.build(vertices, triangles, n_triangles)
            ti.sync()
            times_orig.append(time.perf_counter() - t0)

        # Benchmark Final
        times_final = []
        for _ in range(n_iterations):
            ti.sync()
            t0 = time.perf_counter()
            bvh_final.build(vertices, triangles, n_triangles)
            ti.sync()
            times_final.append(time.perf_counter() - t0)

        orig_ms = np.mean(times_orig) * 1000
        final_ms = np.mean(times_final) * 1000
        speedup = orig_ms / final_ms

        print(f"{n_triangles:>10,} | {orig_ms:>10.2f}ms | {final_ms:>10.2f}ms | {speedup:>9.1f}x")


def main():
    print("=" * 60)
    print("LBVH Final Version Test")
    print("=" * 60)

    ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache")

    # Test correctness
    correctness_ok = test_correctness()

    if correctness_ok:
        # Run benchmark
        benchmark_final()
    else:
        print("\nSkipping benchmark due to correctness test failure")
        return 1

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
