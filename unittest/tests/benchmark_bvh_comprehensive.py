"""
Comprehensive BVH benchmark comparing original vs optimized implementations.

Tests different scales of data and produces detailed analysis.
Compares: Original, Optimized V1, Optimized V2, Optimized V3 (Custom Radix Sort)

Usage:
    python benchmark_bvh_comprehensive.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import taichi as ti
import numpy as np
import time

# Test configurations
SCALES = [1000, 5000, 10000, 50000, 100000, 200000]
N_WARMUP = 2
N_ITERATIONS = 10
COMPARE_V2 = True   # Compare V2 optimized version
COMPARE_V3 = True   # Compare V3 with custom Radix Sort


def create_test_data(n_primitives, n_verts=None):
    """Create random test data for benchmarking."""
    if n_verts is None:
        n_verts = max(n_primitives // 2, 100)

    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    vertices_np = np.random.rand(n_verts, 3).astype(np.float32) * 10.0
    vertices.from_numpy(vertices_np)

    triangles = ti.field(dtype=ti.i32, shape=(n_primitives, 3))
    triangles_np = np.random.randint(0, n_verts, size=(n_primitives, 3)).astype(np.int32)
    triangles.from_numpy(triangles_np)

    edges = ti.field(dtype=ti.i32, shape=(n_primitives, 2))
    edges_np = np.random.randint(0, n_verts, size=(n_primitives, 2)).astype(np.int32)
    edges.from_numpy(edges_np)

    return vertices, triangles, edges, vertices_np


def benchmark_single_scale(n_primitives, verbose=True):
    """Benchmark a single scale and return detailed timing."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking N = {n_primitives:,}")
        print('='*60)

    from algorithm.lbvh import LBVH_Triangles, LBVH_Edges
    from algorithm.lbvh_optimized import LBVH_Triangles_Optimized, LBVH_Edges_Optimized

    if COMPARE_V2:
        from algorithm.lbvh_optimized_v2 import LBVH_Triangles_V2, LBVH_Edges_V2

    if COMPARE_V3:
        from algorithm.lbvh_optimized_v3 import LBVH_Triangles_V3, LBVH_Edges_V3

    vertices, triangles, edges, vertices_np = create_test_data(n_primitives)
    n_verts = vertices.shape[0]

    # Create BVH instances
    bvh_tri_orig = LBVH_Triangles(n_primitives)
    bvh_tri_opt = LBVH_Triangles_Optimized(n_primitives)
    bvh_edge_orig = LBVH_Edges(n_primitives)
    bvh_edge_opt = LBVH_Edges_Optimized(n_primitives)

    if COMPARE_V2:
        bvh_tri_v2 = LBVH_Triangles_V2(n_primitives)
        bvh_edge_v2 = LBVH_Edges_V2(n_primitives)

    if COMPARE_V3:
        bvh_tri_v3 = LBVH_Triangles_V3(n_primitives)
        bvh_edge_v3 = LBVH_Edges_V3(n_primitives)

    results = {
        'n_primitives': n_primitives,
        'n_verts': n_verts,
    }

    # Warmup
    for _ in range(N_WARMUP):
        bvh_tri_orig.build(vertices, triangles, n_primitives)
        bvh_tri_opt.build(vertices, triangles, n_primitives)
        bvh_edge_orig.build(vertices, edges, n_primitives)
        bvh_edge_opt.build(vertices, edges, n_primitives)
        if COMPARE_V2:
            bvh_tri_v2.build(vertices, triangles, n_primitives)
            bvh_edge_v2.build(vertices, edges, n_primitives)
        if COMPARE_V3:
            bvh_tri_v3.build(vertices, triangles, n_primitives)
            bvh_edge_v3.build(vertices, edges, n_primitives)
    ti.sync()

    # ========== Triangle BVH Build ==========
    times_orig = []
    for _ in range(N_ITERATIONS):
        ti.sync()
        t0 = time.perf_counter()
        bvh_tri_orig.build(vertices, triangles, n_primitives)
        ti.sync()
        times_orig.append(time.perf_counter() - t0)

    times_opt = []
    for _ in range(N_ITERATIONS):
        ti.sync()
        t0 = time.perf_counter()
        bvh_tri_opt.build(vertices, triangles, n_primitives)
        ti.sync()
        times_opt.append(time.perf_counter() - t0)

    times_v2 = []
    if COMPARE_V2:
        for _ in range(N_ITERATIONS):
            ti.sync()
            t0 = time.perf_counter()
            bvh_tri_v2.build(vertices, triangles, n_primitives)
            ti.sync()
            times_v2.append(time.perf_counter() - t0)

    times_v3 = []
    if COMPARE_V3:
        for _ in range(N_ITERATIONS):
            ti.sync()
            t0 = time.perf_counter()
            bvh_tri_v3.build(vertices, triangles, n_primitives)
            ti.sync()
            times_v3.append(time.perf_counter() - t0)

    results['tri_build_orig_ms'] = np.mean(times_orig) * 1000
    results['tri_build_opt_ms'] = np.mean(times_opt) * 1000
    results['tri_build_v2_ms'] = np.mean(times_v2) * 1000 if times_v2 else 0
    results['tri_build_v3_ms'] = np.mean(times_v3) * 1000 if times_v3 else 0
    results['tri_build_orig_std'] = np.std(times_orig) * 1000
    results['tri_build_opt_std'] = np.std(times_opt) * 1000
    results['tri_build_v2_std'] = np.std(times_v2) * 1000 if times_v2 else 0
    results['tri_build_v3_std'] = np.std(times_v3) * 1000 if times_v3 else 0

    # ========== Triangle BVH Refit ==========
    times_orig = []
    for i in range(N_ITERATIONS):
        vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
        vertices.from_numpy(vertices_np)
        ti.sync()
        t0 = time.perf_counter()
        bvh_tri_orig.refit(vertices, triangles, n_primitives)
        ti.sync()
        times_orig.append(time.perf_counter() - t0)

    times_opt = []
    for i in range(N_ITERATIONS):
        vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
        vertices.from_numpy(vertices_np)
        ti.sync()
        t0 = time.perf_counter()
        bvh_tri_opt.refit(vertices, triangles, n_primitives)
        ti.sync()
        times_opt.append(time.perf_counter() - t0)

    times_v2 = []
    if COMPARE_V2:
        for i in range(N_ITERATIONS):
            vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
            vertices.from_numpy(vertices_np)
            ti.sync()
            t0 = time.perf_counter()
            bvh_tri_v2.refit(vertices, triangles, n_primitives)
            ti.sync()
            times_v2.append(time.perf_counter() - t0)

    times_v3 = []
    if COMPARE_V3:
        for i in range(N_ITERATIONS):
            vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
            vertices.from_numpy(vertices_np)
            ti.sync()
            t0 = time.perf_counter()
            bvh_tri_v3.refit(vertices, triangles, n_primitives)
            ti.sync()
            times_v3.append(time.perf_counter() - t0)

    results['tri_refit_orig_ms'] = np.mean(times_orig) * 1000
    results['tri_refit_opt_ms'] = np.mean(times_opt) * 1000
    results['tri_refit_v2_ms'] = np.mean(times_v2) * 1000 if times_v2 else 0
    results['tri_refit_v3_ms'] = np.mean(times_v3) * 1000 if times_v3 else 0
    results['tri_refit_orig_std'] = np.std(times_orig) * 1000
    results['tri_refit_opt_std'] = np.std(times_opt) * 1000
    results['tri_refit_v2_std'] = np.std(times_v2) * 1000 if times_v2 else 0
    results['tri_refit_v3_std'] = np.std(times_v3) * 1000 if times_v3 else 0

    # ========== Edge BVH Build ==========
    times_orig = []
    for _ in range(N_ITERATIONS):
        ti.sync()
        t0 = time.perf_counter()
        bvh_edge_orig.build(vertices, edges, n_primitives)
        ti.sync()
        times_orig.append(time.perf_counter() - t0)

    times_opt = []
    for _ in range(N_ITERATIONS):
        ti.sync()
        t0 = time.perf_counter()
        bvh_edge_opt.build(vertices, edges, n_primitives)
        ti.sync()
        times_opt.append(time.perf_counter() - t0)

    times_v2 = []
    if COMPARE_V2:
        for _ in range(N_ITERATIONS):
            ti.sync()
            t0 = time.perf_counter()
            bvh_edge_v2.build(vertices, edges, n_primitives)
            ti.sync()
            times_v2.append(time.perf_counter() - t0)

    times_v3 = []
    if COMPARE_V3:
        for _ in range(N_ITERATIONS):
            ti.sync()
            t0 = time.perf_counter()
            bvh_edge_v3.build(vertices, edges, n_primitives)
            ti.sync()
            times_v3.append(time.perf_counter() - t0)

    results['edge_build_orig_ms'] = np.mean(times_orig) * 1000
    results['edge_build_opt_ms'] = np.mean(times_opt) * 1000
    results['edge_build_v2_ms'] = np.mean(times_v2) * 1000 if times_v2 else 0
    results['edge_build_v3_ms'] = np.mean(times_v3) * 1000 if times_v3 else 0
    results['edge_build_orig_std'] = np.std(times_orig) * 1000
    results['edge_build_opt_std'] = np.std(times_opt) * 1000
    results['edge_build_v2_std'] = np.std(times_v2) * 1000 if times_v2 else 0
    results['edge_build_v3_std'] = np.std(times_v3) * 1000 if times_v3 else 0

    # ========== Edge BVH Refit ==========
    times_orig = []
    for i in range(N_ITERATIONS):
        vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
        vertices.from_numpy(vertices_np)
        ti.sync()
        t0 = time.perf_counter()
        bvh_edge_orig.refit(vertices, edges, n_primitives)
        ti.sync()
        times_orig.append(time.perf_counter() - t0)

    times_opt = []
    for i in range(N_ITERATIONS):
        vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
        vertices.from_numpy(vertices_np)
        ti.sync()
        t0 = time.perf_counter()
        bvh_edge_opt.refit(vertices, edges, n_primitives)
        ti.sync()
        times_opt.append(time.perf_counter() - t0)

    times_v2 = []
    if COMPARE_V2:
        for i in range(N_ITERATIONS):
            vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
            vertices.from_numpy(vertices_np)
            ti.sync()
            t0 = time.perf_counter()
            bvh_edge_v2.refit(vertices, edges, n_primitives)
            ti.sync()
            times_v2.append(time.perf_counter() - t0)

    times_v3 = []
    if COMPARE_V3:
        for i in range(N_ITERATIONS):
            vertices_np += np.random.rand(n_verts, 3).astype(np.float32) * 0.01
            vertices.from_numpy(vertices_np)
            ti.sync()
            t0 = time.perf_counter()
            bvh_edge_v3.refit(vertices, edges, n_primitives)
            ti.sync()
            times_v3.append(time.perf_counter() - t0)

    results['edge_refit_orig_ms'] = np.mean(times_orig) * 1000
    results['edge_refit_opt_ms'] = np.mean(times_opt) * 1000
    results['edge_refit_v2_ms'] = np.mean(times_v2) * 1000 if times_v2 else 0
    results['edge_refit_v3_ms'] = np.mean(times_v3) * 1000 if times_v3 else 0
    results['edge_refit_orig_std'] = np.std(times_orig) * 1000
    results['edge_refit_opt_std'] = np.std(times_opt) * 1000
    results['edge_refit_v2_std'] = np.std(times_v2) * 1000 if times_v2 else 0
    results['edge_refit_v3_std'] = np.std(times_v3) * 1000 if times_v3 else 0

    # Calculate speedups
    results['tri_build_speedup'] = results['tri_build_orig_ms'] / results['tri_build_opt_ms'] if results['tri_build_opt_ms'] > 0 else 0
    results['tri_build_speedup_v2'] = results['tri_build_orig_ms'] / results['tri_build_v2_ms'] if results['tri_build_v2_ms'] > 0 else 0
    results['tri_build_speedup_v3'] = results['tri_build_orig_ms'] / results['tri_build_v3_ms'] if results['tri_build_v3_ms'] > 0 else 0
    results['tri_refit_speedup'] = results['tri_refit_orig_ms'] / results['tri_refit_opt_ms'] if results['tri_refit_opt_ms'] > 0 else 0
    results['tri_refit_speedup_v2'] = results['tri_refit_orig_ms'] / results['tri_refit_v2_ms'] if results['tri_refit_v2_ms'] > 0 else 0
    results['tri_refit_speedup_v3'] = results['tri_refit_orig_ms'] / results['tri_refit_v3_ms'] if results['tri_refit_v3_ms'] > 0 else 0
    results['edge_build_speedup'] = results['edge_build_orig_ms'] / results['edge_build_opt_ms'] if results['edge_build_opt_ms'] > 0 else 0
    results['edge_build_speedup_v2'] = results['edge_build_orig_ms'] / results['edge_build_v2_ms'] if results['edge_build_v2_ms'] > 0 else 0
    results['edge_build_speedup_v3'] = results['edge_build_orig_ms'] / results['edge_build_v3_ms'] if results['edge_build_v3_ms'] > 0 else 0
    results['edge_refit_speedup'] = results['edge_refit_orig_ms'] / results['edge_refit_opt_ms'] if results['edge_refit_opt_ms'] > 0 else 0
    results['edge_refit_speedup_v2'] = results['edge_refit_orig_ms'] / results['edge_refit_v2_ms'] if results['edge_refit_v2_ms'] > 0 else 0
    results['edge_refit_speedup_v3'] = results['edge_refit_orig_ms'] / results['edge_refit_v3_ms'] if results['edge_refit_v3_ms'] > 0 else 0

    if verbose:
        print(f"\n--- Triangle BVH ---")
        print(f"Build:  Orig {results['tri_build_orig_ms']:.3f}ms | V1 {results['tri_build_opt_ms']:.3f}ms ({results['tri_build_speedup']:.2f}x) | V2 {results['tri_build_v2_ms']:.3f}ms ({results['tri_build_speedup_v2']:.2f}x) | V3 {results['tri_build_v3_ms']:.3f}ms ({results['tri_build_speedup_v3']:.2f}x)")
        print(f"Refit:  Orig {results['tri_refit_orig_ms']:.3f}ms | V1 {results['tri_refit_opt_ms']:.3f}ms ({results['tri_refit_speedup']:.2f}x) | V2 {results['tri_refit_v2_ms']:.3f}ms ({results['tri_refit_speedup_v2']:.2f}x) | V3 {results['tri_refit_v3_ms']:.3f}ms ({results['tri_refit_speedup_v3']:.2f}x)")
        print(f"\n--- Edge BVH ---")
        print(f"Build:  Orig {results['edge_build_orig_ms']:.3f}ms | V1 {results['edge_build_opt_ms']:.3f}ms ({results['edge_build_speedup']:.2f}x) | V2 {results['edge_build_v2_ms']:.3f}ms ({results['edge_build_speedup_v2']:.2f}x) | V3 {results['edge_build_v3_ms']:.3f}ms ({results['edge_build_speedup_v3']:.2f}x)")
        print(f"Refit:  Orig {results['edge_refit_orig_ms']:.3f}ms | V1 {results['edge_refit_opt_ms']:.3f}ms ({results['edge_refit_speedup']:.2f}x) | V2 {results['edge_refit_v2_ms']:.3f}ms ({results['edge_refit_speedup_v2']:.2f}x) | V3 {results['edge_refit_v3_ms']:.3f}ms ({results['edge_refit_speedup_v3']:.2f}x)")

    return results


def print_summary_table(all_results):
    """Print a summary table of all results."""
    print("\n" + "=" * 160)
    print("SUMMARY TABLE - Triangle BVH Build")
    print("=" * 160)
    print(f"{'N':>10} | {'Original':>12} | {'V1 Opt':>12} | {'V1 Spdup':>8} | {'V2 Opt':>12} | {'V2 Spdup':>8} | {'V3 Radix':>12} | {'V3 Spdup':>8} | {'Best':>8}")
    print("-" * 160)
    for r in all_results:
        times = {'V1': r['tri_build_opt_ms'], 'V2': r['tri_build_v2_ms'], 'V3': r['tri_build_v3_ms']}
        best = min(times, key=times.get)
        print(f"{r['n_primitives']:>10,} | {r['tri_build_orig_ms']:>10.3f}ms | {r['tri_build_opt_ms']:>10.3f}ms | {r['tri_build_speedup']:>7.2f}x | {r['tri_build_v2_ms']:>10.3f}ms | {r['tri_build_speedup_v2']:>7.2f}x | {r['tri_build_v3_ms']:>10.3f}ms | {r['tri_build_speedup_v3']:>7.2f}x | {best:>8}")

    print("\n" + "=" * 160)
    print("SUMMARY TABLE - Triangle BVH Refit")
    print("=" * 160)
    print(f"{'N':>10} | {'Original':>12} | {'V1 Opt':>12} | {'V1 Spdup':>8} | {'V2 Opt':>12} | {'V2 Spdup':>8} | {'V3 Radix':>12} | {'V3 Spdup':>8} | {'Best':>8}")
    print("-" * 160)
    for r in all_results:
        times = {'V1': r['tri_refit_opt_ms'], 'V2': r['tri_refit_v2_ms'], 'V3': r['tri_refit_v3_ms']}
        best = min(times, key=times.get)
        print(f"{r['n_primitives']:>10,} | {r['tri_refit_orig_ms']:>10.3f}ms | {r['tri_refit_opt_ms']:>10.3f}ms | {r['tri_refit_speedup']:>7.2f}x | {r['tri_refit_v2_ms']:>10.3f}ms | {r['tri_refit_speedup_v2']:>7.2f}x | {r['tri_refit_v3_ms']:>10.3f}ms | {r['tri_refit_speedup_v3']:>7.2f}x | {best:>8}")

    print("\n" + "=" * 160)
    print("SUMMARY TABLE - Edge BVH Build")
    print("=" * 160)
    print(f"{'N':>10} | {'Original':>12} | {'V1 Opt':>12} | {'V1 Spdup':>8} | {'V2 Opt':>12} | {'V2 Spdup':>8} | {'V3 Radix':>12} | {'V3 Spdup':>8} | {'Best':>8}")
    print("-" * 160)
    for r in all_results:
        times = {'V1': r['edge_build_opt_ms'], 'V2': r['edge_build_v2_ms'], 'V3': r['edge_build_v3_ms']}
        best = min(times, key=times.get)
        print(f"{r['n_primitives']:>10,} | {r['edge_build_orig_ms']:>10.3f}ms | {r['edge_build_opt_ms']:>10.3f}ms | {r['edge_build_speedup']:>7.2f}x | {r['edge_build_v2_ms']:>10.3f}ms | {r['edge_build_speedup_v2']:>7.2f}x | {r['edge_build_v3_ms']:>10.3f}ms | {r['edge_build_speedup_v3']:>7.2f}x | {best:>8}")


def analyze_results(all_results):
    """Analyze and print insights from benchmark results."""
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Average speedups - V1
    avg_tri_build_v1 = np.mean([r['tri_build_speedup'] for r in all_results])
    avg_tri_refit_v1 = np.mean([r['tri_refit_speedup'] for r in all_results])
    avg_edge_build_v1 = np.mean([r['edge_build_speedup'] for r in all_results])
    avg_edge_refit_v1 = np.mean([r['edge_refit_speedup'] for r in all_results])

    # Average speedups - V2
    avg_tri_build_v2 = np.mean([r['tri_build_speedup_v2'] for r in all_results])
    avg_tri_refit_v2 = np.mean([r['tri_refit_speedup_v2'] for r in all_results])
    avg_edge_build_v2 = np.mean([r['edge_build_speedup_v2'] for r in all_results])
    avg_edge_refit_v2 = np.mean([r['edge_refit_speedup_v2'] for r in all_results])

    # Average speedups - V3
    avg_tri_build_v3 = np.mean([r['tri_build_speedup_v3'] for r in all_results])
    avg_tri_refit_v3 = np.mean([r['tri_refit_speedup_v3'] for r in all_results])
    avg_edge_build_v3 = np.mean([r['edge_build_speedup_v3'] for r in all_results])
    avg_edge_refit_v3 = np.mean([r['edge_refit_speedup_v3'] for r in all_results])

    print(f"\n1. Average Speedups (vs Original):")
    print(f"                        V1 Optimized    V2 Optimized    V3 Radix Sort")
    print(f"   Triangle BVH Build:    {avg_tri_build_v1:.2f}x           {avg_tri_build_v2:.2f}x           {avg_tri_build_v3:.2f}x")
    print(f"   Triangle BVH Refit:    {avg_tri_refit_v1:.2f}x           {avg_tri_refit_v2:.2f}x           {avg_tri_refit_v3:.2f}x")
    print(f"   Edge BVH Build:        {avg_edge_build_v1:.2f}x           {avg_edge_build_v2:.2f}x           {avg_edge_build_v3:.2f}x")
    print(f"   Edge BVH Refit:        {avg_edge_refit_v1:.2f}x           {avg_edge_refit_v2:.2f}x           {avg_edge_refit_v3:.2f}x")

    # Scaling analysis
    print(f"\n2. Scaling Analysis (how performance changes with N):")

    if len(all_results) >= 2:
        # Compare smallest vs largest
        small = all_results[0]
        large = all_results[-1]
        scale_factor = large['n_primitives'] / small['n_primitives']

        # Ideal O(N log N) scaling
        ideal_factor = scale_factor * np.log2(large['n_primitives']) / np.log2(small['n_primitives'])

        orig_tri_factor = large['tri_build_orig_ms'] / small['tri_build_orig_ms']
        opt_tri_factor = large['tri_build_opt_ms'] / small['tri_build_opt_ms']
        v2_tri_factor = large['tri_build_v2_ms'] / small['tri_build_v2_ms'] if small['tri_build_v2_ms'] > 0 else 0
        v3_tri_factor = large['tri_build_v3_ms'] / small['tri_build_v3_ms'] if small['tri_build_v3_ms'] > 0 else 0

        print(f"   Scale factor: {scale_factor:.1f}x (from {small['n_primitives']:,} to {large['n_primitives']:,})")
        print(f"   Ideal O(N log N) factor: {ideal_factor:.1f}x")
        print(f"   Original Triangle Build factor: {orig_tri_factor:.1f}x")
        print(f"   V1 Optimized Triangle Build factor: {opt_tri_factor:.1f}x")
        print(f"   V2 Optimized Triangle Build factor: {v2_tri_factor:.1f}x")
        print(f"   V3 Radix Sort Triangle Build factor: {v3_tri_factor:.1f}x")

    # Memory bandwidth estimation
    print(f"\n3. Estimated Throughput (primitives/ms):")
    for r in all_results:
        throughput_orig = r['n_primitives'] / r['tri_build_orig_ms']
        throughput_v1 = r['n_primitives'] / r['tri_build_opt_ms']
        throughput_v2 = r['n_primitives'] / r['tri_build_v2_ms'] if r['tri_build_v2_ms'] > 0 else 0
        throughput_v3 = r['n_primitives'] / r['tri_build_v3_ms'] if r['tri_build_v3_ms'] > 0 else 0
        print(f"   N={r['n_primitives']:>7,}: Orig {throughput_orig:>8,.0f}/ms, V1 {throughput_v1:>8,.0f}/ms, V2 {throughput_v2:>8,.0f}/ms, V3 {throughput_v3:>8,.0f}/ms")

    # Best/worst case for V3
    speedups_v3 = [r['tri_build_speedup_v3'] for r in all_results]
    best_idx = np.argmax(speedups_v3)
    worst_idx = np.argmin(speedups_v3)

    print(f"\n4. Best/Worst Cases (V3 vs Original):")
    print(f"   Best speedup:  {speedups_v3[best_idx]:.2f}x at N={all_results[best_idx]['n_primitives']:,}")
    print(f"   Worst speedup: {speedups_v3[worst_idx]:.2f}x at N={all_results[worst_idx]['n_primitives']:,}")

    # Recommendations
    print(f"\n5. Recommendations:")
    best_build = max(['V1', 'V2', 'V3'], key=lambda v: {'V1': avg_tri_build_v1, 'V2': avg_tri_build_v2, 'V3': avg_tri_build_v3}[v])
    best_refit = max(['V1', 'V2', 'V3'], key=lambda v: {'V1': avg_tri_refit_v1, 'V2': avg_tri_refit_v2, 'V3': avg_tri_refit_v3}[v])
    print(f"   Best for Build: {best_build} (V1: {avg_tri_build_v1:.2f}x, V2: {avg_tri_build_v2:.2f}x, V3: {avg_tri_build_v3:.2f}x)")
    print(f"   Best for Refit: {best_refit} (V1: {avg_tri_refit_v1:.2f}x, V2: {avg_tri_refit_v2:.2f}x, V3: {avg_tri_refit_v3:.2f}x)")

    # V3 specific analysis
    print(f"\n6. V3 Custom Radix Sort Analysis:")
    print(f"   - Custom 8-bit Radix Sort (8 passes for 64-bit Morton codes)")
    print(f"   - Per-block histogram computation with atomic increments")
    print(f"   - Block-level prefix sum for scatter destination calculation")

    # Compare V3 vs V2 (radix sort impact)
    v3_vs_v2_build = np.mean([r['tri_build_v2_ms'] / r['tri_build_v3_ms'] if r['tri_build_v3_ms'] > 0 else 0 for r in all_results])
    v3_vs_v2_refit = np.mean([r['tri_refit_v2_ms'] / r['tri_refit_v3_ms'] if r['tri_refit_v3_ms'] > 0 else 0 for r in all_results])
    print(f"\n7. V3 vs V2 Direct Comparison:")
    print(f"   Build: V3 is {v3_vs_v2_build:.2f}x {'faster' if v3_vs_v2_build > 1 else 'slower'} than V2")
    print(f"   Refit: V3 is {v3_vs_v2_refit:.2f}x {'faster' if v3_vs_v2_refit > 1 else 'slower'} than V2 (expected ~same, no sorting in refit)")


def main():
    print("=" * 80)
    print("Comprehensive BVH Benchmark: Original vs V1 vs V2 vs V3 (Custom Radix Sort)")
    print("=" * 80)
    print(f"Scales to test: {SCALES}")
    print(f"Warmup iterations: {N_WARMUP}")
    print(f"Benchmark iterations: {N_ITERATIONS}")

    # Initialize Taichi
    ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache")

    # Run benchmarks for each scale
    all_results = []
    for n in SCALES:
        try:
            results = benchmark_single_scale(n, verbose=True)
            all_results.append(results)
        except Exception as e:
            print(f"Failed at N={n}: {e}")
            import traceback
            traceback.print_exc()
            break

    if all_results:
        print_summary_table(all_results)
        analyze_results(all_results)

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
