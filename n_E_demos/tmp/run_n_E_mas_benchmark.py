"""
MAS Preconditioner Benchmark based on n_E demo (eight_E_drop_demo_contact).

Tests the MAS preconditioner with real mesh data and reports detailed timing breakdown.
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../demo')
import os
os.chdir('../demo')

import taichi as ti
import numpy as np
import time
from collections import defaultdict

# Initialize Taichi with profiler and offline cache
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True, kernel_profiler=True)

print(f"Taichi version: {ti.__version__}")


class TimingReport:
    """Collect and report detailed timing statistics."""

    def __init__(self):
        self.times = defaultdict(list)
        self.labels = {
            'hierarchy': 'Build Hierarchy',
            'assemble': 'Assemble Block Matrices',
            'invert': 'Invert Block Matrices',
            'apply_total': 'Apply Total',
            'restrict': 'Phase1: Restriction',
            'local_solve': 'Phase2: Local Solve',
            'prolong': 'Phase3: Prolongation',
        }

    def add(self, key, time_ms):
        self.times[key].append(time_ms)

    def report(self):
        print("\n" + "="*70)
        print("DETAILED TIMING REPORT")
        print("="*70)
        print(f"{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-"*70)

        total_time = 0
        for key in ['hierarchy', 'assemble', 'invert', 'apply_total']:
            if key in self.times and self.times[key]:
                arr = np.array(self.times[key])
                label = self.labels.get(key, key)
                print(f"{label:<30} {np.mean(arr):>10.3f}   {np.std(arr):>10.3f}   {np.min(arr):>10.3f}   {np.max(arr):>10.3f}")
                if key != 'hierarchy':
                    total_time += np.mean(arr)

        print("-"*70)
        print(f"{'Per-iteration Total':<30} {total_time:>10.3f} ms")

        # Apply breakdown
        print("\n" + "-"*70)
        print("Apply Phase Breakdown:")
        print("-"*70)
        for key in ['restrict', 'local_solve', 'prolong']:
            if key in self.times and self.times[key]:
                arr = np.array(self.times[key])
                label = self.labels.get(key, key)
                print(f"  {label:<28} {np.mean(arr):>10.3f}   {np.std(arr):>10.3f}   {np.min(arr):>10.3f}   {np.max(arr):>10.3f}")

        return total_time


def benchmark_n_E_mas(demo_name='eight_E_drop_demo_contact', n_iterations=10,
                      use_warp_reduction=True, use_cholesky=True,
                      use_blocked=False, use_incomplete=False,
                      use_oneway_gj=False, use_conflict_free=False,
                      use_banded=False):
    """
    Benchmark MAS preconditioner on n_E demo with detailed timing.

    Args:
        demo_name: Demo configuration name
        n_iterations: Number of benchmark iterations (after warmup)
        use_warp_reduction: Use P1 warp reduction optimization
        use_cholesky: Use Cholesky vs Gauss-Jordan inversion
        use_blocked: Use blocked Cholesky (P2 optimization)
        use_incomplete: Use Incomplete Cholesky IC(0) approximation (P3 optimization)
        use_oneway_gj: Use One-way Gauss-Jordan (P4 optimization)
        use_conflict_free: Use Conflict-free SpMV (P5 optimization)
        use_banded: Use Banded Sparse MV (P6 optimization, best with IC(0))
    """
    from util.model_loading import model_loading
    from algorithm.mas_preconditioner import MASPreconditioner

    # Determine inversion method name
    if use_incomplete:
        invert_method = "IC(0)"
    elif use_oneway_gj:
        invert_method = "One-way GJ"
    elif use_blocked and use_cholesky:
        invert_method = "Blocked Cholesky"
    elif use_cholesky:
        invert_method = "Cholesky"
    else:
        invert_method = "Gauss-Jordan"

    # Determine local solve method name
    if use_banded:
        solve_method = "Banded"
    elif use_conflict_free:
        solve_method = "Conflict-free"
    else:
        solve_method = "Standard"

    print(f"\n{'='*70}")
    print(f"MAS Benchmark: {demo_name}")
    print(f"  inversion={invert_method}, local_solve={solve_method}")
    print(f"  warp_reduction={use_warp_reduction}")
    print(f"  iterations={n_iterations}")
    print(f"{'='*70}")

    # Load model
    print("\n[1] Loading model...")
    t0 = time.perf_counter()
    model = model_loading(demo=demo_name)
    mesh = model.mesh
    n_verts = len(mesh.verts)
    n_cells = len(mesh.cells)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"    Model loaded in {load_time:.2f} ms")
    print(f"    Vertices: {n_verts}, Cells: {n_cells}")

    # Place fields
    mesh.verts.place({
        'x': ti.types.vector(3, float),
        'grad': ti.types.vector(3, float),
        'diagH': ti.types.vector(3, float),
        'z': ti.types.vector(3, float),
        'm': float,
    })
    mesh.cells.place({'B': ti.math.mat3, 'W': float})
    mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

    @ti.kernel
    def init_data(mesh: ti.template()):
        for vert in mesh.verts:
            vert.m = 1.0
            vert.grad = ti.Vector([0.0, -1.0, 0.0])
            vert.diagH = ti.Vector([1.0, 1.0, 1.0])
            vert.z = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def precompute_cells(mesh: ti.template()):
        for c in mesh.cells:
            Dm = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Dm.inverse()
            c.W = ti.abs(Dm.determinant()) / 6.0

    @ti.kernel
    def compute_z_stats(mesh: ti.template()) -> ti.types.vector(3, ti.f32):
        z_norm = ti.cast(0.0, ti.f32)
        z_dot_g = ti.cast(0.0, ti.f32)
        z_count = ti.cast(0.0, ti.f32)
        for vert in mesh.verts:
            z_norm += ti.cast(vert.z.norm_sqr(), ti.f32)
            z_dot_g += ti.cast(vert.z.dot(vert.grad), ti.f32)
            z_count += 1.0
        return ti.Vector([ti.sqrt(z_norm), z_dot_g, z_count], dt=ti.f32)

    @ti.kernel
    def reset_z(mesh: ti.template()):
        for vert in mesh.verts:
            vert.z = ti.Vector([0.0, 0.0, 0.0])

    print("\n[2] Initializing data...")
    init_data(mesh)
    precompute_cells(mesh)

    # Create MAS
    print("\n[3] Creating MAS preconditioner...")
    mas = MASPreconditioner(n_verts, n_cells, mesh, use_metis=False)
    mas.elastic_type = 0  # ARAP

    # Create solver proxy
    class SolverProxy:
        def __init__(self):
            self.mesh = mesh
            self.mu = model.mu
            self.la = model.la
            self.dt = model.dt
            self.elastic_type = 0
            self.n_verts = n_verts
            self.n_cells = n_cells

    solver = SolverProxy()
    print(f"    Material: mu={solver.mu:.2f}, la={solver.la:.2f}, dt={solver.dt}")

    timing = TimingReport()

    # Build hierarchy (one-time cost)
    print("\n[4] Building hierarchy...")
    ti.sync()
    t0 = time.perf_counter()
    mas.build_hierarchy()
    ti.sync()
    hierarchy_time = (time.perf_counter() - t0) * 1000
    timing.add('hierarchy', hierarchy_time)
    print(f"    Hierarchy built: {mas.actual_levels} levels, {hierarchy_time:.2f} ms")

    # Warmup run (includes JIT compilation)
    print("\n[5] Warmup run (includes JIT compilation)...")
    ti.profiler.clear_kernel_profiler_info()

    ti.sync()
    t0 = time.perf_counter()
    mas.assemble_block_matrices(solver, use_full_hessian=True, use_optimized_kernel=True)
    ti.sync()
    warmup_assemble = (time.perf_counter() - t0) * 1000
    print(f"    Assemble: {warmup_assemble:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.invert_block_matrices(use_full_inversion=True, use_cholesky=use_cholesky,
                                  use_blocked=use_blocked, use_incomplete=use_incomplete,
                                  use_oneway_gj=use_oneway_gj)
    ti.sync()
    warmup_invert = (time.perf_counter() - t0) * 1000
    print(f"    Invert: {warmup_invert:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.apply(use_full_solve=True, use_warp_reduction=use_warp_reduction,
              use_conflict_free=use_conflict_free, use_banded=use_banded)
    ti.sync()
    warmup_apply = (time.perf_counter() - t0) * 1000
    print(f"    Apply: {warmup_apply:.2f} ms")

    stats = compute_z_stats(mesh)
    print(f"    Result: ||z||={stats[0]:.6f}, z·g={stats[1]:.6f}")
    spd_pass = stats[1] > 0
    print(f"    SPD check: {'PASS' if spd_pass else 'FAIL'}")

    # Benchmark runs
    print(f"\n[6] Benchmark runs ({n_iterations} iterations)...")
    ti.profiler.clear_kernel_profiler_info()

    for i in range(n_iterations):
        reset_z(mesh)

        # Assemble
        ti.sync()
        t0 = time.perf_counter()
        mas.assemble_block_matrices(solver, use_full_hessian=True, use_optimized_kernel=True)
        ti.sync()
        timing.add('assemble', (time.perf_counter() - t0) * 1000)

        # Invert
        ti.sync()
        t0 = time.perf_counter()
        mas.invert_block_matrices(use_full_inversion=True, use_cholesky=use_cholesky,
                                  use_blocked=use_blocked, use_incomplete=use_incomplete,
                                  use_oneway_gj=use_oneway_gj)
        ti.sync()
        timing.add('invert', (time.perf_counter() - t0) * 1000)

        # Apply with detailed breakdown
        ti.sync()
        t_apply_start = time.perf_counter()

        # Phase 1: Restriction
        mas._clear_multi_level_buffers()
        if use_warp_reduction:
            mas._clear_warp_sum_buffer()
            ti.sync()
            t0 = time.perf_counter()
            mas._build_multi_level_r_optimized()
            ti.sync()
        else:
            ti.sync()
            t0 = time.perf_counter()
            mas._build_multi_level_r()
            ti.sync()
        timing.add('restrict', (time.perf_counter() - t0) * 1000)

        # Phase 2: Local solve
        ti.sync()
        t0 = time.perf_counter()
        if use_banded:
            mas._schwarz_local_solve_banded()
        elif use_conflict_free:
            mas._schwarz_local_solve_conflict_free()
        else:
            mas._schwarz_local_solve_full()
        ti.sync()
        timing.add('local_solve', (time.perf_counter() - t0) * 1000)

        # Phase 3: Prolongation
        ti.sync()
        t0 = time.perf_counter()
        mas._collect_final_z()
        ti.sync()
        timing.add('prolong', (time.perf_counter() - t0) * 1000)

        timing.add('apply_total', (time.perf_counter() - t_apply_start) * 1000)

    # Final SPD check
    stats = compute_z_stats(mesh)
    final_spd = stats[1] > 0

    # Print timing report
    total_per_iter = timing.report()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Demo: {demo_name}")
    print(f"Mesh: {n_verts} vertices, {n_cells} cells")
    print(f"Hierarchy: {mas.actual_levels} levels")
    print(f"Inversion method: {invert_method}")
    print(f"Local solve method: {solve_method}")
    print(f"Warp reduction: {use_warp_reduction}")
    print(f"Total per iteration: {total_per_iter:.3f} ms")
    print(f"SPD check: {'PASS' if final_spd else 'FAIL'}")

    # Print kernel profiler
    print(f"\n{'='*70}")
    print("KERNEL PROFILER (top 15)")
    print(f"{'='*70}")
    ti.profiler.print_kernel_profiler_info('count')

    return {
        'demo': demo_name,
        'n_verts': n_verts,
        'n_cells': n_cells,
        'levels': mas.actual_levels,
        'invert_method': invert_method,
        'solve_method': solve_method,
        'warp_reduction': use_warp_reduction,
        'timing': dict(timing.times),
        'total_per_iter_ms': total_per_iter,
        'spd_pass': final_spd,
    }


def compare_warp_reduction(demo_name='eight_E_drop_demo_contact', n_iterations=10):
    """Compare performance with and without warp reduction optimization."""
    print("\n" + "="*70)
    print("COMPARISON: Sequential vs Warp Reduction")
    print("="*70)

    result_seq = benchmark_n_E_mas(demo_name, n_iterations, use_warp_reduction=False)

    print("\n\n" + "="*70)
    print("--- Running Warp Reduction comparison ---")
    print("="*70 + "\n")

    result_warp = benchmark_n_E_mas(demo_name, n_iterations, use_warp_reduction=True)

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Component':<25} {'Sequential (ms)':<18} {'Warp Reduce (ms)':<18} {'Speedup':<10}")
    print("-"*70)

    for key in ['assemble', 'invert', 'restrict', 'local_solve', 'prolong', 'apply_total']:
        if key in result_seq['timing'] and key in result_warp['timing']:
            seq_mean = np.mean(result_seq['timing'][key])
            warp_mean = np.mean(result_warp['timing'][key])
            speedup = seq_mean / warp_mean if warp_mean > 0 else 0
            print(f"{key:<25} {seq_mean:<18.3f} {warp_mean:<18.3f} {speedup:<10.2f}x")

    print("-"*70)
    seq_total = result_seq['total_per_iter_ms']
    warp_total = result_warp['total_per_iter_ms']
    speedup = seq_total / warp_total if warp_total > 0 else 0
    print(f"{'Total per iteration':<25} {seq_total:<18.3f} {warp_total:<18.3f} {speedup:<10.2f}x")

    print(f"\nSPD check: Sequential={'PASS' if result_seq['spd_pass'] else 'FAIL'}, "
          f"Warp={'PASS' if result_warp['spd_pass'] else 'FAIL'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MAS Benchmark on n_E demo')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact',
                        help='Demo name (default: eight_E_drop_demo_contact)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of benchmark iterations')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with/without warp reduction')
    parser.add_argument('--compare-invert', action='store_true',
                        help='Compare all inversion methods')
    parser.add_argument('--compare-solve', action='store_true',
                        help='Compare local solve methods (standard vs conflict-free)')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all optimization combinations (IC(0), P1, P5)')
    parser.add_argument('--no-warp', action='store_true',
                        help='Disable warp reduction optimization')
    parser.add_argument('--gauss-jordan', action='store_true',
                        help='Use Gauss-Jordan instead of Cholesky')
    parser.add_argument('--blocked', action='store_true',
                        help='Use blocked Cholesky (P2 optimization)')
    parser.add_argument('--incomplete', action='store_true',
                        help='Use Incomplete Cholesky IC(0) (P3 optimization)')
    parser.add_argument('--oneway-gj', action='store_true',
                        help='Use One-way Gauss-Jordan (P4 optimization)')
    parser.add_argument('--conflict-free', action='store_true',
                        help='Use Conflict-free SpMV (P5 optimization)')
    parser.add_argument('--banded', action='store_true',
                        help='Use Banded Sparse MV (P6 optimization, best with IC(0))')
    args = parser.parse_args()

    if args.compare:
        compare_warp_reduction(args.demo, args.iterations)
    elif args.compare_invert:
        # Compare all inversion methods
        print("\n" + "="*70)
        print("COMPARISON: All Inversion Methods")
        print("="*70)

        results = {}

        # Standard Cholesky
        print("\n--- Standard Cholesky ---")
        results['cholesky'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=False
        )

        # One-way Gauss-Jordan (P4)
        print("\n--- One-way Gauss-Jordan (P4) ---")
        results['oneway_gj'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=False, use_oneway_gj=True
        )

        # Incomplete Cholesky IC(0) (P3)
        print("\n--- Incomplete Cholesky IC(0) (P3) ---")
        results['incomplete'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=True
        )

        # Blocked Cholesky (P2)
        print("\n--- Blocked Cholesky (P2) ---")
        results['blocked'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=True, use_incomplete=False
        )

        # Gauss-Jordan (baseline)
        print("\n--- Gauss-Jordan (baseline) ---")
        results['gauss_jordan'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=False,
            use_blocked=False, use_incomplete=False
        )

        # Summary comparison
        print("\n" + "="*70)
        print("INVERSION METHOD COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Method':<20} {'Invert (ms)':<15} {'Total (ms)':<15} {'Speedup':<10} {'SPD':<6}")
        print("-"*70)

        baseline = results['cholesky']['total_per_iter_ms']
        for name, res in results.items():
            invert_time = np.mean(res['timing']['invert'])
            total_time = res['total_per_iter_ms']
            speedup = baseline / total_time if total_time > 0 else 0
            spd = 'PASS' if res['spd_pass'] else 'FAIL'
            print(f"{res['invert_method']:<20} {invert_time:<15.3f} {total_time:<15.3f} {speedup:<10.2f}x {spd:<6}")

    elif args.compare_solve:
        # Compare local solve methods
        print("\n" + "="*70)
        print("COMPARISON: Local Solve Methods (Standard vs Conflict-free)")
        print("="*70)

        results = {}

        # Standard local solve
        print("\n--- Standard Local Solve ---")
        results['standard'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_conflict_free=False
        )

        # Conflict-free SpMV (P5)
        print("\n--- Conflict-free SpMV (P5) ---")
        results['conflict_free'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_conflict_free=True
        )

        # Summary comparison
        print("\n" + "="*70)
        print("LOCAL SOLVE METHOD COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Method':<20} {'Local Solve (ms)':<18} {'Total (ms)':<15} {'Speedup':<10} {'SPD':<6}")
        print("-"*70)

        baseline = results['standard']['total_per_iter_ms']
        for name, res in results.items():
            solve_time = np.mean(res['timing']['local_solve'])
            total_time = res['total_per_iter_ms']
            speedup = baseline / total_time if total_time > 0 else 0
            spd = 'PASS' if res['spd_pass'] else 'FAIL'
            print(f"{res['solve_method']:<20} {solve_time:<18.3f} {total_time:<15.3f} {speedup:<10.2f}x {spd:<6}")

    elif args.compare_all:
        # Compare all optimization combinations
        print("\n" + "="*70)
        print("COMPARISON: All Optimization Combinations")
        print("="*70)

        results = {}

        # 1. Baseline: Cholesky + Standard SpMV + Warp Reduction
        print("\n--- [Baseline] Cholesky + P1 (Warp Reduction) ---")
        results['baseline'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=False,
            use_oneway_gj=False, use_conflict_free=False
        )

        # 2. IC(0) only
        print("\n--- [IC(0)] IC(0) + P1 ---")
        results['ic0'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=True,
            use_oneway_gj=False, use_conflict_free=False
        )

        # 3. IC(0) + Conflict-free SpMV
        print("\n--- [IC(0)+P5] IC(0) + P1 + Conflict-free SpMV ---")
        results['ic0_cf'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=True,
            use_oneway_gj=False, use_conflict_free=True
        )

        # 4. IC(0) + Banded SpMV (P6) - NEW
        print("\n--- [IC(0)+P6] IC(0) + P1 + Banded SpMV ---")
        results['ic0_banded'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=True,
            use_oneway_gj=False, use_conflict_free=False, use_banded=True
        )

        # 5. Cholesky + Conflict-free SpMV
        print("\n--- [P5] Cholesky + P1 + Conflict-free SpMV ---")
        results['chol_cf'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=True, use_cholesky=True,
            use_blocked=False, use_incomplete=False,
            use_oneway_gj=False, use_conflict_free=True
        )

        # 6. No warp reduction (for comparison)
        print("\n--- [No P1] Cholesky only (no warp reduction) ---")
        results['no_warp'] = benchmark_n_E_mas(
            args.demo, args.iterations,
            use_warp_reduction=False, use_cholesky=True,
            use_blocked=False, use_incomplete=False,
            use_oneway_gj=False, use_conflict_free=False
        )

        # Summary comparison
        print("\n" + "="*70)
        print("OPTIMIZATION COMBINATION COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Config':<30} {'Invert (ms)':<12} {'Restrict (ms)':<14} {'Solve (ms)':<12} {'Total (ms)':<12} {'Speedup':<8} {'SPD':<6}")
        print("-"*95)

        baseline_total = results['baseline']['total_per_iter_ms']
        config_names = {
            'baseline': 'Cholesky + P1',
            'ic0': 'IC(0) + P1',
            'ic0_cf': 'IC(0) + P1 + P5',
            'ic0_banded': 'IC(0) + P1 + P6',
            'chol_cf': 'Cholesky + P1 + P5',
            'no_warp': 'Cholesky (no P1)',
        }

        for key, res in results.items():
            invert_time = np.mean(res['timing']['invert'])
            restrict_time = np.mean(res['timing']['restrict'])
            solve_time = np.mean(res['timing']['local_solve'])
            total_time = res['total_per_iter_ms']
            speedup = baseline_total / total_time if total_time > 0 else 0
            spd = 'PASS' if res['spd_pass'] else 'FAIL'
            config_name = config_names.get(key, key)
            print(f"{config_name:<30} {invert_time:<12.3f} {restrict_time:<14.3f} {solve_time:<12.3f} {total_time:<12.3f} {speedup:<8.2f}x {spd:<6}")

        # Find best configuration
        best_key = min(results.keys(), key=lambda k: results[k]['total_per_iter_ms'] if results[k]['spd_pass'] else float('inf'))
        best_result = results[best_key]
        print("\n" + "="*70)
        print(f"BEST CONFIGURATION: {config_names.get(best_key, best_key)}")
        print(f"  Total time: {best_result['total_per_iter_ms']:.3f} ms")
        print(f"  Speedup vs baseline: {baseline_total / best_result['total_per_iter_ms']:.2f}x")
        print("="*70)

    else:
        benchmark_n_E_mas(
            args.demo,
            args.iterations,
            use_warp_reduction=not args.no_warp,
            use_cholesky=not args.gauss_jordan,
            use_blocked=args.blocked,
            use_incomplete=args.incomplete,
            use_oneway_gj=args.oneway_gj,
            use_conflict_free=args.conflict_free,
            use_banded=args.banded
        )
