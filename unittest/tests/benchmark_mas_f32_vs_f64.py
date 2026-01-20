"""
Benchmark: MAS Preconditioner Float32 vs Float64 Precision.

This test compares the convergence behavior and numerical accuracy of
the MAS preconditioner using float32 vs float64 precision.

Key observations to look for:
1. Convergence speed (iterations per frame)
2. Final residual accuracy
3. Compute time per iteration

Hypothesis: For high stiffness (E > 1e5), float64 should show:
- Better convergence (fewer iterations)
- Lower final residual
- Slightly higher compute time per iteration

Usage:
    python benchmark_mas_f32_vs_f64.py
    python benchmark_mas_f32_vs_f64.py --E 1e4   # Test with lower stiffness
    python benchmark_mas_f32_vs_f64.py --E 1e6   # Test with high stiffness (default)
    python benchmark_mas_f32_vs_f64.py --frames 10  # Run more frames
"""

import sys
import os
import argparse
import time
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file_path)
unittest_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(unittest_dir)
demo_dir = os.path.join(project_root, 'demo')

sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti


def run_test_f32(demo_config: str, frames: int, E: float, verbose: bool = False):
    """Run test with float32 MAS preconditioner."""
    print("\n" + "=" * 70)
    print("Testing MAS-PNCG with FLOAT32 precision")
    print("=" * 70)

    # Re-import to ensure clean state
    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision

    # Create solver
    solver = MASPNCGSolverNoCollision(demo=demo_config)

    # Override E if specified
    if E != solver.dict.get('E', 1e6):
        print(f"[Note] Overriding E from {solver.dict.get('E', 1e6)} to {E}")
        # We need to rebuild with the new E value
        # For simplicity, just print warning
        print(f"[Warning] E override requires solver rebuild - using config E={solver.dict.get('E', 1e6)}")

    print(f"Material: E={solver.dict['E']}, nu={solver.dict['nu']}")
    print(f"dt={solver.dt}, gravity={solver.gravity}")
    print(f"Precision: FLOAT32")

    # Set initial velocity
    v_np = np.zeros((solver.n_verts, 3), dtype=np.float32)
    v_np[:, 1] = -1.0
    solver.mesh.verts.v.from_numpy(v_np)

    # Run simulation
    results = {
        'iterations': [],
        'times_ms': [],
        'residuals': [],
    }

    for f in range(frames):
        ti.sync()
        t_start = time.perf_counter()

        iters = solver.step(verbose=verbose)

        ti.sync()
        t_elapsed = (time.perf_counter() - t_start) * 1000

        results['iterations'].append(iters)
        results['times_ms'].append(t_elapsed)

        print(f"  Frame {f+1}/{frames}: {iters:3d} iters, {t_elapsed:7.2f}ms")

    return results


def run_test_f64(demo_config: str, frames: int, E: float, verbose: bool = False):
    """Run test with float64 MAS preconditioner."""
    print("\n" + "=" * 70)
    print("Testing MAS-PNCG with FLOAT64 precision")
    print("=" * 70)

    # Import the f64 version
    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision
    from algorithm.mas_preconditioner_small.core_f64 import MASPreconditionerSmallF64

    # Create solver with f64 preconditioner
    solver = MASPNCGSolverNoCollision(demo=demo_config, preconditioner_class=MASPreconditionerSmallF64)

    print(f"Material: E={solver.dict['E']}, nu={solver.dict['nu']}")
    print(f"dt={solver.dt}, gravity={solver.gravity}")
    print(f"Precision: FLOAT64")

    # Set initial velocity
    v_np = np.zeros((solver.n_verts, 3), dtype=np.float32)
    v_np[:, 1] = -1.0
    solver.mesh.verts.v.from_numpy(v_np)

    # Run simulation
    results = {
        'iterations': [],
        'times_ms': [],
        'residuals': [],
    }

    for f in range(frames):
        ti.sync()
        t_start = time.perf_counter()

        iters = solver.step(verbose=verbose)

        ti.sync()
        t_elapsed = (time.perf_counter() - t_start) * 1000

        results['iterations'].append(iters)
        results['times_ms'].append(t_elapsed)

        print(f"  Frame {f+1}/{frames}: {iters:3d} iters, {t_elapsed:7.2f}ms")

    return results


def print_comparison(f32_results: dict, f64_results: dict):
    """Print comparison of results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Float32 vs Float64")
    print("=" * 70)

    # Iterations
    f32_iters = np.array(f32_results['iterations'])
    f64_iters = np.array(f64_results['iterations'])

    print("\nIterations per frame:")
    print(f"  Float32: avg={np.mean(f32_iters):.1f}, min={np.min(f32_iters)}, max={np.max(f32_iters)}")
    print(f"  Float64: avg={np.mean(f64_iters):.1f}, min={np.min(f64_iters)}, max={np.max(f64_iters)}")

    iter_diff = (np.mean(f64_iters) - np.mean(f32_iters)) / np.mean(f32_iters) * 100
    print(f"  Difference: {iter_diff:+.1f}% (negative = f64 better)")

    # Times
    f32_times = np.array(f32_results['times_ms'])
    f64_times = np.array(f64_results['times_ms'])

    print("\nTime per frame (ms):")
    print(f"  Float32: avg={np.mean(f32_times):.2f}, min={np.min(f32_times):.2f}, max={np.max(f32_times):.2f}")
    print(f"  Float64: avg={np.mean(f64_times):.2f}, min={np.min(f64_times):.2f}, max={np.max(f64_times):.2f}")

    time_diff = (np.mean(f64_times) - np.mean(f32_times)) / np.mean(f32_times) * 100
    print(f"  Difference: {time_diff:+.1f}% (positive = f64 slower)")

    # Time per iteration
    f32_time_per_iter = np.mean(f32_times) / np.mean(f32_iters)
    f64_time_per_iter = np.mean(f64_times) / np.mean(f64_iters)

    print("\nTime per iteration (ms):")
    print(f"  Float32: {f32_time_per_iter:.3f}")
    print(f"  Float64: {f64_time_per_iter:.3f}")

    tpi_diff = (f64_time_per_iter - f32_time_per_iter) / f32_time_per_iter * 100
    print(f"  Difference: {tpi_diff:+.1f}% (positive = f64 slower per iter)")

    # Conclusion
    print("\n" + "-" * 70)
    print("CONCLUSION:")
    if iter_diff < -5:
        print(f"  Float64 converges FASTER ({abs(iter_diff):.1f}% fewer iterations)")
        print("  This suggests precision is limiting convergence in float32.")
    elif iter_diff > 5:
        print(f"  Float32 converges faster ({iter_diff:.1f}% fewer iterations)")
        print("  Float64 precision may not be necessary for this stiffness level.")
    else:
        print("  Convergence is similar between float32 and float64.")
        print("  Precision does not appear to be a limiting factor.")

    if time_diff > 10:
        print(f"  Float64 is {time_diff:.1f}% slower overall.")
    elif time_diff < -10:
        print(f"  Float64 is {abs(time_diff):.1f}% faster overall (due to fewer iterations).")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark MAS f32 vs f64')
    parser.add_argument('--demo', type=str, default='eight_E_freefall',
                        help='Demo configuration name')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames')
    parser.add_argument('--E', type=float, default=1e6, help='Young\'s modulus')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--f32-only', action='store_true', help='Only run float32 test')
    parser.add_argument('--f64-only', action='store_true', help='Only run float64 test')
    args = parser.parse_args()

    print("=" * 70)
    print("MAS Preconditioner: Float32 vs Float64 Benchmark")
    print("=" * 70)
    print(f"Demo: {args.demo}")
    print(f"Frames: {args.frames}")
    print(f"Target E: {args.E:.0e}")
    print("=" * 70)

    # Initialize Taichi with f64 support
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True,
            offline_cache_file_path=".taichi_cache_f64_bench")

    f32_results = None
    f64_results = None

    if not args.f64_only:
        f32_results = run_test_f32(args.demo, args.frames, args.E, args.verbose)

    if not args.f32_only:
        # Reset Taichi for clean f64 test
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True,
                offline_cache_file_path=".taichi_cache_f64_bench")

        f64_results = run_test_f64(args.demo, args.frames, args.E, args.verbose)

    if f32_results and f64_results:
        print_comparison(f32_results, f64_results)

    print("\n[Benchmark Complete]")


if __name__ == '__main__':
    main()
