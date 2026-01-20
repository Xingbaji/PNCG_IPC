"""
Demo: Modular Solver Architecture

This demo showcases the new modular solver architecture with:
1. SolverBuilder fluent API
2. Preset solver configurations
3. f32/f64 precision selection
4. Pluggable preconditioners

Usage:
    python demo_modular_solver.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import taichi as ti
import time


def demo_builder_api():
    """Demonstrate the SolverBuilder fluent API."""
    print("\n" + "="*70)
    print("Demo 1: SolverBuilder Fluent API")
    print("="*70)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_demo")

    # Load mesh using existing utility
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Build solver using fluent API
    from algorithm_new import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f32')
        .with_mesh(model.mesh, density=model.density, E=1e4, nu=0.3)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('diagonal')
        .with_solver_params(dt=0.04, epsilon=1e-5, iter_max=50, gravity=-9.8)
        .build())

    print("\nRunning simulation...")
    solver.run(n_frames=10, verbose=False)


def demo_presets():
    """Demonstrate preset solver configurations."""
    print("\n" + "="*70)
    print("Demo 2: Preset Solver Configurations")
    print("="*70)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_demo")

    # Load mesh
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Use preset function
    from algorithm_new import create_collision_free_solver

    solver = create_collision_free_solver(
        mesh=model.mesh,
        density=1000.0,
        E=1e4,
        nu=0.3,
        dt=0.04,
        gravity=-9.8,
        preconditioner='diagonal'
    )

    print("\nRunning simulation with preset configuration...")
    solver.run(n_frames=10, verbose=False)


def demo_mas_preconditioner():
    """Demonstrate MAS preconditioner."""
    print("\n" + "="*70)
    print("Demo 3: MAS Preconditioner")
    print("="*70)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_demo")

    # Load mesh (METIS reordering is applied by model_loading)
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Build solver with MAS preconditioner
    from algorithm_new import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f32')
        .with_mesh(model.mesh, density=model.density, E=1e5, nu=0.3)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('mas', metis_reordered=True)
        .with_solver_params(dt=0.04, epsilon=1e-5, iter_max=50, gravity=-9.8)
        .build())

    print("\nRunning simulation with MAS preconditioner...")

    # Run with timing
    total_time = 0
    total_iters = 0
    n_frames = 10

    for f in range(n_frames):
        t0 = time.perf_counter()
        n_iters = solver.step(verbose=(f == 0))  # Verbose for first frame only
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_iters += n_iters

    print(f"\nResults:")
    print(f"  Total frames: {n_frames}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg time per frame: {total_time/n_frames*1000:.2f}ms")
    print(f"  Total iterations: {total_iters}")
    print(f"  Avg iters per frame: {total_iters/n_frames:.1f}")


def demo_precision_comparison():
    """Compare f32 and f64 precision."""
    print("\n" + "="*70)
    print("Demo 4: Precision Comparison (f32 vs f64)")
    print("="*70)

    from util.model_loading import model_loading
    from algorithm_new import SolverBuilder

    results = {}

    for precision in ['f32', 'f64']:
        print(f"\n--- Testing {precision} precision ---")

        # Initialize Taichi with appropriate precision
        fp_type = ti.f32 if precision == 'f32' else ti.f64
        ti.init(arch=ti.gpu, default_fp=fp_type,
                offline_cache=True,
                offline_cache_file_path=f".taichi_cache_demo_{precision}")

        # Load mesh
        model = model_loading(demo='cube_freefall_10')

        # Build solver
        solver = (SolverBuilder()
            .with_precision(precision)
            .with_mesh(model.mesh, density=model.density, E=1e5, nu=0.3)
            .with_elastic_model('ARAP_filter')
            .with_preconditioner('diagonal')
            .with_solver_params(dt=0.04, epsilon=1e-5, gravity=-9.8)
            .build())

        # Run and time
        n_frames = 5
        total_time = 0
        total_iters = 0

        for f in range(n_frames):
            t0 = time.perf_counter()
            n_iters = solver.step(verbose=False)
            t1 = time.perf_counter()
            total_time += (t1 - t0)
            total_iters += n_iters

        results[precision] = {
            'time': total_time,
            'iters': total_iters,
            'avg_time': total_time / n_frames * 1000,
            'avg_iters': total_iters / n_frames,
        }

        print(f"  Avg time per frame: {results[precision]['avg_time']:.2f}ms")
        print(f"  Avg iters per frame: {results[precision]['avg_iters']:.1f}")

    # Compare
    print("\n--- Comparison ---")
    print(f"f32 avg time: {results['f32']['avg_time']:.2f}ms")
    print(f"f64 avg time: {results['f64']['avg_time']:.2f}ms")
    speedup = results['f64']['avg_time'] / results['f32']['avg_time']
    print(f"f32 is {speedup:.2f}x faster than f64")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Modular Solver Architecture Demo")
    print("="*70)

    demos = [
        ("Builder API", demo_builder_api),
        ("Preset Configurations", demo_presets),
        ("MAS Preconditioner", demo_mas_preconditioner),
        # ("Precision Comparison", demo_precision_comparison),  # Takes longer
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nDemo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)


if __name__ == '__main__':
    main()
