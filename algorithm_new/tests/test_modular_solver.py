"""
Test the modular solver architecture.

This test validates:
1. SolverBuilder creates working solvers
2. Precision selection works (f32/f64)
3. Different preconditioners can be used
4. Simulation produces reasonable results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import taichi as ti
import numpy as np
import time


def test_solver_builder_diagonal():
    """Test SolverBuilder with diagonal preconditioner."""
    print("\n" + "="*60)
    print("Test: SolverBuilder with Diagonal Preconditioner")
    print("="*60)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_test")

    # Load mesh
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Build solver using new architecture
    from algorithm_new.solver import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f32')
        .with_mesh(model.mesh, density=model.density, E=model.mu*2*(1+0.3), nu=0.3)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('diagonal')
        .with_solver_params(dt=model.dt, epsilon=model.epsilon, iter_max=model.iter_max,
                           gravity=model.gravity)
        .build())

    # Get initial position
    pos_init = solver.get_positions().copy()
    y_init = pos_init[:, 1].mean()

    # Run simulation
    n_frames = 5
    for f in range(n_frames):
        t0 = time.perf_counter()
        n_iters = solver.step(verbose=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {n_iters} iters, {(t1-t0)*1000:.2f}ms")

    # Check that objects have fallen
    pos_final = solver.get_positions()
    y_final = pos_final[:, 1].mean()

    print(f"\nInitial Y: {y_init:.4f}")
    print(f"Final Y: {y_final:.4f}")
    print(f"Delta Y: {y_final - y_init:.4f}")

    assert y_final < y_init, "Object should have fallen (y decreased)"
    print("\nTest PASSED: Object fell as expected")


def test_solver_builder_mas():
    """Test SolverBuilder with MAS preconditioner."""
    print("\n" + "="*60)
    print("Test: SolverBuilder with MAS Preconditioner")
    print("="*60)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_test")

    # Load mesh
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Build solver using new architecture with MAS
    from algorithm_new.solver import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f32')
        .with_mesh(model.mesh, density=model.density, E=model.mu*2*(1+0.3), nu=0.3)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('mas', metis_reordered=True)
        .with_solver_params(dt=model.dt, epsilon=model.epsilon, iter_max=model.iter_max,
                           gravity=model.gravity)
        .build())

    # Get initial position
    pos_init = solver.get_positions().copy()
    y_init = pos_init[:, 1].mean()

    # Run simulation
    n_frames = 5
    for f in range(n_frames):
        t0 = time.perf_counter()
        n_iters = solver.step(verbose=True)
        t1 = time.perf_counter()
        print(f"Frame {f}: {n_iters} iters, {(t1-t0)*1000:.2f}ms")

    # Check that objects have fallen
    pos_final = solver.get_positions()
    y_final = pos_final[:, 1].mean()

    print(f"\nInitial Y: {y_init:.4f}")
    print(f"Final Y: {y_final:.4f}")
    print(f"Delta Y: {y_final - y_init:.4f}")

    assert y_final < y_init, "Object should have fallen (y decreased)"
    print("\nTest PASSED: Object fell as expected")


def test_preset_collision_free():
    """Test preset collision-free solver."""
    print("\n" + "="*60)
    print("Test: Preset Collision-Free Solver")
    print("="*60)

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32,
            offline_cache=True, offline_cache_file_path=".taichi_cache_test")

    # Load mesh
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Use preset function
    from algorithm_new.solver import create_collision_free_solver

    solver = create_collision_free_solver(
        mesh=model.mesh,
        density=model.density,
        E=model.mu*2*(1+0.3),
        nu=0.3,
        dt=model.dt,
        gravity=model.gravity,
        preconditioner='diagonal'
    )

    # Run a few frames
    solver.run(n_frames=3, verbose=False)

    print("\nTest PASSED: Preset solver ran successfully")


def test_precision_f64():
    """Test f64 precision configuration."""
    print("\n" + "="*60)
    print("Test: f64 Precision")
    print("="*60)

    # Initialize Taichi with f64
    ti.init(arch=ti.gpu, default_fp=ti.f64,
            offline_cache=True, offline_cache_file_path=".taichi_cache_test_f64")

    # Load mesh
    from util.model_loading import model_loading
    model = model_loading(demo='cube_freefall_10')

    # Build solver with f64
    from algorithm_new.solver import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f64')
        .with_mesh(model.mesh, density=model.density, E=model.mu*2*(1+0.3), nu=0.3)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('diagonal')
        .with_solver_params(dt=model.dt, epsilon=model.epsilon, gravity=model.gravity)
        .build())

    # Run a few frames
    n_frames = 3
    for f in range(n_frames):
        n_iters = solver.step(verbose=False)
        print(f"Frame {f}: {n_iters} iters")

    print("\nTest PASSED: f64 precision works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running All Modular Solver Tests")
    print("="*70)

    tests = [
        test_solver_builder_diagonal,
        # test_solver_builder_mas,  # Requires METIS reordering
        test_preset_collision_free,
        # test_precision_f64,  # Requires f64 Taichi init
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\nTest FAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    # Run single test for quick validation
    test_solver_builder_diagonal()
