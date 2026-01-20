"""
Test MAS-PNCG Solver (No Collision) Convergence

Verifies:
1. Solver converges within max iterations
2. Energy decreases monotonically (or near-monotonically)
3. Gradient norm decreases to below epsilon
4. Physical correctness: free-fall motion matches expected trajectory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import taichi as ti
import numpy as np
import time


def test_convergence_basic():
    """Test basic convergence on cube freefall."""
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision

    print("\n" + "="*80)
    print("Test 1: Basic Convergence (cube_freefall_10)")
    print("="*80)

    solver = MASPNCGSolverNoCollision(demo='cube_freefall_10')

    # Run 10 frames
    total_iters = 0
    converged_frames = 0

    for frame in range(10):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True)
        t1 = time.perf_counter()

        total_iters += iters

        # Check if converged (not hit max iterations)
        if iters < solver.iter_max:
            converged_frames += 1

        print(f"Frame {frame}: {iters} iters, {(t1-t0)*1000:.2f}ms, "
              f"converged={'Yes' if iters < solver.iter_max else 'No'}")

    print(f"\n[Summary]")
    print(f"  Converged frames: {converged_frames}/{10}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Avg iters/frame: {total_iters/10:.1f}")

    # Test passes if at least 80% of frames converge
    success = converged_frames >= 8
    print(f"\n  Test 1 {'PASSED' if success else 'FAILED'}")
    return success


def test_energy_decrease():
    """Test that energy decreases during optimization."""
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision

    print("\n" + "="*80)
    print("Test 2: Energy Decrease Verification")
    print("="*80)

    solver = MASPNCGSolverNoCollision(demo='cube_freefall_10')

    # Manually step through one frame to track energy
    solver.assign_xn_xhat()

    energies = []
    grad_norms = []

    for iter in range(min(50, solver.iter_max)):
        # Compute gradient
        solver.compute_grad()

        # Check convergence
        grad_inf = solver.compute_grad_inf_norm()
        energy = solver.compute_energy()

        energies.append(energy)
        grad_norms.append(grad_inf)

        print(f"  iter {iter:>3}: E={energy:>12.6e}, |g|_inf={grad_inf:>10.4e}")

        if grad_inf < solver.epsilon:
            print(f"  => Converged at iter {iter}")
            break

        # Apply preconditioner
        solver.mas_preconditioner.rebuild(solver)
        solver.apply_preconditioner()
        solver.compute_Hv_z()

        # Compute search direction
        if iter == 0:
            solver.compute_init_search_direction()
        else:
            solver.cache_z_prev()
            solver.compute_subspace_scalars()
            mu, nu = solver.solve_2x2_subspace()
            solver.update_search_direction(mu, nu)

        # Compute w = H * p
        solver.compute_Hv_p()

        # Update position
        solver.update_x(1.0)

    # Check energy decrease
    energy_increases = 0
    for i in range(1, len(energies)):
        if energies[i] > energies[i-1] * 1.001:  # Allow 0.1% tolerance
            energy_increases += 1
            print(f"  Warning: Energy increased at iter {i}: {energies[i-1]:.6e} -> {energies[i]:.6e}")

    # Check gradient decrease trend
    final_grad = grad_norms[-1]
    initial_grad = grad_norms[0]
    grad_reduced = final_grad < initial_grad * 0.01  # At least 100x reduction

    print(f"\n[Summary]")
    print(f"  Initial energy: {energies[0]:.6e}")
    print(f"  Final energy:   {energies[-1]:.6e}")
    print(f"  Energy ratio:   {energies[-1]/energies[0]:.4f}")
    print(f"  Initial |g|:    {initial_grad:.6e}")
    print(f"  Final |g|:      {final_grad:.6e}")
    print(f"  Gradient ratio: {final_grad/initial_grad:.6e}")
    print(f"  Energy increases: {energy_increases}/{len(energies)-1}")

    # Test passes if:
    # 1. No significant energy increases
    # 2. Gradient reduced significantly
    success = energy_increases <= 2 and grad_reduced
    print(f"\n  Test 2 {'PASSED' if success else 'FAILED'}")
    return success


def test_freefall_physics():
    """Test physical correctness: center of mass follows freefall trajectory."""
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision

    print("\n" + "="*80)
    print("Test 3: Freefall Physics Validation")
    print("="*80)

    solver = MASPNCGSolverNoCollision(demo='cube_freefall_10')

    # Get initial center of mass
    x_np = solver.mesh.verts.x.to_numpy()
    com_init = x_np.mean(axis=0)
    print(f"  Initial COM: ({com_init[0]:.4f}, {com_init[1]:.4f}, {com_init[2]:.4f})")

    dt = solver.dt
    g = solver.gravity  # Negative value (e.g., -9.8)

    # Run 20 frames
    n_frames = 20
    com_history = [com_init.copy()]

    for frame in range(n_frames):
        solver.step(verbose=False)
        x_np = solver.mesh.verts.x.to_numpy()
        com = x_np.mean(axis=0)
        com_history.append(com.copy())

    com_history = np.array(com_history)

    # Expected freefall: y(t) = y0 + v0*t + 0.5*g*t^2
    # With v0 = 0, y(t) = y0 + 0.5*g*t^2
    t_total = n_frames * dt
    expected_y_drop = 0.5 * g * t_total**2  # g is negative, so this is negative
    actual_y_drop = com_history[-1, 1] - com_history[0, 1]

    print(f"\n  Simulation time: {t_total:.4f}s ({n_frames} frames x {dt:.4f}s)")
    print(f"  Gravity: {g} m/s^2")
    print(f"  Expected Y drop: {expected_y_drop:.6f}")
    print(f"  Actual Y drop:   {actual_y_drop:.6f}")
    print(f"  Error: {abs(actual_y_drop - expected_y_drop):.6f}")

    # Check X and Z drift (should be minimal)
    x_drift = abs(com_history[-1, 0] - com_history[0, 0])
    z_drift = abs(com_history[-1, 2] - com_history[0, 2])
    print(f"  X drift: {x_drift:.6f}")
    print(f"  Z drift: {z_drift:.6f}")

    # Test passes if:
    # 1. Y drop is within 5% of expected
    # 2. X and Z drift are minimal
    y_error_ratio = abs(actual_y_drop - expected_y_drop) / abs(expected_y_drop) if expected_y_drop != 0 else 0
    success = y_error_ratio < 0.05 and x_drift < 0.01 and z_drift < 0.01

    print(f"  Y error ratio: {y_error_ratio*100:.2f}%")
    print(f"\n  Test 3 {'PASSED' if success else 'FAILED'}")
    return success


def test_larger_mesh():
    """Test convergence on larger mesh (eight_E_freefall)."""
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision

    print("\n" + "="*80)
    print("Test 4: Larger Mesh Convergence (eight_E_freefall)")
    print("="*80)

    solver = MASPNCGSolverNoCollision(demo='eight_E_freefall')

    # Run 5 frames
    total_iters = 0
    converged_frames = 0

    for frame in range(5):
        t0 = time.perf_counter()
        iters = solver.step(verbose=True)
        t1 = time.perf_counter()

        total_iters += iters

        if iters < solver.iter_max:
            converged_frames += 1

        print(f"Frame {frame}: {iters} iters, {(t1-t0)*1000:.2f}ms")

    print(f"\n[Summary]")
    print(f"  Mesh size: {solver.n_verts} verts, {solver.n_cells} cells")
    print(f"  Converged frames: {converged_frames}/{5}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Avg iters/frame: {total_iters/5:.1f}")

    success = converged_frames >= 4
    print(f"\n  Test 4 {'PASSED' if success else 'FAILED'}")
    return success


def main():
    """Run all convergence tests."""
    print("\n" + "="*80)
    print("MAS-PNCG Solver (No Collision) Convergence Tests")
    print("="*80)

    results = {}

    # Test 1: Basic convergence
    try:
        results['basic_convergence'] = test_convergence_basic()
    except Exception as e:
        print(f"Test 1 FAILED with exception: {e}")
        results['basic_convergence'] = False

    # Test 2: Energy decrease
    try:
        results['energy_decrease'] = test_energy_decrease()
    except Exception as e:
        print(f"Test 2 FAILED with exception: {e}")
        results['energy_decrease'] = False

    # Test 3: Freefall physics
    try:
        results['freefall_physics'] = test_freefall_physics()
    except Exception as e:
        print(f"Test 3 FAILED with exception: {e}")
        results['freefall_physics'] = False

    # Test 4: Larger mesh
    try:
        results['larger_mesh'] = test_larger_mesh()
    except Exception as e:
        print(f"Test 4 FAILED with exception: {e}")
        results['larger_mesh'] = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
