"""
Inversion Methods Benchmark Test

Compare different block matrix inversion methods:
- gauss_jordan: Full Gauss-Jordan elimination
- oneway_gj: One-way Gauss-Jordan (faster, less accurate)
- cholesky: Cholesky decomposition (requires SPD)
- incomplete: Incomplete Cholesky (requires SPD)

Tests both accuracy and performance.
"""

import sys
import os
import time
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner


# Available inversion methods
# Format: (key, name, use_cholesky, use_incomplete, use_oneway_gj)
INVERSION_METHODS = [
    ('gauss_jordan', 'Gauss-Jordan', False, False, False),
    ('oneway_gj', 'One-way GJ', False, False, True),
    ('cholesky', 'Cholesky', True, False, False),  # Requires SPD - may fail
    ('incomplete', 'Incomplete Cholesky', True, True, False),  # Requires SPD - may fail
]


@ti.data_oriented
class InversionTestSolver(pncg_ipc_deformer):
    """Solver for testing inversion methods."""

    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)

        # Add z field for preconditioned gradient
        self.mesh.verts.place({'z': ti.types.vector(3, float)})

        # Create MAS without METIS
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient."""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def compute_z_norm(self) -> float:
        """Compute L2 norm of z."""
        z_sum = 0.0
        for vert in self.mesh.verts:
            z_sum += vert.z.norm_sqr()
        return ti.sqrt(z_sum)

    @ti.kernel
    def check_z_has_nan(self) -> int:
        """Check if z contains NaN."""
        has_nan = 0
        for vert in self.mesh.verts:
            if ti.math.isnan(vert.z[0]) or ti.math.isnan(vert.z[1]) or ti.math.isnan(vert.z[2]):
                has_nan = 1
        return has_nan

    @ti.kernel
    def compute_gTz(self) -> float:
        """Compute g^T * z (should be positive for valid preconditioner)."""
        gTz = 0.0
        for vert in self.mesh.verts:
            gTz += vert.grad.dot(vert.z)
        return gTz

    @ti.kernel
    def compute_init_p_mas(self):
        """p = -z"""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    def run_single_step(self, method_name, use_cholesky, use_incomplete, use_oneway_gj):
        """Run a single optimization step with specified inversion method."""
        # Setup
        self.assign_xn_xhat()
        self.compute_grad_and_diagH()
        if self.ground_barrier == 1:
            self.add_grad_and_diagH_ground_barrier()

        grad_init = self.compute_grad_inf_norm()

        # Build hierarchy if needed
        if not self.mas.hierarchy_built:
            self.mas.build_hierarchy()

        # Assemble
        ti.sync()
        t_assemble_start = time.perf_counter()
        self.mas.assemble_block_matrices(self, use_full_hessian=True)
        ti.sync()
        t_assemble = (time.perf_counter() - t_assemble_start) * 1000

        # Invert
        ti.sync()
        t_invert_start = time.perf_counter()
        try:
            self.mas.invert_block_matrices(
                use_full_inversion=True,
                use_cholesky=use_cholesky,
                use_incomplete=use_incomplete,
                use_oneway_gj=use_oneway_gj
            )
            invert_success = True
        except Exception as e:
            print(f"    [ERROR] Inversion failed: {e}")
            invert_success = False
        ti.sync()
        t_invert = (time.perf_counter() - t_invert_start) * 1000

        if not invert_success:
            return None

        # Apply preconditioner
        ti.sync()
        t_apply_start = time.perf_counter()
        self.mas.apply()
        ti.sync()
        t_apply = (time.perf_counter() - t_apply_start) * 1000

        # Check results
        has_nan = self.check_z_has_nan()
        z_norm = self.compute_z_norm()
        gTz = self.compute_gTz()

        return {
            'method': method_name,
            'grad_init': grad_init,
            'z_norm': z_norm,
            'gTz': gTz,
            'has_nan': has_nan,
            't_assemble': t_assemble,
            't_invert': t_invert,
            't_apply': t_apply,
            't_total': t_assemble + t_invert + t_apply,
        }


def run_convergence_test(solver, method_name, use_cholesky, use_incomplete, use_oneway_gj,
                         max_iters=30, grad_tol=1e-4):
    """Run convergence test for a single method."""
    # Reset solver state
    solver.frame = 0
    solver.assign_xn_xhat()

    converged = False
    final_iter = max_iters
    grad_history = []

    for iter in range(max_iters):
        # Compute gradient
        solver.compute_grad_and_diagH()
        if solver.ground_barrier == 1:
            solver.add_grad_and_diagH_ground_barrier()

        grad_inf = solver.compute_grad_inf_norm()
        grad_history.append(grad_inf)

        if grad_inf < grad_tol:
            converged = True
            final_iter = iter
            break

        # Rebuild MAS at first iteration
        if iter == 0:
            if not solver.mas.hierarchy_built:
                solver.mas.build_hierarchy()
            solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
            solver.mas.invert_block_matrices(
                use_full_inversion=True,
                use_cholesky=use_cholesky,
                use_incomplete=use_incomplete,
                use_oneway_gj=use_oneway_gj
            )

        # Apply preconditioner
        solver.mas.apply()

        # Check for NaN
        if solver.check_z_has_nan():
            print(f"    [WARNING] NaN detected at iter {iter}")
            break

        # Search direction
        solver.compute_init_p_mas()

        # Line search
        alpha, gTp, pHp = solver.line_search_newton()
        p_max = solver.compute_p_inf_norm()
        if alpha * p_max > 0.5 * solver.dHat:
            alpha = 0.5 * solver.dHat / p_max

        solver.update_x(alpha)

    return {
        'method': method_name,
        'converged': converged,
        'iterations': final_iter + 1,
        'final_grad': grad_history[-1] if grad_history else float('inf'),
        'grad_history': grad_history,
    }


def run_multi_frame_benchmark(solver, method_name, use_cholesky, use_incomplete, use_oneway_gj,
                               frames=5, max_iters=30, grad_tol=1e-4):
    """Run multi-frame benchmark for timing."""
    solver.frame = 0
    solver.iter_max = max_iters

    iter_history = []
    time_history = []
    assemble_times = []
    invert_times = []
    apply_times = []
    inversion_failed = False

    # Build hierarchy once
    if not solver.mas.hierarchy_built:
        solver.mas.build_hierarchy()

    for f in range(frames):
        solver.assign_xn_xhat()

        ti.sync()
        t_frame_start = time.perf_counter()

        for iter in range(max_iters):
            solver.compute_grad_and_diagH()
            if solver.ground_barrier == 1:
                solver.add_grad_and_diagH_ground_barrier()

            grad_inf = solver.compute_grad_inf_norm()
            if grad_inf < grad_tol:
                break

            # MAS operations - rebuild every frame
            if iter == 0:
                ti.sync()
                t0 = time.perf_counter()
                solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
                ti.sync()
                t1 = time.perf_counter()

                try:
                    solver.mas.invert_block_matrices(
                        use_full_inversion=True,
                        use_cholesky=use_cholesky,
                        use_incomplete=use_incomplete,
                        use_oneway_gj=use_oneway_gj
                    )
                except Exception as e:
                    print(f"    [ERROR] Inversion failed at frame {f}: {e}")
                    inversion_failed = True
                    break

                ti.sync()
                t2 = time.perf_counter()

                assemble_times.append((t1 - t0) * 1000)
                invert_times.append((t2 - t1) * 1000)

            ti.sync()
            t_apply_start = time.perf_counter()
            solver.mas.apply()
            ti.sync()
            apply_times.append((time.perf_counter() - t_apply_start) * 1000)

            if solver.check_z_has_nan():
                print(f"    [WARNING] NaN at frame {f}, iter {iter}")
                break

            solver.compute_init_p_mas()
            alpha, gTp, pHp = solver.line_search_newton()
            p_max = solver.compute_p_inf_norm()
            if alpha * p_max > 0.5 * solver.dHat:
                alpha = 0.5 * solver.dHat / p_max
            solver.update_x(alpha)

        if inversion_failed:
            break

        solver.update_v_and_bound()

        ti.sync()
        frame_time = (time.perf_counter() - t_frame_start) * 1000

        iter_history.append(iter + 1)
        time_history.append(frame_time)
        solver.frame += 1

    return {
        'method': method_name,
        'frames': len(iter_history),
        'success': not inversion_failed and len(iter_history) == frames,
        'avg_iters': np.mean(iter_history) if iter_history else 0,
        'avg_frame_time': np.mean(time_history) if time_history else 0,
        'avg_assemble_time': np.mean(assemble_times) if assemble_times else 0,
        'avg_invert_time': np.mean(invert_times) if invert_times else 0,
        'avg_apply_time': np.mean(apply_times) if apply_times else 0,
        'iter_history': iter_history,
        'time_history': time_history,
    }


def main():
    print("=" * 70)
    print("MAS Preconditioner Inversion Methods Benchmark")
    print("=" * 70)

    # Initialize Taichi
    ti.init(arch=ti.cuda, default_fp=ti.f32)

    # Create solver
    print("\n[Setup] Creating solver...")
    solver = InversionTestSolver(demo='eight_E_stiffness_test')
    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  MAS levels: {solver.mas.level_num}")

    # =========================================================
    # Test 1: Single-step accuracy test
    # =========================================================
    print("\n" + "=" * 70)
    print("Test 1: Single-Step Accuracy Test")
    print("=" * 70)

    single_results = []
    for method_key, method_name, use_cholesky, use_incomplete, use_oneway_gj in INVERSION_METHODS:
        print(f"\n[{method_name}]")

        # Reset state
        solver.frame = 0
        result = solver.run_single_step(method_name, use_cholesky, use_incomplete, use_oneway_gj)

        if result:
            single_results.append(result)
            print(f"  |g|_init:  {result['grad_init']:.4e}")
            print(f"  |z|:       {result['z_norm']:.4e}")
            print(f"  g^T*z:     {result['gTz']:.4e} {'(OK)' if result['gTz'] > 0 else '(BAD - negative!)'}")
            print(f"  Has NaN:   {'Yes' if result['has_nan'] else 'No'}")
            print(f"  Timings:   assemble={result['t_assemble']:.1f}ms, invert={result['t_invert']:.1f}ms, apply={result['t_apply']:.1f}ms")
        else:
            print("  [FAILED]")

    # =========================================================
    # Test 2: Multi-frame benchmark
    # =========================================================
    print("\n" + "=" * 70)
    print("Test 2: Multi-Frame Performance Benchmark (5 frames)")
    print("=" * 70)

    benchmark_results = []
    for method_key, method_name, use_cholesky, use_incomplete, use_oneway_gj in INVERSION_METHODS:
        print(f"\n[{method_name}] Running benchmark...")

        # Reinitialize for clean state
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        solver = InversionTestSolver(demo='eight_E_stiffness_test')

        result = run_multi_frame_benchmark(
            solver, method_name, use_cholesky, use_incomplete, use_oneway_gj,
            frames=5, max_iters=30, grad_tol=1e-6  # Stricter tolerance for more iterations
        )
        benchmark_results.append(result)

        print(f"  Avg iterations: {result['avg_iters']:.1f}")
        print(f"  Avg frame time: {result['avg_frame_time']:.1f}ms ({1000/result['avg_frame_time']:.1f} FPS)")
        print(f"  Timing breakdown:")
        print(f"    - Assemble: {result['avg_assemble_time']:.1f}ms")
        print(f"    - Invert:   {result['avg_invert_time']:.1f}ms")
        print(f"    - Apply:    {result['avg_apply_time']:.1f}ms (per iter)")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n### Single-Step Results ###")
    print(f"{'Method':<20} {'|z|':<12} {'g^T*z':<12} {'NaN':<5} {'Total Time':<12}")
    print("-" * 65)
    for r in single_results:
        nan_str = "Yes" if r['has_nan'] else "No"
        gtz_sign = "+" if r['gTz'] > 0 else "-"
        print(f"{r['method']:<20} {r['z_norm']:<12.4e} {gtz_sign}{abs(r['gTz']):<11.4e} {nan_str:<5} {r['t_total']:<12.1f}ms")

    print("\n### Performance Benchmark (5 frames) ###")
    print(f"{'Method':<20} {'Status':<10} {'Iters':<8} {'Frame Time':<12} {'Invert Time':<12} {'FPS':<8}")
    print("-" * 75)
    for r in benchmark_results:
        status = "OK" if r.get('success', True) else "FAILED"
        if r['avg_frame_time'] > 0:
            fps = 1000 / r['avg_frame_time']
            print(f"{r['method']:<20} {status:<10} {r['avg_iters']:<8.1f} {r['avg_frame_time']:<12.1f}ms {r['avg_invert_time']:<12.1f}ms {fps:<8.1f}")
        else:
            print(f"{r['method']:<20} {status:<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<8}")

    # Recommendation
    print("\n### Recommendation ###")
    successful_results = [r for r in benchmark_results if r.get('success', True) and r['avg_frame_time'] > 0]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['avg_frame_time'])
        most_efficient = min(successful_results, key=lambda x: x['avg_invert_time'])
        print(f"  Fastest overall:    {fastest['method']} ({fastest['avg_frame_time']:.1f}ms/frame)")
        print(f"  Fastest inversion:  {most_efficient['method']} ({most_efficient['avg_invert_time']:.1f}ms)")
    else:
        print("  No successful methods found!")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
