"""
MAS Preconditioner Test Demo

Test MAS preconditioner replacing diagonal Jacobi preconditioner.
Uses 'eight_E_stiffness_test' demo configuration.
Initial version: collision-free (no contact handling).

Usage:
    python mas_preconditioner_test.py                      # Interactive mode
    python mas_preconditioner_test.py --fast --frames 50   # Fast benchmark mode
    python mas_preconditioner_test.py --compare --frames 20  # Compare MAS vs Diag
"""

import sys
import os
import time
import argparse
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner import MASPreconditioner


@ti.data_oriented
class MASPreconditionerTestSolver(pncg_ipc_deformer):
    """
    PNCG solver with MAS preconditioner for testing.

    This solver:
    - Uses MAS preconditioner instead of diagonal Jacobi
    - Disables collision detection for initial testing
    - Tracks iteration counts and timing for comparison
    """

    def __init__(self, demo='eight_E_stiffness_test', use_mas=True):
        # Initialize base solver
        super().__init__(demo=demo)

        self.use_mas = use_mas
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)

        # Per-vertex color for visualization
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self._init_colors()

        # Initialize MAS preconditioner if needed
        if self.use_mas:
            print(f"[MAS Test] Initializing MAS preconditioner...")
            # Add z field for preconditioned gradient if not already present
            if not hasattr(self.mesh.verts, 'z'):
                self.mesh.verts.place({'z': ti.types.vector(3, float)})

            # Get cells as numpy for METIS
            cells_np = self._get_cells_numpy()
            self.mas = MASPreconditioner(
                self.n_verts, self.n_cells, self.mesh,
                use_metis=True, cells_np=cells_np
            )
            print(f"[MAS Test] MAS preconditioner initialized with {self.mas.level_num} levels")
        else:
            self.mas = None
            print(f"[MAS Test] Using diagonal Jacobi preconditioner (baseline)")

        # Performance tracking
        self.iter_history = []
        self.time_history = []
        self.rebuild_count = 0

    def _get_cells_numpy(self):
        """Extract cell connectivity as numpy array."""
        n_cells = self.n_cells
        cells = np.zeros((n_cells, 4), dtype=np.int32)
        self._extract_cells_kernel(cells)
        return cells

    @ti.kernel
    def _extract_cells_kernel(self, cells: ti.types.ndarray()):
        """Kernel to extract cell connectivity."""
        for c in self.mesh.cells:
            cid = c.id
            for i in ti.static(range(4)):
                cells[cid, i] = c.verts[i].id

    @ti.kernel
    def _init_colors(self):
        """Initialize per-vertex colors based on object index."""
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            color_idx = index % 8
            if color_idx == 0:
                self.per_vertex_color[vert.id] = ti.Vector([1.0, 0.5, 0.0])  # Orange
            elif color_idx == 1:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.4, 0.8])  # Blue
            elif color_idx == 2:
                self.per_vertex_color[vert.id] = ti.Vector([0.3, 0.8, 0.3])  # Green
            elif color_idx == 3:
                self.per_vertex_color[vert.id] = ti.Vector([0.7, 0.3, 0.8])  # Purple
            elif color_idx == 4:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.8, 0.8])  # Cyan
            elif color_idx == 5:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.9, 0.2])  # Yellow
            elif color_idx == 6:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.2, 0.2])  # Red
            else:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.5, 0.7])  # Pink

    @ti.kernel
    def compute_init_p_mas(self):
        """Compute initial search direction using MAS preconditioned gradient: p = -z"""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_DK_direction_mas(self) -> float:
        """
        Compute Dai-Kai conjugate direction using MAS preconditioner.

        Standard DK formula:
            beta = (g^T P y - y^T P y * g^T p / y^T p) / y^T p
            p_{k+1} = -P g + beta * p_k

        Where P is approximated by diagH^{-1} for y (gradient difference).
        """
        g_p = 0.0   # g^T p
        g_Py = 0.0  # g^T (P y)
        y_p = 0.0   # y^T p
        y_Py = 0.0  # y^T (P y)

        for vert in self.mesh.verts:
            y = vert.grad - vert.grad_prev
            # Approximate P*y using diagonal Hessian (simple approximation)
            Py = y / vert.diagH

            y_p += y.dot(vert.p)
            g_Py += vert.grad.dot(Py)
            y_Py += y.dot(Py)
            g_p += vert.grad.dot(vert.p)

        # Compute beta (handle division by zero)
        beta = 0.0
        if ti.abs(y_p) > 1e-12:
            beta = (g_Py - y_Py * g_p / y_p) / y_p

        # Update search direction: p = -z + beta * p
        for vert in self.mesh.verts:
            vert.p = -vert.z + beta * vert.p

        return beta

    def step_collision_free(self):
        """
        Optimization step WITHOUT collision detection.
        Only elastic energy and ground barrier.
        """
        print(f'Frame {self.frame}')
        ti.sync()
        t_frame_start = time.perf_counter()

        self.assign_xn_xhat()

        for iter in range(self.iter_max):
            # Skip collision detection - collision free version
            # self.find_cnts(PRINT=False)

            # Compute gradient and diagonal Hessian (elastic + inertia)
            self.compute_grad_and_diagH()

            # Add ground barrier
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()

            # Apply preconditioner and compute search direction
            if self.use_mas and self.mas is not None:
                # MAS preconditioner
                if iter == 0:
                    # Full rebuild on first iteration
                    self.mas.rebuild(self)
                    self.rebuild_count += 1

                # Apply: z = P * grad
                self.mas.apply()

                # Compute search direction
                if iter == 0:
                    self.compute_init_p_mas()
                else:
                    self.compute_DK_direction_mas()
            else:
                # Diagonal Jacobi preconditioner (baseline)
                if iter == 0:
                    self.compute_init_p()
                else:
                    self.compute_DK_direction()

            # Line search (Newton step with clamping)
            alpha, gTp, pHp = self.line_search_newton()
            p_max = self.compute_p_inf_norm()

            if alpha * p_max > 0.5 * self.dHat:
                alpha = 0.5 * self.dHat / p_max

            self.update_x(alpha)

            # Convergence check
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E

            if delta_E < self.epsilon * delta_E_init:
                print(f'  Converged at iter {iter}, rate={delta_E/delta_E_init:.2e}')
                break

        self.update_v_and_bound()

        ti.sync()
        frame_time = (time.perf_counter() - t_frame_start) * 1000

        self.iter_history.append(iter + 1)
        self.time_history.append(frame_time)

        print(f'  Frame {self.frame}: {iter+1} iters, {frame_time:.2f}ms')
        self.frame += 1

        return iter + 1


def run_benchmark(use_mas=True, frames=50, demo='eight_E_stiffness_test'):
    """Run benchmark with specified preconditioner."""
    precond_name = "MAS" if use_mas else "Diagonal"
    print(f"\n{'='*60}")
    print(f"Benchmark: {precond_name} Preconditioner")
    print(f"Demo: {demo}, Frames: {frames}")
    print(f"{'='*60}")

    solver = MASPreconditionerTestSolver(demo=demo, use_mas=use_mas)

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  E (stiffness): {solver.dict['E']}")

    # Warmup
    print("\n[Warmup] Running first frame...")
    ti.sync()
    solver.step_collision_free()
    ti.sync()

    # Reset history after warmup
    solver.iter_history = []
    solver.time_history = []

    # Benchmark
    print(f"\n[Benchmark] Running {frames-1} frames...")
    for _ in range(frames - 1):
        solver.step_collision_free()

    # Summary
    avg_iters = np.mean(solver.iter_history)
    avg_time = np.mean(solver.time_history)
    total_iters = sum(solver.iter_history)

    print(f"\n{'='*60}")
    print(f"Results: {precond_name} Preconditioner")
    print(f"{'='*60}")
    print(f"  Frames: {len(solver.iter_history)}")
    print(f"  Avg iterations: {avg_iters:.1f}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Avg frame time: {avg_time:.2f}ms ({1000/avg_time:.1f} FPS)")
    if use_mas:
        print(f"  MAS rebuilds: {solver.rebuild_count}")
    print(f"{'='*60}")

    return {
        'preconditioner': precond_name,
        'avg_iters': avg_iters,
        'total_iters': total_iters,
        'avg_time_ms': avg_time,
        'iter_history': solver.iter_history,
        'time_history': solver.time_history,
    }


def run_comparison(frames=20, demo='eight_E_stiffness_test'):
    """Compare MAS vs Diagonal preconditioner."""
    print("\n" + "="*70)
    print("COMPARISON: MAS vs Diagonal Preconditioner")
    print("="*70)

    # Run diagonal baseline
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    diag_results = run_benchmark(use_mas=False, frames=frames, demo=demo)

    # Run MAS
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    mas_results = run_benchmark(use_mas=True, frames=frames, demo=demo)

    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Diagonal':<20} {'MAS':<20} {'Speedup':<15}")
    print("-"*70)

    iter_speedup = diag_results['avg_iters'] / mas_results['avg_iters']
    time_speedup = diag_results['avg_time_ms'] / mas_results['avg_time_ms']

    print(f"{'Avg iterations':<25} {diag_results['avg_iters']:<20.1f} {mas_results['avg_iters']:<20.1f} {iter_speedup:<15.2f}x")
    print(f"{'Total iterations':<25} {diag_results['total_iters']:<20} {mas_results['total_iters']:<20} {iter_speedup:<15.2f}x")
    print(f"{'Avg frame time (ms)':<25} {diag_results['avg_time_ms']:<20.2f} {mas_results['avg_time_ms']:<20.2f} {time_speedup:<15.2f}x")
    print("="*70)

    if iter_speedup > 1:
        print(f"\nMAS reduces iterations by {(1 - 1/iter_speedup)*100:.1f}%")
    else:
        print(f"\nDiagonal has {(1/iter_speedup - 1)*100:.1f}% fewer iterations")

    return diag_results, mas_results


def run_interactive(use_mas=True, demo='eight_E_stiffness_test'):
    """Run interactive demo with visualization."""
    from demo_runner import DemoRunner

    solver = MASPreconditionerTestSolver(demo=demo, use_mas=use_mas)

    class MASTestRunner(DemoRunner):
        def __init__(self, solver):
            precond_name = "MAS" if solver.use_mas else "Diagonal"
            super().__init__(solver, demo_name=f"MAS Test ({precond_name})")

        def get_per_vertex_color(self):
            return self.solver.per_vertex_color

        def step(self):
            return self.solver.step_collision_free()

    runner = MASTestRunner(solver)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAS Preconditioner Test')
    parser.add_argument('--fast', action='store_true', help='Fast benchmark mode (MAS only)')
    parser.add_argument('--compare', action='store_true', help='Compare MAS vs Diagonal')
    parser.add_argument('--diag', action='store_true', help='Use diagonal preconditioner instead of MAS')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_test', help='Demo name')
    args = parser.parse_args()

    if args.compare:
        # Comparison mode - don't init taichi here
        run_comparison(frames=args.frames, demo=args.demo)
    else:
        ti.init(arch=ti.gpu, default_fp=ti.f32)

        if args.fast:
            # Benchmark mode
            run_benchmark(use_mas=not args.diag, frames=args.frames, demo=args.demo)
        else:
            # Interactive mode
            run_interactive(use_mas=not args.diag, demo=args.demo)
