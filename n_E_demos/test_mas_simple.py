"""
Simple MAS Preconditioner Test

Test MAS preconditioner in collision-free scenario.
Uses 'eight_E_stiffness_test' demo configuration.
"""

import sys
import os
import time
import argparse
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
from algorithm.mas_preconditioner import MASPreconditioner


@ti.data_oriented
class SimpleMASTestSolver(pncg_ipc_deformer):
    """
    Simple PNCG solver with MAS preconditioner.
    Collision-free version for testing.
    """

    def __init__(self, demo='eight_E_stiffness_test', use_mas=True):
        super().__init__(demo=demo)

        self.use_mas = use_mas
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)

        # Per-vertex color
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self._init_colors()

        # Initialize MAS preconditioner if needed
        if self.use_mas:
            print(f"[MAS Test] Initializing MAS preconditioner...")
            # Add z field for preconditioned gradient
            self.mesh.verts.place({'z': ti.types.vector(3, float)})
            # Create MAS without METIS for simplicity
            self.mas = MASPreconditioner(
                self.n_verts, self.n_cells, self.mesh,
                use_metis=False
            )
            print(f"[MAS Test] MAS preconditioner initialized with {self.mas.level_num} levels")
        else:
            self.mas = None
            print(f"[MAS Test] Using diagonal Jacobi preconditioner (baseline)")

        # Performance tracking
        self.iter_history = []
        self.time_history = []
        self.rebuild_count = 0

    @ti.kernel
    def _init_colors(self):
        """Initialize per-vertex colors based on object index."""
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            color_idx = index % 8
            if color_idx == 0:
                self.per_vertex_color[vert.id] = ti.Vector([1.0, 0.5, 0.0])
            elif color_idx == 1:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.4, 0.8])
            elif color_idx == 2:
                self.per_vertex_color[vert.id] = ti.Vector([0.3, 0.8, 0.3])
            elif color_idx == 3:
                self.per_vertex_color[vert.id] = ti.Vector([0.7, 0.3, 0.8])
            elif color_idx == 4:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.8, 0.8])
            elif color_idx == 5:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.9, 0.2])
            elif color_idx == 6:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.2, 0.2])
            else:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.5, 0.7])

    @ti.kernel
    def compute_init_p_mas(self):
        """Compute initial search direction using MAS preconditioned gradient: p = -z"""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_DK_direction_mas(self):
        """Compute Dai-Kai conjugate direction using MAS preconditioner."""
        g_p = 0.0
        g_Py = 0.0
        y_p = 0.0
        y_Py = 0.0

        for vert in self.mesh.verts:
            y = vert.grad - vert.grad_prev
            Py = y / vert.diagH
            y_p += y.dot(vert.p)
            g_Py += vert.grad.dot(Py)
            y_Py += y.dot(Py)
            g_p += vert.grad.dot(vert.p)

        beta = 0.0
        if ti.abs(y_p) > 1e-12:
            beta = (g_Py - y_Py * g_p / y_p) / y_p

        for vert in self.mesh.verts:
            vert.p = -vert.z + beta * vert.p

    @ti.kernel
    def compute_z_norm(self) -> float:
        """Check z field norm."""
        z_sum = 0.0
        for vert in self.mesh.verts:
            z_sum += vert.z.norm()
        return z_sum

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient: max(|g_i|)"""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    def step_collision_free(self, use_grad_norm_stop=True, grad_tol=1e-3, verbose=True):
        """
        Optimization step WITHOUT collision detection.

        Args:
            use_grad_norm_stop: If True, use |g|_inf as stopping criterion
            grad_tol: Tolerance for gradient infinity norm
            verbose: Print iteration info
        """
        if verbose:
            print(f'Frame {self.frame}')
        ti.sync()
        t_frame_start = time.perf_counter()

        self.assign_xn_xhat()

        grad_inf_init = None

        for iter in range(self.iter_max):
            # Compute gradient and diagonal Hessian
            self.compute_grad_and_diagH()

            # Add ground barrier
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()

            # Check convergence using |g|_inf norm
            if use_grad_norm_stop:
                grad_inf = self.compute_grad_inf_norm()
                if iter == 0:
                    grad_inf_init = grad_inf
                if grad_inf < grad_tol:
                    if verbose:
                        print(f'  Converged at iter {iter}, |g|_inf={grad_inf:.2e} (init={grad_inf_init:.2e})')
                    break

            # Apply preconditioner and compute search direction
            if self.use_mas and self.mas is not None:
                if iter == 0:
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
                if iter == 0:
                    self.compute_init_p()
                else:
                    self.compute_DK_direction()

            # Line search
            alpha, gTp, pHp = self.line_search_newton()
            p_max = self.compute_p_inf_norm()

            if alpha * p_max > 0.5 * self.dHat:
                alpha = 0.5 * self.dHat / p_max

            self.update_x(alpha)

            # Old convergence check (delta_E based) - only if not using grad norm
            if not use_grad_norm_stop:
                delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
                if iter == 0:
                    delta_E_init = delta_E
                if delta_E < self.epsilon * delta_E_init:
                    if verbose:
                        print(f'  Converged at iter {iter}, rate={delta_E/delta_E_init:.2e}')
                    break

        self.update_v_and_bound()

        ti.sync()
        frame_time = (time.perf_counter() - t_frame_start) * 1000

        self.iter_history.append(iter + 1)
        self.time_history.append(frame_time)

        if verbose:
            if use_grad_norm_stop and grad_inf_init is not None:
                final_grad = self.compute_grad_inf_norm()
                print(f'  Frame {self.frame}: {iter+1} iters, {frame_time:.2f}ms, |g|_inf: {grad_inf_init:.2e} -> {final_grad:.2e}')
            else:
                print(f'  Frame {self.frame}: {iter+1} iters, {frame_time:.2f}ms')
        self.frame += 1

        return iter + 1


def run_benchmark(use_mas=True, frames=50, demo='eight_E_stiffness_test', iter_max=200, grad_tol=1e-5):
    """Run benchmark."""
    precond_name = "MAS" if use_mas else "Diagonal"
    print(f"\n{'='*60}")
    print(f"Benchmark: {precond_name} Preconditioner")
    print(f"Demo: {demo}, Frames: {frames}, iter_max: {iter_max}, grad_tol: {grad_tol}")
    print(f"{'='*60}")

    solver = SimpleMASTestSolver(demo=demo, use_mas=use_mas)

    # Override iter_max for testing
    solver.iter_max = iter_max
    solver.grad_tol = grad_tol

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  E (stiffness): {solver.dict['E']}")

    # Warmup
    print("\n[Warmup] Running first frame...")
    ti.sync()
    solver.step_collision_free(grad_tol=grad_tol)
    ti.sync()

    # Reset history after warmup
    solver.iter_history = []
    solver.time_history = []

    # Benchmark
    print(f"\n[Benchmark] Running {frames-1} frames...")
    for _ in range(frames - 1):
        solver.step_collision_free(grad_tol=grad_tol)

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
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple MAS Preconditioner Test')
    parser.add_argument('--diag', action='store_true', help='Use diagonal preconditioner')
    parser.add_argument('--frames', type=int, default=20, help='Number of frames')
    parser.add_argument('--iter_max', type=int, default=200, help='Max iterations per frame')
    parser.add_argument('--grad_tol', type=float, default=1e-5, help='Gradient inf norm tolerance')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_test', help='Demo name')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    run_benchmark(use_mas=not args.diag, frames=args.frames, demo=args.demo,
                  iter_max=args.iter_max, grad_tol=args.grad_tol)
