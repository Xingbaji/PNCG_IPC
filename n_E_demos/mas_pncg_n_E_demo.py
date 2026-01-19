"""
MAS-PNCG n_E Demo - Uses MAS preconditioner with PNCG solver.

This version uses the full MAS-PNCG solver with:
- Multilevel Additive Schwarz (MAS) preconditioner
- 2D subspace minimization
- Powell's restart criterion
- Sparse-Input Woodbury updates

Usage:
    python mas_pncg_n_E_demo.py                     # Interactive mode
    python mas_pncg_n_E_demo.py --headless --frames 50  # Headless with images
    python mas_pncg_n_E_demo.py --fast --frames 50      # Fast mode (no rendering)
    python mas_pncg_n_E_demo.py --profile --frames 20   # Profile mode with detailed timing
"""

import sys
import os
import time
import argparse
import json
from datetime import datetime
from collections import defaultdict

current_file_path = os.path.abspath(__file__)
# Go up one level: n_E_demos -> PNCG_IPC
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
# Also add demo folder for demo_runner import
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
# Logs directory
logs_dir = os.path.join(os.path.dirname(current_file_path), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Change to demo directory for correct relative path resolution (../model/mesh/...)
os.chdir(demo_dir)

import taichi as ti
from algorithm.mas_pncg_solver import MASPNCGSolver

VERSION_NAME = "mas_pncg"
COLLISION_METHOD = "bvh"
BARRIER_TYPE = "cubic"


class PerformanceLogger:
    """Logger for detailed performance timing."""

    def __init__(self, version_name, demo_name):
        self.version_name = version_name
        self.demo_name = demo_name
        self.frame_logs = []
        self.current_frame = {}
        self.current_iter = {}

    def start_frame(self, frame_id):
        self.current_frame = {
            'frame_id': frame_id,
            'iterations': [],
            'total_time_ms': 0,
        }

    def start_iteration(self, iter_id):
        self.current_iter = {
            'iter_id': iter_id,
            'find_cnts_ms': 0,
            'compute_grad_diagH_ms': 0,
            'preconditioner_ms': 0,
            'search_direction_ms': 0,
            'line_search_ms': 0,
            'total_ms': 0,
        }

    def log_time(self, key, time_ms):
        self.current_iter[key] = time_ms

    def end_iteration(self, total_ms):
        self.current_iter['total_ms'] = total_ms
        self.current_frame['iterations'].append(self.current_iter)

    def end_frame(self, total_ms, n_iters):
        self.current_frame['total_time_ms'] = total_ms
        self.current_frame['n_iterations'] = n_iters
        self.frame_logs.append(self.current_frame)

    def get_summary(self):
        """Compute summary statistics."""
        if not self.frame_logs:
            return {}

        # Per-frame stats
        frame_times = [f['total_time_ms'] for f in self.frame_logs]
        iter_counts = [f['n_iterations'] for f in self.frame_logs]

        # Per-component stats (aggregate across all iterations)
        component_times = defaultdict(list)
        for frame in self.frame_logs:
            for it in frame['iterations']:
                for key in ['find_cnts_ms', 'compute_grad_diagH_ms', 'preconditioner_ms',
                           'search_direction_ms', 'line_search_ms']:
                    if key in it:
                        component_times[key].append(it[key])

        import numpy as np
        summary = {
            'version': self.version_name,
            'demo': self.demo_name,
            'n_frames': len(self.frame_logs),
            'frame_time': {
                'avg_ms': float(np.mean(frame_times)),
                'min_ms': float(np.min(frame_times)),
                'max_ms': float(np.max(frame_times)),
                'std_ms': float(np.std(frame_times)),
            },
            'iterations': {
                'avg': float(np.mean(iter_counts)),
                'min': int(np.min(iter_counts)),
                'max': int(np.max(iter_counts)),
            },
            'components': {}
        }

        for key, times in component_times.items():
            if times:
                name = key.replace('_ms', '')
                summary['components'][name] = {
                    'avg_ms': float(np.mean(times)),
                    'total_ms': float(np.sum(times)),
                    'percent': float(np.sum(times) / np.sum(frame_times) * 100),
                }

        return summary

    def save_log(self, filename=None):
        """Save detailed log to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(logs_dir, f'{self.version_name}_{self.demo_name}_{timestamp}.json')

        summary = self.get_summary()
        log_data = {
            'summary': summary,
            'frames': self.frame_logs,
        }

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        return filename

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        if not summary:
            print("No data to summarize")
            return

        print(f"\n{'='*70}")
        print(f"Performance Summary: {self.version_name} ({self.demo_name})")
        print(f"{'='*70}")
        print(f"Frames: {summary['n_frames']}")
        print(f"Frame time: {summary['frame_time']['avg_ms']:.2f}ms avg "
              f"({1000/summary['frame_time']['avg_ms']:.1f} FPS)")
        print(f"           {summary['frame_time']['min_ms']:.2f}ms min, "
              f"{summary['frame_time']['max_ms']:.2f}ms max")
        print(f"Iterations: {summary['iterations']['avg']:.1f} avg "
              f"({summary['iterations']['min']}-{summary['iterations']['max']})")

        print(f"\n{'Component':<25} {'Avg (ms)':<12} {'Total (ms)':<12} {'Percent':<10}")
        print('-' * 60)
        for name, stats in summary['components'].items():
            print(f"{name:<25} {stats['avg_ms']:<12.3f} {stats['total_ms']:<12.1f} {stats['percent']:<10.1f}%")
        print(f"{'='*70}")


@ti.data_oriented
class MASPNCGIndexSolver(MASPNCGSolver):
    """
    MAS-PNCG solver with per-object indexing for multi-object scenes.
    """

    def set_index(self):
        """Initialize per-object index and colors."""
        self.mesh.verts.place({'index': ti.i32})
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self.object_size = 1046  # Vertices per E object
        self.N_object = int(self.n_verts / self.object_size)
        self.init_index()
        self.bvh_initialized = False

    @ti.kernel
    def init_index(self):
        """Assign index and color to each vertex based on object."""
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            vert.index = index
            # Assign color based on object index
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

    def step_with_logging(self, logger=None):
        """
        MAS-PNCG step with optional performance logging.
        """
        print(f'Frame {self.frame}')
        self.assign_xn_xhat()

        do_restart = True
        use_woodbury = False

        n_iters = 0
        for iter in range(self.iter_max):
            if logger:
                logger.start_iteration(iter)
                ti.sync()
                t_iter_start = time.perf_counter()

            # Step 1: Find contacts
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.find_cnts(PRINT=False)
            if logger:
                ti.sync()
                logger.log_time('find_cnts_ms', (time.perf_counter() - t0) * 1000)

            # Step 2: Compute gradient and diagonal Hessian
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.compute_grad_and_diagH()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()
            if logger:
                ti.sync()
                logger.log_time('compute_grad_diagH_ms', (time.perf_counter() - t0) * 1000)

            # Step 3: Preconditioner
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            if do_restart:
                self.mas_preconditioner.rebuild(self)
                if not hasattr(self.mas_preconditioner, 'woodbury_initialized') or \
                   not self.mas_preconditioner.woodbury_initialized:
                    self.mas_preconditioner.init_woodbury_structures()
                self.mas_preconditioner.save_base_contact_state(self)
                use_woodbury = True
            else:
                if use_woodbury:
                    self.mas_preconditioner.woodbury_update(self)
            if logger:
                ti.sync()
                logger.log_time('preconditioner_ms', (time.perf_counter() - t0) * 1000)

            # Step 4: Apply preconditioner
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            if do_restart or not use_woodbury:
                self.mas_preconditioner.apply()
            else:
                self.mas_preconditioner.apply_with_woodbury()

            # Step 5: Compute Hv = H * z
            self.compute_Hv(True)

            # Step 6: Compute search direction
            if iter == 0 or do_restart:
                self.compute_init_search_direction()
                self.compute_Hv(False)
            else:
                self.compute_subspace_scalars()
                mu, nu = self.solve_2x2_subspace()
                self.update_search_direction(mu, nu)
            if logger:
                ti.sync()
                logger.log_time('search_direction_ms', (time.perf_counter() - t0) * 1000)

            # Step 7: Line search
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            alpha, gTp, pHp = self.line_search()

            # Step 8: CCD clamping
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha = 0.5 * self.dHat / p_max

            # Step 9: Update position
            self.update_x(alpha)
            if logger:
                ti.sync()
                logger.log_time('line_search_ms', (time.perf_counter() - t0) * 1000)
                logger.end_iteration((time.perf_counter() - t_iter_start) * 1000)

            # Step 10: Convergence check
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E

            n_iters = iter + 1
            if delta_E < self.epsilon * delta_E_init:
                print(f'converged at iter {iter}, rate={delta_E/delta_E_init:.6f}')
                break

            # Powell's restart criterion
            if iter > 0:
                self.compute_powell_scalars()
                do_restart = self.check_powell_restart()
            else:
                do_restart = False

            # Cache z for next iteration's Powell check
            self.cache_z_prev()

        self.update_v_and_bound()
        self.frame += 1
        return n_iters


class MASPNCGNEDemoRunner:
    """Demo runner for MAS-PNCG n_E demo."""

    def __init__(self, demo='eight_E_stiffness_mas'):
        from demo_runner import DemoRunner
        self.solver = MASPNCGIndexSolver(demo=demo)
        self.solver.set_index()
        self.demo_name = f"n_E-mas_pncg ({demo})"
        self.runner = DemoRunner(self.solver, demo_name=self.demo_name)

        self.version_name = VERSION_NAME
        self.collision_method = COLLISION_METHOD
        self.barrier_type = BARRIER_TYPE

    def get_per_vertex_color(self):
        return self.solver.per_vertex_color

    def run(self):
        self.runner.get_per_vertex_color = self.get_per_vertex_color
        self.runner.run()

    def get_config(self):
        return {
            'version': self.version_name,
            'collision_method': self.collision_method,
            'barrier_type': self.barrier_type,
            'demo': self.solver.demo,
            'n_verts': self.solver.n_verts,
            'n_cells': self.solver.n_cells,
            'n_objects': self.solver.N_object,
        }


def run_fast_mode(frames=50, demo='eight_E_stiffness_mas'):
    """Run in fast mode without rendering overhead."""
    print(f"\n{'='*60}")
    print(f"n_E-mas_pncg (Fast Mode)")
    print(f"{'='*60}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames without rendering overhead...")

    solver = MASPNCGIndexSolver(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  MAS Levels: {solver.mas_preconditioner.n_levels}")

    # Warmup
    print("\n[Warmup] Running first frame...")
    ti.sync()
    solver.step()
    ti.sync()
    print("[Warmup] Done\n")

    # Benchmark
    frame_times = []
    iter_counts = []

    print(f"[Benchmark] Running {frames-1} frames...")
    for i in range(1, frames):
        ti.sync()
        t0 = time.perf_counter()
        iters = solver.step()
        ti.sync()
        t1 = time.perf_counter()

        frame_ms = (t1 - t0) * 1000
        frame_times.append(frame_ms)
        iter_counts.append(iters)

        fps = 1000.0 / frame_ms if frame_ms > 0 else 0
        print(f"  Frame {i}: {frame_ms:.2f}ms ({fps:.1f} FPS), iters={iters}")

    # Summary
    import numpy as np
    avg_time = np.mean(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)
    avg_fps = 1000.0 / avg_time
    avg_iters = np.mean(iter_counts)

    print(f"\n{'='*60}")
    print(f"Summary (excluding warmup frame)")
    print(f"{'='*60}")
    print(f"  Frames: {len(frame_times)}")
    print(f"  Avg time: {avg_time:.2f}ms ({avg_fps:.1f} FPS)")
    print(f"  Min time: {min_time:.2f}ms ({1000/min_time:.1f} FPS)")
    print(f"  Max time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)")
    print(f"  Avg iterations: {avg_iters:.1f}")
    print(f"{'='*60}")

    return {
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'avg_fps': avg_fps,
        'avg_iters': avg_iters,
        'frame_times': frame_times,
        'iter_counts': iter_counts
    }


def run_profile_mode(frames=20, demo='eight_E_stiffness_mas'):
    """Run in profile mode with detailed per-component timing."""
    print(f"\n{'='*70}")
    print(f"n_E-mas_pncg (Profile Mode)")
    print(f"{'='*70}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames with detailed timing...")

    solver = MASPNCGIndexSolver(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  MAS Levels: {solver.mas_preconditioner.n_levels}")

    logger = PerformanceLogger(VERSION_NAME, demo)

    # Warmup
    print("\n[Warmup] Running first frame (not logged)...")
    ti.sync()
    solver.step()
    ti.sync()
    print("[Warmup] Done\n")

    # Profile
    print(f"[Profile] Running {frames-1} frames with detailed timing...")
    for i in range(1, frames):
        ti.sync()
        t_frame_start = time.perf_counter()

        logger.start_frame(i)
        iters = solver.step_with_logging(logger=logger)
        ti.sync()

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        logger.end_frame(frame_ms, iters)

        fps = 1000.0 / frame_ms if frame_ms > 0 else 0
        print(f"  Frame {i}: {frame_ms:.2f}ms ({fps:.1f} FPS), iters={iters}")

    # Summary
    logger.print_summary()
    log_file = logger.save_log()
    print(f"\nDetailed log saved to: {log_file}")

    return logger


def run_comparison_mode(frames=20, demo='eight_E_stiffness_mas'):
    """
    Compare MAS-PNCG vs standard diagonal preconditioner.
    """
    print(f"\n{'='*70}")
    print(f"MAS-PNCG vs Diagonal Preconditioner Comparison")
    print(f"{'='*70}")
    print(f"Demo: {demo}, Frames: {frames}")

    # Run MAS-PNCG
    print("\n--- MAS-PNCG ---")
    mas_results = run_fast_mode(frames=frames, demo=demo)

    # Run standard BVH (diagonal preconditioner)
    print("\n--- Standard (Diagonal Preconditioner) ---")
    from bvh_n_E_demo import run_fast_mode as run_bvh_fast
    # Change demo to non-MAS version
    demo_standard = demo.replace('_mas', '')
    if demo_standard == demo:
        demo_standard = 'eight_E_drop_demo_contact'
    bvh_results = run_fast_mode(frames=frames, demo=demo_standard)

    # Comparison
    print(f"\n{'='*70}")
    print(f"Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'MAS-PNCG':<15} {'Diagonal':<15} {'Speedup':<10}")
    print('-' * 65)
    print(f"{'Avg frame time (ms)':<25} {mas_results['avg_ms']:<15.2f} {bvh_results['avg_ms']:<15.2f} "
          f"{bvh_results['avg_ms']/mas_results['avg_ms']:<10.2f}x")
    print(f"{'Avg iterations':<25} {mas_results['avg_iters']:<15.1f} {bvh_results['avg_iters']:<15.1f} "
          f"{bvh_results['avg_iters']/mas_results['avg_iters']:<10.2f}x")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAS-PNCG n_E Demo')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode with image saving')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (no rendering, accurate timing)')
    parser.add_argument('--profile', action='store_true', help='Run in profile mode with detailed timing')
    parser.add_argument('--compare', action='store_true', help='Compare MAS-PNCG vs diagonal preconditioner')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to run')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_mas', help='Demo name')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    if args.compare:
        run_comparison_mode(frames=args.frames, demo=args.demo)
    elif args.profile:
        run_profile_mode(frames=args.frames, demo=args.demo)
    elif args.fast:
        run_fast_mode(frames=args.frames, demo=args.demo)
    else:
        runner = MASPNCGNEDemoRunner(demo=args.demo)
        runner.run()
