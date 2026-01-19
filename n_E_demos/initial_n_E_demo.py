"""
Initial/Reference n_E Demo - Based on PNCG_IPC_init version.

This is the reference implementation using the ORIGINAL spatial hashing
collision detection (same as PNCG_IPC_init). Uses logarithmic barrier (default).

Usage:
    python initial_n_E_demo.py                     # Interactive mode
    python initial_n_E_demo.py --headless --frames 50  # Headless with images
    python initial_n_E_demo.py --fast --frames 50      # Fast mode (no rendering)
    python initial_n_E_demo.py --profile --frames 20   # Profile mode with detailed timing
"""

import sys
import os
import time
import argparse

current_file_path = os.path.abspath(__file__)
# Go up one level: n_E_demos -> PNCG_IPC
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
# Also add demo folder for demo_runner import
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
# Add n_E_demos folder for perf_logger
n_e_demos_dir = os.path.dirname(current_file_path)
sys.path.append(n_e_demos_dir)

# Change to demo directory for correct relative path resolution (../model/mesh/...)
os.chdir(demo_dir)

import taichi as ti
# Use the ORIGINAL spatial hashing collision detection (same as PNCG_IPC_init)
from algorithm.pncg_base_ipc_spatial_hash import *
from perf_logger import PerformanceLogger

VERSION_NAME = "initial"
COLLISION_METHOD = "spatial_hash"  # Original uses spatial hashing, NOT BVH
BARRIER_TYPE = "log"


@ti.data_oriented
class pncg_index_initial(pncg_ipc_deformer):
    """
    Initial/reference version of per-object optimization solver.
    Based on the original n_E_demo from PNCG_IPC_init.
    """

    def set_index(self):
        # Initialize model
        self.mesh.verts.place({'index': ti.i32})
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)
        self.init_index()
        self.frame = 0
        self.CNT = 0

    @ti.kernel
    def init_index(self):
        # Setting index for each vertex
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            vert.index = index

    @ti.kernel
    def compute_DK_index(self):
        # Compute DK direction for each object
        g_Py_index = ti.Vector.zero(float, self.N_object)
        y_p_index = ti.Vector.zero(float, self.N_object)
        y_Py_index = ti.Vector.zero(float, self.N_object)
        g_p_index = ti.Vector.zero(float, self.N_object)
        beta_index = ti.Vector.zero(float, self.N_object)

        for vert in self.mesh.verts:
            index = vert.index
            y = vert.grad - vert.grad_prev
            Py = y / vert.diagH
            y_p_index[index] += y.dot(vert.p)
            g_Py_index[index] += vert.grad.dot(Py)
            y_Py_index[index] += y.dot(Py)
            g_p_index[index] += vert.grad.dot(vert.p)

        for i in range(self.N_object):
            beta_index[i] = (g_Py_index[i] - y_Py_index[i] * g_p_index[i] / y_p_index[i]) / y_p_index[i]

        for vert in self.mesh.verts:
            index = vert.index
            vert.p = -vert.grad / vert.diagH + beta_index[index] * vert.p

    @ti.kernel
    def compute_alpha_index_and_update_x(self) -> float:
        gTp_index = ti.Vector.zero(float, self.N_object)
        pHp_index = ti.Vector.zero(float, self.N_object)
        alpha_index = ti.Vector.zero(float, self.N_object)
        p_max_index = ti.Vector.zero(float, self.N_object)

        for vert in self.mesh.verts:
            index = vert.index
            gTp_index[index] += vert.grad.dot(vert.p)

        for vert in self.mesh.verts:
            index = vert.index
            pHp_index[index] += vert.p.norm_sqr() * vert.m

        for c in self.mesh.cells:
            index = c.verts[0].index
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            p = ti.Vector.zero(float, 12)
            p[0:3] = c.verts[0].p
            p[3:6] = c.verts[1].p
            p[6:9] = c.verts[2].p
            p[9:12] = c.verts[3].p
            tmp = self.compute_p_d2Psidx2_p(F, B, p, self.mu, self.la)
            pHp_index[index] += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        for k, j in self.cid:
            pair = self.cid[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist
            bg = self.barrier_g(dist)
            para1 = bg / dist
            para0 = (self.barrier_H(dist) - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)
            d_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * d_dtdx.norm_sqr()
            pHp = ti.max(pHp_0 + pHp_1, 0.0)
            index0 = self.mesh.verts.index[ids[0]]
            index1 = self.mesh.verts.index[ids[2]]
            pHp_index[index0] += pHp * 0.5
            pHp_index[index1] += pHp * 0.5

        min_dist = 1e-2 * self.dHat
        for i in range(self.boundary_points.shape[0]):
            p = self.boundary_points[i]
            index = self.mesh.verts.index[p]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    dist = min_dist
                p_tmp = self.mesh.verts.p[p][1]
                ret_value = p_tmp * self.barrier_H(dist) * p_tmp
                pHp_index[index] += ret_value

        for vert in self.mesh.verts:
            index = vert.index
            d_norm = vert.p.norm()
            ti.atomic_max(p_max_index[index], d_norm)

        Delta_E = 0.0
        for i in range(self.N_object):
            alpha_i = -gTp_index[i] / pHp_index[i]
            if alpha_i * p_max_index[i] > 0.5 * self.dHat:
                alpha_index[i] = 0.5 * self.dHat / p_max_index[i]
            else:
                alpha_index[i] = alpha_i
            Delta_E -= (alpha_index[i] * gTp_index[i] + 0.5 * alpha_index[i] ** 2 * pHp_index[i])

        for vert in self.mesh.verts:
            index = vert.index
            alpha = alpha_index[index]
            vert.x += alpha * vert.p

        return Delta_E

    def step(self, logger=None):
        print('Frame', self.frame)
        self.assign_xn_xhat()
        for iter in range(self.iter_max):
            if logger:
                logger.start_iteration(iter)
                ti.sync()
                t_iter_start = time.perf_counter()

            # Collision detection
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.find_cnts()
            if logger:
                ti.sync()
                logger.log_time('find_cnts_ms', (time.perf_counter() - t0) * 1000)

            # Gradient and diagonal Hessian
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.compute_grad_and_diagH()
            if logger:
                ti.sync()
                logger.log_time('compute_grad_diagH_ms', (time.perf_counter() - t0) * 1000)

            # Ground barrier
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()
            if logger:
                ti.sync()
                logger.log_time('ground_barrier_ms', (time.perf_counter() - t0) * 1000)

            # Search direction
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_index()
            if logger:
                ti.sync()
                logger.log_time('compute_direction_ms', (time.perf_counter() - t0) * 1000)

            # Line search / update
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            delta_E = self.compute_alpha_index_and_update_x()
            if logger:
                ti.sync()
                logger.log_time('line_search_ms', (time.perf_counter() - t0) * 1000)
                logger.end_iteration((time.perf_counter() - t_iter_start) * 1000)

            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                break

        print('finish at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E)
        self.update_v_and_bound()
        self.frame += 1
        return iter


class InitialNEDemoRunner:
    """Demo runner for initial n_E demo using DemoRunner framework."""

    def __init__(self, demo='eight_E_drop_demo_contact'):
        from demo_runner import DemoRunner
        self.solver = pncg_index_initial(demo=demo)
        self.solver.set_index()
        self.demo_name = f"n_E-initial ({demo})"
        self.runner = DemoRunner(self.solver, demo_name=self.demo_name)

        # Store version info
        self.version_name = VERSION_NAME
        self.collision_method = COLLISION_METHOD
        self.barrier_type = BARRIER_TYPE

    def get_per_vertex_color(self):
        """Return per-vertex color based on object index."""
        return self.solver.per_vertex_color

    def run(self):
        # Override get_per_vertex_color on the runner
        self.runner.get_per_vertex_color = self.get_per_vertex_color
        self.runner.run()

    def get_config(self):
        """Return configuration info for reporting."""
        return {
            'version': self.version_name,
            'collision_method': self.collision_method,
            'barrier_type': self.barrier_type,
            'demo': self.solver.demo,
            'n_verts': self.solver.n_verts,
            'n_cells': self.solver.n_cells,
            'n_objects': self.solver.N_object,
        }


def run_fast_mode(frames=50, demo='eight_E_drop_demo_contact'):
    """
    Run in fast mode without DemoRunner overhead.
    This gives accurate timing similar to the original visual() loop.
    """
    print(f"\n{'='*60}")
    print(f"n_E-initial (Fast Mode)")
    print(f"{'='*60}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames without rendering overhead...")

    solver = pncg_index_initial(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")

    # Warmup (first frame has compilation overhead)
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


def run_profile_mode(frames=20, demo='eight_E_drop_demo_contact'):
    """
    Run in profile mode with detailed timing for each component.
    Saves detailed log to ./logs folder.
    """
    print(f"\n{'='*70}")
    print(f"n_E-initial (Profile Mode)")
    print(f"{'='*70}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames with detailed timing...")

    solver = pncg_index_initial(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")

    # Create logger
    logger = PerformanceLogger(VERSION_NAME, demo)

    # Warmup (first frame has compilation overhead)
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
        iters = solver.step(logger=logger)
        ti.sync()

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        logger.end_frame(frame_ms, iters + 1)

        fps = 1000.0 / frame_ms if frame_ms > 0 else 0
        print(f"  Frame {i}: {frame_ms:.2f}ms ({fps:.1f} FPS), iters={iters+1}")

    # Print and save summary
    logger.print_summary()

    log_file = logger.save_log()
    print(f"\nDetailed log saved to: {log_file}")

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initial n_E Demo')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode with image saving')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (no rendering, accurate timing)')
    parser.add_argument('--profile', action='store_true', help='Run in profile mode with detailed timing per component')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to run')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact', help='Demo name')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    if args.profile:
        # Profile mode - detailed timing for each component
        run_profile_mode(frames=args.frames, demo=args.demo)
    elif args.fast:
        # Fast mode - no rendering overhead, just timing
        run_fast_mode(frames=args.frames, demo=args.demo)
    else:
        # Use DemoRunner for headless/interactive mode
        runner = InitialNEDemoRunner(demo=args.demo)
        runner.run()
