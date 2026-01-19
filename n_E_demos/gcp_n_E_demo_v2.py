"""
GCP n_E Demo v2 - Uses Geometric Contact Potential (GCP) for collision handling.

This version uses GCP which enables:
- 10x larger dHat/epsilon compared to standard IPC
- NO adjacency matrix needed (gamma filtering auto-excludes adjacent elements)
- Better convergence properties due to smoother barrier functions

Usage:
    python gcp_n_E_demo_v2.py                     # Interactive mode
    python gcp_n_E_demo_v2.py --headless --frames 50  # Headless with images
    python gcp_n_E_demo_v2.py --fast --frames 50      # Fast mode (no rendering)
    python gcp_n_E_demo_v2.py --profile --frames 20   # Profile mode with detailed timing
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
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.gcp_contact_potential import GCPModule, GCPConfig, gcp_barrier_g, gcp_barrier_H, gcp_barrier_E
from math_utils.graphic_util import *

VERSION_NAME = "gcp"
COLLISION_METHOD = "gcp_bvh"
BARRIER_TYPE = "gcp_mollified"


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
            'ground_barrier_ms': 0,
            'compute_direction_ms': 0,
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

        frame_times = [f['total_time_ms'] for f in self.frame_logs]
        iter_counts = [f['n_iterations'] for f in self.frame_logs]

        component_times = defaultdict(list)
        for frame in self.frame_logs:
            for it in frame['iterations']:
                for key in ['find_cnts_ms', 'compute_grad_diagH_ms', 'ground_barrier_ms',
                           'compute_direction_ms', 'line_search_ms']:
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
class GCPIndexSolver(pncg_ipc_deformer):
    """
    GCP-based per-object optimization solver.
    Uses Geometric Contact Potential for collision handling with gamma filtering.
    Inherits from pncg_ipc_deformer for consistent architecture with bvh_n_E_demo.
    """

    def __init__(self, demo='eight_E_drop_demo_contact'):
        super().__init__(demo)

        # Standard IPC dHat from model (used for step size clamping)
        self.standard_dHat = self.dHat

        # GCP epsilon can be larger than dHat for detection range
        # But step size should still be limited by standard_dHat for safety
        self.gcp_epsilon = self.dHat * 2.0  # 2x larger detection range

        # Step size limit - CRITICAL for preventing penetration
        # This should be based on the SMALLER value (standard_dHat) not epsilon
        self.step_limit = self.standard_dHat

        print(f"\n>>> GCP Configuration:")
        print(f"    Standard IPC dHat: {self.standard_dHat}")
        print(f"    GCP epsilon_target: {self.gcp_epsilon} ({self.gcp_epsilon/self.standard_dHat:.1f}x detection range)")
        print(f"    Step size limit: {self.step_limit} (based on standard dHat)")
        print(f"    Adjacency matrix: NOT NEEDED")

        # Initialize GCP module
        self.gcp = GCPModule(
            self.n_boundary_points,
            self.n_boundary_edges,
            self.n_boundary_triangles,
            GCPConfig(
                epsilon_target=self.gcp_epsilon,
                adaptive_epsilon=True,
                alpha=0.1,
                kappa=self.kappa
            )
        )

        # Compute adaptive epsilon from rest configuration
        print("Computing adaptive epsilon from rest configuration...")
        self.gcp.compute_adaptive_epsilon(
            self.mesh,
            self.boundary_points,
            self.boundary_edges,
            self.boundary_triangles
        )

        self.bvh_initialized = False
        print("GCP n_E Demo v2 initialized!\n")

    def set_index(self):
        """Initialize per-object indices and colors."""
        self.mesh.verts.place({'index': ti.i32})
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)
        self.init_index()
        self.frame = 0
        self.CNT = 0

    @ti.kernel
    def init_index(self):
        """Set vertex index and colors based on object."""
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            vert.index = index
            # Color palette for different objects
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

    def find_constraints_gcp(self):
        """Find collision constraints using GCP with gamma filtering."""
        # Build/refit BVH
        if not self.bvh_initialized:
            self.build_bvh()
            self.bvh_initialized = True
        else:
            self.refit_bvh()

        # Find constraints using GCP
        self.gcp.find_constraints_gcp(
            self.mesh,
            self.boundary_points,
            self.boundary_edges,
            self.boundary_triangles,
            self.bvh_triangles,
            self.bvh_edges,
            self.n_verts
        )

    @ti.kernel
    def compute_grad_and_diagH_gcp(self):
        """Compute gradient and diagonal Hessian including GCP contact forces."""
        ti.mesh_local(self.mesh.verts.grad)

        # Inertia term
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)

        # Elastic energy gradient and Hessian
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3*i], dPsidx[3*i+1], dPsidx[3*i+2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3*i], diagH_d2Psidx2[3*i+1], diagH_d2Psidx2[3*i+2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp

        # GCP contact forces
        for k, j in self.gcp.cid_gcp:
            pair = self.gcp.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:  # Only active constraints
                kappa = self.kappa
                bg = gcp_barrier_g(dist, epsilon, gamma, kappa)
                bH = gcp_barrier_H(dist, epsilon, gamma, kappa)
                dist2 = dist * dist
                para = bg / dist
                para0 = (bH - para) / dist2

                for i in range(4):
                    CORD = cord[i]
                    ID = ids[i]
                    self.mesh.verts.grad[ID] += para * CORD * t
                    diag_tmp = CORD * CORD * (para0 * t * t + para * ti.Vector.one(float, 3))
                    diag_tmp_spd = ti.max(diag_tmp, 0.0)
                    self.mesh.verts.diagH[ID] += diag_tmp_spd

    @ti.kernel
    def add_grad_and_diagH_ground_barrier_gcp(self):
        """Add ground barrier gradient and Hessian using GCP barrier."""
        min_dist = 1e-2 * self.gcp_epsilon
        kappa = self.kappa
        for i in range(self.boundary_points.shape[0]):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.gcp_epsilon:
                if dist <= min_dist:
                    dist = min_dist
                # Use standard log barrier for ground (gamma=1)
                bg = gcp_barrier_g(dist, self.gcp_epsilon, 1.0, kappa)
                bH = gcp_barrier_H(dist, self.gcp_epsilon, 1.0, kappa)
                self.mesh.verts.grad[p][1] += bg
                self.mesh.verts.diagH[p][1] += ti.max(bH, 0.0)

    @ti.kernel
    def compute_DK_index(self):
        """Compute DK direction for each object."""
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
    def compute_alpha_index_and_update_x_gcp(self) -> float:
        """Compute per-object step size and update positions."""
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

        # GCP contact contribution to pHp
        for k, j in self.gcp.cid_gcp:
            pair = self.gcp.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:
                kappa = self.kappa
                dist2 = dist * dist
                bg = gcp_barrier_g(dist, epsilon, gamma, kappa)
                para1 = bg / dist
                para0 = (gcp_barrier_H(dist, epsilon, gamma, kappa) - para1) / dist2

                p_tmp = ti.Vector.zero(float, 12)
                p_tmp[0:3] = self.mesh.verts.p[ids[0]]
                p_tmp[3:6] = self.mesh.verts.p[ids[1]]
                p_tmp[6:9] = self.mesh.verts.p[ids[2]]
                p_tmp[9:12] = self.mesh.verts.p[ids[3]]

                dtdx_t = ti.Vector.zero(float, 12)
                for i in ti.static(range(4)):
                    for jj in ti.static(range(3)):
                        dtdx_t[3*i+jj] = cord[i] * t[jj]

                pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)

                p_dtdx = ti.Vector.zero(float, 3)
                for i in ti.static(range(4)):
                    p_dtdx += cord[i] * self.mesh.verts.p[ids[i]]
                pHp_1 = para1 * p_dtdx.norm_sqr()

                pHp = ti.max(pHp_0 + pHp_1, 0.0)
                index0 = self.mesh.verts.index[ids[0]]
                index1 = self.mesh.verts.index[ids[2]]
                pHp_index[index0] += pHp * 0.5
                pHp_index[index1] += pHp * 0.5

        # Ground barrier pHp
        min_dist = 1e-2 * self.gcp_epsilon
        kappa = self.kappa
        for i in range(self.boundary_points.shape[0]):
            p_idx = self.boundary_points[i]
            index = self.mesh.verts.index[p_idx]
            x_a0 = self.mesh.verts.x[p_idx]
            dist = x_a0[1] - self.ground
            if dist < self.gcp_epsilon:
                if dist <= min_dist:
                    dist = min_dist
                p_tmp = self.mesh.verts.p[p_idx][1]
                ret_value = p_tmp * gcp_barrier_H(dist, self.gcp_epsilon, 1.0, kappa) * p_tmp
                pHp_index[index] += ti.max(ret_value, 0.0)

        for vert in self.mesh.verts:
            index = vert.index
            d_norm = vert.p.norm()
            ti.atomic_max(p_max_index[index], d_norm)

        Delta_E = 0.0
        for i in range(self.N_object):
            alpha_i = -gTp_index[i] / pHp_index[i]
            if alpha_i * p_max_index[i] > 0.5 * self.gcp_epsilon:
                alpha_index[i] = 0.5 * self.gcp_epsilon / p_max_index[i]
            else:
                alpha_index[i] = alpha_i
            Delta_E -= (alpha_index[i] * gTp_index[i] + 0.5 * alpha_index[i] ** 2 * pHp_index[i])

        for vert in self.mesh.verts:
            index = vert.index
            alpha = alpha_index[index]
            vert.x += alpha * vert.p

        return Delta_E

    def step(self, logger=None):
        """Perform one simulation step."""
        print('Frame', self.frame)
        self.assign_xn_xhat()

        for iter in range(self.iter_max):
            if logger:
                logger.start_iteration(iter)
                ti.sync()
                t_iter_start = time.perf_counter()

            # Find constraints using GCP
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.find_constraints_gcp()
            if logger:
                ti.sync()
                logger.log_time('find_cnts_ms', (time.perf_counter() - t0) * 1000)

            # Gradient and diagonal Hessian
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            self.compute_grad_and_diagH_gcp()
            if logger:
                ti.sync()
                logger.log_time('compute_grad_diagH_ms', (time.perf_counter() - t0) * 1000)

            # Ground barrier
            if logger:
                ti.sync()
                t0 = time.perf_counter()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier_gcp()
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
            delta_E = self.compute_alpha_index_and_update_x_gcp()
            if logger:
                ti.sync()
                logger.log_time('line_search_ms', (time.perf_counter() - t0) * 1000)
                logger.end_iteration((time.perf_counter() - t_iter_start) * 1000)

            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                break

        print(f'Frame {self.frame}: converged at iter {iter}, rate={delta_E / delta_E_init:.2e}')
        self.update_v_and_bound()
        self.frame += 1
        return iter


class GCPNEDemoRunner:
    """Demo runner for GCP n_E demo using DemoRunner framework."""

    def __init__(self, demo='eight_E_drop_demo_contact'):
        from demo_runner import DemoRunner
        self.solver = GCPIndexSolver(demo=demo)
        self.solver.set_index()
        self.demo_name = f"n_E-gcp ({demo})"
        self.runner = DemoRunner(self.solver, demo_name=self.demo_name)

        self.version_name = VERSION_NAME
        self.collision_method = COLLISION_METHOD
        self.barrier_type = BARRIER_TYPE

    def get_per_vertex_color(self):
        """Return per-vertex color based on object index."""
        return self.solver.per_vertex_color

    def run(self):
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
            'gcp_epsilon': self.solver.gcp_epsilon,
            'standard_dHat': self.solver.standard_dHat,
        }


def run_fast_mode(frames=50, demo='eight_E_drop_demo_contact'):
    """Run in fast mode without DemoRunner overhead."""
    print(f"\n{'='*60}")
    print(f"n_E-gcp (Fast Mode)")
    print(f"{'='*60}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames without rendering overhead...")

    solver = GCPIndexSolver(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  GCP epsilon: {solver.gcp_epsilon} (10x standard)")

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


def run_profile_mode(frames=20, demo='eight_E_drop_demo_contact'):
    """Run in profile mode with detailed timing."""
    print(f"\n{'='*70}")
    print(f"n_E-gcp (Profile Mode)")
    print(f"{'='*70}")
    print(f"Collision: {COLLISION_METHOD}, Barrier: {BARRIER_TYPE}")
    print(f"Running {frames} frames with detailed timing...")

    solver = GCPIndexSolver(demo=demo)
    solver.set_index()

    print(f"  Vertices: {solver.n_verts}")
    print(f"  Cells: {solver.n_cells}")
    print(f"  Objects: {solver.N_object}")
    print(f"  GCP epsilon: {solver.gcp_epsilon} (10x standard)")

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
        iters = solver.step(logger=logger)
        ti.sync()

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        logger.end_frame(frame_ms, iters + 1)

        fps = 1000.0 / frame_ms if frame_ms > 0 else 0
        print(f"  Frame {i}: {frame_ms:.2f}ms ({fps:.1f} FPS), iters={iters+1}")

    logger.print_summary()

    log_file = logger.save_log()
    print(f"\nDetailed log saved to: {log_file}")

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCP n_E Demo v2')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode with image saving')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (no rendering, accurate timing)')
    parser.add_argument('--profile', action='store_true', help='Run in profile mode with detailed timing')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to run')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact', help='Demo name')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    if args.profile:
        run_profile_mode(frames=args.frames, demo=args.demo)
    elif args.fast:
        run_fast_mode(frames=args.frames, demo=args.demo)
    else:
        runner = GCPNEDemoRunner(demo=args.demo)
        runner.run()
