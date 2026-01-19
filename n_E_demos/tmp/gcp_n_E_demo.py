"""
GCP n_E Demo - Uses Geometric Contact Potential with large dHat.

This demo demonstrates the benefits of GCP over standard IPC:
1. No adjacency matrix needed
2. 10x larger detection distance (dHat/epsilon)
3. Automatic filtering via directional factors (gamma)

Usage:
    python gcp_n_E_demo.py                          # Interactive mode
    python gcp_n_E_demo.py --headless --frames 50   # Headless with images
    python gcp_n_E_demo.py --fast --frames 50       # Fast mode (no rendering)
    python gcp_n_E_demo.py --profile --frames 20    # Profile mode with timing
    python gcp_n_E_demo.py --compare                # Compare with standard IPC
"""

import sys
import os
import time
import argparse
import json
from datetime import datetime
from collections import defaultdict

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
logs_dir = os.path.join(os.path.dirname(current_file_path), 'logs')
os.makedirs(logs_dir, exist_ok=True)
imgs_dir = os.path.join(os.path.dirname(current_file_path), 'imgs', 'gcp')
os.makedirs(imgs_dir, exist_ok=True)

os.chdir(demo_dir)

import taichi as ti
from algorithm.collision_detection_bvh import collision_detection_bvh_module
from algorithm.gcp_contact_potential import (
    GCPModule, GCPConfig,
    gcp_barrier_E, gcp_barrier_g, gcp_barrier_H
)
from util.model_loading import model_loading

VERSION_NAME = "gcp"
COLLISION_METHOD = "gcp_gamma"
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
            'collision_method': COLLISION_METHOD,
            'barrier_type': BARRIER_TYPE,
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
                'percent': float(np.sum(times) / np.sum(frame_times) * 100) if np.sum(frame_times) > 0 else 0,
            }

        return summary

    def save_log(self, filename=None):
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
        summary = self.get_summary()
        if not summary:
            print("No performance data collected.")
            return

        print(f"\n{'='*60}")
        print(f"Performance Summary: {summary['version']} ({summary['demo']})")
        print(f"Collision: {summary['collision_method']}, Barrier: {summary['barrier_type']}")
        print(f"{'='*60}")
        print(f"Frames: {summary['n_frames']}")
        print(f"Frame time: {summary['frame_time']['avg_ms']:.1f}ms avg "
              f"({summary['frame_time']['min_ms']:.1f} - {summary['frame_time']['max_ms']:.1f}ms)")
        print(f"Iterations: {summary['iterations']['avg']:.1f} avg "
              f"({summary['iterations']['min']} - {summary['iterations']['max']})")

        print(f"\nComponent breakdown:")
        for name, stats in summary['components'].items():
            print(f"  {name}: {stats['avg_ms']:.2f}ms avg ({stats['percent']:.1f}%)")


@ti.data_oriented
class GCPNEDemoSolver(collision_detection_bvh_module):
    """
    GCP-based n_E Demo Solver.

    Uses Geometric Contact Potential for collision handling with:
    - Large dHat (10x typical IPC)
    - No adjacency matrix needed
    - Automatic gamma filtering
    """

    def __init__(self, demo='eight_E_drop_demo_contact', epsilon_scale=10.0):
        # Load model configuration
        model = model_loading(demo=demo)
        self.demo = demo
        print(f'GCP n_E Demo: {self.demo}')

        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.ground_barrier = model.ground_barrier
        self.frame = 0
        self.SMALL_NUM = 1e-7

        # Initialize mesh fields
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),
            'v': ti.types.vector(3, float),
            'm': float,
            'x_n': ti.types.vector(3, float),
            'x_hat': ti.types.vector(3, float),
            'x_prev': ti.types.vector(3, float),
            'x_init': ti.types.vector(3, float),
            'grad': ti.types.vector(3, float),
            'grad_prev': ti.types.vector(3, float),
            'p': ti.types.vector(3, float),
            'diagH': ti.types.vector(3, float),
            'index': ti.i32,
        })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print(f'n_verts, n_cells: {self.n_verts}, {self.n_cells}')

        # Precompute
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        # Boundary elements
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print(f'Boundary size: {self.n_boundary_points}, {self.n_boundary_edges}, {self.n_boundary_triangles}')
        self.set_point_lights()

        # GCP parameters
        self.kappa = model.kappa
        self.base_dHat = getattr(model, 'dHat', 0.01)
        self.dHat = self.base_dHat  # Keep for compatibility
        self.epsilon_target = self.base_dHat * epsilon_scale  # 10x larger!

        print(f'\n>>> GCP Configuration:')
        print(f'    Standard IPC dHat: {self.base_dHat}')
        print(f'    GCP epsilon_target: {self.epsilon_target} ({epsilon_scale}x larger!)')
        print(f'    Adjacency matrix: NOT NEEDED')

        # Initialize BVH (without adjacency!)
        print('\nInitializing BVH structures...')
        self.init_bvh_gcp()

        # Initialize GCP module
        self.gcp = GCPModule(
            self.n_boundary_points,
            self.n_boundary_edges,
            self.n_boundary_triangles,
            GCPConfig(
                epsilon_target=self.epsilon_target,
                adaptive_epsilon=True,
                alpha=0.1,
                kappa=self.kappa
            )
        )

        # Compute adaptive epsilon
        print('Computing adaptive epsilon from rest configuration...')
        self.gcp.compute_adaptive_epsilon(
            self.mesh,
            self.boundary_points,
            self.boundary_edges,
            self.boundary_triangles
        )

        # Per-vertex color for visualization
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)
        self.init_index()

        print('GCP n_E Demo initialized!')

    def init_bvh_gcp(self):
        """Initialize BVH without adjacency matrix."""
        from algorithm.lbvh import LBVH_Triangles, LBVH_Edges

        self.bvh_triangles = LBVH_Triangles(self.n_boundary_triangles)
        self.bvh_edges = LBVH_Edges(self.n_boundary_edges)

        self.MAX_C = 2 ** 21
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),
            b=float,
            c=ti.types.vector(4, float),
            d=ti.types.vector(3, float)
        )
        self.cid = self.pair.field()
        self.cid_root = ti.root.bitmasked(ti.ij, (2, self.MAX_C)).place(self.cid)

        # No adjacency matrix!
        self.attempt_PT = self.attempt_PT_no_adj
        self.attempt_EE = self.attempt_EE_no_adj

    @ti.kernel
    def init_index(self):
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            vert.index = index
            # Assign color based on index
            color_r = (index * 37 % 256) / 255.0
            color_g = (index * 73 % 256) / 255.0
            color_b = (index * 127 % 256) / 255.0
            self.per_vertex_color[vert.id] = ti.Vector([color_r, color_g, color_b])

    def find_cnts_gcp(self, PRINT=False):
        """Find constraints using GCP filtering."""
        self.build_bvh()
        self.gcp.find_constraints_gcp(
            self.mesh,
            self.boundary_points,
            self.boundary_edges,
            self.boundary_triangles,
            self.bvh_triangles,
            self.bvh_edges,
            self.n_verts
        )
        if PRINT:
            return self.gcp.print_constraints_gcp()

    @ti.kernel
    def compute_E_gcp(self) -> float:
        """Compute total energy with GCP contact potential."""
        E = 0.0

        # Inertia
        for vert in self.mesh.verts:
            E += 0.5 * vert.m * (vert.x - vert.x_hat).norm_sqr()

        # Elastic
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            Psi = self.compute_Psi(F, self.mu, self.la)
            E += (self.dt ** 2) * c.W * Psi

        # GCP contact
        for k, j in self.gcp.cid_gcp:
            pair = self.gcp.cid_gcp[k, j]
            E += gcp_barrier_E(pair.b, pair.epsilon, pair.gamma, self.kappa)

        return E

    @ti.kernel
    def compute_grad_and_diagH_gcp(self):
        """Compute gradient and diagonal Hessian with GCP."""
        ti.mesh_local(self.mesh.verts.grad)

        # Inertia
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)

        # Elastic
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

        # GCP contact
        for k, j in self.gcp.cid_gcp:
            pair = self.gcp.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:
                bg = gcp_barrier_g(dist, epsilon, gamma, self.kappa)
                bH = gcp_barrier_H(dist, epsilon, gamma, self.kappa)

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
    def compute_pHp_gcp(self) -> float:
        """Compute p^T H p for GCP."""
        ret = 0.0

        for vert in self.mesh.verts:
            ret += vert.p.norm_sqr() * vert.m

        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            d = ti.Vector.zero(float, 12)
            d[0:3] = c.verts[0].p
            d[3:6] = c.verts[1].p
            d[6:9] = c.verts[2].p
            d[9:12] = c.verts[3].p
            tmp = self.compute_p_d2Psidx2_p(F, B, d, self.mu, self.la)
            ret += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        for k, j in self.gcp.cid_gcp:
            pair = self.gcp.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:
                bg = gcp_barrier_g(dist, epsilon, gamma, self.kappa)
                bH = gcp_barrier_H(dist, epsilon, gamma, self.kappa)

                dist2 = dist * dist
                para1 = bg / dist
                para0 = (bH - para1) / dist2

                p_tmp = ti.Vector.zero(float, 12)
                p_tmp[0:3] = self.mesh.verts.p[ids[0]]
                p_tmp[3:6] = self.mesh.verts.p[ids[1]]
                p_tmp[6:9] = self.mesh.verts.p[ids[2]]
                p_tmp[9:12] = self.mesh.verts.p[ids[3]]

                dtdx_t = ti.Vector.zero(float, 12)
                for i in ti.static(range(4)):
                    for j in ti.static(range(3)):
                        dtdx_t[3*i+j] = cord[i] * t[j]

                pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)

                p_dtdx = ti.Vector.zero(float, 3)
                for i in ti.static(range(4)):
                    p_dtdx += cord[i] * self.mesh.verts.p[ids[i]]
                pHp_1 = para1 * p_dtdx.norm_sqr()

                ret += ti.max(pHp_0 + pHp_1, 0.0)

        return ret

    @ti.kernel
    def add_grad_diagH_ground_barrier_gcp(self):
        """Add ground barrier (using standard barrier for ground)."""
        min_dist = 1e-2 * self.base_dHat
        for i in range(self.n_boundary_points):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.base_dHat:
                if dist <= min_dist:
                    self.mesh.verts.x[p][1] = self.ground + min_dist
                    dist = min_dist
                # Use standard log barrier for ground
                g = self.kappa * ((dist - self.base_dHat) * ti.log(dist / self.base_dHat) * (-2.0) - (dist - self.base_dHat) ** 2 / dist)
                H = self.kappa * ((-2) * ti.log(dist / self.base_dHat) - 4 + 4 * self.base_dHat / dist + (dist - self.base_dHat) ** 2 / dist ** 2)
                self.mesh.verts.grad[p][1] += g
                self.mesh.verts.diagH[p][1] += H

    @ti.kernel
    def compute_p_inf_norm(self) -> float:
        p_max = 0.0
        for vert in self.mesh.verts:
            ti.atomic_max(p_max, vert.p.norm())
        return p_max

    def step(self, perf_logger=None):
        """Perform one simulation step using GCP."""
        frame_start = time.perf_counter()
        self.assign_xn_xhat()

        for iter in range(self.iter_max):
            iter_start = time.perf_counter()
            if perf_logger:
                perf_logger.start_iteration(iter)

            # Find constraints
            t0 = time.perf_counter()
            ti.sync()
            self.find_cnts_gcp()
            ti.sync()
            if perf_logger:
                perf_logger.log_time('find_cnts_ms', (time.perf_counter() - t0) * 1000)

            # Compute gradient and Hessian
            t0 = time.perf_counter()
            ti.sync()
            self.compute_grad_and_diagH_gcp()
            ti.sync()
            if perf_logger:
                perf_logger.log_time('compute_grad_diagH_ms', (time.perf_counter() - t0) * 1000)

            # Ground barrier
            t0 = time.perf_counter()
            if self.ground_barrier == 1:
                ti.sync()
                self.add_grad_diagH_ground_barrier_gcp()
                ti.sync()
            if perf_logger:
                perf_logger.log_time('ground_barrier_ms', (time.perf_counter() - t0) * 1000)

            # Search direction
            t0 = time.perf_counter()
            ti.sync()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_direction()
            ti.sync()
            if perf_logger:
                perf_logger.log_time('compute_direction_ms', (time.perf_counter() - t0) * 1000)

            # Line search
            t0 = time.perf_counter()
            ti.sync()
            gTp = self.compute_gTp()
            pHp = self.compute_pHp_gcp()
            alpha = -gTp / pHp
            p_max = self.compute_p_inf_norm()

            if alpha * p_max > 0.5 * self.epsilon_target:
                alpha = 0.5 * self.epsilon_target / p_max

            self.update_x(alpha)
            ti.sync()
            if perf_logger:
                perf_logger.log_time('line_search_ms', (time.perf_counter() - t0) * 1000)

            # Convergence check
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E

            if perf_logger:
                perf_logger.end_iteration((time.perf_counter() - iter_start) * 1000)

            if delta_E < self.epsilon * delta_E_init:
                break

        self.update_v_and_bound()

        if perf_logger:
            perf_logger.end_frame((time.perf_counter() - frame_start) * 1000, iter + 1)

        print(f'Frame {self.frame}: converged at iter {iter}, rate={delta_E/delta_E_init:.2e}')
        self.frame += 1
        return iter


def run_interactive(solver):
    """Run interactive visualization."""
    window = ti.ui.Window('GCP n_E Demo', (1024, 768))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(*solver.camera_position)
    camera.lookat(*solver.camera_lookat)

    while window.running:
        solver.step()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.mesh(solver.mesh.verts.x, solver.indices,
                   per_vertex_color=solver.per_vertex_color)
        canvas.scene(scene)
        window.show()


def run_headless(solver, n_frames, save_images=True, perf_logger=None):
    """Run in headless mode with optional image saving."""
    print(f'\nRunning GCP n_E demo in headless mode for {n_frames} frames...')

    if save_images:
        window = ti.ui.Window('GCP n_E Demo', (1024, 768), show_window=False)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(*solver.camera_position)
        camera.lookat(*solver.camera_lookat)

    total_iters = 0
    start_time = time.time()

    for i in range(n_frames):
        if perf_logger:
            perf_logger.start_frame(i)

        iters = solver.step(perf_logger)
        total_iters += iters

        if save_images:
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            scene.mesh(solver.mesh.verts.x, solver.indices,
                       per_vertex_color=solver.per_vertex_color)
            canvas.scene(scene)
            window.save_image(os.path.join(imgs_dir, f'frame_{i:04d}.png'))

    elapsed = time.time() - start_time

    print(f'\n>>> GCP n_E Demo Complete!')
    print(f'    Total frames: {n_frames}')
    print(f'    Total iterations: {total_iters}')
    print(f'    Avg iters/frame: {total_iters/n_frames:.1f}')
    print(f'    Total time: {elapsed:.2f}s')
    print(f'    Avg time/frame: {elapsed/n_frames*1000:.1f}ms')


def main():
    parser = argparse.ArgumentParser(description='GCP n_E Demo')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact',
                        help='Demo configuration')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI, save images')
    parser.add_argument('--fast', action='store_true',
                        help='Run without GUI, no images')
    parser.add_argument('--profile', action='store_true',
                        help='Profile mode with detailed timing')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames')
    parser.add_argument('--epsilon-scale', type=float, default=10.0,
                        help='Scale factor for epsilon relative to base dHat')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with standard IPC')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f64)

    solver = GCPNEDemoSolver(demo=args.demo, epsilon_scale=args.epsilon_scale)

    perf_logger = None
    if args.profile:
        perf_logger = PerformanceLogger(VERSION_NAME, args.demo)

    if args.fast:
        run_headless(solver, args.frames, save_images=False, perf_logger=perf_logger)
    elif args.headless:
        run_headless(solver, args.frames, save_images=True, perf_logger=perf_logger)
    else:
        run_interactive(solver)

    if perf_logger:
        perf_logger.print_summary()
        log_file = perf_logger.save_log()
        print(f'\nPerformance log saved to: {log_file}')


if __name__ == '__main__':
    main()
