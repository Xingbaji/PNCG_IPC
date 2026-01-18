"""
Comparison benchmark: LBVH vs Spatial Hashing for collision detection.
Tests with n_E_demo (8 objects dropping) for 100 frames.
"""
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

import taichi as ti
import numpy as np
import time
import argparse
from math_utils.matrix_util import compute_dtdx_t, compute_d_dtdx


# ============================================================================
# BVH-based IPC Deformer
# ============================================================================
def create_bvh_deformer(demo):
    """Create IPC deformer using LBVH collision detection."""
    from algorithm.pncg_base_ipc import pncg_ipc_deformer as BVH_Deformer
    return BVH_Deformer(demo=demo)


# ============================================================================
# Spatial Hashing IPC Deformer
# ============================================================================
def create_spatial_hash_deformer(demo):
    """Create IPC deformer using spatial hashing collision detection."""
    from algorithm.pncg_base_ipc_spatial_hash import pncg_ipc_deformer as SpatialHash_Deformer
    return SpatialHash_Deformer(demo=demo)


# ============================================================================
# Benchmark n_E Demo with detailed timing
# ============================================================================
@ti.data_oriented
class BenchmarkRunner:
    """Runs n_E demo with detailed timing instrumentation."""

    def __init__(self, deformer, method_name, use_bvh_refit=False):
        self.deformer = deformer
        self.method_name = method_name
        self.use_bvh_refit = use_bvh_refit
        self.object_size = 1046  # vertices per E object
        self.N_object = int(deformer.n_verts / self.object_size)
        self.iter_count = 0  # Track iteration for BVH refit logic

        # Set up index for per-object CG
        deformer.mesh.verts.place({'index': ti.i32})
        self.init_index()

        # Timing storage
        self.collision_times = []
        self.solver_times = []
        self.frame_times = []
        self.constraint_counts = []

    @ti.kernel
    def init_index(self):
        for vert in self.deformer.mesh.verts:
            vert.index = vert.id // self.object_size

    @ti.kernel
    def compute_DK_index(self):
        g_Py_index = ti.Vector.zero(float, self.N_object)
        y_p_index = ti.Vector.zero(float, self.N_object)
        y_Py_index = ti.Vector.zero(float, self.N_object)
        g_p_index = ti.Vector.zero(float, self.N_object)
        beta_index = ti.Vector.zero(float, self.N_object)

        for vert in self.deformer.mesh.verts:
            index = vert.index
            y = vert.grad - vert.grad_prev
            Py = y / vert.diagH
            y_p_index[index] += y.dot(vert.p)
            g_Py_index[index] += vert.grad.dot(Py)
            y_Py_index[index] += y.dot(Py)
            g_p_index[index] += vert.grad.dot(vert.p)

        for i in range(self.N_object):
            beta_index[i] = (g_Py_index[i] - y_Py_index[i] * g_p_index[i] / y_p_index[i]) / y_p_index[i]

        for vert in self.deformer.mesh.verts:
            index = vert.index
            vert.p = -vert.grad / vert.diagH + beta_index[index] * vert.p

    @ti.kernel
    def compute_alpha_index_and_update_x(self) -> float:
        gTp_index = ti.Vector.zero(float, self.N_object)
        pHp_index = ti.Vector.zero(float, self.N_object)
        alpha_index = ti.Vector.zero(float, self.N_object)
        p_max_index = ti.Vector.zero(float, self.N_object)

        for vert in self.deformer.mesh.verts:
            index = vert.index
            gTp_index[index] += vert.grad.dot(vert.p)
            pHp_index[index] += vert.p.norm_sqr() * vert.m

        for c in self.deformer.mesh.cells:
            index = c.verts[0].index
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            p = ti.Vector.zero(float, 12)
            p[0:3] = c.verts[0].p
            p[3:6] = c.verts[1].p
            p[6:9] = c.verts[2].p
            p[9:12] = c.verts[3].p
            tmp = self.deformer.compute_p_d2Psidx2_p(F, c.B, p, self.deformer.mu, self.deformer.la)
            pHp_index[index] += c.W * self.deformer.dt ** 2 * ti.max(tmp, 0.0)

        # IPC contribution
        for k, j in self.deformer.cid:
            pair = self.deformer.cid[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist
            bg = self.deformer.barrier_g(dist)
            para1 = bg / dist
            para0 = (self.deformer.barrier_H(dist) - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.deformer.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.deformer.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.deformer.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.deformer.mesh.verts.p[ids[3]]

            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)
            d_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * d_dtdx.norm_sqr()
            pHp = ti.max(pHp_0 + pHp_1, 0.0)

            index0 = self.deformer.mesh.verts.index[ids[0]]
            index1 = self.deformer.mesh.verts.index[ids[2]]
            pHp_index[index0] += pHp * 0.5
            pHp_index[index1] += pHp * 0.5

        # Ground barrier
        min_dist = 1e-2 * self.deformer.dHat
        for i in range(self.deformer.boundary_points.shape[0]):
            p = self.deformer.boundary_points[i]
            index = self.deformer.mesh.verts.index[p]
            x_a0 = self.deformer.mesh.verts.x[p]
            dist = x_a0[1] - self.deformer.ground
            if dist < self.deformer.dHat:
                if dist <= min_dist:
                    dist = min_dist
                p_tmp = self.deformer.mesh.verts.p[p][1]
                ret_value = p_tmp * self.deformer.barrier_H(dist) * p_tmp
                pHp_index[index] += ret_value

        for vert in self.deformer.mesh.verts:
            index = vert.index
            d_norm = vert.p.norm()
            ti.atomic_max(p_max_index[index], d_norm)

        Delta_E = 0.0
        for i in range(self.N_object):
            alpha_i = -gTp_index[i] / pHp_index[i]
            if alpha_i * p_max_index[i] > 0.5 * self.deformer.dHat:
                alpha_index[i] = 0.5 * self.deformer.dHat / p_max_index[i]
            else:
                alpha_index[i] = alpha_i
            Delta_E -= (alpha_index[i] * gTp_index[i] + 0.5 * alpha_index[i] ** 2 * pHp_index[i])

        for vert in self.deformer.mesh.verts:
            index = vert.index
            vert.x += alpha_index[index] * vert.p

        return Delta_E

    @ti.kernel
    def count_constraints(self) -> ti.i32:
        cnt = 0
        for k, j in self.deformer.cid:
            cnt += 1
        return cnt

    def step_timed(self):
        """Run one simulation step with timing."""
        d = self.deformer
        frame_start = time.perf_counter()

        d.assign_xn_xhat()
        self.iter_count = 0  # Reset iteration count for each frame

        total_collision_time = 0.0
        total_solver_time = 0.0
        n_constraints = 0

        for iter in range(d.iter_max):
            # Collision detection timing
            ti.sync()
            cd_start = time.perf_counter()
            if self.use_bvh_refit and hasattr(d, 'find_cnts_iter'):
                d.find_cnts_iter(self.iter_count)
                self.iter_count += 1
            else:
                d.find_cnts()
            ti.sync()
            cd_end = time.perf_counter()
            total_collision_time += (cd_end - cd_start)

            n_constraints = self.count_constraints()

            # Solver timing
            solver_start = time.perf_counter()
            d.compute_grad_and_diagH()
            if d.ground_barrier == 1:
                d.add_grad_and_diagH_ground_barrier()

            if iter == 0:
                d.compute_init_p()
            else:
                self.compute_DK_index()

            delta_E = self.compute_alpha_index_and_update_x()
            ti.sync()
            solver_end = time.perf_counter()
            total_solver_time += (solver_end - solver_start)

            if iter == 0:
                delta_E_init = delta_E
            if delta_E < d.epsilon * delta_E_init:
                break

        d.update_v_and_bound()
        d.frame += 1

        frame_end = time.perf_counter()

        self.collision_times.append(total_collision_time * 1000)
        self.solver_times.append(total_solver_time * 1000)
        self.frame_times.append((frame_end - frame_start) * 1000)
        self.constraint_counts.append(n_constraints)

        return iter

    def run(self, n_frames):
        """Run simulation for n_frames."""
        print(f"\n{'='*60}")
        print(f"Running {self.method_name} for {n_frames} frames")
        print(f"{'='*60}")

        for frame in range(n_frames):
            iters = self.step_timed()
            if frame % 10 == 0:
                print(f"  Frame {frame:3d}: collision={self.collision_times[-1]:8.2f}ms, "
                      f"solver={self.solver_times[-1]:8.2f}ms, "
                      f"constraints={self.constraint_counts[-1]:6d}, iters={iters}")

        return self.get_statistics()

    def get_statistics(self):
        """Return timing statistics."""
        return {
            'method': self.method_name,
            'collision_mean': np.mean(self.collision_times),
            'collision_std': np.std(self.collision_times),
            'collision_total': np.sum(self.collision_times),
            'solver_mean': np.mean(self.solver_times),
            'solver_std': np.std(self.solver_times),
            'solver_total': np.sum(self.solver_times),
            'frame_mean': np.mean(self.frame_times),
            'frame_std': np.std(self.frame_times),
            'frame_total': np.sum(self.frame_times),
            'constraint_mean': np.mean(self.constraint_counts),
            'constraint_max': np.max(self.constraint_counts),
            'collision_times': self.collision_times,
            'solver_times': self.solver_times,
            'frame_times': self.frame_times,
            'constraint_counts': self.constraint_counts,
        }


def run_comparison(demo, n_frames, methods):
    """Run comparison between methods."""
    results = {}

    for method in methods:
        # Reinitialize Taichi for each method to get clean state
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f32)

        print(f"\n{'#'*70}")
        print(f"# Initializing {method} method")
        print(f"{'#'*70}")

        if method == 'BVH':
            deformer = create_bvh_deformer(demo)
            runner = BenchmarkRunner(deformer, method, use_bvh_refit=False)
        elif method == 'BVH_Refit':
            deformer = create_bvh_deformer(demo)
            runner = BenchmarkRunner(deformer, method, use_bvh_refit=True)
        else:
            deformer = create_spatial_hash_deformer(demo)
            runner = BenchmarkRunner(deformer, method, use_bvh_refit=False)

        results[method] = runner.run(n_frames)

    return results


def print_comparison(results):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    methods = list(results.keys())

    # Header
    print(f"\n{'Metric':<30}", end="")
    for method in methods:
        print(f"{method:>20}", end="")
    print()
    print("-" * (30 + 20 * len(methods)))

    # Collision detection time
    print(f"{'Collision Det. (ms/frame)':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['collision_mean']:>15.2f} ± {r['collision_std']:<4.1f}", end="")
    print()

    # Solver time
    print(f"{'Solver (ms/frame)':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['solver_mean']:>15.2f} ± {r['solver_std']:<4.1f}", end="")
    print()

    # Total frame time
    print(f"{'Total Frame (ms)':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['frame_mean']:>15.2f} ± {r['frame_std']:<4.1f}", end="")
    print()

    # Total collision time
    print(f"{'Total Collision Time (s)':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['collision_total']/1000:>20.2f}", end="")
    print()

    # Constraint count
    print(f"{'Avg Constraints':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['constraint_mean']:>20.0f}", end="")
    print()

    print(f"{'Max Constraints':<30}", end="")
    for method in methods:
        r = results[method]
        print(f"{r['constraint_max']:>20d}", end="")
    print()

    # Speedup
    if len(methods) == 2:
        speedup = results[methods[1]]['collision_mean'] / results[methods[0]]['collision_mean']
        print(f"\n{'Collision Detection Speedup:':<30} {methods[0]} is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than {methods[1]}")

        total_speedup = results[methods[1]]['frame_mean'] / results[methods[0]]['frame_mean']
        print(f"{'Total Frame Speedup:':<30} {methods[0]} is {total_speedup:.2f}x {'faster' if total_speedup > 1 else 'slower'} than {methods[1]}")


def main():
    parser = argparse.ArgumentParser(description='Compare LBVH vs Spatial Hashing')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to run')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact', help='Demo name')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'both', 'bvh', 'bvh_refit', 'spatial'],
                        help='Which method to test (all includes BVH with refit)')
    args = parser.parse_args()

    print("=" * 80)
    print("LBVH vs Spatial Hashing Collision Detection Comparison")
    print("=" * 80)
    print(f"Demo: {args.demo}")
    print(f"Frames: {args.frames}")

    if args.method == 'all':
        methods = ['BVH_Refit', 'SpatialHash']  # Skip slow full BVH build
    elif args.method == 'both':
        methods = ['BVH', 'SpatialHash']
    elif args.method == 'bvh':
        methods = ['BVH']
    elif args.method == 'bvh_refit':
        methods = ['BVH_Refit']
    else:
        methods = ['SpatialHash']

    results = run_comparison(args.demo, args.frames, methods)
    print_comparison(results)

    # Save detailed results
    output_file = f'collision_comparison_{args.demo}_{args.frames}frames.npz'
    np.savez(output_file, **{f"{k}_{m}": v for m, r in results.items() for k, v in r.items() if not isinstance(v, str)})
    print(f"\nDetailed results saved to {output_file}")


if __name__ == '__main__':
    main()
