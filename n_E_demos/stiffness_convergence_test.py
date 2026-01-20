"""
Stiffness Convergence Test - Compare MAS-PNCG convergence across different Young's modulus values.

Tests the Eight E free-fall scenario with varying stiffness (E) to analyze:
- Number of iterations to converge
- Energy convergence rate
- Restart frequency
- Physics validation accuracy

Usage:
    python stiffness_convergence_test.py
    python stiffness_convergence_test.py --frames 10
    python stiffness_convergence_test.py --verbose
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
n_E_demos_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(n_E_demos_dir)
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti


class StiffnessConvergenceTest:
    """
    Test MAS-PNCG convergence with different stiffness values.
    """

    def __init__(self, stiffness_values=None):
        """
        Args:
            stiffness_values: List of Young's modulus values to test
        """
        if stiffness_values is None:
            # Default: test 4 orders of magnitude
            self.stiffness_values = [1e4, 1e5, 1e6, 1e7]
        else:
            self.stiffness_values = stiffness_values

        self.results = {}

    def create_solver_with_stiffness(self, E):
        """
        Create a solver with custom stiffness value.

        We need to modify the model_loading to use custom E.
        """
        from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision
        from util.model_loading import model_loading

        # Create a custom demo config by modifying eight_E_freefall
        class CustomModel:
            pass

        # Load the base config
        demo = 'eight_E_freefall'

        # Manually build the demo_dict with custom E
        demo_dict = {
            'E': E, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
            'epsilon': 1e-6, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
            'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
            'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
            'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
            'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
            'camera_position': [2.02077697, -0.54062709, 2.59427191],
            'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
        }

        # Create model object with custom parameters
        model = model_loading.__new__(model_loading)
        model.set_para(demo_dict)
        model.dict = demo_dict

        # Load mesh using same method as eight_E_freefall
        models = []
        for i in range(8):
            model_i = model.add_object(
                model_path=demo_dict['model_paths'][i],
                scale=demo_dict['scales'][i],
                translation=demo_dict['translations'][i],
                rotation=demo_dict['rotations'][i]
            )
            models.append(model_i)
            if i == 0:
                model.ground = np.min(model_i[0][:, 1]) - demo_dict['height']

        # Import the reorder function
        from util.model_loading import _merge_and_reorder_models
        reordered_dict, model.metis_result = _merge_and_reorder_models(models)

        import meshtaichi_patcher as Patcher
        model.mesh = Patcher.load_mesh(reordered_dict, relations=["CV"])
        model.auto_camera_from_models(models)

        # Now create solver using this model
        solver = MASPNCGSolverNoCollision.__new__(MASPNCGSolverNoCollision)

        # Initialize solver attributes from model
        solver.demo = f'eight_E_freefall_E{E:.0e}'
        solver.dict = model.dict
        solver.mu, solver.la = model.mu, model.la
        solver.density = model.density
        solver.dt = model.dt
        solver.gravity = model.gravity
        solver.ground = model.ground
        solver.mesh = model.mesh
        solver.epsilon = model.epsilon
        solver.iter_max = model.iter_max
        solver.camera_position = model.camera_position
        solver.camera_lookat = model.camera_lookat
        solver.frame = 0

        # Initialize mesh fields
        solver.mesh.verts.place({
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
            'z': ti.types.vector(3, float),
            'z_prev': ti.types.vector(3, float),
            'w': ti.types.vector(3, float),
            'Hv': ti.types.vector(3, float),
        })
        solver.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        solver.mesh.verts.x.from_numpy(solver.mesh.get_position_as_numpy())
        solver.mesh.verts.x_init.copy_from(solver.mesh.verts.x)
        solver.mesh.verts.x_prev.copy_from(solver.mesh.verts.x)
        solver.n_verts = len(solver.mesh.verts)
        solver.n_cells = len(solver.mesh.cells)

        # Precompute
        solver.precompute()

        # Initialize indices
        solver.indices = ti.field(ti.i32, shape=len(solver.mesh.cells) * 4 * 3)
        solver.init_indices()

        # Assign elastic type
        solver.assign_elastic_type(model.elastic_type)

        # Set point lights
        solver.set_point_lights()
        solver.config = model.dict

        # MAS Preconditioner
        from algorithm.mas_preconditioner_small import MASPreconditionerSmall
        solver.mas_preconditioner = MASPreconditionerSmall(solver.mesh, metis_reordered=True)

        # Buffer fields
        solver.hv_input = ti.Vector.field(3, dtype=ti.f32, shape=solver.n_verts)
        solver.hv_output = ti.Vector.field(3, dtype=ti.f32, shape=solver.n_verts)

        # State variables
        from algorithm.mas_pncg_solver_nocolli import RESTART_THRESHOLD, ENERGY_TOL, STAGNANT_WINDOW
        solver.restart_threshold = RESTART_THRESHOLD
        solver.energy_tol = ENERGY_TOL
        solver.stagnant_window = STAGNANT_WINDOW
        solver.energy_history = []

        # Scalar fields
        solver.z_H_z = ti.field(dtype=ti.f32, shape=())
        solver.z_H_p = ti.field(dtype=ti.f32, shape=())
        solver.p_H_p = ti.field(dtype=ti.f32, shape=())
        solver.z_g = ti.field(dtype=ti.f32, shape=())
        solver.p_g = ti.field(dtype=ti.f32, shape=())
        solver.g_z_prev = ti.field(dtype=ti.f32, shape=())
        solver.g_z = ti.field(dtype=ti.f32, shape=())

        return solver

    def set_initial_velocity(self, solver, vy=-1.0):
        """Set initial downward velocity."""
        v_np = np.zeros((solver.n_verts, 3), dtype=np.float32)
        v_np[:, 1] = vy
        solver.mesh.verts.v.from_numpy(v_np)

    def compute_centroid(self, solver):
        """Compute mass-weighted centroid."""
        x_np = solver.mesh.verts.x.to_numpy()
        m_np = solver.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        centroid = np.sum(x_np * m_np[:, np.newaxis], axis=0) / total_mass
        return centroid

    def compute_velocity_centroid(self, solver):
        """Compute mass-weighted velocity centroid."""
        v_np = solver.mesh.verts.v.to_numpy()
        m_np = solver.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        v_centroid = np.sum(v_np * m_np[:, np.newaxis], axis=0) / total_mass
        return v_centroid

    def newton_ground_truth(self, initial_centroid, initial_velocity, gravity, t):
        """Compute ground truth using Newton's laws."""
        g = np.array([0.0, gravity, 0.0])
        pos = initial_centroid + initial_velocity * t + 0.5 * g * t * t
        vel = initial_velocity + g * t
        return pos, vel

    def run_test(self, E, frames=5, initial_vy=-1.0, verbose=False):
        """
        Run convergence test for a specific stiffness value.
        """
        print(f"\n{'='*80}")
        print(f"Testing E = {E:.0e}")
        print(f"{'='*80}")

        # Create solver with this stiffness
        solver = self.create_solver_with_stiffness(E)

        print(f"Material: E={E:.0e}, nu={solver.dict['nu']}")
        print(f"Lame: mu={solver.mu:.2e}, la={solver.la:.2e}")
        print(f"dt={solver.dt}, gravity={solver.gravity}")
        print(f"Mesh: {solver.n_verts} verts, {solver.n_cells} cells")

        # Set initial velocity
        self.set_initial_velocity(solver, initial_vy)

        # Track initial state
        initial_centroid = self.compute_centroid(solver)
        initial_velocity = np.array([0.0, initial_vy, 0.0])
        time_elapsed = 0.0

        print(f"Initial centroid: {initial_centroid}")
        print(f"Initial velocity: {initial_velocity}")

        # Run simulation
        frame_data = []
        total_iters = 0
        total_time = 0.0

        for f in range(frames):
            t_start = time.perf_counter()
            iters = solver.step(verbose=verbose)
            t_elapsed = (time.perf_counter() - t_start) * 1000

            time_elapsed += solver.dt

            # Validate
            sim_centroid = self.compute_centroid(solver)
            sim_velocity = self.compute_velocity_centroid(solver)
            gt_pos, gt_vel = self.newton_ground_truth(
                initial_centroid, initial_velocity, solver.gravity, time_elapsed
            )

            pos_error = np.linalg.norm(sim_centroid - gt_pos)
            vel_error = np.linalg.norm(sim_velocity - gt_vel)

            frame_result = {
                'frame': f,
                'time': time_elapsed,
                'iterations': iters,
                'time_ms': t_elapsed,
                'pos_error': pos_error,
                'vel_error': vel_error,
                'sim_centroid_y': sim_centroid[1],
                'gt_centroid_y': gt_pos[1],
            }
            frame_data.append(frame_result)

            total_iters += iters
            total_time += t_elapsed

            print(f"  Frame {f}: {iters:3d} iters, {t_elapsed:7.2f}ms, "
                  f"pos_err={pos_error:.2e}, vel_err={vel_error:.2e}")

        # Summary
        avg_iters = total_iters / frames
        avg_time = total_time / frames
        max_pos_error = max(d['pos_error'] for d in frame_data)
        max_vel_error = max(d['vel_error'] for d in frame_data)

        # Check if passed
        PASS_THRESHOLD_POS = 1e-2
        PASS_THRESHOLD_VEL = 1e-1
        passed = max_pos_error < PASS_THRESHOLD_POS and max_vel_error < PASS_THRESHOLD_VEL

        result = {
            'E': E,
            'frames': frames,
            'total_iters': total_iters,
            'avg_iters': avg_iters,
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'max_pos_error': max_pos_error,
            'max_vel_error': max_vel_error,
            'passed': passed,
            'frame_data': frame_data,
        }

        print(f"\nSummary for E={E:.0e}:")
        print(f"  Avg iterations: {avg_iters:.1f}")
        print(f"  Avg time: {avg_time:.2f} ms/frame")
        print(f"  Max pos error: {max_pos_error:.2e}")
        print(f"  Max vel error: {max_vel_error:.2e}")
        print(f"  Status: {'PASSED' if passed else 'FAILED'}")

        return result

    def run_all_tests(self, frames=5, initial_vy=-1.0, verbose=False):
        """Run tests for all stiffness values."""
        print("\n" + "="*80)
        print("Stiffness Convergence Test")
        print("="*80)
        print(f"Testing E values: {[f'{e:.0e}' for e in self.stiffness_values]}")
        print(f"Frames per test: {frames}")
        print(f"Initial Vy: {initial_vy}")
        print("="*80)

        for E in self.stiffness_values:
            result = self.run_test(E, frames, initial_vy, verbose)
            self.results[E] = result

        self.print_comparison()
        return self.results

    def print_comparison(self):
        """Print comparison table."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        # Header
        print(f"{'E':>12} {'Avg Iters':>12} {'Avg Time(ms)':>14} "
              f"{'Max Pos Err':>12} {'Max Vel Err':>12} {'Status':>10}")
        print("-"*80)

        for E in sorted(self.results.keys()):
            r = self.results[E]
            status = "PASSED" if r['passed'] else "FAILED"
            print(f"{E:>12.0e} {r['avg_iters']:>12.1f} {r['avg_time_ms']:>14.2f} "
                  f"{r['max_pos_error']:>12.2e} {r['max_vel_error']:>12.2e} {status:>10}")

        print("="*80)

        # Analysis
        print("\n【分析】")

        # Find iteration trends
        E_values = sorted(self.results.keys())
        iters = [self.results[E]['avg_iters'] for E in E_values]

        print(f"刚度范围: {E_values[0]:.0e} ~ {E_values[-1]:.0e} (相差 {E_values[-1]/E_values[0]:.0f}x)")
        print(f"迭代次数: {min(iters):.1f} ~ {max(iters):.1f}")

        if iters[-1] > iters[0]:
            ratio = iters[-1] / iters[0]
            print(f"  => 刚度增加 {E_values[-1]/E_values[0]:.0f}x, 迭代增加 {ratio:.1f}x")
        else:
            print(f"  => 刚度变化对收敛影响不大")

        # Check all passed
        all_passed = all(r['passed'] for r in self.results.values())
        if all_passed:
            print(f"\n✓ 所有测试通过物理验证")
        else:
            failed = [f"{E:.0e}" for E, r in self.results.items() if not r['passed']]
            print(f"\n✗ 以下刚度值未通过验证: {failed}")


def main():
    parser = argparse.ArgumentParser(description='Stiffness Convergence Test')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames per test')
    parser.add_argument('--vy', type=float, default=-1.0, help='Initial downward velocity')
    parser.add_argument('--verbose', action='store_true', help='Show detailed iteration info')
    parser.add_argument('--E', type=float, nargs='+',
                        help='Custom E values to test (e.g., --E 1e4 1e5 1e6)')
    args = parser.parse_args()

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True,
            offline_cache_file_path=".taichi_cache")

    # Parse custom E values
    stiffness_values = None
    if args.E:
        stiffness_values = args.E

    # Run tests
    test = StiffnessConvergenceTest(stiffness_values)
    test.run_all_tests(frames=args.frames, initial_vy=args.vy, verbose=args.verbose)


if __name__ == '__main__':
    main()
