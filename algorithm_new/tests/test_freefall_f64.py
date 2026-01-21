"""
Free-Fall Validation Test for algorithm_new Architecture (Float64)

This demo validates the algorithm_new modular architecture by comparing
simulation results with Newton's laws ground truth in a collision-free free-fall scenario.

Tests:
1. MeshSystem + PNCGOptimizer with diagonal preconditioner
2. MeshSystem + PNCGOptimizer with MAS preconditioner (BANKSIZE=16)
3. MeshSystem + PNCGOptimizer with MAS-8 preconditioner (BANKSIZE=8)

Usage:
    python test_freefall_f64.py --frames 5           # Run for 5 frames
    python test_freefall_f64.py --verbose            # Show detailed iteration info
    python test_freefall_f64.py --preconditioner mas # Use MAS preconditioner
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_new_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(algorithm_new_dir)
sys.path.insert(0, project_root)

import taichi as ti

# Import algorithm_new modules
from algorithm_new.mesh.mesh_system import MeshSystem
from algorithm_new.optimizer.pncg_optimizer import PNCGOptimizer
from algorithm_new.preconditioner.registry import PreconditionerRegistry
from algorithm_new.preconditioner.diagonal import DiagonalPreconditioner

# Import mesh loading utilities
import meshtaichi_patcher as Patcher


class FreeFallValidator:
    """
    Free-fall validator for algorithm_new modular architecture.

    Validates physics correctness by comparing simulated trajectory
    with Newton's laws ground truth.
    """

    def __init__(
        self,
        model_path: str,
        precision: str = 'f64',
        preconditioner_type: str = 'diagonal',
        E: float = 1e4,
        nu: float = 0.4,
        density: float = 1000.0,
        dt: float = 0.04,
        gravity: float = -9.8,
        epsilon: float = 1e-6,
        iter_max: int = 50,
        metis_reordered: bool = True,
    ):
        """
        Initialize the free-fall validator.

        Args:
            model_path: Path to mesh .node file
            precision: Float precision ('f32' or 'f64')
            preconditioner_type: 'diagonal', 'mas', or 'mas8_contact'
            E: Young's modulus
            nu: Poisson's ratio
            density: Material density (kg/m^3)
            dt: Time step
            gravity: Gravity acceleration (negative for downward)
            epsilon: Convergence tolerance
            iter_max: Maximum iterations per step
            metis_reordered: Whether to use METIS reordering
        """
        self.precision = precision
        self.dt = dt
        self.gravity = gravity
        self.epsilon = epsilon
        self.iter_max = iter_max

        # Compute Lame parameters
        mu = E / (2.0 * (1.0 + nu))
        la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        print(f'\n{"="*70}')
        print(f'Free-Fall Validation Test (algorithm_new, {precision})')
        print(f'{"="*70}')
        print(f'Model: {model_path}')
        print(f'Material: E={E:.2e}, nu={nu}, density={density}')
        print(f'Solver: dt={dt}, epsilon={epsilon}, gravity={gravity}')
        print(f'Preconditioner: {preconditioner_type}')
        print(f'{"="*70}\n')

        # Load mesh
        print('[FreeFall] Loading mesh...')
        self.mesh = self._load_mesh(model_path, metis_reordered)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print(f'[FreeFall] Mesh loaded: {self.n_verts} vertices, {self.n_cells} cells')

        # Create mesh system
        print('[FreeFall] Creating MeshSystem...')
        self.mesh_system = MeshSystem(
            mesh=self.mesh,
            density=density,
            mu=mu,
            la=la,
            elastic_type='ARAP_filter',
            precision=precision,
            dt=dt,
            gravity=gravity,
        )

        # Create optimizer
        print('[FreeFall] Creating PNCGOptimizer...')
        self.optimizer = PNCGOptimizer(
            mesh=self.mesh,
            mesh_system=self.mesh_system,
            dt=dt,
            epsilon=epsilon,
            iter_max=iter_max,
            precision=precision,
        )

        # Create preconditioner
        print(f'[FreeFall] Creating preconditioner ({preconditioner_type})...')
        self.preconditioner = PreconditionerRegistry.create(
            preconditioner_type,
            mesh=self.mesh,
            precision=precision,
            metis_reordered=metis_reordered,
        )
        self.optimizer.set_preconditioner(self.preconditioner)

        # Ground truth tracking
        self.initial_centroid = np.zeros(3)
        self.initial_velocity = np.zeros(3)
        self.time_elapsed = 0.0
        self.frame = 0

        # Results storage
        self.frame_results = []

        print('[FreeFall] Initialization complete.\n')

    def _load_mesh(self, model_path: str, metis_reordered: bool):
        """Load mesh with optional METIS reordering."""
        # Load raw mesh data
        model_data = Patcher.load_mesh_rawdata(model_path)
        vertices = model_data[0]
        cells = model_data[3]

        # Apply METIS reordering if available and requested
        if metis_reordered:
            try:
                from algorithm.mas_preconditioner_small import (
                    reorder_mesh_data_metis,
                    check_pymetis_available,
                )
                if check_pymetis_available():
                    print('[FreeFall] Applying METIS reordering...')
                    vertices, cells, _ = reorder_mesh_data_metis(vertices, cells, 16)
            except ImportError:
                print('[FreeFall] METIS not available, using original ordering')

        # Create mesh dict and load
        mesh_dict = {0: vertices, 3: cells.astype(np.int32)}
        mesh = Patcher.load_mesh(mesh_dict, relations=["CV"])
        return mesh

    def compute_centroid(self):
        """Compute mass-weighted centroid of the mesh."""
        x_np = self.mesh.verts.x.to_numpy()
        m_np = self.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        centroid = np.sum(x_np * m_np[:, np.newaxis], axis=0) / total_mass
        return centroid, total_mass

    def compute_velocity_centroid(self):
        """Compute mass-weighted velocity of centroid."""
        v_np = self.mesh.verts.v.to_numpy()
        m_np = self.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        v_centroid = np.sum(v_np * m_np[:, np.newaxis], axis=0) / total_mass
        return v_centroid

    def newton_ground_truth(self, t):
        """Compute ground truth position and velocity using Newton's laws."""
        g = np.array([0.0, self.gravity, 0.0])
        pos = self.initial_centroid + self.initial_velocity * t + 0.5 * g * t * t
        vel = self.initial_velocity + g * t
        return pos, vel

    def set_initial_velocity(self, vy=-1.0):
        """Set initial downward velocity for all vertices."""
        v_np = np.zeros((self.n_verts, 3), dtype=np.float64 if self.precision == 'f64' else np.float32)
        v_np[:, 1] = vy
        self.mesh.verts.v.from_numpy(v_np)

        self.initial_velocity = np.array([0.0, vy, 0.0])
        self.initial_centroid, _ = self.compute_centroid()
        print(f'[FreeFall] Initial centroid: {self.initial_centroid}')
        print(f'[FreeFall] Initial velocity: {self.initial_velocity}')

    def step(self, verbose=False):
        """One time step using the PNCG optimizer."""
        t_start = time.perf_counter()
        iters = self.optimizer.step(verbose=verbose)
        t_elapsed = (time.perf_counter() - t_start) * 1000

        self.time_elapsed += self.dt
        self.frame += 1
        return iters, t_elapsed

    def validate_frame(self, verbose=True):
        """Validate current frame against Newton's ground truth."""
        sim_centroid, total_mass = self.compute_centroid()
        sim_velocity = self.compute_velocity_centroid()
        gt_pos, gt_vel = self.newton_ground_truth(self.time_elapsed)

        pos_error = np.linalg.norm(sim_centroid - gt_pos)
        vel_error = np.linalg.norm(sim_velocity - gt_vel)

        displacement = np.linalg.norm(gt_pos - self.initial_centroid)
        pos_rel_error = pos_error / (displacement + 1e-10) if displacement > 1e-10 else pos_error
        vel_rel_error = vel_error / (np.linalg.norm(gt_vel) + 1e-10)

        result = {
            'frame': self.frame,
            'time': self.time_elapsed,
            'sim_centroid': sim_centroid.copy(),
            'gt_centroid': gt_pos.copy(),
            'sim_velocity': sim_velocity.copy(),
            'gt_velocity': gt_vel.copy(),
            'pos_error': pos_error,
            'vel_error': vel_error,
            'pos_rel_error': pos_rel_error,
            'vel_rel_error': vel_rel_error,
        }

        if verbose:
            print(f'Frame {self.frame} (t={self.time_elapsed:.4f}s):')
            print(f'  Centroid Y: sim={sim_centroid[1]:.6f}, gt={gt_pos[1]:.6f}, err={pos_error:.2e}')
            print(f'  Velocity Y: sim={sim_velocity[1]:.6f}, gt={gt_vel[1]:.6f}, err={vel_error:.2e}')

        self.frame_results.append(result)
        return result


def run_freefall_test(
    model_path: str = None,
    frames: int = 5,
    initial_vy: float = -1.0,
    verbose: bool = True,
    precision: str = 'f64',
    preconditioner_type: str = 'diagonal',
    E: float = 1e4,
    nu: float = 0.4,
    density: float = 1000.0,
    dt: float = 0.04,
    gravity: float = -9.8,
    epsilon: float = 1e-6,
    iter_max: int = 50,
):
    """
    Run free-fall validation test with algorithm_new.

    Args:
        model_path: Path to mesh file (defaults to cube_10)
        frames: Number of frames to simulate
        initial_vy: Initial downward velocity
        verbose: Whether to print detailed iteration info
        precision: Float precision ('f32' or 'f64')
        preconditioner_type: 'diagonal', 'mas', or 'mas8_contact'
        E: Young's modulus
        nu: Poisson's ratio
        density: Material density
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations

    Returns:
        dict with test results
    """
    # Default model path
    if model_path is None:
        model_path = os.path.join(project_root, 'model/mesh/cube_10/cube_10.node')

    validator = FreeFallValidator(
        model_path=model_path,
        precision=precision,
        preconditioner_type=preconditioner_type,
        E=E,
        nu=nu,
        density=density,
        dt=dt,
        gravity=gravity,
        epsilon=epsilon,
        iter_max=iter_max,
    )

    validator.set_initial_velocity(initial_vy)

    print(f'\n[Running {frames} frames...]')
    total_iters = 0
    total_time = 0.0

    for f in range(frames):
        iters, elapsed = validator.step(verbose=verbose)
        result = validator.validate_frame(verbose=verbose)
        total_iters += iters
        total_time += elapsed
        print(f'  => {iters} iters, {elapsed:.2f}ms\n')

    # Summary
    print(f'\n{"="*70}')
    print(f'Test Summary')
    print(f'{"="*70}')

    max_pos_error = max(r['pos_error'] for r in validator.frame_results)
    max_vel_error = max(r['vel_error'] for r in validator.frame_results)
    avg_pos_error = np.mean([r['pos_error'] for r in validator.frame_results])
    avg_vel_error = np.mean([r['vel_error'] for r in validator.frame_results])

    print(f'Position Error: max={max_pos_error:.2e}, avg={avg_pos_error:.2e}')
    print(f'Velocity Error: max={max_vel_error:.2e}, avg={avg_vel_error:.2e}')
    print(f'Total iterations: {total_iters}, Avg per frame: {total_iters/frames:.1f}')
    print(f'Total time: {total_time:.2f}ms, Avg per frame: {total_time/frames:.2f}ms')

    # Check pass/fail
    PASS_THRESHOLD_POS = 1e-2
    PASS_THRESHOLD_VEL = 1e-1

    passed = max_pos_error < PASS_THRESHOLD_POS and max_vel_error < PASS_THRESHOLD_VEL

    if passed:
        print(f'\n[PASSED] algorithm_new ({preconditioner_type}) produces correct physics')
    else:
        print(f'\n[FAILED] Errors exceed threshold')
        print(f'  Position threshold: {PASS_THRESHOLD_POS}, actual: {max_pos_error:.2e}')
        print(f'  Velocity threshold: {PASS_THRESHOLD_VEL}, actual: {max_vel_error:.2e}')

    print(f'{"="*70}\n')

    return {
        'preconditioner': preconditioner_type,
        'precision': precision,
        'passed': passed,
        'max_pos_error': max_pos_error,
        'max_vel_error': max_vel_error,
        'avg_pos_error': avg_pos_error,
        'avg_vel_error': avg_vel_error,
        'total_iters': total_iters,
        'total_time_ms': total_time,
        'frame_results': validator.frame_results,
    }


def run_comparison_test(frames: int = 5, verbose: bool = False):
    """
    Run comparison test across all preconditioner types.

    Args:
        frames: Number of frames to simulate
        verbose: Whether to print detailed iteration info

    Returns:
        dict mapping preconditioner type to results
    """
    print(f'\n{"="*70}')
    print(f'Preconditioner Comparison Test (algorithm_new, f64)')
    print(f'{"="*70}\n')

    preconditioners = ['diagonal', 'mas']

    # Try to include mas8_contact if available
    try:
        from algorithm_new.preconditioner import MASPreconditioner8Contact
        preconditioners.append('mas8_contact')
    except ImportError:
        print('[Warning] MAS-8 preconditioner not available')

    results = {}

    for precond_type in preconditioners:
        print(f'\n{"="*70}')
        print(f'Testing: {precond_type}')
        print(f'{"="*70}\n')

        try:
            results[precond_type] = run_freefall_test(
                frames=frames,
                verbose=verbose,
                precision='f64',
                preconditioner_type=precond_type,
            )
        except Exception as e:
            print(f'[ERROR] {precond_type} failed: {e}')
            results[precond_type] = {'passed': False, 'error': str(e)}

    # Print comparison summary
    print(f'\n{"="*70}')
    print(f'Comparison Summary')
    print(f'{"="*70}')
    print(f'{"Preconditioner":>15} {"Passed":>8} {"Max Pos Err":>12} {"Max Vel Err":>12} {"Iters":>8} {"Time(ms)":>10}')
    print(f'{"-"*70}')

    for precond_type, result in results.items():
        if 'error' in result:
            print(f'{precond_type:>15} {"FAILED":>8} {"N/A":>12} {"N/A":>12} {"N/A":>8} {"N/A":>10}')
        else:
            status = "PASS" if result['passed'] else "FAIL"
            print(f'{precond_type:>15} {status:>8} {result["max_pos_error"]:>12.2e} '
                  f'{result["max_vel_error"]:>12.2e} {result["total_iters"]:>8} '
                  f'{result["total_time_ms"]:>10.2f}')

    print(f'{"="*70}\n')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Free-Fall Validation Test for algorithm_new (f64)'
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to mesh .node file')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames')
    parser.add_argument('--vy', type=float, default=-1.0, help='Initial downward velocity')
    parser.add_argument('--verbose', action='store_true', help='Show detailed iteration info')
    parser.add_argument('--preconditioner', type=str, default='diagonal',
                        choices=['diagonal', 'mas', 'mas8_contact'],
                        help='Preconditioner type')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison test across all preconditioners')
    parser.add_argument('--E', type=float, default=1e4, help='Young\'s modulus')
    parser.add_argument('--nu', type=float, default=0.4, help='Poisson\'s ratio')
    parser.add_argument('--density', type=float, default=1000.0, help='Material density')
    parser.add_argument('--dt', type=float, default=0.04, help='Time step')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Convergence tolerance')

    args = parser.parse_args()

    # Initialize Taichi with f64 precision
    ti.init(
        arch=ti.gpu,
        default_fp=ti.f64,
        offline_cache=True,
        offline_cache_file_path=os.path.join(project_root, '.taichi_cache_test_freefall_f64')
    )

    if args.compare:
        run_comparison_test(frames=args.frames, verbose=args.verbose)
    else:
        run_freefall_test(
            model_path=args.model,
            frames=args.frames,
            initial_vy=args.vy,
            verbose=args.verbose,
            precision='f64',
            preconditioner_type=args.preconditioner,
            E=args.E,
            nu=args.nu,
            density=args.density,
            dt=args.dt,
            epsilon=args.epsilon,
        )
