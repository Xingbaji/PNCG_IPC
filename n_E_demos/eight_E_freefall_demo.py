"""
Eight E Free-Fall Demo - Validates MAS solver with 8 E-shaped objects in free fall.

This demo validates the MAS preconditioner correctness by comparing simulation results
with Newton's laws ground truth in a collision-free free-fall scenario.

Ground Truth (Newton's Laws):
- Position: y(t) = y0 + v0*t + 0.5*g*t^2
- Velocity: v(t) = v0 + g*t

The 8 E-shaped objects fall freely under gravity with an initial downward velocity.
No collision detection, no ground barrier - pure elastic + inertia.

Usage:
    python eight_E_freefall_demo.py                  # Run with default settings
    python eight_E_freefall_demo.py --frames 10      # Run for 10 frames
    python eight_E_freefall_demo.py --verbose        # Show detailed iteration info
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths - find project root (PNCG_IPC) and set up properly
current_file_path = os.path.abspath(__file__)
n_E_demos_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(n_E_demos_dir)
demo_dir = os.path.join(project_root, 'demo')

# Add project root and demo to path
sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision


class EightEFreeFallValidator:
    """
    Free-fall validator wrapper for MASPNCGSolverNoCollision.

    Compares centroid motion with Newton's law prediction to validate
    that the MAS-PNCG solver produces correct physics.
    """

    def __init__(self, demo='eight_E_freefall'):
        """
        Args:
            demo: Demo configuration name
        """
        # Create the MAS-PNCG solver
        self.solver = MASPNCGSolverNoCollision(demo=demo)

        # Ground truth tracking
        self.initial_centroid = np.zeros(3)
        self.initial_velocity = np.zeros(3)
        self.time_elapsed = 0.0

        # Results storage
        self.frame_results = []

        # Calculate number of objects (each E object has 1046 vertices)
        self.object_size = 1046
        self.n_objects = self.solver.n_verts // self.object_size
        print(f"Number of E objects: {self.n_objects}")

    def compute_centroid(self):
        """Compute mass-weighted centroid of the mesh."""
        x_np = self.solver.mesh.verts.x.to_numpy()
        m_np = self.solver.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        centroid = np.sum(x_np * m_np[:, np.newaxis], axis=0) / total_mass
        return centroid, total_mass

    def compute_velocity_centroid(self):
        """Compute mass-weighted velocity of centroid."""
        v_np = self.solver.mesh.verts.v.to_numpy()
        m_np = self.solver.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        v_centroid = np.sum(v_np * m_np[:, np.newaxis], axis=0) / total_mass
        return v_centroid

    def newton_ground_truth(self, t):
        """
        Compute ground truth position and velocity using Newton's laws.

        y(t) = y0 + v0*t + 0.5*g*t^2
        v(t) = v0 + g*t
        """
        g = np.array([0.0, self.solver.gravity, 0.0])
        pos = self.initial_centroid + self.initial_velocity * t + 0.5 * g * t * t
        vel = self.initial_velocity + g * t
        return pos, vel

    def set_initial_velocity(self, vy=-1.0):
        """Set initial downward velocity for all vertices."""
        # Initialize velocity using numpy (avoid taichi kernel issues with nested mesh access)
        v_np = np.zeros((self.solver.n_verts, 3), dtype=np.float32)
        v_np[:, 1] = vy
        self.solver.mesh.verts.v.from_numpy(v_np)

        self.initial_velocity = np.array([0.0, vy, 0.0])
        self.initial_centroid, _ = self.compute_centroid()
        print(f"[Eight E FreeFall] Initial centroid: {self.initial_centroid}")
        print(f"[Eight E FreeFall] Initial velocity: {self.initial_velocity}")

    def step(self, verbose=False):
        """
        One time step using the MAS-PNCG solver.

        Returns:
            (iterations, elapsed_time_ms)
        """
        t_start = time.perf_counter()
        iters = self.solver.step(verbose=verbose)
        t_elapsed = (time.perf_counter() - t_start) * 1000

        # Update time tracking
        self.time_elapsed += self.solver.dt

        return iters, t_elapsed

    def validate_frame(self, verbose=True):
        """
        Validate current frame against Newton's ground truth.

        Returns:
            dict with simulation and ground truth results
        """
        # Get simulation results
        sim_centroid, total_mass = self.compute_centroid()
        sim_velocity = self.compute_velocity_centroid()

        # Get ground truth
        gt_pos, gt_vel = self.newton_ground_truth(self.time_elapsed)

        # Compute errors
        pos_error = np.linalg.norm(sim_centroid - gt_pos)
        vel_error = np.linalg.norm(sim_velocity - gt_vel)

        # Relative errors (use magnitude of ground truth as reference)
        displacement = np.linalg.norm(gt_pos - self.initial_centroid)
        pos_rel_error = pos_error / (displacement + 1e-10) if displacement > 1e-10 else pos_error
        vel_rel_error = vel_error / (np.linalg.norm(gt_vel) + 1e-10)

        result = {
            'frame': self.solver.frame,
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
            print(f"Frame {self.solver.frame} (t={self.time_elapsed:.4f}s):")
            print(f"  Centroid Y: sim={sim_centroid[1]:.6f}, gt={gt_pos[1]:.6f}, err={pos_error:.2e}")
            print(f"  Velocity Y: sim={sim_velocity[1]:.6f}, gt={gt_vel[1]:.6f}, err={vel_error:.2e}")

        self.frame_results.append(result)
        return result


def run_freefall_test(demo='eight_E_freefall', frames=5, initial_vy=-1.0, verbose=True):
    """
    Run free-fall validation test.

    Args:
        demo: Demo configuration name
        frames: Number of frames to simulate
        initial_vy: Initial downward velocity
        verbose: Print detailed output

    Returns:
        dict with test results
    """
    print(f"\n{'='*70}")
    print(f"Eight E Free-Fall Validation Test (MAS-PNCG NoCollision)")
    print(f"{'='*70}")
    print(f"Demo: {demo}")
    print(f"Frames: {frames}, Initial Vy: {initial_vy}")
    print(f"{'='*70}\n")

    # Create validator
    validator = EightEFreeFallValidator(demo=demo)

    print(f"Material: E={validator.solver.dict['E']}, nu={validator.solver.dict['nu']}")
    print(f"dt={validator.solver.dt}, gravity={validator.solver.gravity}")

    # Set initial velocity
    validator.set_initial_velocity(initial_vy)

    # Run simulation
    print(f"\n[Running {frames} frames...]")
    total_iters = 0
    total_time = 0.0

    for f in range(frames):
        iters, elapsed = validator.step(verbose=verbose)
        result = validator.validate_frame(verbose=verbose)
        total_iters += iters
        total_time += elapsed
        print(f"  => {iters} iters, {elapsed:.2f}ms\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}")

    # Compute final errors
    max_pos_error = max(r['pos_error'] for r in validator.frame_results)
    max_vel_error = max(r['vel_error'] for r in validator.frame_results)
    avg_pos_error = np.mean([r['pos_error'] for r in validator.frame_results])
    avg_vel_error = np.mean([r['vel_error'] for r in validator.frame_results])

    print(f"Position Error: max={max_pos_error:.2e}, avg={avg_pos_error:.2e}")
    print(f"Velocity Error: max={max_vel_error:.2e}, avg={avg_vel_error:.2e}")
    print(f"Total iterations: {total_iters}, Avg per frame: {total_iters/frames:.1f}")
    print(f"Total time: {total_time:.2f}ms, Avg per frame: {total_time/frames:.2f}ms")

    # Pass/Fail criteria
    # For a rigid-body-like motion, errors should be very small
    # Allow some tolerance for elastic deformation
    PASS_THRESHOLD_POS = 1e-2  # 1cm position error
    PASS_THRESHOLD_VEL = 1e-1  # 0.1 m/s velocity error

    passed = max_pos_error < PASS_THRESHOLD_POS and max_vel_error < PASS_THRESHOLD_VEL

    if passed:
        print(f"\n[PASSED] MAS-PNCG solver produces correct physics")
    else:
        print(f"\n[FAILED] Errors exceed threshold")
        print(f"  Position threshold: {PASS_THRESHOLD_POS}, actual: {max_pos_error:.2e}")
        print(f"  Velocity threshold: {PASS_THRESHOLD_VEL}, actual: {max_vel_error:.2e}")

    print(f"{'='*70}\n")

    return {
        'demo': demo,
        'passed': passed,
        'max_pos_error': max_pos_error,
        'max_vel_error': max_vel_error,
        'avg_pos_error': avg_pos_error,
        'avg_vel_error': avg_vel_error,
        'total_iters': total_iters,
        'total_time_ms': total_time,
        'avg_time_per_frame_ms': total_time / frames,
        'avg_iters_per_frame': total_iters / frames,
        'frame_results': validator.frame_results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eight E Free-Fall Validation Test')
    parser.add_argument('--demo', type=str, default='eight_E_freefall',
                        help='Demo configuration name')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames')
    parser.add_argument('--vy', type=float, default=-1.0, help='Initial downward velocity')
    parser.add_argument('--verbose', action='store_true', help='Show detailed iteration info')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    run_freefall_test(
        demo=args.demo,
        frames=args.frames,
        initial_vy=args.vy,
        verbose=args.verbose
    )
