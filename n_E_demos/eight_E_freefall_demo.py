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
    python eight_E_freefall_demo.py                  # Interactive mode (default)
    python eight_E_freefall_demo.py --headless       # Headless with image saving
    python eight_E_freefall_demo.py --frames 50      # Run for 50 frames
    python eight_E_freefall_demo.py --validate       # Run validation test only
    python eight_E_freefall_demo.py --verbose        # Show detailed iteration info

Controls (interactive mode):
    SPACE  - Pause/Resume simulation
    R      - Reset simulation
    V      - Toggle validation overlay
    RMB    - Hold and drag to rotate camera
    ESC    - Exit
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


class EightEFreeFallDemo:
    """
    Interactive visualization demo for Eight E Free-Fall.

    Supports both interactive mode (with GGUI) and headless mode (with image saving).
    """

    def __init__(self, demo='eight_E_freefall', initial_vy=-1.0):
        """
        Args:
            demo: Demo configuration name
            initial_vy: Initial downward velocity
        """
        self.validator = EightEFreeFallValidator(demo=demo)
        self.solver = self.validator.solver
        self.initial_vy = initial_vy

        # Set initial velocity
        self.validator.set_initial_velocity(initial_vy)

        # UI state
        self._window = None
        self._scene = None
        self._camera = None
        self._canvas = None

        # Demo state
        self.paused = False
        self.show_validation = True
        self.frame = 0
        self.timing_stats = []

    def _init_window(self, show_window=True):
        """Initialize Taichi UI window."""
        self._window = ti.ui.Window(
            "Eight E Free-Fall Demo",
            (1280, 720),
            vsync=True,
            show_window=show_window
        )
        self._canvas = self._window.get_canvas()
        self._scene = self._window.get_scene()
        self._camera = ti.ui.Camera()

        # Set camera - looking at the center of the 8 E objects
        self._camera.position(2.0, 3.0, 6.0)
        self._camera.lookat(1.5, 2.0, 0.0)
        self._camera.fov(60)
        self._canvas.set_background_color((0.15, 0.15, 0.2))

    def _render_frame(self, track_inputs=False):
        """Render the current frame."""
        if track_inputs and hasattr(self._window, 'RMB'):
            self._camera.track_user_inputs(
                self._window,
                movement_speed=0.2,
                hold_key=self._window.RMB
            )
        self._scene.set_camera(self._camera)
        self._scene.ambient_light((0.3, 0.3, 0.3))

        # Add multiple point lights for better illumination
        self._scene.point_light(pos=(3.0, 5.0, 3.0), color=(1.0, 0.95, 0.9))
        self._scene.point_light(pos=(-2.0, 3.0, 2.0), color=(0.6, 0.6, 0.8))

        # Render mesh with a nice color
        self._scene.mesh_instance(
            self.solver.mesh.verts.x,
            self.solver.indices,
            color=(0.7, 0.5, 0.3),
            show_wireframe=False
        )

        self._canvas.scene(self._scene)

    def _save_image(self, frame, img_dir):
        """Save current frame to image file."""
        filename = f"frame_{frame:05d}.png"
        save_path = os.path.join(img_dir, filename)
        self._window.save_image(save_path)

    def _print_status(self, stats):
        """Print frame status."""
        result = self.validator.frame_results[-1] if self.validator.frame_results else None
        pos_err = result['pos_error'] if result else 0.0
        vel_err = result['vel_error'] if result else 0.0

        print(f"[Frame {self.frame:4d}] "
              f"step: {stats['step_ms']:7.2f}ms | "
              f"iters: {stats['iterations']:3d} | "
              f"pos_err: {pos_err:.2e} | "
              f"vel_err: {vel_err:.2e}")

    def run_interactive(self, frames=200, verbose=False):
        """Run simulation with interactive visualization."""
        print(f"\n[Interactive Mode]")
        print("Controls: SPACE=pause, R=reset, V=toggle validation, RMB+drag=rotate, ESC=exit")

        self._init_window(show_window=True)

        while self._window.running and self.frame < frames:
            # Handle input
            for e in self._window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    self._window.running = False
                elif e.key == ti.ui.SPACE:
                    self.paused = not self.paused
                    print(f"[{'Paused' if self.paused else 'Running'}]")
                elif e.key == 'r':
                    self._reset()
                    print("[Reset]")
                elif e.key == 'v':
                    self.show_validation = not self.show_validation
                    print(f"[Validation overlay: {'ON' if self.show_validation else 'OFF'}]")

            # Step simulation
            if not self.paused:
                stats = self._step_with_timing(verbose)
                self.timing_stats.append(stats)

                if self.show_validation or verbose:
                    self._print_status(stats)

                self.frame += 1

            # Render
            self._render_frame(track_inputs=True)
            self._window.show()

        self._print_summary()

    def run_headless(self, frames=50, verbose=False, save_images=True, log_dir=None):
        """Run simulation in headless mode."""
        print(f"\n[Headless Mode] Running {frames} frames...")

        img_dir = None
        if save_images:
            if log_dir is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join(project_root, "logs", f"eight_E_freefall_{timestamp}")
            img_dir = os.path.join(log_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            self._init_window(show_window=False)
            print(f"[Saving images to: {img_dir}]")

        for f in range(frames):
            ti.sync()
            t_frame_start = time.perf_counter()

            # Step simulation
            stats = self._step_with_timing(verbose)

            # Render and save image
            if save_images and img_dir:
                ti.sync()
                t_render_start = time.perf_counter()
                self._render_frame()
                ti.sync()
                stats['render_ms'] = (time.perf_counter() - t_render_start) * 1000

                t_save_start = time.perf_counter()
                self._save_image(f, img_dir)
                stats['save_ms'] = (time.perf_counter() - t_save_start) * 1000

            ti.sync()
            stats['total_ms'] = (time.perf_counter() - t_frame_start) * 1000

            self.timing_stats.append(stats)
            self._print_status(stats)
            self.frame += 1

        print("\n[Headless Mode] Finished")
        self._print_summary()

        return self._get_validation_results()

    def _step_with_timing(self, verbose=False):
        """Execute one simulation step with timing."""
        ti.sync()
        t_start = time.perf_counter()

        iters, _ = self.validator.step(verbose=verbose)
        self.validator.validate_frame(verbose=False)

        ti.sync()
        step_ms = (time.perf_counter() - t_start) * 1000

        return {
            'frame': self.frame,
            'step_ms': step_ms,
            'iterations': iters,
            'render_ms': 0.0,
            'save_ms': 0.0,
            'total_ms': step_ms,
        }

    def _reset(self):
        """Reset simulation to initial state."""
        if hasattr(self.solver, 'restart'):
            self.solver.restart()

        self.validator.time_elapsed = 0.0
        self.validator.frame_results.clear()
        self.validator.set_initial_velocity(self.initial_vy)

        self.frame = 0
        self.timing_stats.clear()

    def _print_summary(self):
        """Print summary statistics."""
        if not self.timing_stats:
            return

        step_times = [s['step_ms'] for s in self.timing_stats]
        iterations = [s['iterations'] for s in self.timing_stats]

        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Frames completed: {len(self.timing_stats)}")

        print(f"\nTiming (ms):")
        print(f"  Step   - avg: {np.mean(step_times):7.2f}, "
              f"min: {np.min(step_times):7.2f}, max: {np.max(step_times):7.2f}")

        print(f"\nIterations:")
        print(f"  avg: {np.mean(iterations):.2f}, "
              f"min: {np.min(iterations)}, max: {np.max(iterations)}")

        if self.validator.frame_results:
            results = self._get_validation_results()
            print(f"\nValidation (Newton's Laws):")
            print(f"  Position Error: max={results['max_pos_error']:.2e}, avg={results['avg_pos_error']:.2e}")
            print(f"  Velocity Error: max={results['max_vel_error']:.2e}, avg={results['avg_vel_error']:.2e}")
            status = "[PASSED]" if results['passed'] else "[FAILED]"
            print(f"  Status: {status}")

        print("=" * 70)

    def _get_validation_results(self):
        """Get validation results summary."""
        if not self.validator.frame_results:
            return None

        max_pos_error = max(r['pos_error'] for r in self.validator.frame_results)
        max_vel_error = max(r['vel_error'] for r in self.validator.frame_results)
        avg_pos_error = np.mean([r['pos_error'] for r in self.validator.frame_results])
        avg_vel_error = np.mean([r['vel_error'] for r in self.validator.frame_results])

        PASS_THRESHOLD_POS = 1e-2
        PASS_THRESHOLD_VEL = 1e-1
        passed = max_pos_error < PASS_THRESHOLD_POS and max_vel_error < PASS_THRESHOLD_VEL

        return {
            'passed': passed,
            'max_pos_error': max_pos_error,
            'max_vel_error': max_vel_error,
            'avg_pos_error': avg_pos_error,
            'avg_vel_error': avg_vel_error,
            'frame_results': self.validator.frame_results,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Eight E Free-Fall Demo with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (interactive mode):
  SPACE  - Pause/Resume simulation
  R      - Reset simulation
  V      - Toggle validation overlay
  RMB    - Hold and drag to rotate camera
  ESC    - Exit
        """
    )
    parser.add_argument('--demo', type=str, default='eight_E_freefall',
                        help='Demo configuration name')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames')
    parser.add_argument('--vy', type=float, default=-1.0, help='Initial downward velocity')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--validate', action='store_true', help='Run validation test only (no visualization)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed iteration info')
    parser.add_argument('--no-save-images', action='store_true', help='Disable saving images in headless mode')
    parser.add_argument('--log-dir', type=str, default='', help='Custom log directory for headless mode')
    args = parser.parse_args()

    # GPU backend (ti.mesh_local disabled in solver to avoid scalarize bug)
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True, offline_cache_file_path=".taichi_cache")

    if args.validate:
        # Run validation test only (no visualization)
        run_freefall_test(
            demo=args.demo,
            frames=args.frames,
            initial_vy=args.vy,
            verbose=args.verbose
        )
    else:
        # Run with visualization
        demo = EightEFreeFallDemo(demo=args.demo, initial_vy=args.vy)

        if args.headless:
            demo.run_headless(
                frames=args.frames,
                verbose=args.verbose,
                save_images=not args.no_save_images,
                log_dir=args.log_dir if args.log_dir else None
            )
        else:
            demo.run_interactive(
                frames=args.frames,
                verbose=args.verbose
            )
