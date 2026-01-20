"""
Soft Cubes Drop Demo - Tests MAS Contact Solver with Woodbury updates.

This demo validates the MAS-PNCG solver with IPC contact handling by simulating
multiple soft cubes falling and colliding with each other and the ground.

Key features tested:
- MAS preconditioner with contact Hessian assembly
- Woodbury low-rank updates for incremental contact changes
- IPC barrier contact (log or cubic)
- Ground collision handling

Usage:
    python soft_cubes_drop_demo.py                    # Interactive mode
    python soft_cubes_drop_demo.py --headless         # Headless mode
    python soft_cubes_drop_demo.py --frames 100       # Run for 100 frames
    python soft_cubes_drop_demo.py --no-woodbury      # Disable Woodbury updates
    python soft_cubes_drop_demo.py --verbose          # Show detailed iteration info

Controls (interactive mode):
    SPACE  - Pause/Resume simulation
    R      - Reset simulation
    W      - Toggle Woodbury updates
    RMB    - Hold and drag to rotate camera
    ESC    - Exit
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


class SoftCubesDropDemo:
    """
    Soft cubes drop demo with MAS Contact Solver.

    Tests the MAS preconditioner with IPC contact handling and Woodbury updates.
    """

    def __init__(self, demo='soft_cubes_drop', use_woodbury=True):
        """
        Args:
            demo: Demo configuration name
            use_woodbury: Enable Woodbury updates for incremental contact changes
        """
        from algorithm.mas_pncg_solver import MASPNCGSolver

        print(f"\n{'='*70}")
        print(f"Soft Cubes Drop Demo - MAS Contact Solver Test")
        print(f"{'='*70}")
        print(f"Demo: {demo}")
        print(f"Woodbury updates: {'Enabled' if use_woodbury else 'Disabled'}")
        print(f"{'='*70}\n")

        # Create solver
        self.solver = MASPNCGSolver(demo=demo)
        self.use_woodbury = use_woodbury

        # Extract solver parameters
        self.n_verts = self.solver.n_verts
        self.dt = self.solver.dt
        self.gravity = self.solver.gravity

        print(f"Vertices: {self.n_verts}")
        print(f"Material: E={self.solver.dict['E']}, nu={self.solver.dict['nu']}")
        print(f"dt={self.dt}, gravity={self.gravity}")
        print(f"dHat={self.solver.dHat}, kappa={self.solver.kappa}")

        # Calculate number of cubes (cube_10 has 491 vertices)
        self.cube_verts = 491
        self.n_cubes = self.n_verts // self.cube_verts
        print(f"Number of cubes: {self.n_cubes}")

        # Stats tracking
        self.frame = 0
        self.timing_stats = []
        self.contact_stats = []

        # UI state
        self._window = None
        self._scene = None
        self._camera = None
        self._canvas = None
        self.paused = False

    def compute_centroid(self):
        """Compute mass-weighted centroid of the mesh."""
        x_np = self.solver.mesh.verts.x.to_numpy()
        m_np = self.solver.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        centroid = np.sum(x_np * m_np[:, np.newaxis], axis=0) / total_mass
        return centroid

    def step(self, verbose=False):
        """Execute one simulation step."""
        ti.sync()
        t_start = time.perf_counter()

        iters = self.solver.step(verbose=verbose, use_woodbury=self.use_woodbury)

        ti.sync()
        step_ms = (time.perf_counter() - t_start) * 1000

        # Get contact count
        n_contacts = self.solver.n_contacts[None]

        # Get Woodbury stats if available
        woodbury_updates = 0
        if hasattr(self.solver.mas_preconditioner, '_woodbury'):
            stats = self.solver.mas_preconditioner.get_woodbury_stats()
            woodbury_updates = stats.get('n_updates_total', 0)

        return {
            'frame': self.frame,
            'step_ms': step_ms,
            'iterations': iters,
            'n_contacts': n_contacts,
            'woodbury_updates': woodbury_updates,
        }

    def _init_window(self, show_window=True):
        """Initialize Taichi UI window."""
        self._window = ti.ui.Window(
            "Soft Cubes Drop Demo",
            (1280, 720),
            vsync=True,
            show_window=show_window
        )
        self._canvas = self._window.get_canvas()
        self._scene = self._window.get_scene()
        self._camera = ti.ui.Camera()

        # Set camera
        self._camera.position(2.5, 1.5, 3.0)
        self._camera.lookat(0.5, 0.5, 0.5)
        self._camera.fov(60)
        self._canvas.set_background_color((0.1, 0.1, 0.15))

    def _render_frame(self, track_inputs=False):
        """Render the current frame."""
        if track_inputs and hasattr(self._window, 'RMB'):
            self._camera.track_user_inputs(
                self._window,
                movement_speed=0.2,
                hold_key=self._window.RMB
            )
        self._scene.set_camera(self._camera)
        self._scene.ambient_light((0.4, 0.4, 0.4))

        # Lighting
        self._scene.point_light(pos=(3.0, 5.0, 3.0), color=(1.0, 0.95, 0.9))
        self._scene.point_light(pos=(-2.0, 3.0, 2.0), color=(0.5, 0.5, 0.7))

        # Render cubes
        self._scene.mesh_instance(
            self.solver.mesh.verts.x,
            self.solver.indices,
            color=(0.6, 0.4, 0.2),
            show_wireframe=False
        )

        # Render ground plane (simple visual indicator)
        # (Taichi GGUI doesn't have built-in ground plane, so we skip this)

        self._canvas.scene(self._scene)

    def _save_image(self, frame, img_dir):
        """Save current frame to image file."""
        filename = f"frame_{frame:05d}.png"
        save_path = os.path.join(img_dir, filename)
        self._window.save_image(save_path)

    def _print_status(self, stats):
        """Print frame status."""
        print(f"[Frame {stats['frame']:4d}] "
              f"step: {stats['step_ms']:7.2f}ms | "
              f"iters: {stats['iterations']:3d} | "
              f"contacts: {stats['n_contacts']:5d} | "
              f"woodbury: {stats['woodbury_updates']:3d}")

    def run_interactive(self, frames=200, verbose=False):
        """Run simulation with interactive visualization."""
        print(f"\n[Interactive Mode]")
        print("Controls: SPACE=pause, R=reset, W=toggle Woodbury, RMB+drag=rotate, ESC=exit")

        self._init_window(show_window=True)

        while self._window.running and self.frame < frames:
            # Handle input
            for e in self._window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    self._window.running = False
                elif e.key == ti.ui.SPACE:
                    self.paused = not self.paused
                    print(f"[{'Paused' if self.paused else 'Running'}]")
                elif e.key == 'w':
                    self.use_woodbury = not self.use_woodbury
                    print(f"[Woodbury updates: {'ON' if self.use_woodbury else 'OFF'}]")

            # Step simulation
            if not self.paused:
                stats = self.step(verbose=verbose)
                self.timing_stats.append(stats)
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
                log_dir = os.path.join(project_root, "logs", f"soft_cubes_drop_{timestamp}")
            img_dir = os.path.join(log_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            self._init_window(show_window=False)
            print(f"[Saving images to: {img_dir}]")

        for f in range(frames):
            ti.sync()
            t_frame_start = time.perf_counter()

            # Step simulation
            stats = self.step(verbose=verbose)

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

    def _print_summary(self):
        """Print summary statistics."""
        if not self.timing_stats:
            return

        step_times = [s['step_ms'] for s in self.timing_stats]
        iterations = [s['iterations'] for s in self.timing_stats]
        contacts = [s['n_contacts'] for s in self.timing_stats]
        woodbury = [s['woodbury_updates'] for s in self.timing_stats]

        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Frames completed: {len(self.timing_stats)}")
        print(f"Woodbury updates: {'Enabled' if self.use_woodbury else 'Disabled'}")

        print(f"\nTiming (ms):")
        print(f"  Step   - avg: {np.mean(step_times):7.2f}, "
              f"min: {np.min(step_times):7.2f}, max: {np.max(step_times):7.2f}")
        print(f"  Total  - {np.sum(step_times):.2f}ms")

        print(f"\nIterations:")
        print(f"  avg: {np.mean(iterations):.2f}, "
              f"min: {np.min(iterations)}, max: {np.max(iterations)}")

        print(f"\nContacts:")
        print(f"  avg: {np.mean(contacts):.1f}, "
              f"min: {np.min(contacts)}, max: {np.max(contacts)}")

        if self.use_woodbury and any(w > 0 for w in woodbury):
            print(f"\nWoodbury Updates:")
            print(f"  avg: {np.mean(woodbury):.1f}, "
                  f"min: {np.min(woodbury)}, max: {np.max(woodbury)}")

        print("=" * 70)


def run_benchmark(demo='soft_cubes_drop', frames=20, verbose=True):
    """
    Run benchmark comparison between Woodbury enabled and disabled.

    Args:
        demo: Demo configuration name
        frames: Number of frames to run
        verbose: Print detailed output
    """
    print(f"\n{'='*70}")
    print(f"MAS Contact Solver Benchmark")
    print(f"{'='*70}")
    print(f"Demo: {demo}, Frames: {frames}")
    print(f"{'='*70}\n")

    results = {}

    # Test with Woodbury disabled
    print("\n[Test 1: Woodbury DISABLED]")
    demo1 = SoftCubesDropDemo(demo=demo, use_woodbury=False)
    for f in range(frames):
        stats = demo1.step(verbose=verbose)
        demo1.timing_stats.append(stats)
        demo1._print_status(stats)
        demo1.frame += 1

    results['no_woodbury'] = {
        'avg_step_ms': np.mean([s['step_ms'] for s in demo1.timing_stats]),
        'avg_iters': np.mean([s['iterations'] for s in demo1.timing_stats]),
        'total_ms': np.sum([s['step_ms'] for s in demo1.timing_stats]),
    }

    # Test with Woodbury enabled
    print("\n[Test 2: Woodbury ENABLED]")
    demo2 = SoftCubesDropDemo(demo=demo, use_woodbury=True)
    for f in range(frames):
        stats = demo2.step(verbose=verbose)
        demo2.timing_stats.append(stats)
        demo2._print_status(stats)
        demo2.frame += 1

    results['with_woodbury'] = {
        'avg_step_ms': np.mean([s['step_ms'] for s in demo2.timing_stats]),
        'avg_iters': np.mean([s['iterations'] for s in demo2.timing_stats]),
        'total_ms': np.sum([s['step_ms'] for s in demo2.timing_stats]),
    }

    # Summary
    print(f"\n{'='*70}")
    print(f"Benchmark Results")
    print(f"{'='*70}")
    print(f"{'':20} {'No Woodbury':>15} {'With Woodbury':>15} {'Speedup':>10}")
    print(f"{'-'*70}")
    print(f"{'Avg Step (ms)':20} {results['no_woodbury']['avg_step_ms']:>15.2f} "
          f"{results['with_woodbury']['avg_step_ms']:>15.2f} "
          f"{results['no_woodbury']['avg_step_ms']/results['with_woodbury']['avg_step_ms']:>10.2f}x")
    print(f"{'Avg Iterations':20} {results['no_woodbury']['avg_iters']:>15.1f} "
          f"{results['with_woodbury']['avg_iters']:>15.1f} "
          f"{'--':>10}")
    print(f"{'Total Time (ms)':20} {results['no_woodbury']['total_ms']:>15.2f} "
          f"{results['with_woodbury']['total_ms']:>15.2f} "
          f"{results['no_woodbury']['total_ms']/results['with_woodbury']['total_ms']:>10.2f}x")
    print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Soft Cubes Drop Demo - MAS Contact Solver Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (interactive mode):
  SPACE  - Pause/Resume simulation
  W      - Toggle Woodbury updates
  RMB    - Hold and drag to rotate camera
  ESC    - Exit
        """
    )
    parser.add_argument('--demo', type=str, default='soft_cubes_drop',
                        help='Demo configuration name')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--no-woodbury', action='store_true', help='Disable Woodbury updates')
    parser.add_argument('--benchmark', action='store_true', help='Run Woodbury benchmark')
    parser.add_argument('--verbose', action='store_true', help='Show detailed iteration info')
    parser.add_argument('--no-save-images', action='store_true', help='Disable saving images')
    parser.add_argument('--log-dir', type=str, default='', help='Custom log directory')
    args = parser.parse_args()

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True,
            offline_cache_file_path=".taichi_cache")

    if args.benchmark:
        # Run benchmark comparison
        run_benchmark(
            demo=args.demo,
            frames=args.frames,
            verbose=args.verbose
        )
    else:
        # Run demo
        demo = SoftCubesDropDemo(
            demo=args.demo,
            use_woodbury=not args.no_woodbury
        )

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
