"""
Demo runner framework with timing, logging, and headless rendering support.

This module provides a reusable base class for running demos with:
- Debug mode timing for each simulation phase
- Headless mode with automatic image saving to log directory
- Statistics collection and reporting
- Clean, consistent interface across all demos

Usage:
    class MyDemo(DemoRunner):
        def __init__(self, demo='cube'):
            solver = MySolver(demo=demo)
            super().__init__(solver, demo_name=demo)

        def setup(self):
            # Custom setup code
            pass

    if __name__ == '__main__':
        runner = MyDemo(demo='cube')
        runner.run()  # Automatically handles args
"""

import argparse
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TimingStats:
    """Timing statistics for a single frame."""
    frame: int = 0
    total_ms: float = 0.0
    step_ms: float = 0.0
    render_ms: float = 0.0
    collision_ms: float = 0.0
    solver_ms: float = 0.0
    save_ms: float = 0.0
    iterations: int = 0

    @property
    def fps(self) -> float:
        """Calculate FPS from total time."""
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0

    def __str__(self) -> str:
        return (f"Frame {self.frame:4d}: total={self.total_ms:7.2f}ms, "
                f"step={self.step_ms:7.2f}ms, render={self.render_ms:6.2f}ms, "
                f"iters={self.iterations}, FPS={self.fps:.2f}")


@dataclass
class RunConfig:
    """Configuration for demo run."""
    headless: bool = False
    debug: bool = False
    frames: int = 50
    save_images: bool = True
    save_stats: bool = True
    log_dir: str = ""
    demo_name: str = ""

    def __post_init__(self):
        if not self.log_dir and self.demo_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                f"{self.demo_name}_{timestamp}"
            )


class DemoRunner:
    """
    Base class for running simulation demos.

    Provides:
    - Argument parsing for headless/debug/frames
    - Timing instrumentation for debug mode
    - Automatic image saving in headless mode
    - Statistics collection and reporting
    """

    def __init__(self, solver, demo_name: str = "demo"):
        """
        Initialize the demo runner.

        Args:
            solver: The simulation solver instance (must have step() method)
            demo_name: Name of the demo for logging
        """
        self.solver = solver
        self.demo_name = demo_name
        self.config: Optional[RunConfig] = None
        self.timing_stats: List[TimingStats] = []
        self._window = None
        self._scene = None
        self._camera = None
        self._canvas = None

    def setup(self):
        """Override this method for custom setup."""
        pass

    def pre_step(self, frame: int):
        """Override for custom pre-step logic."""
        pass

    def post_step(self, frame: int, iterations: int):
        """Override for custom post-step logic."""
        pass

    def get_render_color(self):
        """Override to customize mesh color. Return (r, g, b) tuple."""
        return (0.6, 0.4, 0.2)

    def get_per_vertex_color(self):
        """Override to return per-vertex color field, or None for uniform color."""
        return None

    def _parse_args(self) -> RunConfig:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=f'{self.demo_name} Demo',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Controls (interactive mode):
  SPACE  - Pause/Resume simulation
  R      - Reset simulation
  RMB    - Hold and drag to rotate camera
  ESC    - Exit
            """
        )
        parser.add_argument(
            '--headless',
            action='store_true',
            help='Run without GUI visualization'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode with detailed timing output'
        )
        parser.add_argument(
            '--frames',
            type=int,
            default=50,
            help='Number of frames to run (default: 50)'
        )
        parser.add_argument(
            '--no-save-images',
            action='store_true',
            help='Disable saving images in headless mode'
        )
        parser.add_argument(
            '--no-save-stats',
            action='store_true',
            help='Disable saving statistics'
        )
        parser.add_argument(
            '--log-dir',
            type=str,
            default='',
            help='Custom log directory (default: logs/<demo>_<timestamp>)'
        )

        args = parser.parse_args()

        return RunConfig(
            headless=args.headless,
            debug=args.debug,
            frames=args.frames,
            save_images=not args.no_save_images,
            save_stats=not args.no_save_stats,
            log_dir=args.log_dir,
            demo_name=self.demo_name
        )

    def _ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)
            img_dir = os.path.join(self.config.log_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            return img_dir
        return None

    def _init_window(self, show_window: bool = True):
        """Initialize Taichi UI window."""
        import taichi as ti

        self._window = ti.ui.Window(
            f"{self.demo_name} Demo",
            (1024, 768),
            vsync=True,
            show_window=show_window
        )
        self._canvas = self._window.get_canvas()
        self._scene = self._window.get_scene()
        self._camera = ti.ui.Camera()

        # Set camera from solver if available
        if hasattr(self.solver, 'camera_position'):
            self._camera.position(*self.solver.camera_position)
        if hasattr(self.solver, 'camera_lookat'):
            self._camera.lookat(*self.solver.camera_lookat)
        self._camera.fov(75)
        self._canvas.set_background_color((0.2, 0.2, 0.2))

    def _render_frame(self, track_inputs: bool = False):
        """Render the current frame."""
        if track_inputs and hasattr(self._window, 'RMB'):
            self._camera.track_user_inputs(
                self._window,
                movement_speed=0.1,
                hold_key=self._window.RMB
            )
        self._scene.set_camera(self._camera)
        self._scene.ambient_light((0.1, 0.1, 0.1))

        # Add point lights
        if hasattr(self.solver, 'point_lights'):
            for light_pos in self.solver.point_lights:
                self._scene.point_light(pos=light_pos, color=(1.0, 0.9, 0.8))
        else:
            self._scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 0.9, 0.8))

        # Render mesh
        per_vertex_color = self.get_per_vertex_color()
        if per_vertex_color is not None:
            self._scene.mesh_instance(
                self.solver.mesh.verts.x,
                self.solver.indices,
                per_vertex_color=per_vertex_color,
                show_wireframe=False
            )
        else:
            self._scene.mesh_instance(
                self.solver.mesh.verts.x,
                self.solver.indices,
                color=self.get_render_color(),
                show_wireframe=False
            )

        self._canvas.scene(self._scene)

    def _save_image(self, frame: int, img_dir: str):
        """Save current frame to image file."""
        filename = f"frame_{frame:05d}.png"
        save_path = os.path.join(img_dir, filename)
        self._window.save_image(save_path)
        if self.config.debug:
            print(f"  Saved: {filename}")

    def _step_with_timing(self, frame: int) -> TimingStats:
        """Execute one simulation step with timing."""
        import taichi as ti

        stats = TimingStats(frame=frame)

        # Always time in headless mode, otherwise only in debug mode
        should_time = self.config.headless or self.config.debug

        if should_time:
            ti.sync()
            t_start = time.perf_counter()

        # Pre-step hook
        self.pre_step(frame)

        if should_time:
            ti.sync()
            t_pre = time.perf_counter()

        # Main step
        result = self.solver.step()
        iterations = result if isinstance(result, int) else 0
        stats.iterations = iterations

        if should_time:
            ti.sync()
            t_step = time.perf_counter()
            stats.step_ms = (t_step - t_pre) * 1000

        # Post-step hook
        self.post_step(frame, iterations)

        if should_time:
            ti.sync()
            t_end = time.perf_counter()
            stats.total_ms = (t_end - t_start) * 1000

        return stats

    def _print_timing(self, stats: TimingStats):
        """Print timing information for a frame."""
        print(f"[Frame {stats.frame:4d}] "
              f"total: {stats.total_ms:7.2f}ms | "
              f"step: {stats.step_ms:7.2f}ms | "
              f"iters: {stats.iterations}")

    def _print_headless_timing(self, stats: TimingStats):
        """Print detailed timing information for headless mode with FPS."""
        parts = [f"[Frame {stats.frame:4d}]"]
        parts.append(f"step: {stats.step_ms:7.2f}ms")
        if stats.render_ms > 0:
            parts.append(f"render: {stats.render_ms:6.2f}ms")
        if stats.save_ms > 0:
            parts.append(f"save: {stats.save_ms:6.2f}ms")
        parts.append(f"total: {stats.total_ms:7.2f}ms")
        parts.append(f"FPS: {stats.fps:6.2f}")
        parts.append(f"iters: {stats.iterations}")
        print(" | ".join(parts))

    def _print_summary(self):
        """Print summary statistics."""
        if not self.timing_stats:
            return

        total_times = [s.total_ms for s in self.timing_stats]
        step_times = [s.step_ms for s in self.timing_stats]
        render_times = [s.render_ms for s in self.timing_stats]
        save_times = [s.save_ms for s in self.timing_stats]
        iterations = [s.iterations for s in self.timing_stats]

        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Frames completed: {len(self.timing_stats)}")

        # Always show timing in headless mode
        should_show_timing = self.config.headless or self.config.debug
        if should_show_timing:
            print(f"\nTiming (ms):")
            print(f"  Step   - avg: {np.mean(step_times):7.2f}, "
                  f"min: {np.min(step_times):7.2f}, max: {np.max(step_times):7.2f}")
            if any(r > 0 for r in render_times):
                print(f"  Render - avg: {np.mean(render_times):7.2f}, "
                      f"min: {np.min(render_times):7.2f}, max: {np.max(render_times):7.2f}")
            if any(s > 0 for s in save_times):
                print(f"  Save   - avg: {np.mean(save_times):7.2f}, "
                      f"min: {np.min(save_times):7.2f}, max: {np.max(save_times):7.2f}")
            print(f"  Total  - avg: {np.mean(total_times):7.2f}, "
                  f"min: {np.min(total_times):7.2f}, max: {np.max(total_times):7.2f}")

            avg_fps = 1000.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0
            min_fps = 1000.0 / np.max(total_times) if np.max(total_times) > 0 else 0
            max_fps = 1000.0 / np.min(total_times) if np.min(total_times) > 0 else 0
            print(f"\nFPS: avg: {avg_fps:.2f}, min: {min_fps:.2f}, max: {max_fps:.2f}")

        print(f"\nIterations:")
        print(f"  avg: {np.mean(iterations):.2f}, "
              f"min: {np.min(iterations)}, max: {np.max(iterations)}")
        print("=" * 60)

    def _save_stats(self):
        """Save statistics to file."""
        if not self.config.save_stats or not self.config.log_dir:
            return

        # Ensure log directory exists
        os.makedirs(self.config.log_dir, exist_ok=True)
        stats_file = os.path.join(self.config.log_dir, "stats.json")

        data = {
            "config": {
                "demo_name": self.config.demo_name,
                "frames": self.config.frames,
                "debug": self.config.debug,
                "headless": self.config.headless
            },
            "frames": [
                {
                    "frame": s.frame,
                    "total_ms": s.total_ms,
                    "step_ms": s.step_ms,
                    "render_ms": s.render_ms,
                    "save_ms": s.save_ms,
                    "fps": s.fps,
                    "iterations": s.iterations
                }
                for s in self.timing_stats
            ],
            "summary": {}
        }

        if self.timing_stats:
            total_times = [s.total_ms for s in self.timing_stats]
            step_times = [s.step_ms for s in self.timing_stats]
            render_times = [s.render_ms for s in self.timing_stats]
            save_times = [s.save_ms for s in self.timing_stats]
            iterations = [s.iterations for s in self.timing_stats]

            avg_fps = 1000.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0
            min_fps = 1000.0 / np.max(total_times) if np.max(total_times) > 0 else 0
            max_fps = 1000.0 / np.min(total_times) if np.min(total_times) > 0 else 0

            data["summary"] = {
                "total_ms": {
                    "avg": float(np.mean(total_times)),
                    "min": float(np.min(total_times)),
                    "max": float(np.max(total_times))
                },
                "step_ms": {
                    "avg": float(np.mean(step_times)),
                    "min": float(np.min(step_times)),
                    "max": float(np.max(step_times))
                },
                "render_ms": {
                    "avg": float(np.mean(render_times)),
                    "min": float(np.min(render_times)),
                    "max": float(np.max(render_times))
                },
                "save_ms": {
                    "avg": float(np.mean(save_times)),
                    "min": float(np.min(save_times)),
                    "max": float(np.max(save_times))
                },
                "fps": {
                    "avg": float(avg_fps),
                    "min": float(min_fps),
                    "max": float(max_fps)
                },
                "iterations": {
                    "avg": float(np.mean(iterations)),
                    "min": int(np.min(iterations)),
                    "max": int(np.max(iterations))
                }
            }

        with open(stats_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nStats saved to: {stats_file}")

    def run_headless(self):
        """Run simulation in headless mode."""
        import taichi as ti

        print(f"\n[Headless Mode] Running {self.config.frames} frames...")
        print("[Timing] Per-frame timing enabled")

        img_dir = None
        if self.config.save_images:
            img_dir = self._ensure_log_dir()
            self._init_window(show_window=False)
            print(f"[Saving images to: {img_dir}]")

        for frame in range(self.config.frames):
            ti.sync()
            t_frame_start = time.perf_counter()

            # Step simulation (always timed in headless mode)
            stats = self._step_with_timing(frame)

            # Render and save image
            if self.config.save_images and img_dir:
                ti.sync()
                t_render_start = time.perf_counter()

                self._render_frame()

                ti.sync()
                t_render_end = time.perf_counter()
                stats.render_ms = (t_render_end - t_render_start) * 1000

                # Save image with timing
                t_save_start = time.perf_counter()
                self._save_image(frame, img_dir)
                t_save_end = time.perf_counter()
                stats.save_ms = (t_save_end - t_save_start) * 1000

            ti.sync()
            t_frame_end = time.perf_counter()
            stats.total_ms = (t_frame_end - t_frame_start) * 1000

            self.timing_stats.append(stats)

            # Always print timing in headless mode
            self._print_headless_timing(stats)

        print("\n[Headless Mode] Finished")
        self._print_summary()
        self._save_stats()

    def run_interactive(self):
        """Run simulation with interactive visualization."""
        import taichi as ti

        print(f"\n[Interactive Mode]")
        print("Controls: SPACE=pause, R=reset, RMB+drag=rotate, ESC=exit")
        if self.config.debug:
            print("[Debug Mode] Timing enabled")

        self._init_window(show_window=True)

        frame = 0
        paused = False

        while self._window.running:
            # Handle input
            for e in self._window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    self._window.running = False
                elif e.key == ti.ui.SPACE:
                    paused = not paused
                    print(f"[{'Paused' if paused else 'Running'}]")
                elif e.key == 'r':
                    if hasattr(self.solver, 'restart'):
                        self.solver.restart()
                    frame = 0
                    self.timing_stats.clear()
                    print("[Reset]")

            # Step simulation
            if not paused:
                stats = self._step_with_timing(frame)
                self.timing_stats.append(stats)

                if self.config.debug:
                    self._print_timing(stats)

                frame += 1

            # Render
            self._render_frame(track_inputs=True)
            self._window.show()

        print(f"\n[Interactive Mode] Finished after {frame} frames")
        self._print_summary()

    def run(self, config: Optional[RunConfig] = None):
        """
        Main entry point - run the demo.

        Args:
            config: Optional RunConfig. If None, parses command line args.
        """
        self.config = config or self._parse_args()

        print("=" * 60)
        print(f"{self.demo_name} Demo")
        print("=" * 60)

        # Print solver info if available
        if hasattr(self.solver, 'n_verts'):
            print(f"  Vertices: {self.solver.n_verts}")
        if hasattr(self.solver, 'n_cells'):
            print(f"  Cells: {self.solver.n_cells}")
        if hasattr(self.solver, 'dt'):
            print(f"  Timestep: {self.solver.dt}")

        # Custom setup
        self.setup()

        # Run in appropriate mode
        if self.config.headless:
            self.run_headless()
        else:
            self.run_interactive()


def create_demo_main(solver_class, demo_name: str, default_demo: str = 'cube'):
    """
    Factory function to create a main() for simple demos.

    Usage:
        from demo_runner import create_demo_main
        from algorithm.pncg_base_ipc import pncg_ipc_deformer

        if __name__ == '__main__':
            create_demo_main(pncg_ipc_deformer, 'My Demo', 'cube')()
    """
    def main():
        import argparse
        import taichi as ti

        # Pre-parse to get demo name before ti.init
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument('--demo', type=str, default=default_demo)
        pre_args, _ = pre_parser.parse_known_args()

        ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

        solver = solver_class(demo=pre_args.demo)
        runner = DemoRunner(solver, demo_name=f"{demo_name} ({pre_args.demo})")
        runner.run()

    return main
