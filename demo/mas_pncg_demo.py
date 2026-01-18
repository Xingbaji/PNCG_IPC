"""
Demo script for the MAS-PNCG solver.

This demonstrates the full MAS-PNCG algorithm from the paper:
"An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework
for Incremental Potential Contact"

Usage:
    python mas_pncg_demo.py                    # Interactive mode with visualization
    python mas_pncg_demo.py --headless         # Headless mode for benchmarking
    python mas_pncg_demo.py --headless --frames 100  # Specific number of frames
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

from algorithm.mas_pncg_solver import MASPNCGSolver


class MASPNCGDemo(MASPNCGSolver):
    """Demo class extending MASPNCGSolver with visualization."""

    def __init__(self, demo='cube_0'):
        super().__init__(demo=demo)
        print(f"\n[MAS-PNCG Demo] Initialized with demo: {demo}")
        print(f"  - Vertices: {self.n_verts}")
        print(f"  - Cells: {self.n_cells}")
        print(f"  - dHat: {self.dHat}")
        print(f"  - kappa: {self.kappa}")
        print(f"  - dt: {self.dt}")
        print(f"  - mu: {self.mu}, la: {self.la}")

    def run_interactive(self):
        """Run simulation with GGUI visualization."""
        print("\n[MAS-PNCG Demo] Starting interactive mode...")

        window = ti.ui.Window("MAS-PNCG Demo", (1024, 768), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        # Set camera
        camera.position(*self.camera_position)
        camera.lookat(*self.camera_lookat)

        frame = 0
        paused = False

        while window.running:
            # Handle input
            for e in window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    window.running = False
                elif e.key == ti.ui.SPACE:
                    paused = not paused
                    print(f"[MAS-PNCG Demo] {'Paused' if paused else 'Running'}")
                elif e.key == 'r':
                    # Reset simulation
                    self.mesh.verts.x.copy_from(self.mesh.verts.x_init)
                    self.mesh.verts.v.fill(0.0)
                    self.frame = 0
                    frame = 0
                    print("[MAS-PNCG Demo] Reset")

            # Step simulation
            if not paused:
                try:
                    self.step()
                    frame += 1
                except Exception as e:
                    print(f"[MAS-PNCG Demo] Error in step: {e}")
                    paused = True

            # Render
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.5, 0.5, 0.5))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

            # Draw mesh
            scene.mesh(self.mesh.verts.x,
                      indices=self.indices,
                      per_vertex_color=None,
                      color=(0.7, 0.5, 0.3))

            # Draw ground plane
            scene.particles(ti.Vector.field(3, dtype=float, shape=1),
                           radius=0.001, color=(0.3, 0.3, 0.3))

            canvas.scene(scene)
            window.show()

        print(f"[MAS-PNCG Demo] Finished after {frame} frames")


def main():
    parser = argparse.ArgumentParser(description='MAS-PNCG Demo')
    parser.add_argument('--demo', type=str, default='cube',
                       help='Demo configuration to run (e.g., cube, cube_10, cube_20, cube_40, eight_E_drop_demo_contact)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without visualization')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to run in headless mode')

    args = parser.parse_args()

    print("=" * 60)
    print("MAS-PNCG Demo")
    print("=" * 60)
    print("\nAlgorithm components:")
    print("  1. MAS Preconditioner with connectivity-aware hierarchy")
    print("  2. Sparse-Input Woodbury updates for Level-0")
    print("  3. Optimal 2D Subspace Minimization")
    print("  4. Powell's Restart Criterion")
    print("  5. Conservative CCD with per-subdomain step sizes")
    print("=" * 60)

    demo = MASPNCGDemo(demo=args.demo)

    if args.headless:
        print(f"\n[Headless Mode] Running {args.frames} frames...")
        demo.run_headless(n_frames=args.frames)
    else:
        print("\n[Interactive Mode]")
        print("  Controls:")
        print("    SPACE - Pause/Resume")
        print("    R - Reset simulation")
        print("    RMB + drag - Rotate camera")
        print("    ESC - Exit")
        demo.run_interactive()


if __name__ == "__main__":
    main()
