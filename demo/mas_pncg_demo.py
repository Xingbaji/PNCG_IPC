"""
Demo script for the MAS-PNCG solver.

This demonstrates the full MAS-PNCG algorithm from the paper:
"An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework
for Incremental Potential Contact"

Usage:
    python mas_pncg_demo.py                              # Interactive, 50 frames
    python mas_pncg_demo.py --headless                   # Headless, 50 frames
    python mas_pncg_demo.py --headless --frames 100      # Headless, 100 frames
    python mas_pncg_demo.py --debug                      # With timing info
    python mas_pncg_demo.py --demo cube_10               # Different demo config
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import taichi as ti

    # Pre-parse to get all relevant args before ti.init
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--demo', type=str, default='cube')
    pre_parser.add_argument('--headless', action='store_true')
    pre_parser.add_argument('--debug', action='store_true')
    pre_parser.add_argument('--frames', type=int, default=50)
    pre_parser.add_argument('--no-save-images', action='store_true')
    pre_parser.add_argument('--no-save-stats', action='store_true')
    pre_parser.add_argument('--log-dir', type=str, default='')
    pre_args, _ = pre_parser.parse_known_args()

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

    from algorithm.mas_pncg_solver import MASPNCGSolver
    from demo.demo_runner import DemoRunner, RunConfig

    class MASPNCGDemo(DemoRunner):
        """Demo runner for MAS-PNCG solver with custom setup."""

        def __init__(self, demo='cube'):
            solver = MASPNCGSolver(demo=demo)
            super().__init__(solver, demo_name=f"MAS-PNCG ({demo})")

        def setup(self):
            """Print algorithm components info."""
            print("\nAlgorithm components:")
            print("  1. MAS Preconditioner with connectivity-aware hierarchy")
            print("  2. Sparse-Input Woodbury updates for Level-0")
            print("  3. Optimal 2D Subspace Minimization")
            print("  4. Powell's Restart Criterion")
            print("  5. Conservative CCD with per-subdomain step sizes")

            # Print IPC parameters
            if hasattr(self.solver, 'dHat'):
                print(f"\nIPC parameters:")
                print(f"  dHat: {self.solver.dHat}")
                print(f"  kappa: {self.solver.kappa}")

            # Print material parameters
            if hasattr(self.solver, 'mu'):
                print(f"\nMaterial parameters:")
                print(f"  mu: {self.solver.mu}, la: {self.solver.la}")
                print(f"  dt: {self.solver.dt}")

    # Create config directly from pre-parsed args
    config = RunConfig(
        headless=pre_args.headless,
        debug=pre_args.debug,
        frames=pre_args.frames,
        save_images=not pre_args.no_save_images,
        save_stats=not pre_args.no_save_stats,
        log_dir=pre_args.log_dir,
        demo_name=f"MAS-PNCG ({pre_args.demo})"
    )

    # Create and run demo with config
    demo = MASPNCGDemo(demo=pre_args.demo)
    demo.run(config=config)


if __name__ == "__main__":
    main()
