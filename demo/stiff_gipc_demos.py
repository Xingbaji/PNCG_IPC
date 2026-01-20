"""
Stiff-GIPC Demo Runner

Runs demos ported from the Stiff-GIPC codebase.

Usage:
    python stiff_gipc_demos.py --demo stiff_single_bunny
    python stiff_gipc_demos.py --demo stiff_two_cubes_drop --headless --frames 50
    python stiff_gipc_demos.py --list  # List all available demos
"""

import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

from algorithm.pncg_base_ipc import pncg_ipc_deformer
import taichi as ti


def get_stiff_gipc_demos():
    """Get list of available Stiff-GIPC demos from YAML configs."""
    try:
        from demo_settings import list_demos
        demos = list_demos(by_category=True)
        return demos.get('stiff_gipc', [])
    except ImportError:
        # Fallback to hardcoded list
        return [
            'stiff_octopus_stack',
            'stiff_single_bunny',
            'stiff_two_bunnies',
            'stiff_dragon_high',
            'stiff_two_cubes_drop',
            'stiff_teapots',
        ]


# List of all Stiff-GIPC demos (loaded dynamically)
STIFF_GIPC_DEMOS = get_stiff_gipc_demos()


@ti.data_oriented
class StiffGIPCDemo(pncg_ipc_deformer):
    """
    Demo class for Stiff-GIPC scenes.
    Uses the standard PNCG-IPC solver with configurations from model_loading.py.
    Inherits the step() method from pncg_ipc_deformer.
    """
    pass


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Stiff-GIPC Demo Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--demo', type=str, default='stiff_single_bunny',
                        help='Demo name to run')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to run in headless mode')
    parser.add_argument('--list', action='store_true',
                        help='List all available demos')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    if args.list:
        print("Available Stiff-GIPC demos:")
        for demo in STIFF_GIPC_DEMOS:
            print(f"  - {demo}")
        return

    if args.demo not in STIFF_GIPC_DEMOS:
        print(f"Error: Unknown demo '{args.demo}'")
        print(f"Available demos: {', '.join(STIFF_GIPC_DEMOS)}")
        return

    print(f"Running demo: {args.demo}")

    # Initialize Taichi
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    # Create and run demo
    demo = StiffGIPCDemo(demo=args.demo)

    if args.headless:
        demo.run_headless(args.frames)
    else:
        demo.visual()


if __name__ == '__main__':
    main()
