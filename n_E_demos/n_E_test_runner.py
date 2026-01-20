#!/usr/bin/env python
"""
n_E Demo Test Runner - Run all algorithm variants and generate comparison report.

This script runs different versions of the n_E demo (multi-object collision test)
with different collision detection methods and barrier functions, collects
performance metrics, and generates a comparison report.

Usage:
    python n_E_test_runner.py                    # Run all versions, 50 frames each
    python n_E_test_runner.py --frames 100       # Run with 100 frames
    python n_E_test_runner.py --versions bvh,spatial_hash  # Run specific versions
    python n_E_test_runner.py --report-only      # Generate report from existing results

Versions:
    - initial: Reference implementation (BVH + log barrier)
    - bvh: BVH collision detection (LBVH + Morton codes)
    - spatial_hash: Spatial hashing collision detection
    - cubic_barrier: BVH + cubic barrier function
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

current_file_path = os.path.abspath(__file__)
# Go up one level: n_E_demos -> PNCG_IPC
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
# Also add demo folder for demo_runner import
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
# Add current folder for importing demo modules
sys.path.append(os.path.dirname(current_file_path))

# Change to demo directory for correct relative path resolution (../model/mesh/...)
os.chdir(demo_dir)

# Version configurations
VERSIONS = {
    'initial': {
        'module': 'initial_n_E_demo',
        'runner_class': 'InitialNEDemoRunner',
        'description': 'Reference implementation (BVH + log barrier)',
    },
    'bvh': {
        'module': 'bvh_n_E_demo',
        'runner_class': 'BVHNEDemoRunner',
        'description': 'BVH collision (LBVH + Morton codes)',
    },
    'spatial_hash': {
        'module': 'spatial_hash_n_E_demo',
        'runner_class': 'SpatialHashNEDemoRunner',
        'description': 'Spatial hashing collision detection',
    },
    'cubic_barrier': {
        'module': 'cubic_barrier_n_E_demo',
        'runner_class': 'CubicBarrierNEDemoRunner',
        'description': 'BVH + cubic barrier function',
    },
}

DEFAULT_FRAMES = 50
DEFAULT_DEMO = 'eight_E_drop_demo_contact'


def get_available_demos():
    """Get list of available demos from YAML configs."""
    try:
        from demo_settings import list_demos
        return list_demos()
    except ImportError:
        return [DEFAULT_DEMO]


def run_version(version_name: str, n_frames: int, output_dir: str, demo: str = DEFAULT_DEMO) -> Dict:
    """
    Run a single test version and return metrics.

    Args:
        version_name: Name of the version to run
        n_frames: Number of frames to simulate
        output_dir: Directory to save images
        demo: Demo configuration name

    Returns:
        Dictionary with metrics and timing stats
    """
    import taichi as ti

    print(f"\n{'='*60}")
    print(f"Running {version_name} version")
    print(f"Description: {VERSIONS[version_name]['description']}")
    print(f"{'='*60}")

    # Reset Taichi for clean state
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    # Import and create runner
    module_name = VERSIONS[version_name]['module']
    runner_class_name = VERSIONS[version_name]['runner_class']

    module = __import__(module_name, fromlist=[runner_class_name])
    runner_class = getattr(module, runner_class_name)
    runner = runner_class(demo=demo)

    # Setup output directory for this version
    version_output_dir = os.path.join(output_dir, version_name)
    os.makedirs(version_output_dir, exist_ok=True)

    # Configure runner for headless mode with custom output directory
    from demo_runner import RunConfig
    config = RunConfig(
        headless=True,
        debug=False,
        frames=n_frames,
        save_images=True,
        save_stats=True,
        log_dir=version_output_dir,
        demo_name=f"n_E-{version_name}"
    )

    # Override get_per_vertex_color
    runner.runner.get_per_vertex_color = runner.get_per_vertex_color

    # Run with timing
    start_time = time.perf_counter()
    runner.runner.run(config=config)
    total_time = time.perf_counter() - start_time

    # Collect metrics
    metrics = {
        'version': version_name,
        'collision_method': runner.collision_method,
        'barrier_type': runner.barrier_type,
        'n_frames': n_frames,
        'total_time_s': total_time,
        'timing_stats': [
            {
                'frame': s.frame,
                'total_ms': s.total_ms,
                'step_ms': s.step_ms,
                'render_ms': s.render_ms,
                'save_ms': s.save_ms,
                'iterations': s.iterations,
            }
            for s in runner.runner.timing_stats
        ]
    }

    # Calculate summary statistics
    if runner.runner.timing_stats:
        step_times = [s.step_ms for s in runner.runner.timing_stats]
        total_times = [s.total_ms for s in runner.runner.timing_stats]
        iterations = [s.iterations for s in runner.runner.timing_stats]

        metrics['summary'] = {
            'step_ms': {
                'mean': float(np.mean(step_times)),
                'std': float(np.std(step_times)),
                'min': float(np.min(step_times)),
                'max': float(np.max(step_times)),
            },
            'total_ms': {
                'mean': float(np.mean(total_times)),
                'std': float(np.std(total_times)),
                'min': float(np.min(total_times)),
                'max': float(np.max(total_times)),
            },
            'iterations': {
                'mean': float(np.mean(iterations)),
                'std': float(np.std(iterations)),
                'min': int(np.min(iterations)),
                'max': int(np.max(iterations)),
            },
            'fps': {
                'mean': 1000.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0,
            }
        }

    return metrics


def generate_report(results: Dict[str, Dict], output_dir: str, n_frames: int):
    """
    Generate a Markdown performance report.

    Args:
        results: Dictionary of version name -> metrics
        output_dir: Directory to save the report
        n_frames: Number of frames used in testing
    """
    report_path = os.path.join(os.path.dirname(output_dir), 'n_E_PERFORMANCE_REPORT.md')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        '# n_E Demo Performance Report',
        '',
        f'**Generated:** {timestamp}',
        f'**Frames:** {n_frames}',
        f'**Demo:** {DEFAULT_DEMO}',
        '',
        '## Algorithm Variants',
        '',
        '| Version | Collision Method | Barrier Type | Description |',
        '|---------|------------------|--------------|-------------|',
    ]

    for version_name, config in VERSIONS.items():
        if version_name in results:
            r = results[version_name]
            lines.append(f'| {version_name} | {r["collision_method"]} | {r["barrier_type"]} | {config["description"]} |')

    lines.extend([
        '',
        '## Performance Summary',
        '',
        '| Version | Avg Frame (ms) | Avg Step (ms) | Avg Iterations | Avg FPS |',
        '|---------|----------------|---------------|----------------|---------|',
    ])

    for version_name in VERSIONS.keys():
        if version_name in results and 'summary' in results[version_name]:
            s = results[version_name]['summary']
            lines.append(
                f'| {version_name} | '
                f'{s["total_ms"]["mean"]:.2f} | '
                f'{s["step_ms"]["mean"]:.2f} | '
                f'{s["iterations"]["mean"]:.1f} | '
                f'{s["fps"]["mean"]:.2f} |'
            )

    lines.extend([
        '',
        '## Detailed Timing Statistics',
        '',
        '### Step Time (ms)',
        '',
        '| Version | Mean | Std Dev | Min | Max |',
        '|---------|------|---------|-----|-----|',
    ])

    for version_name in VERSIONS.keys():
        if version_name in results and 'summary' in results[version_name]:
            s = results[version_name]['summary']['step_ms']
            lines.append(
                f'| {version_name} | '
                f'{s["mean"]:.2f} | '
                f'{s["std"]:.2f} | '
                f'{s["min"]:.2f} | '
                f'{s["max"]:.2f} |'
            )

    lines.extend([
        '',
        '### Iterations per Frame',
        '',
        '| Version | Mean | Std Dev | Min | Max |',
        '|---------|------|---------|-----|-----|',
    ])

    for version_name in VERSIONS.keys():
        if version_name in results and 'summary' in results[version_name]:
            s = results[version_name]['summary']['iterations']
            lines.append(
                f'| {version_name} | '
                f'{s["mean"]:.1f} | '
                f'{s["std"]:.2f} | '
                f'{s["min"]} | '
                f'{s["max"]} |'
            )

    # Speedup analysis
    lines.extend([
        '',
        '## Speedup Analysis',
        '',
    ])

    if 'bvh' in results and 'spatial_hash' in results:
        if 'summary' in results['bvh'] and 'summary' in results['spatial_hash']:
            bvh_time = results['bvh']['summary']['step_ms']['mean']
            hash_time = results['spatial_hash']['summary']['step_ms']['mean']
            if bvh_time > 0:
                speedup = hash_time / bvh_time
                lines.append(f'- **BVH vs Spatial Hash:** {speedup:.2f}x {"faster" if speedup > 1 else "slower"}')

    if 'bvh' in results and 'cubic_barrier' in results:
        if 'summary' in results['bvh'] and 'summary' in results['cubic_barrier']:
            log_time = results['bvh']['summary']['step_ms']['mean']
            cubic_time = results['cubic_barrier']['summary']['step_ms']['mean']
            if log_time > 0:
                speedup = log_time / cubic_time
                lines.append(f'- **Cubic vs Log Barrier:** {speedup:.2f}x {"faster" if speedup > 1 else "slower"}')

    lines.extend([
        '',
        '## Image Outputs',
        '',
        'Images saved to:',
    ])

    for version_name in VERSIONS.keys():
        if version_name in results:
            lines.append(f'- `./imgs/{version_name}/images/frame_XXXXX.png`')

    lines.extend([
        '',
        '## Raw Data',
        '',
        f'Full timing data saved to: `./imgs/<version>/stats.json`',
        '',
        '## Notes',
        '',
        '- **initial**: Reference implementation from PNCG_IPC_init, uses BVH collision',
        '- **bvh**: LBVH (Linear BVH) with Morton codes for spatial sorting',
        '- **spatial_hash**: Uses Bresenham line algorithm for edge-grid rasterization',
        '- **cubic_barrier**: Uses cubic barrier function: psi = -2k/(3*dHat) * (d - dHat)^3',
        '- MAS preconditioner is NOT included in these tests',
        '',
    ])

    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved to: {report_path}")

    # Also save raw results as JSON
    json_path = os.path.join(output_dir, 'all_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw data saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run n_E demo algorithm comparison tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--frames', type=int, default=DEFAULT_FRAMES,
        help=f'Number of frames to run (default: {DEFAULT_FRAMES})'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./imgs',
        help='Output directory for images (default: ./imgs)'
    )
    parser.add_argument(
        '--versions', type=str, default='all',
        help='Comma-separated list of versions to run: all, or: initial,bvh,spatial_hash,cubic_barrier'
    )
    parser.add_argument(
        '--demo', type=str, default=DEFAULT_DEMO,
        help=f'Demo configuration name (default: {DEFAULT_DEMO})'
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help='Generate report from existing results (skip running tests)'
    )
    parser.add_argument(
        '--list-demos', action='store_true',
        help='List available demos from YAML configs'
    )

    args = parser.parse_args()

    if args.list_demos:
        print("Available demos:")
        for demo in get_available_demos():
            print(f"  - {demo}")
        sys.exit(0)

    # Determine versions to run
    if args.versions == 'all':
        versions = list(VERSIONS.keys())
    else:
        versions = [v.strip() for v in args.versions.split(',')]
        # Validate versions
        for v in versions:
            if v not in VERSIONS:
                print(f"Error: Unknown version '{v}'")
                print(f"Available versions: {', '.join(VERSIONS.keys())}")
                sys.exit(1)

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("n_E Demo Algorithm Comparison Test")
    print("=" * 60)
    print(f"Frames: {args.frames}")
    print(f"Output: {output_dir}")
    print(f"Versions: {', '.join(versions)}")
    print("=" * 60)

    if args.report_only:
        # Load existing results
        json_path = os.path.join(output_dir, 'all_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results = json.load(f)
            generate_report(results, output_dir, args.frames)
        else:
            print(f"Error: No existing results found at {json_path}")
            sys.exit(1)
    else:
        # Run tests
        results = {}
        for version in versions:
            try:
                results[version] = run_version(
                    version, args.frames, output_dir, args.demo
                )
            except Exception as e:
                print(f"Error running {version}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Generate report
        if results:
            generate_report(results, output_dir, args.frames)
        else:
            print("No results to report.")

    print("\nDone!")


if __name__ == '__main__':
    main()
