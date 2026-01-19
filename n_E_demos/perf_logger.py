"""
Performance Logger for n_E Demos.

Provides detailed timing for each component of the simulation step.
Saves logs to ./logs folder.
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict

# Logs directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)


class PerformanceLogger:
    """Logger for detailed performance timing."""

    def __init__(self, version_name, demo_name):
        self.version_name = version_name
        self.demo_name = demo_name
        self.frame_logs = []
        self.current_frame = {}
        self.current_iter = {}

    def start_frame(self, frame_id):
        self.current_frame = {
            'frame_id': frame_id,
            'iterations': [],
            'total_time_ms': 0,
        }

    def start_iteration(self, iter_id):
        self.current_iter = {
            'iter_id': iter_id,
            'find_cnts_ms': 0,
            'compute_grad_diagH_ms': 0,
            'ground_barrier_ms': 0,
            'compute_direction_ms': 0,  # compute_init_p or compute_DK
            'line_search_ms': 0,  # compute_alpha_and_update_x
            'total_ms': 0,
        }

    def log_time(self, key, time_ms):
        self.current_iter[key] = time_ms

    def end_iteration(self, total_ms):
        self.current_iter['total_ms'] = total_ms
        self.current_frame['iterations'].append(self.current_iter)

    def end_frame(self, total_ms, n_iters):
        self.current_frame['total_time_ms'] = total_ms
        self.current_frame['n_iterations'] = n_iters
        self.frame_logs.append(self.current_frame)

    def get_summary(self):
        """Compute summary statistics."""
        if not self.frame_logs:
            return {}

        import numpy as np

        # Per-frame stats
        frame_times = [f['total_time_ms'] for f in self.frame_logs]
        iter_counts = [f['n_iterations'] for f in self.frame_logs]

        # Per-component stats (aggregate across all iterations)
        component_times = defaultdict(list)
        for frame in self.frame_logs:
            for it in frame['iterations']:
                for key in ['find_cnts_ms', 'compute_grad_diagH_ms', 'ground_barrier_ms',
                           'compute_direction_ms', 'line_search_ms']:
                    component_times[key].append(it[key])

        summary = {
            'version': self.version_name,
            'demo': self.demo_name,
            'n_frames': len(self.frame_logs),
            'frame_time': {
                'avg_ms': float(np.mean(frame_times)),
                'min_ms': float(np.min(frame_times)),
                'max_ms': float(np.max(frame_times)),
                'std_ms': float(np.std(frame_times)),
            },
            'iterations': {
                'avg': float(np.mean(iter_counts)),
                'min': int(np.min(iter_counts)),
                'max': int(np.max(iter_counts)),
            },
            'components': {}
        }

        total_frame_time = np.sum(frame_times)
        for key, times in component_times.items():
            name = key.replace('_ms', '')
            summary['components'][name] = {
                'avg_ms': float(np.mean(times)),
                'total_ms': float(np.sum(times)),
                'percent': float(np.sum(times) / total_frame_time * 100) if total_frame_time > 0 else 0,
            }

        return summary

    def save_log(self, filename=None):
        """Save detailed log to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(logs_dir, f'{self.version_name}_{self.demo_name}_{timestamp}.json')

        summary = self.get_summary()
        log_data = {
            'summary': summary,
            'frames': self.frame_logs,
        }

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        return filename

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        if not summary:
            print("No data to summarize")
            return

        print(f"\n{'='*70}")
        print(f"Performance Summary: {self.version_name} ({self.demo_name})")
        print(f"{'='*70}")
        print(f"Frames: {summary['n_frames']}")
        print(f"Frame time: {summary['frame_time']['avg_ms']:.2f}ms avg "
              f"({1000/summary['frame_time']['avg_ms']:.1f} FPS)")
        print(f"           {summary['frame_time']['min_ms']:.2f}ms min, "
              f"{summary['frame_time']['max_ms']:.2f}ms max")
        print(f"Iterations: {summary['iterations']['avg']:.1f} avg "
              f"({summary['iterations']['min']}-{summary['iterations']['max']})")

        print(f"\n{'Component':<25} {'Avg (ms)':<12} {'Total (ms)':<12} {'Percent':<10}")
        print('-' * 60)
        for name, stats in summary['components'].items():
            print(f"{name:<25} {stats['avg_ms']:<12.3f} {stats['total_ms']:<12.1f} {stats['percent']:<10.1f}%")
        print(f"{'='*70}")
