"""
METIS Reordering Performance Test for MAS Preconditioner Small.

This test measures the performance impact of METIS reordering on:
1. build_hierarchy() - one-time initialization
2. assemble_block_matrices() - per-iteration Hessian assembly
3. invert_block_matrices() - per-iteration block inversion
4. apply() - per-iteration preconditioner application
5. hessian_matvec() - optional Hessian-vector product

Key insights:
- METIS reordering is computed ONCE at simulation start
- The goal is to improve per-iteration operations through better data locality
- METIS groups spatially close vertices into the same block

Usage:
    python test_metis_performance.py                    # Quick test (cube)
    python test_metis_performance.py --demo bunny10    # Larger mesh
    python test_metis_performance.py --iterations 20   # More iterations
    python test_metis_performance.py --verbose         # Detailed output
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file_path)
unittest_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(unittest_dir)
demo_dir = os.path.join(project_root, 'demo')

sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

# Suppress verbose output
import builtins
_original_print = builtins.print
_verbose = False

def _filtered_print(*args, **kwargs):
    """Print filter that suppresses MAS/METIS messages unless verbose mode."""
    if args:
        msg = str(args[0])
        if msg.startswith("[MAS") or msg.startswith("[METIS]"):
            if not _verbose:
                return
    _original_print(*args, **kwargs)

builtins.print = _filtered_print

import taichi as ti
from math_utils.elastic_util import *
from util.model_loading import model_loading
from algorithm.mas_preconditioner_small import (
    MASPreconditionerSmall,
    compute_metis_reorder,
    compute_optimized_cell_data,
    check_pymetis_available,
    BANKSIZE,
)


class TimingResult:
    """Container for timing results."""
    def __init__(self, name: str):
        self.name = name
        self.times = []

    def add(self, t: float):
        self.times.append(t)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        return np.min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return np.max(self.times) if self.times else 0.0

    def __repr__(self):
        return f"{self.name}: {self.mean*1000:.3f}ms ± {self.std*1000:.3f}ms (n={len(self.times)})"


@ti.data_oriented
class MetisPerformanceTester:
    """
    Tests METIS reordering performance for MAS preconditioner.

    Compares:
    - Sequential ordering (no METIS)
    - METIS-based ordering
    """

    def __init__(self, demo='cube_freefall_10'):
        """Initialize tester with given demo configuration."""
        # Load model
        model = model_loading(demo=demo)
        self.demo = demo
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.mesh = model.mesh
        self.epsilon = model.epsilon

        # Place vertex fields
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),
            'v': ti.types.vector(3, float),
            'm': float,
            'x_n': ti.types.vector(3, float),
            'x_hat': ti.types.vector(3, float),
            'grad': ti.types.vector(3, float),
            'diagH': ti.types.vector(3, float),
            'z': ti.types.vector(3, float),
        })

        # Place cell fields
        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})

        # Initialize positions
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])

        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        self.ndof = self.n_verts * 3

        _original_print(f"Mesh: {self.n_verts} vertices, {self.n_cells} cells, {self.ndof} DOFs")

        # Precompute mass, B, W
        self.precompute()

        # Assign elastic type
        self.assign_elastic_type(model.elastic_type)

        # Extract cell connectivity for METIS
        self.cells_np = self.extract_cell_verts()

        # Temporary buffers for hessian_matvec
        self.v_buffer = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.result_buffer = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    def assign_elastic_type(self, elastic):
        """Set elastic type functions."""
        if elastic == 'ARAP':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP
        elif elastic == 'ARAP_filter':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_filter
        else:
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_filter
        self.elastic_type_str = elastic

    @ti.kernel
    def precompute(self):
        """Precompute mass, B matrix, and cell volumes."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4.0

    @ti.kernel
    def assign_xn_xhat(self):
        """Assign x_n and x_hat for implicit time integration."""
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.gravity * self.dt * self.dt

    @ti.kernel
    def compute_grad(self):
        """Compute gradient."""
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad = m * (vert.x - vert.x_hat)

        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2

            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)

            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3*i], dPsidx[3*i+1], dPsidx[3*i+2]], float)

    def extract_cell_verts(self):
        """Extract cell-vertex connectivity."""
        cell_verts_field = ti.field(dtype=ti.i32, shape=(self.n_cells, 4))

        @ti.kernel
        def _extract(cell_verts: ti.template()):
            for c in self.mesh.cells:
                cell_verts[c.id, 0] = c.verts[0].id
                cell_verts[c.id, 1] = c.verts[1].id
                cell_verts[c.id, 2] = c.verts[2].id
                cell_verts[c.id, 3] = c.verts[3].id

        _extract(cell_verts_field)
        return cell_verts_field.to_numpy()

    @ti.kernel
    def _set_random_grad(self):
        """Set random gradient for testing."""
        for vert in self.mesh.verts:
            vert.grad = ti.Vector([
                ti.random() - 0.5,
                ti.random() - 0.5,
                ti.random() - 0.5
            ])

    @ti.kernel
    def _copy_z_to_v_buffer(self):
        """Copy z to v_buffer for hessian_matvec test."""
        for vert in self.mesh.verts:
            self.v_buffer[vert.id] = vert.z

    def run_benchmark(self, n_iterations: int = 10, warmup: int = 3):
        """
        Run performance benchmark comparing sequential vs METIS ordering.

        Args:
            n_iterations: Number of iterations to time
            warmup: Number of warmup iterations

        Returns:
            dict: Timing results for both orderings
        """
        results = {
            'sequential': {},
            'metis': {}
        }

        # Check METIS availability
        has_metis = check_pymetis_available()
        _original_print(f"\nMETIS available: {has_metis}")

        # Setup state
        self.assign_xn_xhat()
        self.compute_grad()

        # ====================================================================
        # Test Sequential Ordering (No METIS)
        # ====================================================================
        _original_print("\n" + "=" * 70)
        _original_print("Testing Sequential Ordering (No METIS)")
        _original_print("=" * 70)

        # Create preconditioner without METIS
        t_start = time.perf_counter()
        mas_seq = MASPreconditionerSmall(self.mesh, metis_result=None)
        t_init_seq = time.perf_counter() - t_start

        # Build hierarchy
        t_start = time.perf_counter()
        mas_seq.build_hierarchy()
        t_build_seq = time.perf_counter() - t_start

        _original_print(f"  Init time: {t_init_seq*1000:.2f}ms")
        _original_print(f"  Build hierarchy: {t_build_seq*1000:.2f}ms")

        # Benchmark assemble
        assemble_times = TimingResult("assemble")
        for i in range(warmup + n_iterations):
            ti.sync()
            t_start = time.perf_counter()
            mas_seq.assemble_block_matrices(self)
            ti.sync()
            t_end = time.perf_counter()
            if i >= warmup:
                assemble_times.add(t_end - t_start)

        # Benchmark invert
        invert_times = TimingResult("invert")
        for i in range(warmup + n_iterations):
            ti.sync()
            t_start = time.perf_counter()
            mas_seq.invert_block_matrices()
            ti.sync()
            t_end = time.perf_counter()
            if i >= warmup:
                invert_times.add(t_end - t_start)

        # Benchmark apply
        apply_times = TimingResult("apply")
        for i in range(warmup + n_iterations):
            self._set_random_grad()
            ti.sync()
            t_start = time.perf_counter()
            mas_seq.apply()
            ti.sync()
            t_end = time.perf_counter()
            if i >= warmup:
                apply_times.add(t_end - t_start)

        # Benchmark hessian_matvec
        matvec_times = TimingResult("hessian_matvec")
        for i in range(warmup + n_iterations):
            self._copy_z_to_v_buffer()
            ti.sync()
            t_start = time.perf_counter()
            mas_seq.hessian_matvec(self.v_buffer, self.result_buffer)
            ti.sync()
            t_end = time.perf_counter()
            if i >= warmup:
                matvec_times.add(t_end - t_start)

        results['sequential'] = {
            'init': t_init_seq,
            'build': t_build_seq,
            'assemble': assemble_times,
            'invert': invert_times,
            'apply': apply_times,
            'hessian_matvec': matvec_times,
        }

        _original_print(f"\n  {assemble_times}")
        _original_print(f"  {invert_times}")
        _original_print(f"  {apply_times}")
        _original_print(f"  {matvec_times}")

        # ====================================================================
        # Test METIS Ordering
        # ====================================================================
        if has_metis:
            _original_print("\n" + "=" * 70)
            _original_print("Testing METIS Ordering")
            _original_print("=" * 70)

            # Compute METIS reordering (one-time cost)
            t_start = time.perf_counter()
            metis_result = compute_metis_reorder(self.n_verts, self.cells_np)
            t_metis = time.perf_counter() - t_start
            _original_print(f"  METIS computation: {t_metis*1000:.2f}ms (ONE-TIME)")
            _original_print(f"  METIS partitions: {metis_result.n_parts}")
            _original_print(f"  Max partition size: {metis_result.stats['max_partition_size']}")
            _original_print(f"  Avg partition size: {metis_result.stats['avg_partition_size']:.1f}")

            # Create preconditioner with METIS
            t_start = time.perf_counter()
            mas_metis = MASPreconditionerSmall(self.mesh, metis_result=metis_result)
            t_init_metis = time.perf_counter() - t_start

            # Build hierarchy
            t_start = time.perf_counter()
            mas_metis.build_hierarchy()
            t_build_metis = time.perf_counter() - t_start

            _original_print(f"  Init time: {t_init_metis*1000:.2f}ms")
            _original_print(f"  Build hierarchy: {t_build_metis*1000:.2f}ms")

            # Benchmark assemble
            assemble_times_metis = TimingResult("assemble")
            for i in range(warmup + n_iterations):
                ti.sync()
                t_start = time.perf_counter()
                mas_metis.assemble_block_matrices(self)
                ti.sync()
                t_end = time.perf_counter()
                if i >= warmup:
                    assemble_times_metis.add(t_end - t_start)

            # Benchmark invert
            invert_times_metis = TimingResult("invert")
            for i in range(warmup + n_iterations):
                ti.sync()
                t_start = time.perf_counter()
                mas_metis.invert_block_matrices()
                ti.sync()
                t_end = time.perf_counter()
                if i >= warmup:
                    invert_times_metis.add(t_end - t_start)

            # Benchmark apply
            apply_times_metis = TimingResult("apply")
            for i in range(warmup + n_iterations):
                self._set_random_grad()
                ti.sync()
                t_start = time.perf_counter()
                mas_metis.apply()
                ti.sync()
                t_end = time.perf_counter()
                if i >= warmup:
                    apply_times_metis.add(t_end - t_start)

            # Benchmark hessian_matvec
            matvec_times_metis = TimingResult("hessian_matvec")
            for i in range(warmup + n_iterations):
                self._copy_z_to_v_buffer()
                ti.sync()
                t_start = time.perf_counter()
                mas_metis.hessian_matvec(self.v_buffer, self.result_buffer)
                ti.sync()
                t_end = time.perf_counter()
                if i >= warmup:
                    matvec_times_metis.add(t_end - t_start)

            results['metis'] = {
                'metis_compute': t_metis,
                'init': t_init_metis,
                'build': t_build_metis,
                'assemble': assemble_times_metis,
                'invert': invert_times_metis,
                'apply': apply_times_metis,
                'hessian_matvec': matvec_times_metis,
            }

            _original_print(f"\n  {assemble_times_metis}")
            _original_print(f"  {invert_times_metis}")
            _original_print(f"  {apply_times_metis}")
            _original_print(f"  {matvec_times_metis}")

            # ====================================================================
            # Test METIS Optimized Assembly
            # ====================================================================
            _original_print("\n" + "=" * 70)
            _original_print("Testing METIS Optimized Assembly")
            _original_print("=" * 70)

            # Create preconditioner with METIS and optimized assembly
            mas_metis_opt = MASPreconditionerSmall(self.mesh, metis_result=metis_result)
            mas_metis_opt.init_optimized_assembly(self.cells_np)
            mas_metis_opt.build_hierarchy()

            # Benchmark optimized assemble
            assemble_times_opt = TimingResult("assemble_optimized")
            for i in range(warmup + n_iterations):
                ti.sync()
                t_start = time.perf_counter()
                mas_metis_opt.assemble_block_matrices(self)
                ti.sync()
                t_end = time.perf_counter()
                if i >= warmup:
                    assemble_times_opt.add(t_end - t_start)

            # Compare optimized vs regular METIS assembly
            opt_speedup = assemble_times_metis.mean / assemble_times_opt.mean if assemble_times_opt.mean > 0 else 0
            _original_print(f"  Regular METIS assemble: {assemble_times_metis.mean*1000:.3f}ms")
            _original_print(f"  Optimized METIS assemble: {assemble_times_opt.mean*1000:.3f}ms")
            _original_print(f"  Optimized speedup: {opt_speedup:.2f}x")

            results['metis_optimized'] = {
                'assemble': assemble_times_opt,
            }

        else:
            _original_print("\n[SKIP] METIS not available - install pymetis for METIS testing")

        return results

    def print_comparison(self, results: dict):
        """Print comparison table."""
        _original_print("\n" + "=" * 70)
        _original_print("Performance Comparison Summary")
        _original_print("=" * 70)

        if 'metis' not in results or not results['metis']:
            _original_print("METIS results not available.")
            return

        seq = results['sequential']
        metis = results['metis']

        _original_print(f"\n{'Operation':<25} {'Sequential':>12} {'METIS':>12} {'Speedup':>10}")
        _original_print("-" * 60)

        # One-time costs
        _original_print(f"\n{'--- One-time Costs ---':<25}")
        _original_print(f"{'METIS computation':<25} {'N/A':>12} {metis['metis_compute']*1000:>10.2f}ms {'':>10}")
        _original_print(f"{'Init preconditioner':<25} {seq['init']*1000:>10.2f}ms {metis['init']*1000:>10.2f}ms {seq['init']/metis['init']:>10.2f}x")
        _original_print(f"{'Build hierarchy':<25} {seq['build']*1000:>10.2f}ms {metis['build']*1000:>10.2f}ms {seq['build']/metis['build'] if metis['build'] > 0 else 0:>10.2f}x")

        # Per-iteration costs
        _original_print(f"\n{'--- Per-iteration (mean) ---':<25}")

        ops = ['assemble', 'invert', 'apply', 'hessian_matvec']
        for op in ops:
            seq_mean = seq[op].mean * 1000
            metis_mean = metis[op].mean * 1000
            speedup = seq_mean / metis_mean if metis_mean > 0 else 0
            _original_print(f"{op:<25} {seq_mean:>10.3f}ms {metis_mean:>10.3f}ms {speedup:>10.2f}x")

        # Total per-iteration
        seq_total = sum(seq[op].mean for op in ops) * 1000
        metis_total = sum(metis[op].mean for op in ops) * 1000
        total_speedup = seq_total / metis_total if metis_total > 0 else 0
        _original_print("-" * 60)
        _original_print(f"{'TOTAL per-iteration':<25} {seq_total:>10.3f}ms {metis_total:>10.3f}ms {total_speedup:>10.2f}x")

        # Amortization analysis
        _original_print(f"\n{'--- Amortization Analysis ---':<25}")
        metis_overhead = metis['metis_compute'] * 1000  # ms
        time_saved_per_iter = seq_total - metis_total
        if time_saved_per_iter > 0:
            breakeven = int(np.ceil(metis_overhead / time_saved_per_iter))
            _original_print(f"METIS overhead: {metis_overhead:.2f}ms")
            _original_print(f"Time saved per iteration: {time_saved_per_iter:.3f}ms")
            _original_print(f"Break-even after: {breakeven} iterations")
        else:
            _original_print(f"METIS does not provide speedup for per-iteration operations.")


def main():
    parser = argparse.ArgumentParser(description='METIS Performance Test')
    parser.add_argument('--demo', type=str, default='cube_freefall_10',
                        help='Demo name (default: cube_freefall_10)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to time (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    global _verbose
    _verbose = args.verbose

    # Initialize Taichi
    ti.init(arch=ti.gpu)

    _original_print("=" * 70)
    _original_print("METIS Reordering Performance Test")
    _original_print("=" * 70)
    _original_print(f"Demo: {args.demo}")
    _original_print(f"Iterations: {args.iterations}")
    _original_print(f"Warmup: {args.warmup}")

    # Run test
    tester = MetisPerformanceTester(demo=args.demo)
    results = tester.run_benchmark(n_iterations=args.iterations, warmup=args.warmup)
    tester.print_comparison(results)

    _original_print("\n" + "=" * 70)
    _original_print("Test Complete")
    _original_print("=" * 70)


if __name__ == '__main__':
    main()
