"""
Test: Banded O(5) vs Full O(16) Local Solve Accuracy

This test compares the accuracy of:
1. Banded solve (NODE_BANDWIDTH=2, O(5) per vertex) - current default
2. Full solve (O(16) per vertex) - more accurate but slower

Metrics:
- cos(z_banded, z_full): Direction alignment
- |z_banded - z_full| / |z_full|: Relative error
- g^T z comparison: Descent direction quality
- Timing comparison

Usage:
    python test_banded_vs_full.py
    python test_banded_vs_full.py --demo eight_E_freefall
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

import taichi as ti
from util.model_loading import model_loading
from algorithm.mas_preconditioner_small import MASPreconditionerSmall


@ti.data_oriented
class BandedVsFullTester:
    """Test banded vs full block solve accuracy."""

    def __init__(self, demo='cube_freefall_10'):
        # Load model
        model = model_loading(demo=demo)
        self.demo = demo
        self.mu, self.la = model.mu, model.la
        self.dt = model.dt
        self.gravity = model.gravity
        self.mesh = model.mesh

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

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())

        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)

        print(f"Mesh: {self.n_verts} vertices, {self.n_cells} cells")

        # Precompute
        self.precompute()

        # Initialize MAS preconditioner
        print("Initializing MAS preconditioner...")
        self.mas = MASPreconditionerSmall(self.mesh)
        print(f"MAS initialized with {self.mas.level_num} levels")

    @ti.kernel
    def precompute(self):
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += 1000.0 * c.W / 4.0

    @ti.kernel
    def init_state(self, vy: float):
        for vert in self.mesh.verts:
            vert.v[1] = vy
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.gravity * self.dt * self.dt

    @ti.kernel
    def compute_grad(self):
        """Compute simple gradient (inertia only for this test)."""
        for vert in self.mesh.verts:
            vert.grad = vert.m * (vert.x - vert.x_hat)

    def run_test(self, n_timing_iters: int = 10):
        """Run the banded vs full comparison test."""
        print(f"\n{'='*70}")
        print("Banded O(5) vs Full O(16) Local Solve Comparison")
        print(f"{'='*70}\n")

        # Initialize state
        self.init_state(-1.0)
        self.compute_grad()

        # Build MAS hierarchy and matrices
        if not self.mas.hierarchy_built:
            self.mas.build_hierarchy()
        self.mas.assemble_block_matrices(self)
        self.mas.invert_block_matrices()

        # Get gradient
        g_np = self.mesh.verts.grad.to_numpy().flatten()
        print(f"|g| = {np.linalg.norm(g_np):.6e}")

        # ====================================================================
        # Test 1: Banded solve O(5)
        # ====================================================================
        print(f"\n[1] Banded Solve O(5) (NODE_BANDWIDTH=2)")

        # Warmup
        self.mas.apply(mode='banded')
        ti.sync()

        # Timing
        t0 = time.perf_counter()
        for _ in range(n_timing_iters):
            self.mas.apply(mode='banded')
            ti.sync()
        t_banded = (time.perf_counter() - t0) / n_timing_iters * 1000

        z_banded = self.mesh.verts.z.to_numpy().flatten().copy()
        print(f"  |z_banded| = {np.linalg.norm(z_banded):.6e}")
        print(f"  Time: {t_banded:.3f} ms")

        # ====================================================================
        # Test 2: Full solve O(16)
        # ====================================================================
        print(f"\n[2] Full Solve O(16)")

        # Warmup
        self.mas.apply(mode='full')
        ti.sync()

        # Timing
        t0 = time.perf_counter()
        for _ in range(n_timing_iters):
            self.mas.apply(mode='full')
            ti.sync()
        t_full = (time.perf_counter() - t0) / n_timing_iters * 1000

        z_full = self.mesh.verts.z.to_numpy().flatten().copy()
        print(f"  |z_full| = {np.linalg.norm(z_full):.6e}")
        print(f"  Time: {t_full:.3f} ms")

        # ====================================================================
        # Comparison
        # ====================================================================
        print(f"\n{'='*70}")
        print("Comparison Results")
        print(f"{'='*70}")

        # Norms
        z_banded_norm = np.linalg.norm(z_banded)
        z_full_norm = np.linalg.norm(z_full)

        # Cosine similarity
        cos_sim = np.dot(z_banded, z_full) / (z_banded_norm * z_full_norm + 1e-15)
        angle_deg = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi

        # Relative error
        rel_error = np.linalg.norm(z_banded - z_full) / (z_full_norm + 1e-15)

        # g^T z (descent quality)
        gTz_banded = np.dot(g_np, z_banded)
        gTz_full = np.dot(g_np, z_full)

        print(f"\n1. Direction Alignment:")
        print(f"   cos(z_banded, z_full) = {cos_sim:.6f}")
        print(f"   Angle = {angle_deg:.2f} degrees")

        print(f"\n2. Relative Error:")
        print(f"   |z_banded - z_full| / |z_full| = {rel_error:.6e}")

        print(f"\n3. Descent Direction Quality (g^T z):")
        print(f"   g^T z_banded = {gTz_banded:.6e}")
        print(f"   g^T z_full   = {gTz_full:.6e}")
        print(f"   Ratio = {gTz_banded / gTz_full:.4f}")

        print(f"\n4. Norm Comparison:")
        print(f"   |z_banded| / |z_full| = {z_banded_norm / z_full_norm:.4f}")

        print(f"\n5. Timing:")
        print(f"   Banded: {t_banded:.3f} ms")
        print(f"   Full:   {t_full:.3f} ms")
        print(f"   Speedup (full/banded): {t_full / t_banded:.2f}x")

        # ====================================================================
        # Per-vertex analysis
        # ====================================================================
        print(f"\n{'='*70}")
        print("Per-Vertex Error Analysis")
        print(f"{'='*70}")

        z_banded_2d = z_banded.reshape(-1, 3)
        z_full_2d = z_full.reshape(-1, 3)
        per_vertex_diff = np.linalg.norm(z_banded_2d - z_full_2d, axis=1)
        per_vertex_full_norm = np.linalg.norm(z_full_2d, axis=1)
        per_vertex_rel_error = per_vertex_diff / (per_vertex_full_norm + 1e-15)

        print(f"\nPer-vertex relative error statistics:")
        print(f"  Min:    {per_vertex_rel_error.min():.6e}")
        print(f"  Max:    {per_vertex_rel_error.max():.6e}")
        print(f"  Mean:   {per_vertex_rel_error.mean():.6e}")
        print(f"  Median: {np.median(per_vertex_rel_error):.6e}")
        print(f"  Std:    {per_vertex_rel_error.std():.6e}")

        # Count vertices with significant error
        thresholds = [0.01, 0.05, 0.1, 0.5]
        print(f"\nVertices with relative error > threshold:")
        for thresh in thresholds:
            count = np.sum(per_vertex_rel_error > thresh)
            pct = 100.0 * count / self.n_verts
            print(f"  > {thresh*100:.0f}%: {count} vertices ({pct:.1f}%)")

        # ====================================================================
        # Summary
        # ====================================================================
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")

        if cos_sim > 0.99:
            status = "EXCELLENT (cos > 0.99)"
        elif cos_sim > 0.95:
            status = "GOOD (cos > 0.95)"
        elif cos_sim > 0.9:
            status = "ACCEPTABLE (cos > 0.9)"
        else:
            status = "POOR (cos <= 0.9)"

        print(f"\nAccuracy Status: {status}")
        print(f"Banded approximation is {t_full/t_banded:.1f}x faster than full solve")

        if cos_sim > 0.95:
            print("\nRecommendation: Banded solve is acceptable for this mesh.")
        else:
            print("\nRecommendation: Consider using full solve for better accuracy.")

        return {
            'cos_sim': cos_sim,
            'rel_error': rel_error,
            'gTz_ratio': gTz_banded / gTz_full,
            't_banded': t_banded,
            't_full': t_full,
            'status': status,
        }


def main():
    parser = argparse.ArgumentParser(description='Banded vs Full Block Solve Test')
    parser.add_argument('--demo', type=str, default='cube_freefall_10',
                        help='Demo configuration')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of timing iterations')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    tester = BandedVsFullTester(demo=args.demo)
    results = tester.run_test(n_timing_iters=args.iters)

    # Return 0 if acceptable accuracy
    if results['cos_sim'] > 0.9:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
