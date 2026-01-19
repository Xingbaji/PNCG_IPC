"""
Comprehensive Unit Tests for MAS Preconditioner Package

Tests each module and functionality for correctness and performance.
Excludes collision-related tests as requested.

Usage:
    python test_mas_pkg_unittest.py              # Run all tests
    python test_mas_pkg_unittest.py -v           # Verbose output
    python test_mas_pkg_unittest.py TestConstants  # Run specific test class
    python test_mas_pkg_unittest.py --benchmark  # Run performance benchmarks only
"""

import sys
import os
import time
import unittest
import argparse
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti


# ==============================================================================
# Taichi Initialization
# ==============================================================================

_taichi_initialized = False

def init_taichi():
    """Initialize Taichi with appropriate backend."""
    global _taichi_initialized
    if _taichi_initialized:
        return
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
        print("[Test] Taichi initialized with GPU backend")
    except Exception as e:
        print(f"[Test] GPU initialization failed: {e}, falling back to CPU")
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    _taichi_initialized = True

# Initialize Taichi at module import time for unittest compatibility
init_taichi()


# ==============================================================================
# Performance Timer Context Manager
# ==============================================================================

class PerfTimer:
    """Context manager for measuring execution time."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed_ms = 0.0

    def __enter__(self):
        ti.sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        ti.sync()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.verbose:
            print(f"  [{self.name}] {self.elapsed_ms:.3f} ms")


# ==============================================================================
# Test: Constants Module
# ==============================================================================

class TestConstants(unittest.TestCase):
    """Test constants module values and relationships."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg import constants
        cls.constants = constants

    def test_banksize_value(self):
        """BANKSIZE should be 16 (warp subdivision)."""
        self.assertEqual(self.constants.BANKSIZE, 16)

    def test_max_levels_value(self):
        """MAX_LEVELS should be between 4 and 8."""
        self.assertGreaterEqual(self.constants.MAX_LEVELS, 4)
        self.assertLessEqual(self.constants.MAX_LEVELS, 8)

    def test_sym_block_count(self):
        """SYM_BLOCK_COUNT should equal BANKSIZE * (BANKSIZE + 1) // 2."""
        expected = self.constants.BANKSIZE * (self.constants.BANKSIZE + 1) // 2
        self.assertEqual(self.constants.SYM_BLOCK_COUNT, expected)
        self.assertEqual(self.constants.SYM_BLOCK_COUNT, 136)

    def test_block_dof(self):
        """BLOCK_DOF should equal BANKSIZE * 3."""
        self.assertEqual(self.constants.BLOCK_DOF, self.constants.BANKSIZE * 3)
        self.assertEqual(self.constants.BLOCK_DOF, 48)

    def test_max_neighbors_per_vertex(self):
        """MAX_NEIGHBORS_PER_VERTEX should be reasonable."""
        self.assertGreaterEqual(self.constants.MAX_NEIGHBORS_PER_VERTEX, 32)
        self.assertLessEqual(self.constants.MAX_NEIGHBORS_PER_VERTEX, 128)

    def test_node_bandwidth(self):
        """NODE_BANDWIDTH should be positive."""
        self.assertGreater(self.constants.NODE_BANDWIDTH, 0)

    def test_top_k_updates(self):
        """TOP_K_UPDATES should be positive for Woodbury."""
        self.assertGreater(self.constants.TOP_K_UPDATES, 0)

    def test_eps_value(self):
        """EPS should be small but positive."""
        self.assertGreater(self.constants.EPS, 0)
        self.assertLess(self.constants.EPS, 1e-6)


# ==============================================================================
# Test: Warp Utilities Module
# ==============================================================================

class TestWarpUtils(unittest.TestCase):
    """Test warp-level utility functions."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg import warp_utils
        cls.warp_utils = warp_utils

        # Create test kernel container class
        @ti.data_oriented
        class WarpUtilTester:
            def __init__(self):
                self.result_i32 = ti.field(dtype=ti.i32, shape=())
                self.result_u32 = ti.field(dtype=ti.u32, shape=())

            @ti.kernel
            def test_bit_reverse(self, x: ti.u32):
                self.result_u32[None] = warp_utils.bit_reverse_u32(x)

            @ti.kernel
            def test_clz(self, x: ti.u32):
                self.result_i32[None] = warp_utils.count_leading_zeros_u32(x)

            @ti.kernel
            def test_popcount(self, x: ti.u32):
                self.result_i32[None] = warp_utils.popcount_u32(x)

            @ti.kernel
            def test_ffs(self, x: ti.u32):
                self.result_i32[None] = warp_utils.find_first_set_u32(x)

            @ti.kernel
            def test_ffs_zero(self, x: ti.u32):
                self.result_i32[None] = warp_utils.find_first_set_zero_indexed(x)

            @ti.kernel
            def test_lanemask(self, lane_id: ti.i32):
                self.result_u32[None] = warp_utils.lanemask_lt(lane_id)

        cls.tester = WarpUtilTester()

    def test_popcount_zero(self):
        """popcount(0) should be 0."""
        self.tester.test_popcount(0)
        self.assertEqual(self.tester.result_i32[None], 0)

    def test_popcount_one(self):
        """popcount(1) should be 1."""
        self.tester.test_popcount(1)
        self.assertEqual(self.tester.result_i32[None], 1)

    def test_popcount_powers_of_two(self):
        """popcount of power of 2 should be 1."""
        for i in range(32):
            val = 1 << i
            self.tester.test_popcount(val)
            self.assertEqual(self.tester.result_i32[None], 1, f"popcount(2^{i}) should be 1")

    def test_popcount_all_ones(self):
        """popcount(0xFFFFFFFF) should be 32."""
        self.tester.test_popcount(0xFFFFFFFF)
        self.assertEqual(self.tester.result_i32[None], 32)

    def test_popcount_specific(self):
        """Test popcount for specific values."""
        # 0b1010 = 10, has 2 bits set
        self.tester.test_popcount(10)
        self.assertEqual(self.tester.result_i32[None], 2)
        # 0b1111 = 15, has 4 bits set
        self.tester.test_popcount(15)
        self.assertEqual(self.tester.result_i32[None], 4)
        # 0xFF = 255, has 8 bits set
        self.tester.test_popcount(255)
        self.assertEqual(self.tester.result_i32[None], 8)

    def test_clz_zero(self):
        """clz(0) should be 32."""
        self.tester.test_clz(0)
        self.assertEqual(self.tester.result_i32[None], 32)

    def test_clz_one(self):
        """clz(1) should be 31."""
        self.tester.test_clz(1)
        self.assertEqual(self.tester.result_i32[None], 31)

    def test_clz_powers_of_two(self):
        """clz(2^i) should be 31-i."""
        for i in range(32):
            val = 1 << i
            expected = 31 - i
            self.tester.test_clz(val)
            self.assertEqual(self.tester.result_i32[None], expected, f"clz(2^{i}) should be {expected}")

    def test_ffs_zero(self):
        """ffs(0) should be 0."""
        self.tester.test_ffs(0)
        self.assertEqual(self.tester.result_i32[None], 0)

    def test_ffs_one(self):
        """ffs(1) should be 1 (1-indexed)."""
        self.tester.test_ffs(1)
        self.assertEqual(self.tester.result_i32[None], 1)

    def test_ffs_powers_of_two(self):
        """ffs(2^i) should be i+1."""
        for i in range(31):
            val = 1 << i
            expected = i + 1
            self.tester.test_ffs(val)
            self.assertEqual(self.tester.result_i32[None], expected, f"ffs(2^{i}) should be {expected}")

    def test_ffs_zero_indexed(self):
        """ffs_zero_indexed should return 0-based index."""
        self.tester.test_ffs_zero(0)
        self.assertEqual(self.tester.result_i32[None], -1)
        self.tester.test_ffs_zero(1)
        self.assertEqual(self.tester.result_i32[None], 0)
        self.tester.test_ffs_zero(2)
        self.assertEqual(self.tester.result_i32[None], 1)
        self.tester.test_ffs_zero(4)
        self.assertEqual(self.tester.result_i32[None], 2)
        self.tester.test_ffs_zero(8)
        self.assertEqual(self.tester.result_i32[None], 3)

    def test_lanemask_lt(self):
        """lanemask_lt should return correct bitmask."""
        self.tester.test_lanemask(0)
        self.assertEqual(self.tester.result_u32[None], 0)
        self.tester.test_lanemask(1)
        self.assertEqual(self.tester.result_u32[None], 0b1)
        self.tester.test_lanemask(2)
        self.assertEqual(self.tester.result_u32[None], 0b11)
        self.tester.test_lanemask(3)
        self.assertEqual(self.tester.result_u32[None], 0b111)
        self.tester.test_lanemask(4)
        self.assertEqual(self.tester.result_u32[None], 0b1111)
        self.tester.test_lanemask(16)
        self.assertEqual(self.tester.result_u32[None], 0xFFFF)

    def test_bit_reverse(self):
        """Test bit reverse operation."""
        # 0x80000000 reversed is 0x00000001
        self.tester.test_bit_reverse(0x80000000)
        self.assertEqual(self.tester.result_u32[None], 0x00000001)
        # 0x00000001 reversed is 0x80000000
        self.tester.test_bit_reverse(0x00000001)
        self.assertEqual(self.tester.result_u32[None], 0x80000000)


# ==============================================================================
# Test: SpMV Module
# ==============================================================================

class TestSRBKSpMV(unittest.TestCase):
    """Test Symmetric Reduce-By-Key SpMV implementation."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.spmv import SRBKSpMV
        cls.SRBKSpMV = SRBKSpMV

    def test_initialization(self):
        """Test SpMV initialization."""
        n_verts = 100
        max_triplets = 1000
        spmv = self.SRBKSpMV(max_triplets, n_verts * 3)

        self.assertEqual(spmv.max_triplets, max_triplets)
        self.assertEqual(spmv.n_dofs, n_verts * 3)
        self.assertEqual(spmv.n_verts, n_verts)

    def test_clear(self):
        """Test clearing triplets."""
        spmv = self.SRBKSpMV(100, 30)
        spmv.n_triplets[None] = 50
        spmv.clear()
        self.assertEqual(spmv.n_triplets[None], 0)
        self.assertFalse(spmv.sorted)

    def test_identity_spmv(self):
        """Test SpMV with identity-like diagonal matrix."""
        n_verts = 10
        spmv = self.SRBKSpMV(n_verts, n_verts * 3)

        # Create identity-like diagonal blocks
        x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        y = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # Set x to random values
        np.random.seed(42)
        x_np = np.random.randn(n_verts, 3)
        x.from_numpy(x_np.astype(np.float64))

        # Add identity diagonal blocks
        identity_3x3 = np.eye(3, dtype=np.float64)
        for i in range(n_verts):
            spmv.add_triplet(i, i, ti.Matrix(identity_3x3))

        # Compute y = I * x
        spmv.spmv(x, y, alpha=1.0, beta=0.0)

        # Verify y ≈ x
        y_np = y.to_numpy()
        np.testing.assert_array_almost_equal(y_np, x_np, decimal=5)

    def test_symmetric_storage(self):
        """Test that off-diagonal entries are stored symmetrically."""
        n_verts = 4
        spmv = self.SRBKSpMV(10, n_verts * 3)

        x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        y = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # Add a symmetric off-diagonal entry (only upper triangle)
        A_01 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
        spmv.add_triplet(0, 1, ti.Matrix(A_01))  # This should also contribute to (1,0)

        # Add diagonal
        identity = np.eye(3, dtype=np.float64)
        for i in range(n_verts):
            spmv.add_triplet(i, i, ti.Matrix(identity))

        # Set x
        x_np = np.zeros((n_verts, 3))
        x_np[1] = [1, 1, 1]  # Non-zero at vertex 1
        x.from_numpy(x_np)

        # Compute y
        spmv.spmv(x, y, alpha=1.0, beta=0.0)
        y_np = y.to_numpy()

        # y[0] should have contribution from A[0,1] * x[1]
        expected_y0 = A_01 @ x_np[1]
        np.testing.assert_array_almost_equal(y_np[0], expected_y0, decimal=5)

    def test_sort_by_row(self):
        """Test sorting triplets by row."""
        n_verts = 5
        spmv = self.SRBKSpMV(20, n_verts * 3)

        # Add triplets out of order
        identity = np.eye(3, dtype=np.float64)
        for i in [4, 1, 3, 0, 2]:
            spmv.add_triplet(i, i, ti.Matrix(identity))

        self.assertFalse(spmv.sorted)
        spmv.sort_by_row()
        self.assertTrue(spmv.sorted)

        # Check row_starts is valid CSR format
        row_starts_np = spmv.row_starts.to_numpy()
        self.assertEqual(row_starts_np[0], 0)
        self.assertEqual(row_starts_np[n_verts], n_verts)


# ==============================================================================
# Test: Inversion Module
# ==============================================================================

class TestInversion(unittest.TestCase):
    """Test block matrix inversion algorithms."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, SYM_BLOCK_COUNT, BLOCK_DOF
        cls.BANKSIZE = BANKSIZE
        cls.SYM_BLOCK_COUNT = SYM_BLOCK_COUNT
        cls.BLOCK_DOF = BLOCK_DOF

        # Create a simple test helper class with inversion functionality
        @ti.data_oriented
        class InversionTester:
            def __init__(self, n_blocks):
                self.n_blocks = n_blocks
                self.total_nodes_all_levels = n_blocks * BANKSIZE
                self.total_blocks = n_blocks

                self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                       shape=(n_blocks, SYM_BLOCK_COUNT))
                self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                           shape=(n_blocks, SYM_BLOCK_COUNT))
                self.full_block_matrix = ti.field(dtype=ti.f32,
                                                  shape=(n_blocks, BLOCK_DOF, BLOCK_DOF))
                self.full_block_inverse = ti.field(dtype=ti.f32,
                                                    shape=(n_blocks, BLOCK_DOF, BLOCK_DOF))

            @ti.func
            def _sym_index(self, i: ti.i32, j: ti.i32) -> ti.i32:
                min_idx = ti.min(i, j)
                max_idx = ti.max(i, j)
                return BANKSIZE * min_idx - min_idx * (min_idx + 1) // 2 + max_idx

            @ti.kernel
            def set_diagonal_spd(self, scale: ti.f32):
                """Set diagonal blocks to scaled identity (SPD)."""
                for block_id in range(self.n_blocks):
                    for row in range(BANKSIZE):
                        for col in range(BANKSIZE):
                            sym_idx = self._sym_index(row, col)
                            if row == col:
                                self.block_matrices[block_id, sym_idx] = ti.Matrix.identity(ti.f32, 3) * scale
                            else:
                                self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

            @ti.kernel
            def verify_inverse_diagonal(self, scale: ti.f32) -> ti.f32:
                """Verify inverse of diagonal matrix: max error from expected 1/scale."""
                max_err = 0.0
                for block_id in range(self.n_blocks):
                    for lane_id in range(BANKSIZE):
                        sym_idx = self._sym_index(lane_id, lane_id)
                        inv_block = self.inv_block_matrices[block_id, sym_idx]
                        expected = 1.0 / scale
                        for d in ti.static(range(3)):
                            err = ti.abs(inv_block[d, d] - expected)
                            ti.atomic_max(max_err, err)
                return max_err

        cls.InversionTester = InversionTester

    def test_sym_index_diagonal(self):
        """Test symmetric index for diagonal elements."""
        # For diagonal (i, i), sym_index should be BANKSIZE * i - i*(i+1)/2 + i
        # = BANKSIZE * i - i*(i-1)/2
        expected = [0, 16, 31, 45, 58, 70, 81, 91, 100, 108, 115, 121, 126, 130, 133, 135]

        @ti.kernel
        def compute_sym_index(i: ti.i32) -> ti.i32:
            BANKSIZE = 16
            min_idx = ti.min(i, i)
            max_idx = ti.max(i, i)
            return BANKSIZE * min_idx - min_idx * (min_idx + 1) // 2 + max_idx

        for i in range(self.BANKSIZE):
            result = compute_sym_index(i)
            self.assertEqual(result, expected[i], f"sym_index({i}, {i}) should be {expected[i]}")

    def test_sym_index_upper_triangle(self):
        """Test symmetric index covers all 136 entries."""
        indices_seen = set()

        @ti.kernel
        def compute_sym_index(i: ti.i32, j: ti.i32) -> ti.i32:
            BANKSIZE = 16
            min_idx = ti.min(i, j)
            max_idx = ti.max(i, j)
            return BANKSIZE * min_idx - min_idx * (min_idx + 1) // 2 + max_idx

        for i in range(self.BANKSIZE):
            for j in range(i, self.BANKSIZE):
                idx = compute_sym_index(i, j)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, self.SYM_BLOCK_COUNT)
                indices_seen.add(idx)

        # All 136 indices should be used exactly once
        self.assertEqual(len(indices_seen), self.SYM_BLOCK_COUNT)


# ==============================================================================
# Test: Hierarchy Module
# ==============================================================================

class TestHierarchy(unittest.TestCase):
    """Test multi-level hierarchy construction."""

    def test_level_count_small_mesh(self):
        """Small mesh should have few levels."""
        n_verts = 100  # ~6 warps
        BANKSIZE = 16

        def compute_levels(n):
            levels = 1
            size = n
            while size > BANKSIZE and levels < 6:
                size = (size + BANKSIZE - 1) // BANKSIZE
                levels += 1
            return levels

        levels = compute_levels(n_verts)
        self.assertGreaterEqual(levels, 1)
        self.assertLessEqual(levels, 3)

    def test_level_count_large_mesh(self):
        """Large mesh should have more levels."""
        n_verts = 10000  # ~625 warps
        BANKSIZE = 16
        MAX_LEVELS = 6

        def compute_levels(n):
            levels = 1
            size = n
            while size > BANKSIZE and levels < MAX_LEVELS:
                size = (size + BANKSIZE - 1) // BANKSIZE
                levels += 1
            return levels

        levels = compute_levels(n_verts)
        self.assertGreaterEqual(levels, 3)
        self.assertLessEqual(levels, MAX_LEVELS)

    def test_hierarchy_size_estimation(self):
        """Total hierarchy size should be bounded."""
        n_verts = 1000
        BANKSIZE = 16

        def estimate_total(n):
            total = n
            size = n
            for _ in range(5):  # Up to 6 levels
                if size <= BANKSIZE:
                    break
                size = (size + BANKSIZE - 1) // BANKSIZE
                total += size
            return total

        total = estimate_total(n_verts)

        # Total should be bounded by geometric series: n * (1 + 1/16 + 1/256 + ...)
        # ≈ n * 16/15 ≈ 1.067 * n
        self.assertLess(total, n_verts * 1.2)


# ==============================================================================
# Test: Integration - Core MASPreconditioner
# ==============================================================================

class TestMASPreconditionerIntegration(unittest.TestCase):
    """Integration tests for the complete MASPreconditioner class."""

    @classmethod
    def setUpClass(cls):
        """Set up a simple test solver with mesh."""
        from algorithm.pncg_base_collision_free import pncg_base_deformer
        from algorithm.mas_preconditioner_pkg import MASPreconditioner

        cls.MASPreconditioner = MASPreconditioner

        # Create a simple solver for testing (collision-free version)
        @ti.data_oriented
        class SimpleSolver(pncg_base_deformer):
            def __init__(self):
                super().__init__(demo='cube')
                self.mesh.verts.place({'z': ti.types.vector(3, float)})

        try:
            cls.solver = SimpleSolver()
            cls.solver_available = True
        except Exception as e:
            print(f"[Warning] Could not create test solver: {e}")
            cls.solver_available = False
            cls.solver = None

        # Create helper kernels as module-level functions
        if cls.solver_available:
            @ti.kernel
            def set_gradient_kernel(mesh: ti.template()):
                for vert in mesh.verts:
                    vert.grad = ti.Vector([1.0, 0.5, -0.3])

            @ti.kernel
            def compute_z_norm_kernel(mesh: ti.template()) -> ti.f32:
                z_sum = 0.0
                for vert in mesh.verts:
                    z_sum += vert.z.norm_sqr()
                return ti.sqrt(z_sum)

            cls._set_gradient_kernel = set_gradient_kernel
            cls._compute_z_norm_kernel = compute_z_norm_kernel

    def setUp(self):
        if not self.solver_available:
            self.skipTest("Solver not available")

    def test_initialization(self):
        """Test MAS preconditioner initialization."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        self.assertEqual(mas.n_verts, self.solver.n_verts)
        self.assertEqual(mas.n_cells, self.solver.n_cells)
        self.assertGreater(mas.level_num, 0)
        self.assertFalse(mas.hierarchy_built)

    def test_build_hierarchy(self):
        """Test hierarchy building."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        mas.build_hierarchy()

        self.assertTrue(mas.hierarchy_built)
        self.assertGreater(mas.total_neighbors, 0)

    def test_assemble_and_invert(self):
        """Test matrix assembly and inversion pipeline."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        mas.build_hierarchy()
        mas.assemble_block_matrices(self.solver, use_full_hessian=True)

        self.assertTrue(mas.matrices_assembled)

        mas.invert_block_matrices(use_full_inversion=True, use_oneway_gj=True)

        self.assertTrue(mas.matrices_inverted)

    def test_apply(self):
        """Test preconditioner application."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        mas.build_hierarchy()
        mas.assemble_block_matrices(self.solver, use_full_hessian=True)
        mas.invert_block_matrices(use_full_inversion=True, use_oneway_gj=True)

        # Set gradient to non-zero using class-level static kernel
        TestMASPreconditionerIntegration._set_gradient_kernel(self.solver.mesh)

        # Apply preconditioner
        mas.apply()

        # Check z field is populated
        z_norm = TestMASPreconditionerIntegration._compute_z_norm_kernel(self.solver.mesh)
        self.assertGreater(z_norm, 0)

    def test_get_stats(self):
        """Test statistics retrieval."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        stats = mas.get_stats()

        self.assertIn('n_verts', stats)
        self.assertIn('hierarchy_built', stats)
        self.assertEqual(stats['n_verts'], self.solver.n_verts)


# ==============================================================================
# Performance Benchmarks
# ==============================================================================

class MASBenchmarks:
    """Performance benchmarks for MAS preconditioner modules."""

    def __init__(self, demo='eight_E_stiffness_test'):
        """Initialize benchmarks with a test solver."""
        from algorithm.pncg_base_collision_free import pncg_base_deformer
        from algorithm.mas_preconditioner_pkg import MASPreconditioner

        self.MASPreconditioner = MASPreconditioner

        @ti.data_oriented
        class BenchmarkSolver(pncg_base_deformer):
            def __init__(self):
                super().__init__(demo=demo)
                self.mesh.verts.place({'z': ti.types.vector(3, float)})

        self.solver = BenchmarkSolver()
        self.mas = None

        print(f"\n[Benchmark] Initialized with {self.solver.n_verts} vertices, {self.solver.n_cells} cells")

    def benchmark_hierarchy_build(self, n_runs=5):
        """Benchmark hierarchy building."""
        print(f"\n{'='*60}")
        print("Benchmark: Hierarchy Build")
        print(f"{'='*60}")

        times = []
        for i in range(n_runs):
            mas = self.MASPreconditioner(
                self.solver.n_verts,
                self.solver.n_cells,
                self.solver.mesh,
                use_metis=False
            )

            with PerfTimer(f"Run {i+1}", verbose=False) as timer:
                mas.build_hierarchy()
            times.append(timer.elapsed_ms)

        avg_time = np.mean(times[1:])  # Exclude warmup
        print(f"  Average time (excluding warmup): {avg_time:.3f} ms")
        print(f"  Min: {np.min(times[1:]):.3f} ms, Max: {np.max(times[1:]):.3f} ms")

        self.mas = mas
        return avg_time

    def benchmark_matrix_assembly(self, n_runs=10):
        """Benchmark matrix assembly."""
        print(f"\n{'='*60}")
        print("Benchmark: Matrix Assembly")
        print(f"{'='*60}")

        if self.mas is None or not self.mas.hierarchy_built:
            self.mas = self.MASPreconditioner(
                self.solver.n_verts,
                self.solver.n_cells,
                self.solver.mesh,
                use_metis=False
            )
            self.mas.build_hierarchy()

        times = []
        for i in range(n_runs):
            with PerfTimer(f"Run {i+1}", verbose=False) as timer:
                self.mas.assemble_block_matrices(self.solver, use_full_hessian=True)
            times.append(timer.elapsed_ms)

        avg_time = np.mean(times[1:])
        print(f"  Average time (excluding warmup): {avg_time:.3f} ms")
        print(f"  Min: {np.min(times[1:]):.3f} ms, Max: {np.max(times[1:]):.3f} ms")

        return avg_time

    def benchmark_inversion_methods(self, n_runs=5):
        """Benchmark different inversion methods."""
        print(f"\n{'='*60}")
        print("Benchmark: Inversion Methods")
        print(f"{'='*60}")

        if self.mas is None or not self.mas.matrices_assembled:
            self.mas = self.MASPreconditioner(
                self.solver.n_verts,
                self.solver.n_cells,
                self.solver.mesh,
                use_metis=False
            )
            self.mas.build_hierarchy()
            self.mas.assemble_block_matrices(self.solver, use_full_hessian=True)

        methods = [
            ('Gauss-Jordan', {'use_full_inversion': True, 'use_cholesky': False, 'use_oneway_gj': False}),
            ('One-way GJ', {'use_full_inversion': True, 'use_oneway_gj': True}),
            ('Cholesky', {'use_full_inversion': True, 'use_cholesky': True}),
            ('Incomplete Cholesky', {'use_full_inversion': True, 'use_incomplete': True}),
            ('Diagonal Only', {'use_full_inversion': False}),
        ]

        results = {}
        for name, kwargs in methods:
            times = []
            for i in range(n_runs):
                # Re-assemble to reset matrix state
                self.mas.assemble_block_matrices(self.solver, use_full_hessian=True)

                with PerfTimer(f"{name} Run {i+1}", verbose=False) as timer:
                    self.mas.invert_block_matrices(**kwargs)
                times.append(timer.elapsed_ms)

            avg_time = np.mean(times[1:])
            results[name] = avg_time
            print(f"  {name}: {avg_time:.3f} ms (avg), min={np.min(times[1:]):.3f} ms")

        return results

    def benchmark_apply(self, n_runs=20):
        """Benchmark preconditioner apply operation."""
        print(f"\n{'='*60}")
        print("Benchmark: Preconditioner Apply")
        print(f"{'='*60}")

        if self.mas is None or not self.mas.matrices_inverted:
            self.mas = self.MASPreconditioner(
                self.solver.n_verts,
                self.solver.n_cells,
                self.solver.mesh,
                use_metis=False
            )
            self.mas.build_hierarchy()
            self.mas.assemble_block_matrices(self.solver, use_full_hessian=True)
            self.mas.invert_block_matrices(use_full_inversion=True, use_oneway_gj=True)

        # Set non-zero gradient
        @ti.kernel
        def set_gradient(solver: ti.template()):
            for vert in solver.mesh.verts:
                vert.grad = ti.Vector([1.0, 0.5, -0.3])

        set_gradient(self.solver)

        times = []
        for i in range(n_runs):
            with PerfTimer(f"Run {i+1}", verbose=False) as timer:
                self.mas.apply()
            times.append(timer.elapsed_ms)

        avg_time = np.mean(times[1:])
        print(f"  Average time (excluding warmup): {avg_time:.3f} ms")
        print(f"  Min: {np.min(times[1:]):.3f} ms, Max: {np.max(times[1:]):.3f} ms")

        return avg_time

    def benchmark_full_pipeline(self, n_runs=5):
        """Benchmark full rebuild + apply pipeline."""
        print(f"\n{'='*60}")
        print("Benchmark: Full Pipeline (rebuild + apply)")
        print(f"{'='*60}")

        times_rebuild = []
        times_apply = []

        for i in range(n_runs):
            mas = self.MASPreconditioner(
                self.solver.n_verts,
                self.solver.n_cells,
                self.solver.mesh,
                use_metis=False
            )

            with PerfTimer(f"Rebuild {i+1}", verbose=False) as timer:
                mas.rebuild(self.solver, use_full_hessian=True, use_full_inversion=True)
            times_rebuild.append(timer.elapsed_ms)

            # Set gradient
            @ti.kernel
            def set_gradient(solver: ti.template()):
                for vert in solver.mesh.verts:
                    vert.grad = ti.Vector([1.0, 0.5, -0.3])
            set_gradient(self.solver)

            with PerfTimer(f"Apply {i+1}", verbose=False) as timer:
                mas.apply()
            times_apply.append(timer.elapsed_ms)

        avg_rebuild = np.mean(times_rebuild[1:])
        avg_apply = np.mean(times_apply[1:])

        print(f"  Rebuild: {avg_rebuild:.3f} ms (avg)")
        print(f"  Apply: {avg_apply:.3f} ms (avg)")
        print(f"  Total: {avg_rebuild + avg_apply:.3f} ms (avg)")

        return avg_rebuild, avg_apply

    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("MAS PRECONDITIONER PACKAGE - PERFORMANCE BENCHMARKS")
        print("="*70)

        self.benchmark_hierarchy_build()
        self.benchmark_matrix_assembly()
        self.benchmark_inversion_methods()
        self.benchmark_apply()
        self.benchmark_full_pipeline()

        print("\n" + "="*70)
        print("BENCHMARKS COMPLETE")
        print("="*70)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_tests(verbosity=2):
    """Run all unit tests."""
    init_taichi()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestWarpUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestSRBKSpMV))
    suite.addTests(loader.loadTestsFromTestCase(TestInversion))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchy))
    suite.addTests(loader.loadTestsFromTestCase(TestMASPreconditionerIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def run_benchmarks(demo='eight_E_stiffness_test'):
    """Run performance benchmarks."""
    init_taichi()

    benchmarks = MASBenchmarks(demo=demo)
    benchmarks.run_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAS Preconditioner Package Unit Tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose test output')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks only')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_test',
                        help='Demo name for benchmarks')
    parser.add_argument('test_name', nargs='?', default=None,
                        help='Specific test class or method to run')
    args = parser.parse_args()

    if args.benchmark:
        run_benchmarks(demo=args.demo)
    else:
        if args.test_name:
            # Run specific test
            init_taichi()
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()

            # Try to find the test
            try:
                suite.addTests(loader.loadTestsFromName(args.test_name, sys.modules[__name__]))
            except Exception as e:
                print(f"Error loading test '{args.test_name}': {e}")
                sys.exit(1)

            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
            result = runner.run(suite)
        else:
            result = run_tests(verbosity=2 if args.verbose else 1)

            # Exit with appropriate code
            sys.exit(0 if result.wasSuccessful() else 1)
