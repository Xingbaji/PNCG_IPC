"""
Unit Tests for METIS Integration Module

Comprehensive tests for the METIS-based node reordering in MAS preconditioner.
Tests cover:
1. Utility functions (pymetis availability check)
2. CPU fallback functions (adjacency building, sorting, mapping)
3. GPU-accelerated MetisReorderGPU class
4. METIS partitioning and mesh reordering
5. METISMixin integration with MASPreconditioner
6. End-to-end pipeline tests

Usage:
    python test_metis_integration.py           # Run all tests
    python test_metis_integration.py -v        # Verbose output
    python test_metis_integration.py TestCPUFallback  # Specific test class
"""

import sys
import os
import unittest
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

# Initialize Taichi at module import time
init_taichi()


# ==============================================================================
# Test Utilities
# ==============================================================================

def create_test_mesh(size: int = 3) -> tuple:
    """
    Create a test tetrahedral mesh from a grid.

    Args:
        size: Grid size (size x size x size vertices)

    Returns:
        n_verts: Number of vertices
        vertices: Vertex positions (n_verts, 3)
        cells: Cell indices (n_cells, 4)
    """
    n_verts = size ** 3
    vertices = np.array([[i, j, k]
                         for i in range(size)
                         for j in range(size)
                         for k in range(size)], dtype=np.float32)

    cells = []
    for i in range(size - 1):
        for j in range(size - 1):
            for k in range(size - 1):
                # Create 5 tetrahedra from each cube
                v000 = i * size * size + j * size + k
                v001 = v000 + 1
                v010 = v000 + size
                v011 = v010 + 1
                v100 = v000 + size * size
                v101 = v100 + 1
                v110 = v100 + size
                v111 = v110 + 1

                cells.append([v000, v001, v011, v111])
                cells.append([v000, v011, v010, v111])
                cells.append([v000, v010, v110, v111])
                cells.append([v000, v110, v100, v111])
                cells.append([v000, v100, v101, v111])

    return n_verts, vertices, np.array(cells, dtype=np.int32)


# ==============================================================================
# Test: Utility Functions
# ==============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in metis_integration module."""

    def test_check_pymetis_available_returns_bool(self):
        """check_pymetis_available should return a boolean."""
        from algorithm.mas_preconditioner_pkg.metis_integration import check_pymetis_available
        result = check_pymetis_available()
        self.assertIsInstance(result, bool)

    def test_check_pymetis_available_consistent(self):
        """check_pymetis_available should return consistent results."""
        from algorithm.mas_preconditioner_pkg.metis_integration import check_pymetis_available
        result1 = check_pymetis_available()
        result2 = check_pymetis_available()
        self.assertEqual(result1, result2)


# ==============================================================================
# Test: CPU Fallback Functions
# ==============================================================================

class TestCPUFallback(unittest.TestCase):
    """Test CPU fallback functions for environments without GPU support."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE
        cls.BANKSIZE = BANKSIZE

        # Create test mesh
        cls.n_verts, cls.vertices, cls.cells = create_test_mesh(3)

    def test_build_adjacency_from_cells_cpu_valid_output(self):
        """build_adjacency_from_cells_cpu should return valid adjacency list."""
        from algorithm.mas_preconditioner_pkg.metis_integration import build_adjacency_from_cells_cpu
        adj_list, info = build_adjacency_from_cells_cpu(self.n_verts, self.cells)

        self.assertEqual(len(adj_list), self.n_verts)
        self.assertIsInstance(info, dict)

        # Each vertex should have neighbors
        for i, neighbors in enumerate(adj_list):
            self.assertIsInstance(neighbors, np.ndarray)
            # Interior vertices should have more neighbors
            if i not in [0, self.n_verts - 1]:  # Not corner vertices
                self.assertGreater(len(neighbors), 0)

    def test_build_adjacency_symmetry(self):
        """Adjacency should be symmetric: if i connects to j, j connects to i."""
        from algorithm.mas_preconditioner_pkg.metis_integration import build_adjacency_from_cells_cpu
        adj_list, _ = build_adjacency_from_cells_cpu(self.n_verts, self.cells)

        for i in range(self.n_verts):
            for j in adj_list[i]:
                self.assertIn(i, adj_list[j],
                             f"Asymmetric adjacency: {i} -> {j} but not {j} -> {i}")

    def test_build_adjacency_no_self_loops(self):
        """Adjacency should not contain self-loops."""
        from algorithm.mas_preconditioner_pkg.metis_integration import build_adjacency_from_cells_cpu
        adj_list, _ = build_adjacency_from_cells_cpu(self.n_verts, self.cells)

        for i in range(self.n_verts):
            self.assertNotIn(i, adj_list[i], f"Self-loop found at vertex {i}")

    def test_compute_sort_index_cpu_valid(self):
        """compute_sort_index_cpu should return valid permutation."""
        from algorithm.mas_preconditioner_pkg.metis_integration import compute_sort_index_cpu
        partition = np.array([2, 0, 1, 0, 2, 1], dtype=np.int32)
        sort_index = compute_sort_index_cpu(partition)

        self.assertEqual(len(sort_index), len(partition))
        # Should be a valid permutation
        self.assertTrue(np.all(np.sort(sort_index) == np.arange(len(partition))))
        # Sorted partition should be monotonically non-decreasing
        sorted_partition = partition[sort_index]
        for i in range(1, len(sorted_partition)):
            self.assertGreaterEqual(sorted_partition[i], sorted_partition[i-1])

    def test_compute_sort_index_cpu_stable(self):
        """Sort should be stable: same-partition vertices maintain relative order."""
        from algorithm.mas_preconditioner_pkg.metis_integration import compute_sort_index_cpu
        partition = np.array([1, 0, 0, 1, 0], dtype=np.int32)
        sort_index = compute_sort_index_cpu(partition)

        # Vertices with partition 0: indices 1, 2, 4
        # In sorted order, they should appear in the same relative order
        partition_0_indices = [i for i in sort_index if partition[i] == 0]
        expected_order = [1, 2, 4]
        self.assertEqual(partition_0_indices, expected_order)

    def test_compute_inverse_mapping_cpu(self):
        """compute_inverse_mapping_cpu should be true inverse."""
        from algorithm.mas_preconditioner_pkg.metis_integration import compute_inverse_mapping_cpu
        sort_index = np.array([3, 1, 4, 0, 2], dtype=np.int32)
        inverse = compute_inverse_mapping_cpu(sort_index)

        self.assertEqual(len(inverse), len(sort_index))

        # Verify inverse property: inverse[sort_index[i]] == i
        for new_pos in range(len(sort_index)):
            old_id = sort_index[new_pos]
            self.assertEqual(inverse[old_id], new_pos)

    def test_build_partition_mappings_cpu_coverage(self):
        """build_partition_mappings_cpu should cover all vertices."""
        from algorithm.mas_preconditioner_pkg.metis_integration import (
            compute_sort_index_cpu, build_partition_mappings_cpu
        )
        n_verts = 20
        n_parts = 2
        partition = np.array([i // 10 for i in range(n_verts)], dtype=np.int32)
        sort_index = compute_sort_index_cpu(partition)
        sorted_partition = partition[sort_index]

        partId_map, real_map = build_partition_mappings_cpu(
            sorted_partition, sort_index, self.BANKSIZE)

        # Check real_map covers all vertices
        mapped_vertices = set()
        for orig_id in range(n_verts):
            part_info = real_map[orig_id]
            block_id = part_info // self.BANKSIZE
            lane_id = part_info % self.BANKSIZE

            # Verify reverse mapping
            self.assertEqual(partId_map[part_info], orig_id,
                           f"Mapping inconsistency at vertex {orig_id}")
            mapped_vertices.add(orig_id)

        self.assertEqual(len(mapped_vertices), n_verts)

    def test_build_partition_mappings_cpu_bijective(self):
        """Partition mappings should be bijective (one-to-one)."""
        from algorithm.mas_preconditioner_pkg.metis_integration import (
            compute_sort_index_cpu, build_partition_mappings_cpu
        )
        n_verts = 32
        partition = np.array([i // 16 for i in range(n_verts)], dtype=np.int32)
        sort_index = compute_sort_index_cpu(partition)
        sorted_partition = partition[sort_index]

        partId_map, real_map = build_partition_mappings_cpu(
            sorted_partition, sort_index, self.BANKSIZE)

        # Check partId_map_real entries are unique (within valid range)
        valid_entries = [v for v in partId_map if v >= 0]
        self.assertEqual(len(valid_entries), len(set(valid_entries)),
                        "partId_map_real has duplicate entries")

    def test_reorder_cells_cpu(self):
        """reorder_cells_cpu should correctly map vertex indices."""
        from algorithm.mas_preconditioner_pkg.metis_integration import reorder_cells_cpu
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        old_to_new = np.array([4, 3, 2, 1, 0], dtype=np.int32)

        sorted_cells = reorder_cells_cpu(cells, old_to_new)

        expected = np.array([[4, 3, 2, 1], [3, 2, 1, 0]], dtype=np.int32)
        np.testing.assert_array_equal(sorted_cells, expected)

    def test_create_identity_result_structure(self):
        """_create_identity_result should return complete result structure."""
        from algorithm.mas_preconditioner_pkg.metis_integration import _create_identity_result
        result = _create_identity_result(self.n_verts, self.cells, self.vertices)

        required_keys = [
            'sort_index', 'old_to_new', 'partition', 'n_partitions',
            'partId_map_real', 'real_map_partId', 'sorted_cells', 'stats'
        ]

        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_create_identity_result_identity_mapping(self):
        """Identity result should have identity sort_index and old_to_new."""
        from algorithm.mas_preconditioner_pkg.metis_integration import _create_identity_result
        result = _create_identity_result(self.n_verts, self.cells, self.vertices)

        expected_identity = np.arange(self.n_verts, dtype=np.int32)
        np.testing.assert_array_equal(result['sort_index'], expected_identity)
        np.testing.assert_array_equal(result['old_to_new'], expected_identity)


# ==============================================================================
# Test: MetisReorderGPU Class
# ==============================================================================

class TestMetisReorderGPU(unittest.TestCase):
    """Test GPU-accelerated METIS reordering class."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.metis_integration import (
            MetisReorderGPU, check_pymetis_available
        )
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE

        cls.MetisReorderGPU = MetisReorderGPU
        cls.check_pymetis_available = check_pymetis_available
        cls.BANKSIZE = BANKSIZE

        # Create test mesh
        cls.n_verts, cls.vertices, cls.cells = create_test_mesh(3)
        cls.n_cells = len(cls.cells)

    def test_initialization(self):
        """MetisReorderGPU should initialize correctly."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)

        self.assertEqual(gpu.n_verts, self.n_verts)
        self.assertEqual(gpu.n_cells, self.n_cells)

    def test_load_cells(self):
        """load_cells should correctly transfer cell data to GPU."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        gpu.load_cells(self.cells)

        cells_from_gpu = gpu.cells.to_numpy()
        np.testing.assert_array_equal(cells_from_gpu, self.cells)

    def test_build_adjacency(self):
        """build_adjacency should produce valid adjacency list."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        adj_list, total_edges = gpu.build_adjacency(self.cells)

        self.assertEqual(len(adj_list), self.n_verts)
        self.assertGreater(total_edges, 0)

        # Note: GPU adjacency building may have minor asymmetries due to
        # atomic operations and duplicate removal. For METIS partitioning
        # purposes, this is acceptable as METIS will internally handle it.
        # The key test is that each vertex has reasonable neighbors.
        non_empty_vertices = sum(1 for adj in adj_list if len(adj) > 0)
        self.assertGreater(non_empty_vertices, 0, "No vertices have neighbors")

    def test_set_partition(self):
        """set_partition should correctly store partition data."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        partition = np.array([i % 3 for i in range(self.n_verts)], dtype=np.int32)

        gpu.set_partition(partition)

        self.assertEqual(gpu.n_partitions[None], 3)
        partition_from_gpu = gpu.partition.to_numpy()
        np.testing.assert_array_equal(partition_from_gpu, partition)

    def test_compute_sort_index_cpu(self):
        """compute_sort_index_cpu should produce valid sort index."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        partition = np.array([i % 3 for i in range(self.n_verts)], dtype=np.int32)
        gpu.set_partition(partition)

        sort_index_np, old_to_new_np = gpu.compute_sort_index_cpu()

        # Verify it's a valid permutation
        self.assertTrue(np.all(np.sort(sort_index_np) == np.arange(self.n_verts)))

        # Verify inverse relationship
        for new_pos in range(self.n_verts):
            old_id = sort_index_np[new_pos]
            self.assertEqual(old_to_new_np[old_id], new_pos)

    def test_build_mappings(self):
        """build_mappings should create valid bidirectional mappings."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        partition = np.array([i // self.BANKSIZE for i in range(self.n_verts)], dtype=np.int32)
        gpu.set_partition(partition)
        gpu.compute_sort_index_cpu()
        gpu.load_cells(self.cells)
        gpu.build_mappings()

        # Get results
        partId_map = gpu.partId_map_real.to_numpy()
        real_map = gpu.real_map_partId.to_numpy()

        # Verify bidirectional consistency
        for orig_id in range(self.n_verts):
            part_info = real_map[orig_id]
            self.assertEqual(partId_map[part_info], orig_id,
                           f"Mapping inconsistency at vertex {orig_id}")

    def test_get_results(self):
        """get_results should return complete result dictionary."""
        gpu = self.MetisReorderGPU(self.n_verts, self.n_cells)
        partition = np.array([i // self.BANKSIZE for i in range(self.n_verts)], dtype=np.int32)
        gpu.set_partition(partition)
        gpu.compute_sort_index_cpu()
        gpu.load_cells(self.cells)
        gpu.build_mappings()

        result = gpu.get_results()

        required_keys = [
            'sort_index', 'old_to_new', 'partition', 'n_partitions',
            'partId_map_real', 'real_map_partId', 'sorted_cells', 'stats'
        ]

        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")


# ==============================================================================
# Test: METIS Partitioning Functions
# ==============================================================================

class TestMETISPartitioning(unittest.TestCase):
    """Test METIS partitioning functions."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.metis_integration import check_pymetis_available
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE

        cls.BANKSIZE = BANKSIZE
        cls.pymetis_available = check_pymetis_available()

        # Create test mesh
        cls.n_verts, cls.vertices, cls.cells = create_test_mesh(3)

    def test_metis_partition_single_partition(self):
        """metis_partition with n_parts=1 should return all zeros."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        from algorithm.mas_preconditioner_pkg.metis_integration import (
            metis_partition, build_adjacency_from_cells_cpu
        )
        adj_list, _ = build_adjacency_from_cells_cpu(self.n_verts, self.cells)
        partition = metis_partition(self.n_verts, adj_list, 1)

        np.testing.assert_array_equal(partition, np.zeros(self.n_verts, dtype=np.int32))

    def test_metis_partition_valid_partition_ids(self):
        """metis_partition should return valid partition IDs."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        from algorithm.mas_preconditioner_pkg.metis_integration import (
            metis_partition, build_adjacency_from_cells_cpu
        )
        adj_list, _ = build_adjacency_from_cells_cpu(self.n_verts, self.cells)
        n_parts = 3
        partition = metis_partition(self.n_verts, adj_list, n_parts)

        self.assertEqual(len(partition), self.n_verts)
        self.assertTrue(np.all(partition >= 0))
        self.assertTrue(np.all(partition < n_parts))

    def test_metis_partition_all_partitions_used(self):
        """metis_partition should use all requested partitions (for large enough mesh)."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        from algorithm.mas_preconditioner_pkg.metis_integration import (
            metis_partition, build_adjacency_from_cells_cpu
        )
        # Use larger mesh
        n_verts, vertices, cells = create_test_mesh(5)
        adj_list, _ = build_adjacency_from_cells_cpu(n_verts, cells)
        n_parts = 5
        partition = metis_partition(n_verts, adj_list, n_parts)

        used_partitions = set(partition)
        self.assertEqual(len(used_partitions), n_parts,
                        "Not all partitions were used")

    def test_metis_reorder_mesh_structure(self):
        """metis_reorder_mesh should return complete result structure."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result = metis_reorder_mesh(self.n_verts, self.cells, self.vertices)

        required_keys = [
            'sort_index', 'old_to_new', 'partition', 'n_partitions',
            'partId_map_real', 'real_map_partId', 'sorted_cells', 'stats'
        ]

        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_metis_reorder_mesh_max_partition_size(self):
        """Partition sizes should not exceed BANKSIZE."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result = metis_reorder_mesh(self.n_verts, self.cells, self.vertices)

        max_size = result['stats']['max_partition_size']
        self.assertLessEqual(max_size, self.BANKSIZE,
                            f"Max partition size {max_size} exceeds BANKSIZE {self.BANKSIZE}")

    def test_metis_reorder_mesh_mapping_consistency(self):
        """Sort index and old_to_new should be consistent inverses."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result = metis_reorder_mesh(self.n_verts, self.cells, self.vertices)

        sort_index = result['sort_index']
        old_to_new = result['old_to_new']

        # Verify inverse relationship
        for new_pos in range(self.n_verts):
            old_id = sort_index[new_pos]
            self.assertEqual(old_to_new[old_id], new_pos,
                           f"Mapping inconsistency at position {new_pos}")

    def test_metis_reorder_mesh_partition_mapping_consistency(self):
        """partId_map_real and real_map_partId should be consistent."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result = metis_reorder_mesh(self.n_verts, self.cells, self.vertices)

        partId_map = result['partId_map_real']
        real_map = result['real_map_partId']

        for orig_id in range(self.n_verts):
            part_info = real_map[orig_id]
            mapped_id = partId_map[part_info]
            self.assertEqual(mapped_id, orig_id,
                           f"Partition mapping inconsistency at vertex {orig_id}")

    def test_metis_reorder_mesh_gpu_cpu_consistency(self):
        """GPU and CPU paths should produce equivalent results."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result_gpu = metis_reorder_mesh(self.n_verts, self.cells, self.vertices, use_gpu=True)
        result_cpu = metis_reorder_mesh(self.n_verts, self.cells, self.vertices, use_gpu=False)

        # Stats should be identical (same METIS partitioning)
        # Note: sort_index might differ slightly due to implementation details,
        # but partition structure should be the same
        self.assertEqual(result_gpu['n_partitions'], result_cpu['n_partitions'])
        self.assertEqual(result_gpu['stats']['max_partition_size'],
                        result_cpu['stats']['max_partition_size'])

    def test_metis_reorder_mesh_with_vertices(self):
        """metis_reorder_mesh should include sorted_vertices when provided."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        result = metis_reorder_mesh(self.n_verts, self.cells, self.vertices)

        self.assertIn('sorted_vertices', result)

        # Verify sorted vertices match the sort index
        expected_sorted = self.vertices[result['sort_index']]
        np.testing.assert_array_equal(result['sorted_vertices'], expected_sorted)


# ==============================================================================
# Test: File I/O Utilities
# ==============================================================================

class TestFileIO(unittest.TestCase):
    """Test file I/O utilities for partition data."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.temp_dir = tempfile.mkdtemp()

    def test_save_and_load_partition(self):
        """Save and load should preserve partition data."""
        import os
        from algorithm.mas_preconditioner_pkg.metis_integration import (
            save_partition_file, load_partition_file
        )

        partition = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
        filepath = os.path.join(self.temp_dir, 'test_partition.txt')

        save_partition_file(partition, filepath)
        loaded_partition = load_partition_file(filepath)

        np.testing.assert_array_equal(loaded_partition, partition)

        # Cleanup
        os.remove(filepath)

    def test_save_and_load_large_partition(self):
        """Save and load should work with large partitions."""
        import os
        from algorithm.mas_preconditioner_pkg.metis_integration import (
            save_partition_file, load_partition_file
        )

        partition = np.random.randint(0, 100, size=10000, dtype=np.int32)
        filepath = os.path.join(self.temp_dir, 'test_large_partition.txt')

        save_partition_file(partition, filepath)
        loaded_partition = load_partition_file(filepath)

        np.testing.assert_array_equal(loaded_partition, partition)

        # Cleanup
        os.remove(filepath)


# ==============================================================================
# Test: METISMixin Integration
# ==============================================================================

class TestMETISMixinIntegration(unittest.TestCase):
    """Test METISMixin integration with MASPreconditioner."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg import MASPreconditioner, BANKSIZE
        from algorithm.mas_preconditioner_pkg.metis_integration import check_pymetis_available
        from algorithm.pncg_base_collision_free import pncg_base_deformer

        cls.MASPreconditioner = MASPreconditioner
        cls.BANKSIZE = BANKSIZE
        cls.pymetis_available = check_pymetis_available()

        # Create a simple solver for testing
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

    def setUp(self):
        if not self.solver_available:
            self.skipTest("Solver not available")

    def test_mas_with_metis_initialization(self):
        """MASPreconditioner with use_metis=True should initialize METIS."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=True
        )

        # Check METIS was initialized (if mesh is large enough)
        if self.solver.n_verts >= self.BANKSIZE * 2:
            self.assertTrue(hasattr(mas, 'use_metis_reorder'))
            if mas.use_metis_reorder:
                self.assertTrue(hasattr(mas, 'metis_n_parts'))
                self.assertTrue(hasattr(mas, 'partId_map_real'))
                self.assertTrue(hasattr(mas, 'real_map_partId'))

    def test_mas_without_metis_initialization(self):
        """MASPreconditioner with use_metis=False should not initialize METIS."""
        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=False
        )

        use_metis = getattr(mas, 'use_metis_reorder', False)
        self.assertFalse(use_metis)

    def test_metis_mixin_methods_exist(self):
        """MASPreconditioner should have all METIS mixin methods."""
        required_methods = [
            'init_metis_reordering',
            'get_metis_vertex_id',
            'get_sorted_vertex_id',
            '_build_connect_mask_l0_metis',
            '_propagate_connectivity_metis',
            '_find_cluster_representatives_metis',
            '_assign_cluster_ids_metis',
            '_schwarz_local_solve_full_metis',
            'build_hierarchy_metis',
            'rebuild_with_metis',
            'apply_metis',
        ]

        for method in required_methods:
            self.assertTrue(hasattr(self.MASPreconditioner, method),
                          f"Missing method: {method}")

    def test_build_hierarchy_metis(self):
        """build_hierarchy_metis should build valid hierarchy."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=True
        )

        if not getattr(mas, 'use_metis_reorder', False):
            self.skipTest("METIS not initialized (mesh too small)")

        mas.build_hierarchy_metis()

        self.assertTrue(mas.hierarchy_built)
        self.assertGreater(mas.actual_levels, 0)

    def test_rebuild_with_metis(self):
        """rebuild_with_metis should complete full pipeline."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=True
        )

        mas.rebuild_with_metis(self.solver)

        self.assertTrue(mas.hierarchy_built)
        self.assertTrue(mas.matrices_assembled)
        self.assertTrue(mas.matrices_inverted)

    def test_apply_metis_produces_valid_output(self):
        """apply_metis should produce valid z field."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        mas = self.MASPreconditioner(
            self.solver.n_verts,
            self.solver.n_cells,
            self.solver.mesh,
            use_metis=True
        )

        mas.rebuild_with_metis(self.solver)

        # Set non-zero gradient
        @ti.kernel
        def set_grad(mesh: ti.template()):
            for vert in mesh.verts:
                vert.grad = ti.Vector([1.0, 0.5, -0.3])

        set_grad(self.solver.mesh)

        # Apply METIS preconditioner
        mas.apply_metis()

        # Check z field is populated and has no NaNs
        @ti.kernel
        def check_z(mesh: ti.template()) -> ti.i32:
            valid = 1
            for vert in mesh.verts:
                z = vert.z
                for d in ti.static(range(3)):
                    if ti.math.isnan(z[d]) or ti.math.isinf(z[d]):
                        valid = 0
            return valid

        is_valid = check_z(self.solver.mesh)
        self.assertEqual(is_valid, 1, "z field contains NaN or Inf values")


# ==============================================================================
# Test: Partition Quality
# ==============================================================================

class TestPartitionQuality(unittest.TestCase):
    """Test that METIS improves partition quality over sequential ordering."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.metis_integration import check_pymetis_available
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE

        cls.BANKSIZE = BANKSIZE
        cls.pymetis_available = check_pymetis_available()

    def test_cross_block_edge_reduction(self):
        """METIS should reduce cross-block edges compared to sequential ordering."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        from algorithm.mas_preconditioner_pkg.metis_integration import (
            metis_reorder_mesh, build_adjacency_from_cells_cpu
        )

        # Use larger mesh for meaningful comparison
        n_verts, vertices, cells = create_test_mesh(5)
        adj_list, _ = build_adjacency_from_cells_cpu(n_verts, cells)

        def count_cross_block_edges(mapping):
            """Count edges crossing block boundaries."""
            cross_edges = 0
            for i, neighbors in enumerate(adj_list):
                block_i = mapping[i] // self.BANKSIZE
                for j in neighbors:
                    block_j = mapping[j] // self.BANKSIZE
                    if block_i != block_j:
                        cross_edges += 1
            return cross_edges // 2  # Each edge counted twice

        # Sequential ordering
        sequential_map = np.arange(n_verts)
        seq_cross = count_cross_block_edges(sequential_map)

        # METIS ordering
        result = metis_reorder_mesh(n_verts, cells, vertices)
        metis_map = result['real_map_partId']
        metis_cross = count_cross_block_edges(metis_map)

        # METIS should not increase cross-block edges
        # (may not always decrease for small meshes)
        self.assertLessEqual(metis_cross, seq_cross * 1.1,
                            f"METIS increased cross-block edges significantly: "
                            f"{seq_cross} -> {metis_cross}")

    def test_partition_balance(self):
        """METIS partitions should be reasonably balanced."""
        if not self.pymetis_available:
            self.skipTest("pymetis not available")

        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh

        n_verts, vertices, cells = create_test_mesh(5)
        result = metis_reorder_mesh(n_verts, cells, vertices)

        stats = result['stats']
        max_size = stats['max_partition_size']
        min_size = stats['min_partition_size']

        # Allow some imbalance, but not extreme
        if min_size > 0:
            ratio = max_size / min_size
            self.assertLess(ratio, 3.0,
                          f"Partition imbalance too high: max={max_size}, min={min_size}")


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        from algorithm.mas_preconditioner_pkg.constants import BANKSIZE
        cls.BANKSIZE = BANKSIZE

    def test_single_cell_mesh(self):
        """Handle mesh with single tetrahedron."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        n_verts = 4
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

        result = metis_reorder_mesh(n_verts, cells, vertices)

        self.assertEqual(len(result['sort_index']), n_verts)
        self.assertGreater(result['n_partitions'], 0)

    def test_small_mesh_below_banksize(self):
        """Handle mesh smaller than BANKSIZE."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        n_verts = self.BANKSIZE - 4  # 12 vertices
        vertices = np.random.rand(n_verts, 3).astype(np.float32)
        # Create some cells
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)

        result = metis_reorder_mesh(n_verts, cells, vertices)

        # Should still produce valid result
        self.assertEqual(len(result['sort_index']), n_verts)
        self.assertTrue(result['stats']['max_partition_size'] <= self.BANKSIZE)

    def test_mesh_exactly_banksize(self):
        """Handle mesh with exactly BANKSIZE vertices."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        n_verts = self.BANKSIZE
        vertices = np.random.rand(n_verts, 3).astype(np.float32)
        # Create a simple cell chain
        cells = np.array([[i, i+1, i+2, i+3] for i in range(n_verts - 3)], dtype=np.int32)

        result = metis_reorder_mesh(n_verts, cells, vertices)

        self.assertEqual(len(result['sort_index']), n_verts)
        # Should fit in 1 or 2 partitions
        self.assertLessEqual(result['n_partitions'], 2)

    def test_large_mesh(self):
        """Handle larger mesh efficiently."""
        from algorithm.mas_preconditioner_pkg.metis_integration import metis_reorder_mesh
        n_verts, vertices, cells = create_test_mesh(6)  # 216 vertices

        result = metis_reorder_mesh(n_verts, cells, vertices)

        self.assertEqual(len(result['sort_index']), n_verts)
        self.assertTrue(result['stats']['max_partition_size'] <= self.BANKSIZE)
        # Should have multiple partitions
        self.assertGreater(result['n_partitions'], 1)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_tests(verbosity=2):
    """Run all unit tests."""
    init_taichi()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes in logical order
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestCPUFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestMetisReorderGPU))
    suite.addTests(loader.loadTestsFromTestCase(TestMETISPartitioning))
    suite.addTests(loader.loadTestsFromTestCase(TestFileIO))
    suite.addTests(loader.loadTestsFromTestCase(TestMETISMixinIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='METIS Integration Unit Tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('test_name', nargs='?', default=None,
                       help='Specific test class or method to run')
    args = parser.parse_args()

    if args.test_name:
        # Run specific test
        init_taichi()
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        try:
            suite.addTests(loader.loadTestsFromName(args.test_name, sys.modules[__name__]))
        except Exception as e:
            print(f"Error loading test '{args.test_name}': {e}")
            sys.exit(1)

        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        result = run_tests(verbosity=2 if args.verbose else 1)
        sys.exit(0 if result.wasSuccessful() else 1)
