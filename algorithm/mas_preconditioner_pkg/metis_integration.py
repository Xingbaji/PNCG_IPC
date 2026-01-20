"""
METIS Integration Module: METIS-based node reordering for improved MAS quality.

This module implements the CEMAS (Connectivity-Enhanced MAS) approach from
Stiff-GIPC that uses METIS graph partitioning to group topologically connected
vertices together in blocks for better preconditioner quality.

This module combines:
1. MetisReorderGPU class - GPU-accelerated METIS reordering pipeline
2. METISMixin class - Integration with MASPreconditioner
3. CPU fallback functions for environments without GPU support

Reference: MASPreconditioner.cu METIS integration
Reference: /root/Stiff-GIPC_init/MeshProcess/metis_partition/
"""

import taichi as ti
import numpy as np
from typing import Tuple, List, Optional, Dict

from .constants import BANKSIZE, MAX_LEVELS


# ============================================================================
# Utility Functions
# ============================================================================

def check_pymetis_available() -> bool:
    """Check if pymetis is available."""
    try:
        import pymetis
        return True
    except ImportError:
        return False


# ============================================================================
# GPU-Accelerated METIS Reordering
# ============================================================================

@ti.data_oriented
class MetisReorderGPU:
    """
    GPU-accelerated METIS reordering for MAS preconditioner.

    This class provides Taichi kernels for parallel operations in the
    METIS reordering pipeline:
    1. Build adjacency graph from mesh topology
    2. Compute sort index and inverse mapping
    3. Build partition mappings for runtime use
    """

    def __init__(self, n_verts: int, n_cells: int, max_neighbors_per_vert: int = 64):
        """
        Initialize GPU reordering structures.

        Args:
            n_verts: Number of vertices
            n_cells: Number of cells (tetrahedra)
            max_neighbors_per_vert: Maximum neighbors per vertex
        """
        self.n_verts = n_verts
        self.n_cells = n_cells
        self.max_neighbors = max_neighbors_per_vert

        # Adjacency structures (CSR format)
        self.adj_count = ti.field(dtype=ti.i32, shape=n_verts)
        self.adj_offset = ti.field(dtype=ti.i32, shape=n_verts + 1)
        self.adj_list = ti.field(dtype=ti.i32, shape=n_verts * max_neighbors_per_vert)

        # Cell data for adjacency building
        self.cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

        # Partition and sorting
        self.partition = ti.field(dtype=ti.i32, shape=n_verts)
        self.sort_index = ti.field(dtype=ti.i32, shape=n_verts)  # sort_index[new] = old
        self.old_to_new = ti.field(dtype=ti.i32, shape=n_verts)  # old_to_new[old] = new

        # Partition mappings for MAS
        self.n_partitions = ti.field(dtype=ti.i32, shape=())
        # These will be allocated after we know n_partitions
        self.partId_map_real = None
        self.real_map_partId = ti.field(dtype=ti.i32, shape=n_verts)

        # Local index counter per partition (for building mappings)
        self.local_index = None

        # Sorted cells output
        self.sorted_cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

    def load_cells(self, cells_np: np.ndarray):
        """Load cell data from numpy array."""
        self.cells.from_numpy(cells_np.astype(np.int32))

    @ti.kernel
    def _count_neighbors(self):
        """Count neighbors for each vertex (first pass)."""
        # Reset counts
        for i in range(self.n_verts):
            self.adj_count[i] = 0

        # Count cell-based neighbors
        for c in range(self.n_cells):
            v0, v1, v2, v3 = self.cells[c][0], self.cells[c][1], self.cells[c][2], self.cells[c][3]

            # Each vertex connects to 3 others in the tet
            # Use atomic to handle race conditions
            ti.atomic_add(self.adj_count[v0], 3)
            ti.atomic_add(self.adj_count[v1], 3)
            ti.atomic_add(self.adj_count[v2], 3)
            ti.atomic_add(self.adj_count[v3], 3)

    @ti.kernel
    def _compute_adj_offset(self):
        """Compute adjacency offset (prefix sum) - sequential for correctness."""
        self.adj_offset[0] = 0
        for i in range(self.n_verts):
            # Clamp count to max_neighbors
            count = ti.min(self.adj_count[i], self.max_neighbors)
            self.adj_offset[i + 1] = self.adj_offset[i] + count
            # Reset count for fill pass
            self.adj_count[i] = 0

    @ti.kernel
    def _fill_adjacency(self):
        """Fill adjacency list (second pass)."""
        for c in range(self.n_cells):
            v = ti.Vector([self.cells[c][0], self.cells[c][1],
                          self.cells[c][2], self.cells[c][3]])

            # Add edges for each vertex pair
            for i in ti.static(range(4)):
                vi = v[i]
                for j in ti.static(range(4)):
                    if i != j:
                        vj = v[j]
                        # Get slot atomically
                        slot = ti.atomic_add(self.adj_count[vi], 1)
                        if slot < self.max_neighbors:
                            offset = self.adj_offset[vi] + slot
                            self.adj_list[offset] = vj

    def build_adjacency(self, cells_np: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """
        Build adjacency list from cell connectivity using GPU.

        Args:
            cells_np: Cell array of shape (n_cells, 4)

        Returns:
            adjacency_list: List of arrays for pymetis
            total_edges: Total number of edges
        """
        self.load_cells(cells_np)

        # Two-pass algorithm
        self._count_neighbors()
        self._compute_adj_offset()
        self._fill_adjacency()

        # Convert to list format for pymetis
        adj_count_np = self.adj_count.to_numpy()
        adj_offset_np = self.adj_offset.to_numpy()
        adj_list_np = self.adj_list.to_numpy()

        adjacency_list = []
        for i in range(self.n_verts):
            start = adj_offset_np[i]
            end = start + min(adj_count_np[i], self.max_neighbors)
            neighbors = adj_list_np[start:end]
            # Remove duplicates
            neighbors = np.unique(neighbors)
            adjacency_list.append(neighbors.astype(np.int32))

        total_edges = sum(len(adj) for adj in adjacency_list)
        return adjacency_list, total_edges

    def set_partition(self, partition_np: np.ndarray):
        """Set partition from numpy array (result from METIS)."""
        self.partition.from_numpy(partition_np.astype(np.int32))
        n_parts = int(np.max(partition_np)) + 1
        self.n_partitions[None] = n_parts

        # Allocate partition mapping arrays
        self.partId_map_real = ti.field(dtype=ti.i32, shape=n_parts * BANKSIZE)
        self.local_index = ti.field(dtype=ti.i32, shape=n_parts)

    @ti.kernel
    def _compute_sort_index(self):
        """
        Compute sort index using counting sort.
        This is a GPU-parallel stable sort by partition ID.
        """
        n_parts = self.n_partitions[None]

        # Reset local index counters
        for p in range(n_parts):
            self.local_index[p] = 0

        # Count vertices per partition
        for i in range(self.n_verts):
            part_id = self.partition[i]
            ti.atomic_add(self.local_index[part_id], 1)

    def compute_sort_index_cpu(self):
        """
        Compute sort index on CPU (stable sort by partition ID).
        Called after partition is set.
        """
        partition_np = self.partition.to_numpy()
        n_verts = len(partition_np)

        # Create (index, partition) pairs and sort
        indexed = [(i, partition_np[i]) for i in range(n_verts)]
        indexed.sort(key=lambda x: x[1])  # Stable sort by partition

        # Extract sort index
        sort_index_np = np.array([p[0] for p in indexed], dtype=np.int32)
        self.sort_index.from_numpy(sort_index_np)

        # Compute inverse mapping
        old_to_new_np = np.zeros(n_verts, dtype=np.int32)
        for new_pos in range(n_verts):
            old_id = sort_index_np[new_pos]
            old_to_new_np[old_id] = new_pos
        self.old_to_new.from_numpy(old_to_new_np)

        return sort_index_np, old_to_new_np

    @ti.kernel
    def _build_partition_mappings(self):
        """
        Build bidirectional partition mappings on GPU.

        Following the reference implementation in gl_main.cu:setMAS_partition()

        partId_map_real[part_id * BANKSIZE + local_idx] = original_vertex_id
        real_map_partId[original_vertex_id] = part_id * BANKSIZE + local_idx

        NOTE: Both mappings use ORIGINAL vertex indices (not sorted indices)
        to allow direct access in MAS preconditioner without index conversion.
        """
        n_parts = self.n_partitions[None]

        # Initialize partId_map_real to -1
        for i in range(n_parts * BANKSIZE):
            self.partId_map_real[i] = -1

        # Reset local index counters
        for p in range(n_parts):
            self.local_index[p] = 0

        # Build mappings - iterate in sorted order to ensure consistent lane assignment
        for sorted_idx in range(self.n_verts):
            # Get original vertex ID
            orig_id = self.sort_index[sorted_idx]
            part_id = self.partition[orig_id]

            # Get local index within partition (atomically)
            local_idx = ti.atomic_add(self.local_index[part_id], 1)

            if local_idx < BANKSIZE:
                # Map: (partition_id, local_index) -> ORIGINAL vertex index
                self.partId_map_real[part_id * BANKSIZE + local_idx] = orig_id

                # Reverse map: ORIGINAL vertex index -> (partition_id, local_index)
                self.real_map_partId[orig_id] = part_id * BANKSIZE + local_idx

    @ti.kernel
    def _reorder_cells(self):
        """Reorder cell vertex indices using GPU parallel."""
        for c in range(self.n_cells):
            for i in ti.static(range(4)):
                old_id = self.cells[c][i]
                new_id = self.old_to_new[old_id]
                self.sorted_cells[c][i] = new_id

    def build_mappings(self):
        """Build all partition mappings on GPU."""
        self._build_partition_mappings()
        self._reorder_cells()

    def get_results(self) -> dict:
        """
        Get all reordering results as numpy arrays.

        Returns:
            Dictionary with all reordering data
        """
        partition_np = self.partition.to_numpy()
        sort_index_np = self.sort_index.to_numpy()
        old_to_new_np = self.old_to_new.to_numpy()

        # Get sorted partition (partition IDs in sorted vertex order)
        sorted_partition = partition_np[sort_index_np]

        n_parts = self.n_partitions[None]

        # Compute statistics
        partition_sizes = np.bincount(sorted_partition)
        stats = {
            'n_vertices': self.n_verts,
            'n_partitions': n_parts,
            'max_partition_size': int(np.max(partition_sizes)),
            'min_partition_size': int(np.min(partition_sizes)),
            'avg_partition_size': float(np.mean(partition_sizes)),
            'non_full_partitions': int(np.sum(partition_sizes < BANKSIZE)),
        }

        return {
            'sort_index': sort_index_np,
            'old_to_new': old_to_new_np,
            'partition': sorted_partition,
            'n_partitions': n_parts,
            'partId_map_real': self.partId_map_real.to_numpy(),
            'real_map_partId': self.real_map_partId.to_numpy(),
            'sorted_cells': self.sorted_cells.to_numpy(),
            'stats': stats,
        }


# ============================================================================
# CPU Fallback Functions
# ============================================================================

def build_adjacency_from_cells_cpu(n_verts: int, cells: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
    """
    Build adjacency list from cell connectivity (CPU version).

    Args:
        n_verts: Number of vertices
        cells: Cell array of shape (n_cells, 4)

    Returns:
        adjacency_list: List of neighbor arrays
        info: Empty dict (for API compatibility)
    """
    adj_dict = {i: set() for i in range(n_verts)}

    for cell in cells:
        v0, v1, v2, v3 = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
        vertices = [v0, v1, v2, v3]

        for i in range(4):
            for j in range(4):
                if i != j:
                    adj_dict[vertices[i]].add(vertices[j])

    adjacency_list = []
    for i in range(n_verts):
        neighbors = np.array(list(adj_dict[i]), dtype=np.int32)
        adjacency_list.append(neighbors)

    return adjacency_list, {}


def compute_sort_index_cpu(partition: np.ndarray) -> np.ndarray:
    """Compute sort index on CPU."""
    n_verts = len(partition)
    indexed = [(i, partition[i]) for i in range(n_verts)]
    indexed.sort(key=lambda x: x[1])
    return np.array([p[0] for p in indexed], dtype=np.int32)


def compute_inverse_mapping_cpu(sort_index: np.ndarray) -> np.ndarray:
    """Compute inverse mapping on CPU."""
    n_verts = len(sort_index)
    inverse = np.zeros(n_verts, dtype=np.int32)
    for new_pos in range(n_verts):
        old_id = sort_index[new_pos]
        inverse[old_id] = new_pos
    return inverse


def build_partition_mappings_cpu(partition: np.ndarray,
                                  sort_index: np.ndarray,
                                  block_size: int = BANKSIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build partition mappings on CPU.

    Both mappings use ORIGINAL vertex indices (not sorted indices)
    to allow direct access in MAS preconditioner without index conversion.

    Args:
        partition: Partition IDs for each vertex (in sorted vertex order)
        sort_index: sort_index[sorted_pos] = original_vertex_id
        block_size: Size of each block (default: BANKSIZE)

    Returns:
        partId_map_real: partId_map_real[part_id * block_size + lane] = original_vertex_id
        real_map_partId: real_map_partId[original_vertex_id] = part_id * block_size + lane
    """
    n_verts = len(sort_index)
    n_parts = int(np.max(partition)) + 1

    partId_map_real = np.full(n_parts * block_size, -1, dtype=np.int32)
    real_map_partId = np.zeros(n_verts, dtype=np.int32)
    local_index = np.zeros(n_parts, dtype=np.int32)

    # Iterate in sorted order to ensure consistent lane assignment
    for sorted_idx in range(n_verts):
        orig_id = sort_index[sorted_idx]
        part_id = partition[sorted_idx]  # partition is already in sorted order
        local_idx = local_index[part_id]

        if local_idx < block_size:
            # Map using ORIGINAL vertex indices
            partId_map_real[part_id * block_size + local_idx] = orig_id
            real_map_partId[orig_id] = part_id * block_size + local_idx
            local_index[part_id] += 1

    return partId_map_real, real_map_partId


def reorder_cells_cpu(cells: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """Reorder cell vertex indices on CPU."""
    sorted_cells = np.zeros_like(cells)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            old_id = int(cells[i, j])
            sorted_cells[i, j] = old_to_new[old_id]
    return sorted_cells


def _create_identity_result(n_verts: int, cells: np.ndarray,
                            vertices: Optional[np.ndarray],
                            block_size: int = BANKSIZE) -> dict:
    """Create identity mapping result when METIS is not available."""
    n_parts = (n_verts + block_size - 1) // block_size

    sort_index = np.arange(n_verts, dtype=np.int32)
    old_to_new = np.arange(n_verts, dtype=np.int32)

    partition = np.zeros(n_verts, dtype=np.int32)
    for i in range(n_verts):
        partition[i] = i // block_size

    partId_map_real, real_map_partId = build_partition_mappings_cpu(partition, sort_index, block_size)

    result = {
        'sort_index': sort_index,
        'old_to_new': old_to_new,
        'partition': partition,
        'n_partitions': n_parts,
        'partId_map_real': partId_map_real,
        'real_map_partId': real_map_partId,
        'sorted_cells': cells.copy(),
        'stats': {
            'n_vertices': n_verts,
            'n_partitions': n_parts,
            'max_partition_size': min(block_size, n_verts),
            'min_partition_size': n_verts % block_size if n_verts % block_size > 0 else block_size,
            'avg_partition_size': n_verts / n_parts,
            'non_full_partitions': 1 if n_verts % block_size > 0 else 0,
        },
    }

    if vertices is not None:
        result['sorted_vertices'] = vertices.copy()

    return result


# ============================================================================
# Partition Quality Metrics
# ============================================================================

def compute_partition_quality(n_verts: int, partition: np.ndarray,
                               adjacency_list: List[np.ndarray]) -> Dict:
    """
    Compute partition quality metrics including ribbon ratio and boundary vertices.

    This function evaluates partition quality by measuring:
    1. Ribbon ratio: fraction of vertices with neighbors in other partitions
    2. Boundary vertex count per partition
    3. Edge cut: number of edges crossing partition boundaries
    4. Internal connectivity: average internal neighbors per vertex

    Args:
        n_verts: Number of vertices
        partition: Partition assignment for each vertex
        adjacency_list: Adjacency list for the mesh graph

    Returns:
        Dictionary with quality metrics:
        - ribbon_ratio: fraction of boundary vertices (lower is better)
        - boundary_count: number of boundary vertices
        - edge_cut: number of edges crossing partitions
        - avg_internal_neighbors: average internal connectivity
        - partition_ribbon_ratios: ribbon ratio per partition
    """
    n_parts = int(np.max(partition)) + 1

    # Count boundary vertices and edge cuts
    boundary_count = 0
    edge_cut = 0
    internal_neighbors_total = 0

    # Per-partition statistics
    partition_sizes = np.zeros(n_parts, dtype=np.int32)
    partition_boundary_counts = np.zeros(n_parts, dtype=np.int32)

    for v in range(n_verts):
        my_part = partition[v]
        partition_sizes[my_part] += 1

        is_boundary = False
        internal_count = 0

        for neighbor in adjacency_list[v]:
            if partition[neighbor] != my_part:
                edge_cut += 1
                is_boundary = True
            else:
                internal_count += 1

        if is_boundary:
            boundary_count += 1
            partition_boundary_counts[my_part] += 1

        internal_neighbors_total += internal_count

    # Edge cut is counted twice (once from each endpoint)
    edge_cut //= 2

    # Compute ribbon ratios per partition
    partition_ribbon_ratios = np.zeros(n_parts, dtype=np.float32)
    for p in range(n_parts):
        if partition_sizes[p] > 0:
            partition_ribbon_ratios[p] = partition_boundary_counts[p] / partition_sizes[p]

    return {
        'ribbon_ratio': boundary_count / n_verts if n_verts > 0 else 0.0,
        'boundary_count': int(boundary_count),
        'edge_cut': int(edge_cut),
        'avg_internal_neighbors': internal_neighbors_total / n_verts if n_verts > 0 else 0.0,
        'partition_ribbon_ratios': partition_ribbon_ratios,
        'avg_partition_ribbon_ratio': float(np.mean(partition_ribbon_ratios)),
        'max_partition_ribbon_ratio': float(np.max(partition_ribbon_ratios)),
    }


def identify_boundary_vertices(n_verts: int, partition: np.ndarray,
                                adjacency_list: List[np.ndarray]) -> np.ndarray:
    """
    Identify vertices on partition boundaries.

    A vertex is on the boundary if it has at least one neighbor in a different partition.

    Args:
        n_verts: Number of vertices
        partition: Partition assignment for each vertex
        adjacency_list: Adjacency list for the mesh graph

    Returns:
        Boolean array where True indicates boundary vertex
    """
    is_boundary = np.zeros(n_verts, dtype=bool)

    for v in range(n_verts):
        my_part = partition[v]
        for neighbor in adjacency_list[v]:
            if partition[neighbor] != my_part:
                is_boundary[v] = True
                break

    return is_boundary


# ============================================================================
# Greedy Partition (Alternative to METIS)
# ============================================================================

def greedy_partition(n_verts: int, adjacency_list: List[np.ndarray],
                     block_size: int = BANKSIZE) -> np.ndarray:
    """
    Greedy graph partitioning that minimizes boundary vertices.

    This algorithm grows partitions greedily by selecting vertices with
    the highest connectivity to the current partition. It's useful as a
    fallback when METIS is not available or for comparison.

    Algorithm:
    1. Start with the highest-connectivity unassigned vertex as seed
    2. Greedily add vertices that maximize internal connectivity
    3. Stop when partition reaches block_size
    4. Repeat until all vertices are assigned

    Args:
        n_verts: Number of vertices
        adjacency_list: Adjacency list for the mesh graph
        block_size: Target partition size (default: BANKSIZE=16)

    Returns:
        Partition assignment array
    """
    partition = np.full(n_verts, -1, dtype=np.int32)
    assigned = np.zeros(n_verts, dtype=bool)

    # Compute initial connectivity scores
    connectivity = np.array([len(adj) for adj in adjacency_list])

    # For very sparse graphs, use simple sequential assignment
    avg_connectivity = np.mean(connectivity)
    if avg_connectivity < 2:
        for v in range(n_verts):
            partition[v] = v // block_size
        return partition

    part_id = 0

    while not assigned.all():
        # Find seed: highest connectivity unassigned vertex
        unassigned_conn = connectivity * (~assigned)
        seed = np.argmax(unassigned_conn)

        if assigned[seed]:
            # All remaining vertices have zero connectivity, assign sequentially
            for v in range(n_verts):
                if not assigned[v]:
                    partition[v] = part_id
                    assigned[v] = True
                    if np.sum(partition == part_id) >= block_size:
                        part_id += 1
            break

        # Grow partition from seed
        partition[seed] = part_id
        assigned[seed] = True
        current_size = 1

        # Build candidate set from seed's neighbors
        candidates = set()
        for neighbor in adjacency_list[seed]:
            if not assigned[neighbor]:
                candidates.add(neighbor)

        while current_size < block_size and (candidates or not assigned.all()):
            best_score = -1
            best_vertex = -1

            # Score candidates by connection to current partition
            for v in candidates:
                if assigned[v]:
                    continue
                # Count connections to current partition
                score = sum(1 for n in adjacency_list[v]
                           if assigned[n] and partition[n] == part_id)
                # Small bonus for high connectivity
                score += len(adjacency_list[v]) * 0.01
                if score > best_score:
                    best_score = score
                    best_vertex = v

            if best_vertex == -1:
                # No valid candidate from neighbors, find any unassigned
                for v in range(n_verts):
                    if not assigned[v]:
                        best_vertex = v
                        break
                if best_vertex == -1:
                    break

            # Add to partition
            partition[best_vertex] = part_id
            assigned[best_vertex] = True
            current_size += 1
            candidates.discard(best_vertex)

            # Add new candidates from this vertex's neighbors
            for neighbor in adjacency_list[best_vertex]:
                if not assigned[neighbor]:
                    candidates.add(neighbor)

        part_id += 1

    # Handle any remaining unassigned vertices
    for v in range(n_verts):
        if partition[v] == -1:
            partition[v] = max(0, part_id - 1)

    return partition


# ============================================================================
# METIS Partitioning Functions
# ============================================================================

def metis_partition(n_verts: int, adjacency_list: List[np.ndarray],
                    n_parts: int) -> np.ndarray:
    """
    Partition vertices using METIS k-way partitioning.

    Args:
        n_verts: Number of vertices
        adjacency_list: Adjacency list for the graph
        n_parts: Number of partitions to create

    Returns:
        partition: Array of partition IDs for each vertex
    """
    try:
        import pymetis
    except ImportError:
        raise ImportError(
            "pymetis is required for METIS-based reordering. "
            "Install it with: pip install pymetis"
        )

    if n_parts <= 1:
        # Single partition - all vertices in partition 0
        return np.zeros(n_verts, dtype=np.int32)

    try:
        # pymetis.part_graph expects adjacency as list of numpy arrays
        n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
        return np.array(membership, dtype=np.int32)
    except Exception as e:
        print(f"[METIS] WARNING: pymetis.part_graph failed: {e}")
        print("[METIS] Falling back to simple sequential partitioning")
        # Simple fallback: sequential assignment
        partition = np.zeros(n_verts, dtype=np.int32)
        for i in range(n_verts):
            partition[i] = i % n_parts
        return partition


def metis_reorder_mesh(n_verts: int, cells: np.ndarray,
                       vertices: Optional[np.ndarray] = None,
                       block_size: int = BANKSIZE,
                       use_gpu: bool = True) -> dict:
    """
    Perform METIS-based mesh reordering for MAS preconditioner.

    This is the main entry point that performs the complete reordering pipeline:
    1. Build adjacency graph from mesh topology (GPU accelerated if use_gpu=True)
    2. Compute number of partitions to ensure max partition size <= block_size
    3. Partition using METIS with iterative refinement
    4. Compute sort index and inverse mapping (GPU accelerated)
    5. Build partition mappings for runtime use (GPU accelerated)

    Args:
        n_verts: Number of vertices
        cells: Cell array of shape (n_cells, 4) containing vertex indices
        vertices: Optional vertex positions of shape (n_verts, 3)
        block_size: Maximum partition size (default: BANKSIZE=16)
        use_gpu: Use GPU-accelerated operations (default: True)

    Returns:
        Dictionary containing all reordering data:
        - sort_index: sort_index[new] = old
        - old_to_new: old_to_new[old] = new
        - partition: partition IDs in sorted order
        - n_partitions: number of partitions
        - partId_map_real: mapping from (part_id, lane) to original vertex
        - real_map_partId: mapping from original vertex to (part_id, lane)
        - sorted_cells: cells with reordered vertex indices
        - stats: partition statistics
    """
    print(f"[METIS] Starting mesh reordering for {n_verts} vertices (GPU={use_gpu})...")

    # Check if pymetis is available
    if not check_pymetis_available():
        print("[METIS] WARNING: pymetis not available, returning identity mapping")
        return _create_identity_result(n_verts, cells, vertices, block_size)

    n_cells = cells.shape[0]

    if use_gpu:
        # GPU-accelerated path
        reorder_gpu = MetisReorderGPU(n_verts, n_cells)

        # Build adjacency on GPU
        print("[METIS] Building adjacency graph (GPU)...")
        adjacency_list, total_edges = reorder_gpu.build_adjacency(cells)
        print(f"[METIS] Built adjacency: {total_edges} edges")
    else:
        # CPU path (fallback)
        print("[METIS] Building adjacency graph (CPU)...")
        adjacency_list, _ = build_adjacency_from_cells_cpu(n_verts, cells)

    # Compute number of partitions with iterative refinement
    partition = None
    final_n_parts = 1

    for metis_offset in range(block_size):
        n_parts = (n_verts + block_size - metis_offset - 1) // (block_size - metis_offset)

        if n_parts <= 1:
            partition = np.zeros(n_verts, dtype=np.int32)
            final_n_parts = 1
            break

        print(f"[METIS] Trying {n_parts} partitions (offset={metis_offset})...")
        partition = metis_partition(n_verts, adjacency_list, n_parts)

        # Check max partition size
        partition_sizes = np.bincount(partition)
        max_size = int(np.max(partition_sizes))

        if max_size <= block_size:
            final_n_parts = n_parts
            print(f"[METIS] Success: max partition size = {max_size}")
            break

        print(f"[METIS] Max partition size {max_size} > {block_size}, retrying...")

    if use_gpu:
        # GPU-accelerated sort and mapping
        print("[METIS] Computing sort index and mappings (GPU)...")
        reorder_gpu.set_partition(partition)
        reorder_gpu.compute_sort_index_cpu()  # Sort is more efficient on CPU
        reorder_gpu.build_mappings()

        result = reorder_gpu.get_results()
    else:
        # CPU path
        print("[METIS] Computing sort index and mappings (CPU)...")
        sort_index = compute_sort_index_cpu(partition)
        old_to_new = compute_inverse_mapping_cpu(sort_index)
        sorted_partition = partition[sort_index]
        partId_map_real, real_map_partId = build_partition_mappings_cpu(
            sorted_partition, sort_index, block_size)
        sorted_cells = reorder_cells_cpu(cells, old_to_new)

        partition_sizes = np.bincount(sorted_partition)
        stats = {
            'n_vertices': n_verts,
            'n_partitions': final_n_parts,
            'max_partition_size': int(np.max(partition_sizes)),
            'min_partition_size': int(np.min(partition_sizes)),
            'avg_partition_size': float(np.mean(partition_sizes)),
            'non_full_partitions': int(np.sum(partition_sizes < block_size)),
        }

        result = {
            'sort_index': sort_index,
            'old_to_new': old_to_new,
            'partition': sorted_partition,
            'n_partitions': final_n_parts,
            'partId_map_real': partId_map_real,
            'real_map_partId': real_map_partId,
            'sorted_cells': sorted_cells,
            'stats': stats,
        }

    # Add sorted vertices if provided
    if vertices is not None:
        result['sorted_vertices'] = vertices[result['sort_index']].copy()

    print(f"[METIS] Partitioning complete:")
    print(f"  - Partitions: {result['stats']['n_partitions']}")
    print(f"  - Max size: {result['stats']['max_partition_size']}")
    print(f"  - Min size: {result['stats']['min_partition_size']}")
    print(f"  - Avg size: {result['stats']['avg_partition_size']:.1f}")
    print(f"  - Non-full: {result['stats']['non_full_partitions']}")

    return result


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_partition_file(partition: np.ndarray, filepath: str):
    """Save partition to file."""
    with open(filepath, 'w') as f:
        for p in partition:
            f.write(f"{p}\n")


def load_partition_file(filepath: str) -> np.ndarray:
    """Load partition from file."""
    partition = []
    with open(filepath, 'r') as f:
        for line in f:
            partition.append(int(line.strip()))
    return np.array(partition, dtype=np.int32)


# ============================================================================
# Integration with MAS Preconditioner
# ============================================================================

def apply_metis_reordering_to_mas(mas_preconditioner, reorder_result: dict):
    """
    Apply METIS reordering mappings to MAS preconditioner.

    Args:
        mas_preconditioner: MASPreconditioner instance
        reorder_result: Result from metis_reorder_mesh()
    """
    # Store mappings in the preconditioner
    mas_preconditioner.use_metis_reorder = True
    mas_preconditioner.metis_sort_index = reorder_result['sort_index']
    mas_preconditioner.metis_old_to_new = reorder_result['old_to_new']
    mas_preconditioner.metis_partition = reorder_result['partition']
    mas_preconditioner.metis_n_parts = reorder_result['n_partitions']
    mas_preconditioner.metis_stats = reorder_result['stats']

    # Create Taichi fields for GPU access
    n_verts = len(reorder_result['sort_index'])
    n_parts = reorder_result['n_partitions']

    mas_preconditioner.partId_map_real = ti.field(dtype=ti.i32,
                                                   shape=n_parts * BANKSIZE)
    mas_preconditioner.real_map_partId = ti.field(dtype=ti.i32,
                                                   shape=n_verts)

    # Copy mappings to Taichi fields
    mas_preconditioner.partId_map_real.from_numpy(reorder_result['partId_map_real'])
    mas_preconditioner.real_map_partId.from_numpy(reorder_result['real_map_partId'])

    print(f"[METIS] Applied reordering to MAS preconditioner")
    print(f"  - {n_parts} partitions for {n_verts} vertices")


# ============================================================================
# METISMixin Class - Integration with MASPreconditioner
# ============================================================================

class METISMixin:
    """
    Mixin class providing METIS-based reordering and hierarchy construction.

    METIS partitions the mesh graph so that topologically connected vertices
    are grouped together, improving the quality of subdomain Hessian blocks.

    Required attributes from main class:
        - n_verts: int
        - mesh: meshtaichi mesh
        - neighbor_num, neighbor_start, neighbor_list: neighbor fields
        - fine_connect_mask: ti.field(u32)
        - prefix_original, prefix_sum_original: ti.field(i32)
        - elected_mask: ti.field(u32)
        - going_next: ti.field(i32)
        - coarse_space_table_0: ti.field(i32)
        - level_size: ti.Vector.field(2, i32)
        - n_blocks_per_level: ti.field(i32)
        - multi_level_r, multi_level_z: ti.Vector.field(3, f32)
        - block_matrices, inv_block_matrices: matrix fields
    """

    def _init_metis_from_mesh(self):
        """
        Initialize METIS reordering automatically from mesh topology.

        This is called during __init__ when use_metis=True (the default).
        It extracts cell data from the MeshTaichi mesh and performs METIS
        partitioning for improved MAS block quality.
        """
        # Skip METIS for very small meshes (no benefit)
        if self.n_verts < BANKSIZE * 2:
            print(f"[MAS] Mesh too small ({self.n_verts} verts), skipping METIS")
            self.use_metis_reorder = False
            return

        try:
            # Extract cells from mesh using neighbor list
            # Build cells from the already-constructed neighbor topology
            cells_np = self._extract_cells_from_neighbors()

            if cells_np is None or len(cells_np) == 0:
                print("[MAS] WARNING: Could not extract cells, skipping METIS reordering")
                return

            # Get vertex positions
            vertices_np = self.mesh.get_position_as_numpy()

            # Call the main METIS initialization
            self.init_metis_reordering(cells_np, vertices_np)

        except Exception as e:
            print(f"[MAS] WARNING: Failed to initialize METIS from mesh: {e}")
            print("[MAS] Falling back to standard hierarchy (no METIS reordering)")
            self.use_metis_reorder = False

    def _extract_cells_from_neighbors(self) -> np.ndarray:
        """
        Extract cell connectivity from neighbor list.

        For tetrahedral meshes, we can reconstruct cells by finding
        4-cliques in the neighbor graph. However, this is expensive.
        Instead, we use the neighbor information directly for METIS
        partitioning, which only needs the graph structure.

        Returns:
            np.ndarray: Cell array of shape (n_cells, 4) or None if extraction fails
        """
        # Get neighbor data
        neighbor_num_np = self.neighbor_num.to_numpy()
        neighbor_start_np = self.neighbor_start.to_numpy()
        neighbor_list_np = self.neighbor_list.to_numpy()

        # Build cells by finding tetrahedra from mesh topology
        # For MeshTaichi, iterate through cells directly using the mesh relation
        cells_list = []

        try:
            # Try to access cells directly through mesh.cells relation
            # This works if the mesh has CV (cell-vertex) relation
            n_cells = len(self.mesh.cells)
            for c_idx in range(n_cells):
                # Access cell's vertices through the relation
                cell = self.mesh.cells[c_idx]
                v0 = cell.verts[0].id if hasattr(cell.verts[0], 'id') else 0
                v1 = cell.verts[1].id if hasattr(cell.verts[1], 'id') else 1
                v2 = cell.verts[2].id if hasattr(cell.verts[2], 'id') else 2
                v3 = cell.verts[3].id if hasattr(cell.verts[3], 'id') else 3
                cells_list.append([v0, v1, v2, v3])
        except Exception:
            # Fall back: construct pseudo-cells from neighbor list for METIS
            # Each vertex with its 3 most connected neighbors forms a pseudo-cell
            print("[MAS] Using neighbor-based pseudo-cells for METIS")
            for i in range(self.n_verts):
                n_neighbors = neighbor_num_np[i]
                if n_neighbors >= 3:
                    start = neighbor_start_np[i]
                    neighbors = neighbor_list_np[start:start+min(n_neighbors, 3)]
                    cells_list.append([i] + list(neighbors[:3]))

        if len(cells_list) == 0:
            return None

        return np.array(cells_list, dtype=np.int32)

    def init_metis_reordering(self, cells_np: np.ndarray, vertices_np: np.ndarray = None):
        """
        Initialize METIS-based node reordering for improved MAS quality.

        This implements the CEMAS (Connectivity-Enhanced MAS) approach from
        the Stiff-GIPC reference. METIS partitions the mesh graph so that
        topologically connected vertices are grouped together in blocks.

        Args:
            cells_np: Cell array of shape (n_cells, 4) containing vertex indices
            vertices_np: Optional vertex positions of shape (n_verts, 3)
        """
        print("[MAS] Initializing METIS-based node reordering...")

        # Perform METIS partitioning using module function
        reorder_result = metis_reorder_mesh(
            n_verts=self.n_verts,
            cells=cells_np,
            vertices=vertices_np,
            block_size=BANKSIZE
        )

        # Store reordering data
        self.use_metis_reorder = True
        self.metis_sort_index = reorder_result['sort_index']
        self.metis_old_to_new = reorder_result['old_to_new']
        self.metis_partition = reorder_result['partition']
        self.metis_n_parts = reorder_result['n_partitions']
        self.metis_stats = reorder_result['stats']

        # Create Taichi fields for GPU access
        n_parts = reorder_result['n_partitions']

        self.partId_map_real = ti.field(dtype=ti.i32, shape=n_parts * BANKSIZE)
        self.real_map_partId = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Copy mappings to Taichi fields
        self.partId_map_real.from_numpy(reorder_result['partId_map_real'])
        self.real_map_partId.from_numpy(reorder_result['real_map_partId'])

        # Store sorted cells for reference
        self.metis_sorted_cells = reorder_result['sorted_cells']

        print(f"[MAS] METIS reordering initialized:")
        print(f"  - {n_parts} partitions for {self.n_verts} vertices")
        print(f"  - Max partition size: {self.metis_stats['max_partition_size']}")
        print(f"  - Avg partition size: {self.metis_stats['avg_partition_size']:.1f}")

    def get_metis_vertex_id(self, sorted_idx: int) -> int:
        """
        Get original vertex ID from sorted index.

        Args:
            sorted_idx: Index in the sorted (METIS-reordered) array

        Returns:
            Original vertex ID
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            return sorted_idx
        return int(self.metis_sort_index[sorted_idx])

    def get_sorted_vertex_id(self, original_idx: int) -> int:
        """
        Get sorted index from original vertex ID.

        Args:
            original_idx: Original vertex ID

        Returns:
            Index in the sorted (METIS-reordered) array
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            return original_idx
        return int(self.metis_old_to_new[original_idx])

    # ========================================================================
    # METIS-specific Connectivity and Hierarchy Kernels
    # ========================================================================

    @ti.kernel
    def _build_connect_mask_l0_metis(self):
        """
        Build connectivity bitmask at Level 0 using METIS partition info.

        When METIS reordering is used, vertices within each partition are
        already grouped by connectivity, so we use the partition structure
        directly to determine block membership.
        """
        for idx in range(self.n_verts):
            # Get partition info from METIS mapping
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            # Start with self-connectivity
            connect_mask = ti.u32(1) << ti.u32(lane_id)

            # Check neighbors - only mark as connected if in same partition
            num_neighbor = self.neighbor_num[idx]
            start_id = self.neighbor_start[idx]

            for i in range(num_neighbor):
                neighbor_id = self.neighbor_list[start_id + i]
                neighbor_part_info = self.real_map_partId[neighbor_id]
                neighbor_block = neighbor_part_info // BANKSIZE

                if block_id == neighbor_block:
                    # Same partition/block: add to connectivity mask
                    neighbor_lane = neighbor_part_info % BANKSIZE
                    connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))

            self.fine_connect_mask[idx] = connect_mask

    @ti.kernel
    def _propagate_connectivity_metis(self):
        """
        Propagate connectivity using METIS partition structure.

        This is the METIS version of _propagate_connectivity() that uses
        METIS partition IDs (block_id from real_map_partId) instead of
        warp IDs (idx // BANKSIZE) to compute prefix_original.

        The key difference is that prefix_original[block_id] counts the number
        of connected components (cluster representatives) within each METIS partition,
        not within each sequential warp of 16 vertices.
        """
        n_parts = self.metis_n_parts

        # Reset prefix counts for METIS partitions
        for p in range(n_parts):
            self.prefix_original[p] = 0

        for idx in range(self.n_verts):
            # Get METIS partition info
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # BFS-style transitive closure within the METIS partition
            visited = ti.u32(1) << ti.u32(lane_id)

            max_iter = BANKSIZE
            for _ in range(max_iter):
                todo = visited ^ connect_mask

                if todo == 0:
                    break

                next_visit = self._find_first_set(todo)
                if next_visit < 0:
                    break

                visited |= ti.u32(1) << ti.u32(next_visit)

                # Find the vertex in this METIS partition with the given lane
                other_part_idx = block_id * BANKSIZE + next_visit
                other_idx = self.partId_map_real[other_part_idx]

                if other_idx >= 0 and other_idx < self.n_verts:
                    connect_mask |= self.fine_connect_mask[other_idx]

            # Store final transitive closure
            self.fine_connect_mask[idx] = connect_mask

            # Count elected representatives using METIS partition (not warp)
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This node is the representative of its connected component
                ti.atomic_add(self.prefix_original[block_id], 1)

    @ti.kernel
    def _find_cluster_representatives_metis(self):
        """
        Find cluster representatives using METIS partition structure.

        Sets elected_mask based on METIS partitions rather than sequential warps.
        """
        n_parts = self.metis_n_parts

        # Reset elected mask
        for p in range(n_parts):
            self.elected_mask[p] = ti.u32(0)

        # Build elected mask
        for idx in range(self.n_verts):
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]

            # Count how many connected nodes have lower lane ID
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is the representative of its cluster
                ti.atomic_or(self.elected_mask[block_id], ti.u32(1) << ti.u32(lane_id))

    @ti.kernel
    def _assign_cluster_ids_metis(self, level_1_offset: ti.i32):
        """
        Assign cluster IDs using METIS partition structure.

        Maps each vertex to its coarse-level cluster based on METIS partitions.
        """
        for idx in range(self.n_verts):
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_mask = self.elected_mask[block_id]

            # Find which elected node this vertex belongs to
            my_cluster_mask = connect_mask & elected_mask

            if my_cluster_mask != 0:
                # Find the lowest set bit (the representative)
                rep_lane = self._find_first_set(my_cluster_mask)

                # Count elected nodes with lower lane ID
                prefix_in_warp = self._popcount(elected_mask & self._lanemask_lt(rep_lane))

                # Global cluster ID
                cluster_id = self.prefix_sum_original[block_id] + prefix_in_warp

                # Store mapping
                self.going_next[idx] = level_1_offset + cluster_id
                self.coarse_space_table_0[idx] = level_1_offset + cluster_id

    @ti.kernel
    def _add_inertia_contribution_metis(self, dt: ti.f32):
        """Add mass matrix to diagonal blocks using METIS mapping.

        Note: The inertia Hessian is just 'm' (not m/dt²) to match the gradient scaling.
        """
        for idx in range(self.n_verts):
            # Get block and lane from METIS partition
            part_info = self.real_map_partId[idx]
            block_id = part_info // BANKSIZE
            lane_id = part_info % BANKSIZE

            # Get mass from mesh (using original vertex index)
            m = self.mesh.verts.m[idx]

            # Diagonal block index in symmetric storage
            sym_idx = self._sym_index(lane_id, lane_id)

            # Use m directly (not m/dt²) to match gradient scaling
            mass_val = m
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[block_id, sym_idx][d, d], mass_val)

    @ti.kernel
    def _schwarz_local_solve_full_metis(self):
        """
        Solve z_d = B_d^{-1} * r_d using METIS partition mapping.

        partId_map_real[part_id * BANKSIZE + lane] = original_vertex_id
        This kernel iterates over all METIS partitions (blocks) and applies
        the inverse block matrix to compute z = B^{-1} * r.
        """
        n_blocks = self.metis_n_parts

        for block_id in range(n_blocks):
            for lane_i in range(BANKSIZE):
                # Get original vertex ID from partition mapping
                part_idx_i = block_id * BANKSIZE + lane_i
                idx_i = self.partId_map_real[part_idx_i]

                # Check if this slot is valid (not all partitions are full)
                if idx_i >= 0 and idx_i < self.n_verts:
                    # Initialize z to zero
                    z = ti.Vector.zero(ti.f32, 3)

                    # Multiply by full inverse block
                    for lane_j in range(BANKSIZE):
                        part_idx_j = block_id * BANKSIZE + lane_j
                        idx_j = self.partId_map_real[part_idx_j]

                        if idx_j >= 0 and idx_j < self.n_verts:
                            sym_idx = self._sym_index(lane_i, lane_j)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            r_j = self.multi_level_r[idx_j]

                            if lane_i <= lane_j:
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * r_j[dj]
                            else:
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * r_j[dj]

                    self.multi_level_z[idx_i] = z

        # Coarse level solve (same as before)
        level_1_offset = self.n_verts
        level_1_size = self.level_size[1][0]

        for i in range(level_1_size):
            idx = level_1_offset + i
            r = self.multi_level_r[idx]
            r_norm = r.norm()
            if r_norm > 1e-10:
                scale = 1.0 / r_norm
                self.multi_level_z[idx] = r * ti.min(scale, 1.0)
            else:
                self.multi_level_z[idx] = r

    # ========================================================================
    # METIS Hierarchy Building
    # ========================================================================

    def build_hierarchy_metis(self):
        """
        Build hierarchy using METIS partition information.

        This is an optimized version that uses METIS partition structure
        directly for the hierarchy construction. It uses METIS-specific
        versions of connectivity propagation and cluster assignment that
        work with METIS partition IDs instead of sequential warp IDs.
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            print("[MAS] WARNING: METIS reordering not initialized, using standard hierarchy")
            self.build_hierarchy()
            return

        print("[MAS] Building hierarchy with METIS partitioning...")

        # Use METIS partition for Level 0 connectivity
        self._build_connect_mask_l0_metis()

        # Propagate connectivity using METIS partition structure
        # This sets prefix_original[block_id] for each METIS partition
        self._propagate_connectivity_metis()

        # Find cluster representatives using METIS partitions
        self._find_cluster_representatives_metis()

        # Compute prefix sum for cluster IDs
        prefix_np = self.prefix_original.to_numpy()
        n_parts = self.metis_n_parts
        prefix_sum_np = np.zeros(n_parts + 1, dtype=np.int32)
        prefix_sum_np[1:n_parts+1] = np.cumsum(prefix_np[:n_parts])
        self.prefix_sum_original.from_numpy(prefix_sum_np)

        # Level 1 size and offset
        level_1_size = int(prefix_sum_np[n_parts])
        level_1_offset = self.n_verts

        # Store level info
        self.level_size[0] = ti.Vector([self.n_verts, 0])
        self.level_size[1] = ti.Vector([level_1_size, level_1_offset])

        # Assign cluster IDs using METIS partition structure
        self._assign_cluster_ids_metis(level_1_offset)

        # Set block counts
        self.n_blocks_per_level[0] = self.metis_n_parts

        print(f"[MAS] Level 0: {self.n_verts} nodes ({self.metis_n_parts} METIS partitions)")
        print(f"[MAS] Level 1: {level_1_size} nodes")

        # Build additional coarse levels (same as standard)
        actual_levels = 2
        current_size = level_1_size
        current_offset = level_1_offset

        for level in range(2, MAX_LEVELS):
            if current_size <= BANKSIZE:
                print(f"[MAS] Stopping at level {level-1}: {current_size} nodes <= BANKSIZE")
                break

            self._build_connect_mask_lx(current_offset, current_size)
            self._propagate_connectivity_lx(current_offset, current_size)

            next_n_warps = (current_size + BANKSIZE - 1) // BANKSIZE
            self._assign_coarse_cluster_ids(level - 1, current_offset, current_size, 0)

            prefix_np = self.prefix_original.to_numpy()
            prefix_sum_np = np.zeros(next_n_warps + 1, dtype=np.int32)
            prefix_sum_np[1:next_n_warps+1] = np.cumsum(prefix_np[:next_n_warps])
            self.prefix_sum_original.from_numpy(prefix_sum_np)

            next_size = int(prefix_sum_np[next_n_warps])
            next_offset = current_offset + current_size

            if next_size >= current_size * 0.9:
                print(f"[MAS] Stopping at level {level}: insufficient reduction")
                break

            self.level_size[level] = ti.Vector([next_size, next_offset])
            self._update_going_next_lx(current_offset, current_size, next_offset, level - 1)

            print(f"[MAS] Level {level}: {next_size} nodes")

            current_size = next_size
            current_offset = next_offset
            actual_levels = level + 1

            n_blocks = (current_size + BANKSIZE - 1) // BANKSIZE
            self.n_blocks_per_level[level - 1] = n_blocks

        self._build_aggregation_table()

        # P1 Optimization: Cache warp prefix values for fast access during restriction
        self._cache_warp_prefix()

        print(f"[MAS] METIS hierarchy built with {actual_levels} levels")

        self.hierarchy_built = True
        self.actual_levels = actual_levels

    # ========================================================================
    # METIS Apply and Rebuild Methods
    # ========================================================================

    def rebuild_with_metis(self, solver, use_full_hessian: bool = True,
                           method: str = 'ic', adaptive_regularization: float = 0.05):
        """
        Full rebuild of preconditioner using METIS-based partitioning.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling
            method: Inversion method ('ic', 'cholesky', 'gauss_jordan', etc.)
            adaptive_regularization: Per-block regularization (default: 0.05)
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            print("[MAS] WARNING: METIS not initialized, using standard rebuild")
            self.rebuild(solver, use_full_hessian, method, adaptive_regularization)
            return

        if not self.hierarchy_built:
            self.build_hierarchy_metis()

        if hasattr(solver, 'elastic_type'):
            self.elastic_type = solver.elastic_type

        self.assemble_block_matrices(solver, use_full_hessian)
        self.invert_block_matrices(method=method, adaptive_regularization=adaptive_regularization)

    def apply_metis(self, solve_method: str = 'banded'):
        """
        Apply MAS preconditioner using METIS partition structure.

        Args:
            solve_method: Local solve method (see apply() for options)
        """
        if not hasattr(self, 'use_metis_reorder') or not self.use_metis_reorder:
            self.apply(solve_method)
            return

        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction
        self._build_multi_level_r()

        # Phase 2: Local solve with METIS mapping
        if solve_method.lower() != 'diagonal':
            self._schwarz_local_solve_full_metis()
        else:
            self._schwarz_local_solve()

        # Phase 3: Prolongation
        self._collect_final_z()
