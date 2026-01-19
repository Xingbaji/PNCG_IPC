"""
METIS-based node reordering for MAS preconditioner with GPU acceleration.

This module implements METIS graph partitioning to reorder mesh vertices
so that vertices in the same partition are contiguous. This improves
the quality of the MAS preconditioner by ensuring that vertices within
each BANKSIZE block are topologically connected.

The approach follows the Stiff-GIPC reference implementation:
1. Build adjacency graph from mesh topology (GPU accelerated)
2. Use METIS k-way partitioning to partition vertices into ~BANKSIZE groups
3. Sort vertices by partition ID to make partitions contiguous (GPU accelerated)
4. Build bidirectional mappings for runtime use (GPU accelerated)

Reference: /root/Stiff-GIPC_init/MeshProcess/metis_partition/
"""

import taichi as ti
import numpy as np
from typing import Tuple, List, Optional, Dict

# Constants matching CUDA reference
BANKSIZE = 16  # Nodes per subdomain (warp subdivision)


def check_pymetis_available() -> bool:
    """Check if pymetis is available."""
    try:
        import pymetis
        return True
    except ImportError:
        return False


@ti.data_oriented
class MetisReorderGPU:
    """
    GPU-accelerated METIS reordering for MAS preconditioner.

    This class provides Taichi kernels for parallel operations in the
    METIS reordering pipeline.
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
        partition_count = ti.Vector.zero(ti.i32, 1)  # placeholder
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
        Dictionary containing all reordering data
    """
    print(f"[METIS] Starting mesh reordering for {n_verts} vertices (GPU={use_gpu})...")

    # Check if pymetis is available
    if not check_pymetis_available():
        print("[METIS] WARNING: pymetis not available, returning identity mapping")
        return _create_identity_result(n_verts, cells, vertices)

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
        partId_map_real, real_map_partId = build_partition_mappings_cpu(sorted_partition, sort_index)
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
# CPU Fallback Functions
# ============================================================================

def build_adjacency_from_cells_cpu(n_verts: int, cells: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
    """
    Build adjacency list from cell connectivity (CPU version).
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
                                  sort_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build partition mappings on CPU.

    Both mappings use ORIGINAL vertex indices (not sorted indices)
    to allow direct access in MAS preconditioner without index conversion.

    Args:
        partition: Partition IDs for each vertex (in original vertex order)
        sort_index: sort_index[sorted_pos] = original_vertex_id

    Returns:
        partId_map_real: partId_map_real[part_id * BANKSIZE + lane] = original_vertex_id
        real_map_partId: real_map_partId[original_vertex_id] = part_id * BANKSIZE + lane
    """
    n_verts = len(sort_index)
    n_parts = int(np.max(partition)) + 1

    partId_map_real = np.full(n_parts * BANKSIZE, -1, dtype=np.int32)
    real_map_partId = np.zeros(n_verts, dtype=np.int32)
    local_index = np.zeros(n_parts, dtype=np.int32)

    # Iterate in sorted order to ensure consistent lane assignment
    for sorted_idx in range(n_verts):
        orig_id = sort_index[sorted_idx]
        part_id = partition[sorted_idx]  # partition is already in sorted order
        local_idx = local_index[part_id]

        if local_idx < BANKSIZE:
            # Map using ORIGINAL vertex indices
            partId_map_real[part_id * BANKSIZE + local_idx] = orig_id
            real_map_partId[orig_id] = part_id * BANKSIZE + local_idx
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
                            vertices: Optional[np.ndarray]) -> dict:
    """Create identity mapping result when METIS is not available."""
    n_parts = (n_verts + BANKSIZE - 1) // BANKSIZE

    sort_index = np.arange(n_verts, dtype=np.int32)
    old_to_new = np.arange(n_verts, dtype=np.int32)

    partition = np.zeros(n_verts, dtype=np.int32)
    for i in range(n_verts):
        partition[i] = i // BANKSIZE

    partId_map_real, real_map_partId = build_partition_mappings_cpu(partition, sort_index)

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
            'max_partition_size': min(BANKSIZE, n_verts),
            'min_partition_size': n_verts % BANKSIZE if n_verts % BANKSIZE > 0 else BANKSIZE,
            'avg_partition_size': n_verts / n_parts,
            'non_full_partitions': 1 if n_verts % BANKSIZE > 0 else 0,
        },
    }

    if vertices is not None:
        result['sorted_vertices'] = vertices.copy()

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
