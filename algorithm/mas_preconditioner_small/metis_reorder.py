"""
METIS Reordering for MAS Preconditioner Small.

Simplified METIS integration that only needs to run ONCE during simulation setup.
The reordering maps are computed at initialization and reused throughout simulation.

Key features:
- One-time METIS computation at initialization
- CPU-based for simplicity (METIS is already fast enough)
- Provides partition mappings for block assembly
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

BANKSIZE = 16


def check_pymetis_available() -> bool:
    """Check if pymetis is available."""
    try:
        import pymetis
        return True
    except ImportError:
        return False


def build_adjacency_from_cells(n_verts: int, cells: np.ndarray) -> List[np.ndarray]:
    """
    Build adjacency list from cell connectivity.

    Args:
        n_verts: Number of vertices
        cells: Cell array of shape (n_cells, 4)

    Returns:
        adjacency_list: List of neighbor arrays for each vertex
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

    return adjacency_list


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
        # Fallback: sequential partitioning
        partition = np.zeros(n_verts, dtype=np.int32)
        for i in range(n_verts):
            partition[i] = i // BANKSIZE
        return partition

    if n_parts <= 1:
        return np.zeros(n_verts, dtype=np.int32)

    try:
        n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
        return np.array(membership, dtype=np.int32)
    except Exception as e:
        print(f"[METIS] WARNING: pymetis failed: {e}, using sequential partition")
        partition = np.zeros(n_verts, dtype=np.int32)
        for i in range(n_verts):
            partition[i] = i // BANKSIZE
        return partition


class MetisReorderResult:
    """
    Container for METIS reordering results.

    Computed once at initialization, reused throughout simulation.
    """

    def __init__(self, n_verts: int, n_parts: int):
        self.n_verts = n_verts
        self.n_parts = n_parts

        # Mappings
        self.sort_index = None      # sort_index[new_idx] = old_idx
        self.old_to_new = None      # old_to_new[old_idx] = new_idx
        self.partition = None       # partition[sorted_idx] = part_id

        # Block mappings for MAS
        # partId_map_real[part_id * BANKSIZE + lane] = original_vertex_id
        self.partId_map_real = None
        # real_map_partId[original_vertex_id] = part_id * BANKSIZE + lane
        self.real_map_partId = None

        # Statistics
        self.stats = {}

    def is_valid(self) -> bool:
        """Check if result is valid."""
        return (self.sort_index is not None and
                self.partId_map_real is not None and
                self.real_map_partId is not None)


def compute_metis_reorder(n_verts: int, cells: np.ndarray,
                          block_size: int = BANKSIZE) -> MetisReorderResult:
    """
    Compute METIS-based reordering for MAS preconditioner.

    This function should be called ONCE at simulation initialization.
    The result is then passed to MASPreconditionerSmall for reuse.

    Args:
        n_verts: Number of vertices
        cells: Cell array of shape (n_cells, 4)
        block_size: Maximum partition size (default: BANKSIZE=16)

    Returns:
        MetisReorderResult containing all reordering data
    """
    print(f"[METIS] Computing reordering for {n_verts} vertices...")

    result = MetisReorderResult(n_verts, 0)

    # Check if pymetis is available
    if not check_pymetis_available():
        print("[METIS] WARNING: pymetis not available, using identity mapping")
        return _create_identity_result(n_verts, block_size)

    # Build adjacency list
    print("[METIS] Building adjacency graph...")
    adjacency_list = build_adjacency_from_cells(n_verts, cells)

    # Iteratively find partition count that satisfies block_size constraint
    partition = None
    n_parts = 1

    for metis_offset in range(block_size):
        n_parts = (n_verts + block_size - metis_offset - 1) // (block_size - metis_offset)

        if n_parts <= 1:
            partition = np.zeros(n_verts, dtype=np.int32)
            n_parts = 1
            break

        partition = metis_partition(n_verts, adjacency_list, n_parts)

        # Check max partition size
        partition_sizes = np.bincount(partition)
        max_size = int(np.max(partition_sizes))

        if max_size <= block_size:
            print(f"[METIS] Success with {n_parts} partitions, max size = {max_size}")
            break

    result.n_parts = n_parts

    # Compute sort index (stable sort by partition ID)
    indexed = [(i, partition[i]) for i in range(n_verts)]
    indexed.sort(key=lambda x: x[1])

    sort_index = np.array([p[0] for p in indexed], dtype=np.int32)
    result.sort_index = sort_index

    # Compute inverse mapping
    old_to_new = np.zeros(n_verts, dtype=np.int32)
    for new_pos in range(n_verts):
        old_id = sort_index[new_pos]
        old_to_new[old_id] = new_pos
    result.old_to_new = old_to_new

    # Sorted partition
    sorted_partition = partition[sort_index]
    result.partition = sorted_partition

    # Build partition mappings for MAS
    partId_map_real = np.full(n_parts * block_size, -1, dtype=np.int32)
    real_map_partId = np.zeros(n_verts, dtype=np.int32)
    local_index = np.zeros(n_parts, dtype=np.int32)

    for sorted_idx in range(n_verts):
        orig_id = sort_index[sorted_idx]
        part_id = sorted_partition[sorted_idx]
        local_idx = local_index[part_id]

        if local_idx < block_size:
            partId_map_real[part_id * block_size + local_idx] = orig_id
            real_map_partId[orig_id] = part_id * block_size + local_idx
            local_index[part_id] += 1

    result.partId_map_real = partId_map_real
    result.real_map_partId = real_map_partId

    # Compute statistics
    partition_sizes = np.bincount(sorted_partition)
    result.stats = {
        'n_vertices': n_verts,
        'n_partitions': n_parts,
        'max_partition_size': int(np.max(partition_sizes)),
        'min_partition_size': int(np.min(partition_sizes)),
        'avg_partition_size': float(np.mean(partition_sizes)),
        'non_full_partitions': int(np.sum(partition_sizes < block_size)),
    }

    print(f"[METIS] Reordering complete:")
    print(f"  - Partitions: {n_parts}")
    print(f"  - Max size: {result.stats['max_partition_size']}")
    print(f"  - Avg size: {result.stats['avg_partition_size']:.1f}")

    return result


def _create_identity_result(n_verts: int, block_size: int) -> MetisReorderResult:
    """Create identity mapping when METIS is not available."""
    n_parts = (n_verts + block_size - 1) // block_size

    result = MetisReorderResult(n_verts, n_parts)

    result.sort_index = np.arange(n_verts, dtype=np.int32)
    result.old_to_new = np.arange(n_verts, dtype=np.int32)

    partition = np.zeros(n_verts, dtype=np.int32)
    for i in range(n_verts):
        partition[i] = i // block_size
    result.partition = partition

    # Build identity mappings
    partId_map_real = np.full(n_parts * block_size, -1, dtype=np.int32)
    real_map_partId = np.zeros(n_verts, dtype=np.int32)

    for i in range(n_verts):
        part_id = i // block_size
        local_idx = i % block_size
        partId_map_real[part_id * block_size + local_idx] = i
        real_map_partId[i] = part_id * block_size + local_idx

    result.partId_map_real = partId_map_real
    result.real_map_partId = real_map_partId

    result.stats = {
        'n_vertices': n_verts,
        'n_partitions': n_parts,
        'max_partition_size': min(block_size, n_verts),
        'min_partition_size': n_verts % block_size if n_verts % block_size > 0 else block_size,
        'avg_partition_size': n_verts / n_parts,
        'non_full_partitions': 1 if n_verts % block_size > 0 else 0,
    }

    return result


def extract_cells_from_mesh(mesh) -> np.ndarray:
    """
    Extract cell connectivity from MeshTaichi mesh.

    Args:
        mesh: MeshTaichi mesh object

    Returns:
        cells: numpy array of shape (n_cells, 4)
    """
    n_cells = mesh.cells.size
    cells_list = []

    try:
        # Access cells through mesh.cells relation
        for c_idx in range(n_cells):
            cell = mesh.cells[c_idx]
            v0 = cell.verts[0].id
            v1 = cell.verts[1].id
            v2 = cell.verts[2].id
            v3 = cell.verts[3].id
            cells_list.append([v0, v1, v2, v3])
    except Exception as e:
        print(f"[METIS] WARNING: Could not extract cells: {e}")
        return None

    return np.array(cells_list, dtype=np.int32)


def compute_optimized_cell_data(cells: np.ndarray, metis_result: MetisReorderResult,
                                 block_size: int = BANKSIZE) -> dict:
    """
    Compute optimized cell data for METIS-aware assembly.

    This function precomputes:
    1. Reordered cell connectivity (vertex IDs mapped to METIS order)
    2. Cell processing order (sorted by main partition to reduce atomic conflicts)

    Args:
        cells: Original cell array of shape (n_cells, 4)
        metis_result: Pre-computed MetisReorderResult
        block_size: Block size (default: BANKSIZE=16)

    Returns:
        dict with:
            'sorted_cells': np.ndarray of shape (n_cells, 4) with METIS-reordered vertex IDs,
                           sorted by main partition
            'cell_order': np.ndarray of shape (n_cells,) mapping sorted index to original index
            'cell_main_partition': np.ndarray of shape (n_cells,) with main partition per cell
            'stats': dict with statistics
    """
    if cells is None or metis_result is None:
        return None

    n_cells = len(cells)
    old_to_new = metis_result.old_to_new

    # Step 1: Reorder cell vertex IDs and compute main partition
    reordered_cells = np.zeros((n_cells, 4), dtype=np.int32)
    cell_main_partition = np.zeros(n_cells, dtype=np.int32)

    same_partition_count = 0

    for c_idx in range(n_cells):
        parts = []
        for i in range(4):
            old_v = cells[c_idx, i]
            new_v = old_to_new[old_v]
            reordered_cells[c_idx, i] = new_v
            parts.append(new_v // block_size)  # partition = block

        # Main partition: most common among the 4 vertices
        from collections import Counter
        part_counts = Counter(parts)
        cell_main_partition[c_idx] = part_counts.most_common(1)[0][0]

        # Count cells with all 4 vertices in same partition
        if len(set(parts)) == 1:
            same_partition_count += 1

    # Step 2: Sort cells by main partition (stable sort preserves original order within partition)
    cell_order = np.argsort(cell_main_partition, kind='stable')

    # Step 3: Reorder cells according to sorted order
    sorted_cells = reordered_cells[cell_order]

    # Compute statistics
    stats = {
        'n_cells': n_cells,
        'same_partition_cells': same_partition_count,
        'same_partition_ratio': same_partition_count / n_cells if n_cells > 0 else 0,
        'n_partitions': metis_result.n_parts,
    }

    print(f"[METIS] Cell optimization:")
    print(f"  - Same partition cells: {same_partition_count}/{n_cells} ({stats['same_partition_ratio']*100:.1f}%)")

    return {
        'sorted_cells': sorted_cells,
        'cell_order': cell_order,
        'cell_main_partition': cell_main_partition,
        'stats': stats,
    }
