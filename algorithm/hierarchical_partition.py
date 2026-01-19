"""
Hierarchical Unified Partition (HUP) for MAS-MeshTaichi integration.

This module implements a two-level partitioning strategy that combines:
1. MeshTaichi-style Patch partitioning (minimize boundary overhead)
2. METIS-based block partitioning within each Patch (maximize block connectivity)

The resulting layout enables:
- Coalesced memory access (Patch-aligned data)
- Shared memory utilization (Patch fits in GPU shared memory)
- High-quality MAS blocks (METIS-partitioned within Patch)

Reference: ref_doc/MAS_MESHTAICHI_INTEGRATION_DESIGN.md
"""

import taichi as ti
import numpy as np
from typing import Tuple, List, Optional, Dict

# Constants
BANKSIZE = 16           # MAS block size (vertices per block)
PATCH_SIZE = 2048       # MeshTaichi-compatible patch size
BLOCKS_PER_PATCH = PATCH_SIZE // BANKSIZE  # = 128


def check_pymetis_available() -> bool:
    """Check if pymetis is available."""
    try:
        import pymetis
        return True
    except ImportError:
        return False


@ti.data_oriented
class HierarchicalPartitionGPU:
    """
    GPU-accelerated hierarchical partition for MAS-MeshTaichi integration.

    This class provides:
    1. Greedy Patch partitioning (MeshTaichi-style)
    2. METIS block partitioning within each Patch
    3. GPU-parallel data reordering
    4. Prologue-Epilogue data transfer kernels
    """

    def __init__(self, n_verts: int, n_cells: int, patch_size: int = PATCH_SIZE):
        """
        Initialize hierarchical partition structures.

        Args:
            n_verts: Number of vertices
            n_cells: Number of cells (tetrahedra)
            patch_size: Size of each Patch (default: 2048)
        """
        self.n_verts = n_verts
        self.n_cells = n_cells
        self.patch_size = patch_size
        self.blocks_per_patch = patch_size // BANKSIZE

        # Compute number of patches (will be updated after partitioning)
        self.n_patches = (n_verts + patch_size - 1) // patch_size
        # Use n_verts as upper bound for blocks (worst case: 1 vertex per block)
        self.max_blocks = (n_verts + BANKSIZE - 1) // BANKSIZE
        self.n_blocks = self.max_blocks  # Will be updated after partitioning

        print(f"[HUP] Initializing: {n_verts} verts, {self.n_patches} patches (initial), "
              f"{self.max_blocks} max blocks")

        # Adjacency structures (CSR format)
        self.max_neighbors = 64
        self.adj_count = ti.field(dtype=ti.i32, shape=n_verts)
        self.adj_offset = ti.field(dtype=ti.i32, shape=n_verts + 1)
        self.adj_list = ti.field(dtype=ti.i32, shape=n_verts * self.max_neighbors)

        # Cell data
        self.cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

        # Partition assignments
        self.patch_assignment = ti.field(dtype=ti.i32, shape=n_verts)  # vertex -> patch_id
        self.block_assignment = ti.field(dtype=ti.i32, shape=n_verts)  # vertex -> global_block_id

        # Reordering mappings
        self.sort_index = ti.field(dtype=ti.i32, shape=n_verts)      # new -> old
        self.old_to_new = ti.field(dtype=ti.i32, shape=n_verts)      # old -> new

        # Patch-block mappings for MAS (use max_blocks for allocation)
        self.block_to_patch = ti.field(dtype=ti.i32, shape=self.max_blocks)
        self.block_local_id = ti.field(dtype=ti.i32, shape=self.max_blocks)

        # Vertex mappings within blocks (use max_blocks for allocation)
        self.partId_map_real = ti.field(dtype=ti.i32, shape=self.max_blocks * BANKSIZE)
        self.real_map_partId = ti.field(dtype=ti.i32, shape=n_verts)

        # Local index counters
        self.local_index = ti.field(dtype=ti.i32, shape=self.max_blocks)

        # Sorted cells
        self.sorted_cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

        # Patch boundary info
        self.is_boundary_vertex = ti.field(dtype=ti.i32, shape=n_verts)
        self.patch_boundary_count = ti.field(dtype=ti.i32, shape=self.n_patches)

        # Reordered data buffers for Prologue-Epilogue
        self.reordered_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.reordered_grad = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.reordered_z = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

    def load_cells(self, cells_np: np.ndarray):
        """Load cell data from numpy array."""
        self.cells.from_numpy(cells_np.astype(np.int32))

    @ti.kernel
    def _count_neighbors(self):
        """Count neighbors for each vertex."""
        for i in range(self.n_verts):
            self.adj_count[i] = 0

        for c in range(self.n_cells):
            v0, v1, v2, v3 = self.cells[c][0], self.cells[c][1], self.cells[c][2], self.cells[c][3]
            ti.atomic_add(self.adj_count[v0], 3)
            ti.atomic_add(self.adj_count[v1], 3)
            ti.atomic_add(self.adj_count[v2], 3)
            ti.atomic_add(self.adj_count[v3], 3)

    @ti.kernel
    def _compute_adj_offset(self):
        """Compute adjacency offset (prefix sum)."""
        self.adj_offset[0] = 0
        for i in range(self.n_verts):
            count = ti.min(self.adj_count[i], self.max_neighbors)
            self.adj_offset[i + 1] = self.adj_offset[i] + count
            self.adj_count[i] = 0

    @ti.kernel
    def _fill_adjacency(self):
        """Fill adjacency list."""
        for c in range(self.n_cells):
            v = ti.Vector([self.cells[c][0], self.cells[c][1],
                          self.cells[c][2], self.cells[c][3]])
            for i in ti.static(range(4)):
                vi = v[i]
                for j in ti.static(range(4)):
                    if i != j:
                        vj = v[j]
                        slot = ti.atomic_add(self.adj_count[vi], 1)
                        if slot < self.max_neighbors:
                            offset = self.adj_offset[vi] + slot
                            self.adj_list[offset] = vj

    def build_adjacency(self, cells_np: np.ndarray):
        """Build adjacency list from cells."""
        self.load_cells(cells_np)
        self._count_neighbors()
        self._compute_adj_offset()
        self._fill_adjacency()

    def _get_adjacency_for_metis(self) -> List[np.ndarray]:
        """Convert adjacency to format for pymetis."""
        adj_count_np = self.adj_count.to_numpy()
        adj_offset_np = self.adj_offset.to_numpy()
        adj_list_np = self.adj_list.to_numpy()

        adjacency_list = []
        for i in range(self.n_verts):
            start = adj_offset_np[i]
            end = start + min(adj_count_np[i], self.max_neighbors)
            neighbors = np.unique(adj_list_np[start:end])
            adjacency_list.append(neighbors.astype(np.int32))

        return adjacency_list

    def _greedy_patch_partition(self, adjacency_list: List[np.ndarray]) -> np.ndarray:
        """
        Greedy patch partitioning (MeshTaichi-style).

        Minimizes ribbon elements by greedily selecting most connected vertices.
        Falls back to simple sequential assignment for sparse graphs.
        """
        n = self.n_verts
        patch_assignment = np.full(n, -1, dtype=np.int32)

        # For small meshes or sparse graphs, use simple sequential assignment
        avg_connectivity = np.mean([len(adj) for adj in adjacency_list])
        if n < self.patch_size * 2 or avg_connectivity < 2:
            # Simple sequential assignment: each patch gets patch_size vertices
            for v in range(n):
                patch_assignment[v] = v // self.patch_size
            return patch_assignment

        assigned = np.zeros(n, dtype=bool)
        patch_id = 0

        # Build connectivity scores
        connectivity = np.array([len(adj) for adj in adjacency_list])

        while not assigned.all():
            # Start new patch from highest connectivity unassigned vertex
            unassigned_mask = ~assigned
            unassigned_conn = connectivity * unassigned_mask
            seed = np.argmax(unassigned_conn)

            if assigned[seed]:
                # All done
                break

            # Grow patch greedily
            patch_verts = []
            candidates = {seed}
            patch_set = set()  # For fast lookup

            while len(patch_verts) < self.patch_size and candidates:
                # Score candidates by connection to current patch
                best_score = -1
                best_vert = -1

                for v in candidates:
                    if assigned[v]:
                        continue
                    # Count connections to patch
                    score = sum(1 for neighbor in adjacency_list[v] if neighbor in patch_set)
                    # Bonus for high connectivity
                    score += len(adjacency_list[v]) * 0.01
                    if score > best_score:
                        best_score = score
                        best_vert = v

                if best_vert == -1:
                    # No valid candidate from neighbors, find any unassigned
                    for v in range(n):
                        if not assigned[v]:
                            best_vert = v
                            break
                    if best_vert == -1:
                        break

                # Add to patch
                patch_verts.append(best_vert)
                patch_set.add(best_vert)
                patch_assignment[best_vert] = patch_id
                assigned[best_vert] = True
                candidates.discard(best_vert)

                # Add neighbors to candidates
                for neighbor in adjacency_list[best_vert]:
                    if not assigned[neighbor]:
                        candidates.add(neighbor)

            patch_id += 1

        # Handle any remaining vertices
        for v in range(n):
            if patch_assignment[v] == -1:
                patch_assignment[v] = max(0, patch_id - 1)  # Add to last patch

        return patch_assignment

    def _metis_partition_within_patch(self, patch_verts: np.ndarray,
                                       adjacency_list: List[np.ndarray],
                                       n_blocks: int) -> np.ndarray:
        """
        METIS partition within a single patch.

        Args:
            patch_verts: Global vertex IDs in this patch
            adjacency_list: Full mesh adjacency
            n_blocks: Number of blocks to create

        Returns:
            local_block: Local block assignment for each vertex in patch
        """
        n_local = len(patch_verts)

        if n_blocks <= 1 or n_local <= BANKSIZE:
            # Single block or too small
            return np.zeros(n_local, dtype=np.int32)

        # Build local adjacency (only within patch)
        global_to_local = {v: i for i, v in enumerate(patch_verts)}
        local_adj = []

        for i, v in enumerate(patch_verts):
            neighbors = []
            for n in adjacency_list[v]:
                if n in global_to_local:
                    neighbors.append(global_to_local[n])
            local_adj.append(np.array(neighbors, dtype=np.int32))

        # Use METIS for partitioning
        if check_pymetis_available():
            try:
                import pymetis
                _, membership = pymetis.part_graph(n_blocks, adjacency=local_adj)
                return np.array(membership, dtype=np.int32)
            except Exception as e:
                print(f"[HUP] METIS failed for patch: {e}")

        # Fallback: sequential assignment
        return np.arange(n_local, dtype=np.int32) % n_blocks

    def build_hierarchical_partition(self, cells_np: np.ndarray) -> dict:
        """
        Build complete hierarchical partition.

        Steps:
        1. Build adjacency graph
        2. Greedy patch partition (minimize boundary)
        3. METIS partition within each patch (maximize block connectivity)
        4. Compute global sort index and mappings

        Returns:
            Dictionary with partition results
        """
        print("[HUP] Building adjacency graph...")
        self.build_adjacency(cells_np)
        adjacency_list = self._get_adjacency_for_metis()

        print("[HUP] Phase 1: Greedy patch partitioning...")
        patch_assignment = self._greedy_patch_partition(adjacency_list)

        # Count actual patches created
        actual_n_patches = int(np.max(patch_assignment)) + 1
        self.n_patches = actual_n_patches
        self.n_blocks = actual_n_patches * self.blocks_per_patch

        print(f"[HUP] Created {actual_n_patches} patches")

        print("[HUP] Phase 2: METIS partition within patches...")
        block_assignment = np.zeros(self.n_verts, dtype=np.int32)

        for patch_id in range(actual_n_patches):
            patch_verts = np.where(patch_assignment == patch_id)[0]
            n_local = len(patch_verts)
            n_blocks_in_patch = max(1, (n_local + BANKSIZE - 1) // BANKSIZE)
            n_blocks_in_patch = min(n_blocks_in_patch, self.blocks_per_patch)

            local_blocks = self._metis_partition_within_patch(
                patch_verts, adjacency_list, n_blocks_in_patch
            )

            # Map to global block IDs
            for i, v in enumerate(patch_verts):
                global_block = patch_id * self.blocks_per_patch + local_blocks[i]
                block_assignment[v] = global_block

        print("[HUP] Phase 3: Computing sort index...")
        # Sort by block assignment
        sort_index = np.argsort(block_assignment)
        old_to_new = np.zeros(self.n_verts, dtype=np.int32)
        for new_idx, old_idx in enumerate(sort_index):
            old_to_new[old_idx] = new_idx

        # Upload to GPU
        self.patch_assignment.from_numpy(patch_assignment)
        self.block_assignment.from_numpy(block_assignment)
        self.sort_index.from_numpy(sort_index)
        self.old_to_new.from_numpy(old_to_new)

        print("[HUP] Phase 4: Building partition mappings...")
        self._build_mappings_cpu(sort_index, block_assignment[sort_index])
        self._reorder_cells()
        self._identify_boundary_vertices()

        # Compute statistics
        sorted_blocks = block_assignment[sort_index]
        block_sizes = np.bincount(sorted_blocks)

        # Compute ribbon ratio per patch
        ribbon_counts = []
        for patch_id in range(actual_n_patches):
            patch_verts = np.where(patch_assignment == patch_id)[0]
            ribbon_count = 0
            for v in patch_verts:
                for n in adjacency_list[v]:
                    if patch_assignment[n] != patch_id:
                        ribbon_count += 1
                        break
            ribbon_counts.append(ribbon_count)

        avg_ribbon_ratio = np.mean([r / len(np.where(patch_assignment == p)[0])
                                    for p, r in enumerate(ribbon_counts)])

        stats = {
            'n_vertices': self.n_verts,
            'n_patches': actual_n_patches,
            'n_blocks': len(block_sizes),
            'avg_patch_size': self.n_verts / actual_n_patches,
            'max_block_size': int(np.max(block_sizes)) if len(block_sizes) > 0 else 0,
            'min_block_size': int(np.min(block_sizes)) if len(block_sizes) > 0 else 0,
            'avg_block_size': float(np.mean(block_sizes)) if len(block_sizes) > 0 else 0,
            'avg_ribbon_ratio': avg_ribbon_ratio,
        }

        print(f"[HUP] Partition complete:")
        print(f"  - Patches: {stats['n_patches']}")
        print(f"  - Blocks: {stats['n_blocks']}")
        print(f"  - Avg block size: {stats['avg_block_size']:.1f}")
        print(f"  - Avg ribbon ratio: {stats['avg_ribbon_ratio']:.2%}")

        return {
            'sort_index': sort_index,
            'old_to_new': old_to_new,
            'patch_assignment': patch_assignment,
            'block_assignment': block_assignment[sort_index],
            'n_patches': actual_n_patches,
            'n_blocks': len(block_sizes),
            'stats': stats,
        }

    def _build_mappings_cpu(self, sort_index: np.ndarray, sorted_blocks: np.ndarray):
        """Build partition mappings on CPU."""
        # Use max_blocks for allocation (worst case)
        partId_map_real = np.full(self.max_blocks * BANKSIZE, -1, dtype=np.int32)
        real_map_partId = np.zeros(self.n_verts, dtype=np.int32)
        local_index = np.zeros(self.max_blocks, dtype=np.int32)

        for sorted_idx in range(self.n_verts):
            block_id = sorted_blocks[sorted_idx]
            if block_id >= self.max_blocks:
                continue
            local_idx = local_index[block_id]

            if local_idx < BANKSIZE:
                partId_map_real[block_id * BANKSIZE + local_idx] = sorted_idx
                real_map_partId[sorted_idx] = block_id * BANKSIZE + local_idx
                local_index[block_id] += 1

        self.partId_map_real.from_numpy(partId_map_real)
        self.real_map_partId.from_numpy(real_map_partId)

    @ti.kernel
    def _reorder_cells(self):
        """Reorder cell vertex indices."""
        for c in range(self.n_cells):
            for i in ti.static(range(4)):
                old_id = self.cells[c][i]
                new_id = self.old_to_new[old_id]
                self.sorted_cells[c][i] = new_id

    @ti.kernel
    def _identify_boundary_vertices(self):
        """Identify vertices on patch boundaries."""
        for patch_id in range(self.n_patches):
            self.patch_boundary_count[patch_id] = 0

        for i in range(self.n_verts):
            self.is_boundary_vertex[i] = 0

        for i in range(self.n_verts):
            my_patch = self.patch_assignment[i]
            num_neighbors = self.adj_count[i]
            start = self.adj_offset[i]

            is_boundary = 0
            for j in range(num_neighbors):
                neighbor = self.adj_list[start + j]
                if self.patch_assignment[neighbor] != my_patch:
                    is_boundary = 1
                    break

            self.is_boundary_vertex[i] = is_boundary
            if is_boundary:
                ti.atomic_add(self.patch_boundary_count[my_patch], 1)

    # ========================================================================
    # Prologue-Epilogue Data Transfer (MeshTaichi-style)
    # ========================================================================

    @ti.kernel
    def prologue_reorder_to_patches(self, x: ti.template(), grad: ti.template()):
        """
        Prologue: Reorder vertex data from original to patch-aligned order.

        This is called before MAS preconditioner apply().
        """
        for new_idx in range(self.n_verts):
            old_idx = self.sort_index[new_idx]
            self.reordered_x[new_idx] = x[old_idx]
            self.reordered_grad[new_idx] = grad[old_idx]

    @ti.kernel
    def epilogue_reorder_from_patches(self, z: ti.template()):
        """
        Epilogue: Reorder solution from patch-aligned to original order.

        This is called after MAS preconditioner solve.
        """
        for new_idx in range(self.n_verts):
            old_idx = self.sort_index[new_idx]
            z[old_idx] = self.reordered_z[new_idx]

    @ti.kernel
    def clear_reordered_z(self):
        """Clear reordered z buffer."""
        for i in range(self.n_verts):
            self.reordered_z[i] = ti.Vector.zero(ti.f64, 3)

    # ========================================================================
    # Patch-Parallel Accessors
    # ========================================================================

    @ti.func
    def get_patch_offset(self, patch_id: ti.i32) -> ti.i32:
        """Get vertex offset for a patch in reordered array."""
        return patch_id * self.patch_size

    @ti.func
    def get_block_in_patch(self, patch_id: ti.i32, local_block: ti.i32) -> ti.i32:
        """Get global block ID from patch and local block."""
        return patch_id * self.blocks_per_patch + local_block

    @ti.func
    def get_vertex_in_block(self, block_id: ti.i32, lane_id: ti.i32) -> ti.i32:
        """Get sorted vertex index from block and lane."""
        return self.partId_map_real[block_id * BANKSIZE + lane_id]


def hierarchical_partition_mesh(n_verts: int, cells: np.ndarray,
                                 patch_size: int = PATCH_SIZE) -> dict:
    """
    Perform hierarchical unified partition for MAS-MeshTaichi integration.

    This is the main entry point for HUP.

    Args:
        n_verts: Number of vertices
        cells: Cell array of shape (n_cells, 4)
        patch_size: Size of each patch (default: 2048)

    Returns:
        Dictionary with partition results and GPU structure
    """
    n_cells = cells.shape[0]

    # Create GPU structure
    hup = HierarchicalPartitionGPU(n_verts, n_cells, patch_size)

    # Build partition
    result = hup.build_hierarchical_partition(cells)

    # Add GPU structure to result
    result['gpu'] = hup

    return result
