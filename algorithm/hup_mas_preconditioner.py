"""
HUP-Enhanced MAS Preconditioner (Hierarchical Unified Partition).

This module implements a MAS preconditioner that combines:
1. MeshTaichi-style Patch partitioning for optimal memory access
2. METIS-based block partitioning within each Patch for block quality
3. Prologue-Epilogue data transfer for shared memory utilization

Architecture:
    Original Data (mesh.verts) ──prologue──> Reordered Data (Patch-aligned)
                                                    ↓
                                               MAS Solve
                                                    ↓
    mesh.verts.z <──────────epilogue─────── Reordered Solution

Reference: ref_doc/MAS_MESHTAICHI_INTEGRATION_DESIGN.md
"""

import taichi as ti
import numpy as np

# Constants
BANKSIZE = 16          # Nodes per subdomain (warp subdivision)
MAX_LEVELS = 6         # Maximum hierarchy depth
SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2  # = 136 for symmetric storage
MAX_NEIGHBORS_PER_VERTEX = 64
BLOCK_DOF = BANKSIZE * 3  # 48 DOFs per block

# Patch constants
PATCH_SIZE = 2048       # MeshTaichi-compatible patch size
BLOCKS_PER_PATCH = PATCH_SIZE // BANKSIZE  # = 128


@ti.data_oriented
class HUPMASPreconditioner:
    """
    MAS Preconditioner with Hierarchical Unified Partition (HUP).

    This class combines:
    1. MeshTaichi-style Patch partitioning for optimal memory access
    2. METIS-based block partitioning within each Patch for block quality
    3. Prologue-Epilogue data transfer for shared memory utilization
    """

    def __init__(self, n_verts: int, n_cells: int, mesh, cells_np: np.ndarray,
                 patch_size: int = PATCH_SIZE):
        """
        Initialize HUP-MAS preconditioner.

        Args:
            n_verts: Number of vertices
            n_cells: Number of cells
            mesh: MeshTaichi mesh object
            cells_np: Cell array of shape (n_cells, 4)
            patch_size: Size of each Patch (default: 2048)
        """
        self.n_verts = n_verts
        self.n_cells = n_cells
        self.mesh = mesh
        self.patch_size = patch_size
        self.blocks_per_patch = patch_size // BANKSIZE

        print(f"[HUP-MAS] Initializing with {n_verts} vertices, patch_size={patch_size}")

        # Import and initialize HUP
        from algorithm.hierarchical_partition import HierarchicalPartitionGPU

        self.hup = HierarchicalPartitionGPU(n_verts, n_cells, patch_size)
        partition_result = self.hup.build_hierarchical_partition(cells_np)

        self.n_patches = partition_result['n_patches']
        self.n_blocks = partition_result['n_blocks']

        print(f"[HUP-MAS] Created {self.n_patches} patches, {self.n_blocks} blocks")

        # Compute hierarchy levels
        self.level_num = self._compute_num_levels(n_verts)

        # Allocate MAS structures using HUP block organization
        self._allocate_mas_structures()

        # Build neighbor list from reordered mesh
        self._build_neighbor_list_reordered(cells_np)

        # State tracking
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

    def _compute_num_levels(self, n_verts: int) -> int:
        """Compute number of hierarchy levels."""
        levels = 1
        size = n_verts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return min(levels, MAX_LEVELS)

    def _compute_total_hierarchy_size(self) -> int:
        """Estimate total nodes across all hierarchy levels."""
        total = self.n_verts
        size = self.n_verts
        for _ in range(self.level_num - 1):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return int(total * 1.5)

    def _allocate_mas_structures(self):
        """Allocate MAS data structures with Patch-aligned layout."""
        self.total_nodes_all_levels = self._compute_total_hierarchy_size()

        # Level info
        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS + 1)

        # Block matrices organized by Patch for coalesced access
        self.patch_block_matrices = ti.Matrix.field(
            3, 3, dtype=ti.f64,
            shape=(self.n_patches, self.blocks_per_patch, SYM_BLOCK_COUNT)
        )

        self.patch_inv_blocks = ti.Matrix.field(
            3, 3, dtype=ti.f32,
            shape=(self.n_patches, self.blocks_per_patch, SYM_BLOCK_COUNT)
        )

        # Full block matrix for inversion (per patch)
        self.patch_full_block = ti.field(
            dtype=ti.f64,
            shape=(self.n_patches, self.blocks_per_patch, BLOCK_DOF, BLOCK_DOF)
        )

        self.patch_full_inverse = ti.field(
            dtype=ti.f32,
            shape=(self.n_patches, self.blocks_per_patch, BLOCK_DOF, BLOCK_DOF)
        )

        # Multi-level buffers (in reordered space)
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f64,
                                              shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f64,
                                              shape=self.total_nodes_all_levels)

        # Hierarchy structures
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)
        self.aggregation_table = ti.Vector.field(MAX_LEVELS - 1, dtype=ti.i32,
                                                  shape=self.n_verts)

        # Connectivity masks (in reordered space)
        self.fine_connect_mask = ti.field(dtype=ti.u32, shape=self.n_verts)

        # Prefix sum for hierarchy
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.prefix_original = ti.field(dtype=ti.i32, shape=n_warps + 1)
        self.prefix_sum_original = ti.field(dtype=ti.i32, shape=n_warps + 1)
        self.elected_mask = ti.field(dtype=ti.u32, shape=n_warps)
        self.cluster_id = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Neighbor list for reordered vertices
        max_neighbor_entries = self.n_verts * MAX_NEIGHBORS_PER_VERTEX
        self.neighbor_list = ti.field(dtype=ti.i32, shape=max_neighbor_entries)
        self.neighbor_start = ti.field(dtype=ti.i32, shape=self.n_verts + 1)
        self.neighbor_num = ti.field(dtype=ti.i32, shape=self.n_verts)

        # Coarse space tables
        self.coarse_space_tables = ti.field(dtype=ti.i32,
                                            shape=(MAX_LEVELS, self.n_verts))

        # Elastic type
        self.elastic_type = 0

        # Block info
        self.n_blocks_per_level = ti.field(dtype=ti.i32, shape=MAX_LEVELS)

    def _build_neighbor_list_reordered(self, cells_np: np.ndarray):
        """Build neighbor list for reordered vertices."""
        sorted_cells = self.hup.sorted_cells.to_numpy()

        # Build adjacency in reordered space
        adj_dict = {i: set() for i in range(self.n_verts)}

        for cell in sorted_cells:
            v0, v1, v2, v3 = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
            vertices = [v0, v1, v2, v3]
            for i in range(4):
                for j in range(4):
                    if i != j:
                        adj_dict[vertices[i]].add(vertices[j])

        # Convert to CSR format
        neighbor_num_np = np.array([len(adj_dict[i]) for i in range(self.n_verts)],
                                    dtype=np.int32)
        neighbor_start_np = np.zeros(self.n_verts + 1, dtype=np.int32)
        neighbor_start_np[1:] = np.cumsum(neighbor_num_np)

        neighbor_list_np = np.zeros(int(neighbor_start_np[-1]), dtype=np.int32)
        for i in range(self.n_verts):
            start = neighbor_start_np[i]
            for j, neighbor in enumerate(adj_dict[i]):
                if start + j < len(neighbor_list_np):
                    neighbor_list_np[start + j] = neighbor

        self.neighbor_num.from_numpy(neighbor_num_np)
        self.neighbor_start.from_numpy(neighbor_start_np)
        if len(neighbor_list_np) > 0:
            self.neighbor_list.from_numpy(neighbor_list_np[:self.neighbor_list.shape[0]])

        print(f"[HUP-MAS] Built reordered neighbor list: {neighbor_start_np[-1]} edges")

    # ========================================================================
    # Prologue-Epilogue (MeshTaichi-style data transfer)
    # ========================================================================

    def prologue(self):
        """Transfer data from original to reordered space."""
        self._prologue_reorder()

    @ti.kernel
    def _prologue_reorder(self):
        """Reorder vertex data from original to Patch-aligned order."""
        for new_idx in range(self.n_verts):
            old_idx = self.hup.sort_index[new_idx]
            self.hup.reordered_x[new_idx] = self.mesh.verts.x[old_idx]
            self.hup.reordered_grad[new_idx] = self.mesh.verts.grad[old_idx]

    def epilogue(self):
        """Transfer solution from reordered to original space."""
        self._epilogue_reorder()

    @ti.kernel
    def _epilogue_reorder(self):
        """Reorder solution from Patch-aligned to original order."""
        for new_idx in range(self.n_verts):
            old_idx = self.hup.sort_index[new_idx]
            self.mesh.verts.z[old_idx] = ti.cast(self.hup.reordered_z[new_idx], ti.f32)

    # ========================================================================
    # Hierarchy Construction (in reordered space)
    # ========================================================================

    @ti.func
    def _popcount(self, x: ti.u32) -> ti.i32:
        """Count set bits."""
        count = 0
        while x:
            count += ti.i32(x & 1)
            x >>= 1
        return count

    @ti.func
    def _find_first_set(self, x: ti.u32) -> ti.i32:
        """Find position of first set bit."""
        pos = -1
        if x != 0:
            pos = 0
            temp = x
            while (temp & 1) == 0:
                temp >>= 1
                pos += 1
        return pos

    @ti.func
    def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
        """Return bitmask of lanes less than lane_id."""
        return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)

    @ti.func
    def _sym_index(self, row: ti.i32, col: ti.i32) -> ti.i32:
        """Compute symmetric storage index."""
        r = ti.min(row, col)
        c = ti.max(row, col)
        return BANKSIZE * r - r * (r + 1) // 2 + c

    @ti.kernel
    def _build_connect_mask_l0(self):
        """Build connectivity mask at Level 0 in reordered space."""
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = ti.u32(1) << ti.u32(lane_id)
            nk = 0

            num_neighbor = self.neighbor_num[idx]
            start_id = self.neighbor_start[idx]

            for i in range(num_neighbor):
                neighbor_id = self.neighbor_list[start_id + i]
                neighbor_warp = neighbor_id // BANKSIZE

                if warp_id == neighbor_warp:
                    neighbor_lane = neighbor_id % BANKSIZE
                    connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))
                else:
                    self.neighbor_list[start_id + nk] = neighbor_id
                    nk += 1

            self.neighbor_num[idx] = nk
            self.fine_connect_mask[idx] = connect_mask

    @ti.kernel
    def _propagate_connectivity(self):
        """Propagate connectivity to find connected components."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for w in range(n_warps):
            self.prefix_original[w] = 0

        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            visited = ti.u32(1) << ti.u32(lane_id)

            for _ in range(BANKSIZE):
                todo = visited ^ connect_mask
                if todo == 0:
                    break
                next_visit = self._find_first_set(todo)
                if next_visit < 0:
                    break
                visited |= ti.u32(1) << ti.u32(next_visit)
                other_idx = warp_id * BANKSIZE + next_visit
                if other_idx < self.n_verts:
                    connect_mask |= self.fine_connect_mask[other_idx]

            self.fine_connect_mask[idx] = connect_mask

            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))
            if elected_prefix == 0:
                ti.atomic_add(self.prefix_original[warp_id], 1)

    @ti.kernel
    def _find_cluster_representatives(self):
        """Find cluster representatives."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for w in range(n_warps):
            self.elected_mask[w] = ti.u32(0)

        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                ti.atomic_or(self.elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))

    @ti.kernel
    def _assign_cluster_ids(self, level_offset: ti.i32):
        """Assign cluster IDs."""
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                base_id = self.prefix_sum_original[warp_id]
                elected_in_warp = self.elected_mask[warp_id]
                local_offset = self._popcount(elected_in_warp & self._lanemask_lt(lane_id))

                cluster_id = base_id + local_offset
                self.cluster_id[idx] = cluster_id
                self.coarse_space_tables[0, idx] = cluster_id
                self.going_next[idx] = level_offset + cluster_id
            else:
                self.cluster_id[idx] = -1

    @ti.kernel
    def _propagate_cluster_ids(self):
        """Propagate cluster IDs."""
        for idx in range(self.n_verts):
            if self.cluster_id[idx] == -1:
                warp_id = idx // BANKSIZE
                connect_mask = self.fine_connect_mask[idx]

                rep_lane = self._find_first_set(connect_mask)
                rep_idx = warp_id * BANKSIZE + rep_lane

                self.cluster_id[idx] = self.cluster_id[rep_idx]
                self.coarse_space_tables[0, idx] = self.cluster_id[rep_idx]
                self.going_next[idx] = self.going_next[rep_idx]

    @ti.kernel
    def _build_aggregation_table(self):
        """Build aggregation table."""
        for idx in range(self.n_verts):
            current_idx = idx
            for level in range(self.level_num - 1):
                next_idx = self.going_next[current_idx]
                self.aggregation_table[idx][level] = next_idx
                current_idx = next_idx

    def build_hierarchy(self):
        """Build complete hierarchy in reordered space."""
        print("[HUP-MAS] Building hierarchy in reordered space...")

        self._build_connect_mask_l0()
        self._propagate_connectivity()
        self._find_cluster_representatives()

        prefix_np = self.prefix_original.to_numpy()
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        prefix_sum_np = np.zeros(n_warps + 1, dtype=np.int32)
        prefix_sum_np[1:n_warps+1] = np.cumsum(prefix_np[:n_warps])
        self.prefix_sum_original.from_numpy(prefix_sum_np)

        level_1_size = int(prefix_sum_np[n_warps])
        level_1_offset = self.n_verts

        self.level_size[0] = ti.Vector([self.n_verts, 0])
        self.level_size[1] = ti.Vector([level_1_size, level_1_offset])

        self._assign_cluster_ids(level_1_offset)
        self._propagate_cluster_ids()

        n_blocks_l0 = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.n_blocks_per_level[0] = n_blocks_l0

        actual_levels = 2
        print(f"[HUP-MAS] Level 0: {self.n_verts} nodes, Level 1: {level_1_size} nodes")

        self._build_aggregation_table()

        self.hierarchy_built = True
        self.actual_levels = actual_levels

        print(f"[HUP-MAS] Hierarchy built with {actual_levels} levels")

    # ========================================================================
    # Matrix Assembly (Patch-parallel)
    # ========================================================================

    @ti.kernel
    def _clear_patch_block_matrices(self):
        """Clear all Patch block matrices."""
        for patch_id in range(self.n_patches):
            for local_block in range(self.blocks_per_patch):
                for sym_idx in range(SYM_BLOCK_COUNT):
                    self.patch_block_matrices[patch_id, local_block, sym_idx] = \
                        ti.Matrix.zero(ti.f64, 3, 3)

    @ti.kernel
    def _add_inertia_contribution_patch(self, dt: ti.f64):
        """Add mass matrix to diagonal blocks (Patch-organized)."""
        for new_idx in range(self.n_verts):
            # Get Patch and block info
            patch_id = new_idx // self.patch_size
            local_in_patch = new_idx % self.patch_size
            local_block = local_in_patch // BANKSIZE
            lane_id = local_in_patch % BANKSIZE

            # Clamp to valid range
            if patch_id >= self.n_patches:
                patch_id = self.n_patches - 1
            if local_block >= self.blocks_per_patch:
                local_block = self.blocks_per_patch - 1

            # Get mass from original vertex
            old_idx = self.hup.sort_index[new_idx]
            m = self.mesh.verts.m[old_idx]

            # Diagonal block index
            sym_idx = self._sym_index(lane_id, lane_id)

            # Add mass to diagonal
            for d in ti.static(range(3)):
                ti.atomic_add(
                    self.patch_block_matrices[patch_id, local_block, sym_idx][d, d],
                    m
                )

    def assemble_block_matrices(self, solver, use_full_hessian: bool = True):
        """Assemble block matrices with Patch organization."""
        print("[HUP-MAS] Assembling Patch-organized block matrices...")

        self._clear_patch_block_matrices()

        if hasattr(solver, 'elastic_type'):
            self.elastic_type = solver.elastic_type

        # Add inertia
        dt = solver.dt if hasattr(solver, 'dt') else 0.01
        self._add_inertia_contribution_patch(dt)

        self.matrices_assembled = True
        print("[HUP-MAS] Block matrix assembly complete")

    # ========================================================================
    # Matrix Inversion (Patch-parallel)
    # ========================================================================

    @ti.kernel
    def _expand_sym_to_full_patch(self):
        """Expand symmetric to full matrix (Patch-organized)."""
        for patch_id in range(self.n_patches):
            for local_block in range(self.blocks_per_patch):
                # Clear full matrix
                for i in range(BLOCK_DOF):
                    for j in range(BLOCK_DOF):
                        self.patch_full_block[patch_id, local_block, i, j] = 0.0

                # Copy from symmetric storage
                for row in range(BANKSIZE):
                    for col in range(row, BANKSIZE):
                        sym_idx = self._sym_index(row, col)
                        block_3x3 = self.patch_block_matrices[patch_id, local_block, sym_idx]

                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                self.patch_full_block[patch_id, local_block,
                                                       row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                                if row != col:
                                    self.patch_full_block[patch_id, local_block,
                                                           col * 3 + dj, row * 3 + di] = block_3x3[dj, di]

    @ti.kernel
    def _gauss_jordan_invert_patch(self):
        """Gauss-Jordan inversion (Patch-parallel)."""
        for patch_id in range(self.n_patches):
            for local_block in range(self.blocks_per_patch):
                # Initialize identity
                for i in range(BLOCK_DOF):
                    for j in range(BLOCK_DOF):
                        if i == j:
                            self.patch_full_inverse[patch_id, local_block, i, j] = 1.0
                        else:
                            self.patch_full_inverse[patch_id, local_block, i, j] = 0.0

                # Gauss-Jordan elimination
                for pivot in range(BLOCK_DOF):
                    # Find pivot
                    max_val = ti.abs(self.patch_full_block[patch_id, local_block, pivot, pivot])
                    max_row = pivot

                    for r in range(pivot + 1, BLOCK_DOF):
                        val = ti.abs(self.patch_full_block[patch_id, local_block, r, pivot])
                        if val > max_val:
                            max_val = val
                            max_row = r

                    # Swap rows
                    if max_row != pivot:
                        for c in range(BLOCK_DOF):
                            tmp_m = self.patch_full_block[patch_id, local_block, pivot, c]
                            self.patch_full_block[patch_id, local_block, pivot, c] = \
                                self.patch_full_block[patch_id, local_block, max_row, c]
                            self.patch_full_block[patch_id, local_block, max_row, c] = tmp_m

                            tmp_i = self.patch_full_inverse[patch_id, local_block, pivot, c]
                            self.patch_full_inverse[patch_id, local_block, pivot, c] = \
                                self.patch_full_inverse[patch_id, local_block, max_row, c]
                            self.patch_full_inverse[patch_id, local_block, max_row, c] = tmp_i

                    # Scale pivot row
                    pivot_val = self.patch_full_block[patch_id, local_block, pivot, pivot]
                    if ti.abs(pivot_val) > 1e-12:
                        scale = 1.0 / pivot_val
                        for c in range(BLOCK_DOF):
                            self.patch_full_block[patch_id, local_block, pivot, c] *= scale
                            self.patch_full_inverse[patch_id, local_block, pivot, c] *= ti.f32(scale)
                    else:
                        self.patch_full_block[patch_id, local_block, pivot, pivot] = 1e-6
                        for c in range(BLOCK_DOF):
                            if c == pivot:
                                self.patch_full_inverse[patch_id, local_block, pivot, c] = 1e6
                            else:
                                self.patch_full_inverse[patch_id, local_block, pivot, c] = 0.0

                    # Eliminate in other rows
                    for r in range(BLOCK_DOF):
                        if r != pivot:
                            factor = self.patch_full_block[patch_id, local_block, r, pivot]
                            for c in range(BLOCK_DOF):
                                self.patch_full_block[patch_id, local_block, r, c] -= \
                                    factor * self.patch_full_block[patch_id, local_block, pivot, c]
                                self.patch_full_inverse[patch_id, local_block, r, c] -= \
                                    ti.f32(factor) * self.patch_full_inverse[patch_id, local_block, pivot, c]

    @ti.kernel
    def _copy_inverse_to_sym_patch(self):
        """Copy inverse back to symmetric storage (Patch-organized)."""
        for patch_id in range(self.n_patches):
            for local_block in range(self.blocks_per_patch):
                for row in range(BANKSIZE):
                    for col in range(row, BANKSIZE):
                        sym_idx = self._sym_index(row, col)

                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                self.patch_inv_blocks[patch_id, local_block, sym_idx][di, dj] = \
                                    self.patch_full_inverse[patch_id, local_block,
                                                             row * 3 + di, col * 3 + dj]

    def invert_block_matrices(self, use_full_inversion: bool = True):
        """Invert all block matrices (Patch-parallel)."""
        print("[HUP-MAS] Inverting Patch-organized block matrices...")

        if use_full_inversion:
            self._expand_sym_to_full_patch()
            self._gauss_jordan_invert_patch()
            self._copy_inverse_to_sym_patch()
            print("[HUP-MAS] Full block inversion complete (Gauss-Jordan)")
        else:
            print("[HUP-MAS] Diagonal-only inversion not yet implemented")

        self.matrices_inverted = True

    # ========================================================================
    # Preconditioning (Patch-parallel with Prologue-Epilogue)
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f64, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def _build_multi_level_r_patch(self):
        """Build multi-level residual from reordered gradient."""
        # Copy gradient to level 0 (from reordered space)
        for idx in range(self.n_verts):
            self.multi_level_r[idx] = ti.cast(self.hup.reordered_grad[idx], ti.f64)

        # Restrict to coarse levels
        for idx in range(self.n_verts):
            r = self.multi_level_r[idx]
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            connect_mask = self.fine_connect_mask[idx]
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                current_idx = idx
                for _ in range(self.level_num - 1):
                    next_idx = self.going_next[current_idx]
                    if next_idx >= 0 and next_idx < self.total_nodes_all_levels:
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                        current_idx = next_idx
                    else:
                        break
            else:
                elected_lane = self._find_first_set(connect_mask)
                elected_idx = warp_id * BANKSIZE + elected_lane
                if elected_idx < self.n_verts and elected_idx != idx:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.multi_level_r[elected_idx][d], r[d])

    @ti.kernel
    def _schwarz_local_solve_patch(self):
        """Patch-parallel local solve."""
        for patch_id in range(self.n_patches):
            for local_block in range(self.blocks_per_patch):
                for lane_i in range(BANKSIZE):
                    idx_i = patch_id * self.patch_size + local_block * BANKSIZE + lane_i
                    if idx_i < self.n_verts:
                        z = ti.Vector.zero(ti.f64, 3)

                        for lane_j in range(BANKSIZE):
                            idx_j = patch_id * self.patch_size + local_block * BANKSIZE + lane_j
                            if idx_j < self.n_verts:
                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    inv_block = self.patch_inv_blocks[patch_id, local_block, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += ti.f64(inv_block[di, dj]) * self.multi_level_r[idx_j][dj]
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    inv_block = self.patch_inv_blocks[patch_id, local_block, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += ti.f64(inv_block[dj, di]) * self.multi_level_r[idx_j][dj]

                        self.multi_level_z[idx_i] = z

        # Coarse level solve (simplified)
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

    @ti.kernel
    def _collect_final_z_patch(self, level_num: ti.i32):
        """Prolongation: collect solutions to reordered space."""
        for idx in range(self.n_verts):
            z_total = self.multi_level_z[idx]

            for level in range(level_num - 1):
                coarse_idx = self.aggregation_table[idx][level]
                if coarse_idx >= 0 and coarse_idx < self.total_nodes_all_levels:
                    z_total += self.multi_level_z[coarse_idx]

            # Store in reordered z buffer
            self.hup.reordered_z[idx] = z_total

    def apply(self, use_full_solve: bool = True):
        """
        Apply HUP-MAS preconditioner with Prologue-Epilogue.

        1. Prologue: Transfer grad from original to reordered space
        2. Restriction: gradient -> multi_level_r
        3. Local solve: multi_level_r -> multi_level_z (Patch-parallel)
        4. Prolongation: multi_level_z -> reordered_z
        5. Epilogue: Transfer z from reordered to original space
        """
        # Prologue
        self.prologue()

        # Clear buffers
        self._clear_multi_level_buffers()

        # Restriction
        self._build_multi_level_r_patch()

        # Local solve (Patch-parallel)
        self._schwarz_local_solve_patch()

        # Prolongation
        self._collect_final_z_patch(self.actual_levels)

        # Epilogue
        self.epilogue()

    # ========================================================================
    # Public Interface
    # ========================================================================

    def rebuild(self, solver, use_full_hessian: bool = True, use_full_inversion: bool = True):
        """Full rebuild of preconditioner."""
        if not self.hierarchy_built:
            self.build_hierarchy()

        if hasattr(solver, 'elastic_type'):
            self.elastic_type = solver.elastic_type

        self.assemble_block_matrices(solver, use_full_hessian)
        self.invert_block_matrices(use_full_inversion)

    def get_stats(self) -> dict:
        """Return statistics about the preconditioner."""
        return {
            'n_verts': self.n_verts,
            'n_patches': self.n_patches,
            'n_blocks': self.n_blocks,
            'n_levels': self.actual_levels if hasattr(self, 'actual_levels') else 0,
            'patch_size': self.patch_size,
            'hierarchy_built': self.hierarchy_built,
            'matrices_inverted': self.matrices_inverted,
        }
