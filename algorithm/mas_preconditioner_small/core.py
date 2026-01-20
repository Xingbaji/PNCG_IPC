"""
MAS Preconditioner Small - Core implementation.

Simplified single-file implementation with METIS support.
METIS reordering is computed ONCE at initialization and reused.
"""

import taichi as ti

# Constants
BANKSIZE = 16
SYM_BLOCK_COUNT = 136  # BANKSIZE * (BANKSIZE + 1) // 2
BLOCK_DOF = BANKSIZE * 3  # 48
MAX_LEVELS = 6

# Import ARAP Hessian computation
from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter


@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """Compute symmetric storage index for upper triangle."""
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


@ti.data_oriented
class MASPreconditionerSmall:
    """
    Simplified MAS Preconditioner with METIS support.

    Key features:
    - METIS reordering computed ONCE at initialization
    - ARAP elastic Hessian
    - IC(0) inversion
    - Banded local solve
    - Correct prolongation using going_next hierarchy
    """

    def __init__(self, mesh, metis_reordered: bool = True, max_verts: int = None):
        """
        Initialize MAS Preconditioner.

        Args:
            mesh: MeshTaichi mesh object (must be pre-reordered with METIS)
            metis_reordered: Must be True. Mesh data must be reordered with
                            reorder_mesh_data_metis() BEFORE creating mesh.
                            - Vertex IDs directly correspond to METIS partitions
                            - block_id = vertex_id // BANKSIZE
                            - No runtime mapping lookups needed (fastest mode)
            max_verts: Maximum number of vertices (default: mesh.verts.size)

        Note: Non-METIS and runtime mapping modes have been removed for compile optimization.
              Use util/model_loading.py which applies METIS reordering automatically.
        """
        if not metis_reordered:
            raise ValueError("MASPreconditionerSmall requires metis_reordered=True. "
                           "Use util/model_loading.py which applies METIS reordering automatically.")

        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        if max_verts is None:
            max_verts = self.n_verts

        # METIS pre-reordered mode only
        self.metis_reordered = True
        self.use_metis = True

        # Pre-reordered mode: vertex IDs are already in METIS partition order
        # block_id = vertex_id // BANKSIZE, lane_id = vertex_id % BANKSIZE
        self.n_parts = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        print(f"[MAS-Small] METIS pre-reordered mode: {self.n_parts} partitions")

        # Compute hierarchy sizes
        self.level_num = min(MAX_LEVELS, self._compute_level_num(self.n_verts))
        self.total_nodes_all_levels = self._compute_total_nodes(self.n_verts, self.level_num)
        self.total_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        # Level info: [size, offset] for each level
        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS)
        self._init_level_sizes()

        # Block matrices (symmetric storage)
        self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                              shape=(self.total_blocks, SYM_BLOCK_COUNT))
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                   shape=(self.total_blocks, SYM_BLOCK_COUNT))

        # Full block matrices for inversion
        self.full_block_matrix = ti.field(dtype=ti.f32,
                                          shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f32,
                                           shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))

        # Hierarchy mapping
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)

        # Multi-level buffers
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)

        # Pre-reordered mode: no runtime mapping needed
        # Vertex IDs directly map to partitions: block_id = vertex_id // BANKSIZE
        self.partId_map_real = None
        self.real_map_partId = None

        # State flags
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

        # ====================================================================
        # Cross-block coupling storage (triplet format for exact Hessian matvec)
        # ====================================================================
        # Estimate max cross-block entries: each tet can have up to 6 cross-block pairs
        # (4 choose 2 = 6), each pair is a 3x3 block
        max_cross_block_entries = self.n_cells * 6  # Upper bound

        # Triplet storage: (row_vertex, col_vertex, 3x3 matrix)
        self.cross_block_row = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_col = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_val = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_cross_block_entries)
        self.cross_block_count = ti.field(dtype=ti.i32, shape=())  # Atomic counter
        self.max_cross_block_entries = max_cross_block_entries
        self.has_cross_block_data = False

        print(f"[MAS-Small] Initialized: {self.n_verts} verts, {self.level_num} levels, "
              f"{self.total_blocks} blocks")

    def _compute_level_num(self, n_verts: int) -> int:
        """Compute number of hierarchy levels."""
        levels = 1
        size = n_verts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return levels

    def _compute_total_nodes(self, n_verts: int, level_num: int) -> int:
        """Compute total nodes across all levels."""
        total = n_verts
        size = n_verts
        for _ in range(level_num - 1):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return total

    def _init_level_sizes(self):
        """Initialize level size and offset arrays."""
        sizes = []
        offsets = []

        size = self.n_verts
        offset = 0
        for level in range(self.level_num):
            sizes.append(size)
            offsets.append(offset)
            offset += size
            size = (size + BANKSIZE - 1) // BANKSIZE

        for i in range(self.level_num):
            self.level_size[i] = ti.Vector([sizes[i], offsets[i]])
        for i in range(self.level_num, MAX_LEVELS):
            self.level_size[i] = ti.Vector([0, offset])

    # ========================================================================
    # Hierarchy Building
    # ========================================================================

    @ti.kernel
    def _build_going_next(self, level_num: ti.i32):
        """
        Build going_next mapping for METIS pre-reordered mesh.

        When mesh is pre-reordered, vertex IDs directly correspond to partitions:
        - partition_id = vertex_id // BANKSIZE
        No mapping lookup needed!
        """
        if level_num == 1:
            for i in range(self.n_verts):
                self.going_next[i] = -1
        else:
            # Level 0: vertices map to coarse based on position (direct calculation)
            for i in range(self.n_verts):
                # In pre-reordered mode, partition = vertex_id // BANKSIZE
                part_id = i // BANKSIZE
                # Coarse node index = level_1_offset + partition_id
                coarse_idx = self.level_size[1][1] + part_id
                self.going_next[i] = coarse_idx

            # Higher levels: sequential mapping (same as non-METIS)
            for level in range(1, level_num - 1):
                level_offset = self.level_size[level][1]
                level_size_val = self.level_size[level][0]
                next_offset = self.level_size[level + 1][1]

                for i in range(level_size_val):
                    idx = level_offset + i
                    coarse_idx = next_offset + i // BANKSIZE
                    self.going_next[idx] = coarse_idx

            # Last level: map to -1
            last_offset = self.level_size[level_num - 1][1]
            last_size = self.level_size[level_num - 1][0]
            for i in range(last_size):
                self.going_next[last_offset + i] = -1

    def build_hierarchy(self):
        """Build the multi-level hierarchy for METIS pre-reordered mesh."""
        # Recompute level sizes for METIS
        # Level 1 size = number of METIS partitions
        level_1_size = self.n_parts
        level_1_offset = self.n_verts

        sizes = [self.n_verts, level_1_size]
        offsets = [0, level_1_offset]

        # Compute higher levels
        size = level_1_size
        offset = level_1_offset + size
        for _ in range(2, self.level_num):
            size = (size + BANKSIZE - 1) // BANKSIZE
            sizes.append(size)
            offsets.append(offset)
            offset += size

        # Update level_size field
        for i in range(len(sizes)):
            self.level_size[i] = ti.Vector([sizes[i], offsets[i]])

        # Pre-reordered mode: direct partition calculation
        self._build_going_next(self.level_num)
        print(f"[MAS-Small] Hierarchy built: {self.level_num} levels, "
              f"L0={self.n_verts}, L1={level_1_size}")

        self.hierarchy_built = True

    # ========================================================================
    # Matrix Assembly
    # ========================================================================

    @ti.kernel
    def _clear_block_matrices(self):
        """Zero out all block matrices."""
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _clear_cross_block_storage(self):
        """Clear cross-block triplet storage counter."""
        self.cross_block_count[None] = 0

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f32):
        """Add mass matrix to diagonal blocks using MeshTaichi iterator.

        Note: The inertia Hessian is just 'm' (not m/dt²) to match the gradient scaling.
        Energy: E_inertia = 0.5 * m * ||x - x_hat||²
        Gradient: g_inertia = m * (x - x_hat)
        Hessian: H_inertia = m * I
        """
        for vert in self.mesh.verts:
            idx = vert.id
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Access mass directly from vertex
            m = vert.m
            sym_idx = sym_index(lane_id, lane_id)
            # Use m directly (not m/dt²) to match gradient scaling
            mass_val = m
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], mass_val)

    @ti.kernel
    def _add_elastic_contribution_arap(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """Add ARAP elastic Hessian contribution."""
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt

            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                                  v2 // BANKSIZE, v3 // BANKSIZE])
            lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                                  v2 % BANKSIZE, v3 % BANKSIZE])

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Compute element Hessian
            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Assemble to block matrices
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):
                    warp_i = warp_ids[i]
                    warp_j = warp_ids[j]
                    lane_i = lane_ids[i]
                    lane_j = lane_ids[j]

                    if warp_i == warp_j:
                        # Same warp: direct assembly
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        if lane_i <= lane_j:
                            s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, s_idx][di, dj],
                                                  sub_block[di, dj])
                        else:
                            s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, s_idx][di, dj],
                                                  sub_block[dj, di])
                    else:
                        # Cross-warp: propagate to coarse level AND store in triplet format
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # Store cross-block entry in triplet format for exact Hessian matvec
                        # Use original vertex IDs (not coarse), store upper triangle (i < j)
                        orig_i = v_ids[i]
                        orig_j = v_ids[j]
                        triplet_idx = ti.atomic_add(self.cross_block_count[None], 1)
                        if triplet_idx < self.max_cross_block_entries:
                            if orig_i <= orig_j:
                                self.cross_block_row[triplet_idx] = orig_i
                                self.cross_block_col[triplet_idx] = orig_j
                                self.cross_block_val[triplet_idx] = sub_block
                            else:
                                # Store as (j, i) with transposed block
                                self.cross_block_row[triplet_idx] = orig_j
                                self.cross_block_col[triplet_idx] = orig_i
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        self.cross_block_val[triplet_idx][di, dj] = sub_block[dj, di]

                        # Also propagate to coarse levels for preconditioning
                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                coarse_lane_i = vert_i % BANKSIZE
                                coarse_lane_j = vert_j % BANKSIZE

                                if coarse_lane_i <= coarse_lane_j:
                                    s_idx = BANKSIZE * coarse_lane_i - coarse_lane_i * (coarse_lane_i + 1) // 2 + coarse_lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                          sub_block[di, dj])
                                            if coarse_lane_i == coarse_lane_j:
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                              sub_block[dj, di])
                                else:
                                    s_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                          sub_block[dj, di])
                                break

    @ti.kernel
    def _aggregate_fine_to_coarse(self):
        """Aggregate fine-level block entries to coarse levels."""
        n_fine_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_fine_blocks):
            for lane_row in range(BANKSIZE):
                for lane_col in range(lane_row, BANKSIZE):
                    row_idx = block_id * BANKSIZE + lane_row
                    col_idx = block_id * BANKSIZE + lane_col

                    if row_idx >= self.n_verts or col_idx >= self.n_verts:
                        continue

                    s_idx = BANKSIZE * lane_row - lane_row * (lane_row + 1) // 2 + lane_col
                    mat3 = self.block_matrices[block_id, s_idx]

                    mat_norm = ti.f32(0.0)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            mat_norm += ti.abs(mat3[di, dj])
                    if mat_norm < 1e-12:
                        continue

                    rdx = row_idx
                    cdx = col_idx

                    for level in range(self.level_num - 1):
                        rdx = self.going_next[rdx]
                        cdx = self.going_next[cdx]

                        if rdx < 0 or cdx < 0:
                            break

                        coarse_block_r = rdx // BANKSIZE
                        coarse_block_c = cdx // BANKSIZE

                        if coarse_block_r == coarse_block_c:
                            coarse_lane_r = rdx % BANKSIZE
                            coarse_lane_c = cdx % BANKSIZE

                            if coarse_lane_r <= coarse_lane_c:
                                coarse_s_idx = BANKSIZE * coarse_lane_r - coarse_lane_r * (coarse_lane_r + 1) // 2 + coarse_lane_c
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_r, coarse_s_idx][di, dj], mat3[di, dj])
                                        if coarse_lane_r == coarse_lane_c and row_idx != col_idx:
                                            ti.atomic_add(self.block_matrices[coarse_block_r, coarse_s_idx][di, dj], mat3[dj, di])
                            else:
                                coarse_s_idx = BANKSIZE * coarse_lane_c - coarse_lane_c * (coarse_lane_c + 1) // 2 + coarse_lane_r
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_c, coarse_s_idx][di, dj], mat3[dj, di])

    @ti.kernel
    def _add_elastic_contribution_arap_optimized_full(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """
        Optimized ARAP elastic Hessian contribution using sorted cells.

        Key optimizations:
        1. Cells are sorted by main partition (reduces atomic conflicts)
        2. METIS-reordered vertex IDs allow direct block/lane computation
        3. B and W are pre-extracted and sorted, avoiding mesh cell access

        This kernel uses pre-sorted data for maximum efficiency.
        """
        for sorted_idx in range(self.n_cells):
            # Access pre-sorted W and B
            W = self.sorted_W[sorted_idx]
            para = W * dt * dt

            # Get pre-computed METIS-reordered vertex IDs
            cell = self.sorted_cells[sorted_idx]
            new_v0, new_v1, new_v2, new_v3 = cell[0], cell[1], cell[2], cell[3]

            # Block and lane computed directly from METIS-ordered IDs
            # No need for real_map_partId lookup!
            block_ids = ti.Vector([new_v0 // BANKSIZE, new_v1 // BANKSIZE,
                                   new_v2 // BANKSIZE, new_v3 // BANKSIZE])
            lane_ids = ti.Vector([new_v0 % BANKSIZE, new_v1 % BANKSIZE,
                                  new_v2 % BANKSIZE, new_v3 % BANKSIZE])
            v_ids = ti.Vector([new_v0, new_v1, new_v2, new_v3])

            # Access pre-sorted B matrix and original vertex IDs for position access
            B = self.sorted_B[sorted_idx]
            orig_cell = self.sorted_orig_cells[sorted_idx]

            # Get vertex positions using original vertex IDs
            x0 = self.mesh.verts.x[orig_cell[0]]
            x1 = self.mesh.verts.x[orig_cell[1]]
            x2 = self.mesh.verts.x[orig_cell[2]]
            x3 = self.mesh.verts.x[orig_cell[3]]

            # Compute deformation gradient
            Ds = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
            F = Ds @ B

            # Compute element Hessian
            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Assemble to block matrices
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):
                    block_i = block_ids[i]
                    block_j = block_ids[j]
                    lane_i = lane_ids[i]
                    lane_j = lane_ids[j]

                    if block_i == block_j:
                        # Same METIS partition: direct assembly
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        if lane_i <= lane_j:
                            s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                  sub_block[di, dj])
                        else:
                            s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                  sub_block[dj, di])
                    else:
                        # Cross-partition: propagate to coarse level
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                coarse_lane_i = vert_i % BANKSIZE
                                coarse_lane_j = vert_j % BANKSIZE

                                if coarse_lane_i <= coarse_lane_j:
                                    s_idx = BANKSIZE * coarse_lane_i - coarse_lane_i * (coarse_lane_i + 1) // 2 + coarse_lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                          sub_block[di, dj])
                                            if coarse_lane_i == coarse_lane_j:
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                              sub_block[dj, di])
                                else:
                                    s_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj],
                                                          sub_block[dj, di])
                                break

    def assemble_block_matrices(self, solver):
        """Assemble Hessian contributions into block matrices.

        Also stores cross-block coupling in triplet format for exact Hessian matvec.
        Uses METIS pre-reordered mode where vertex IDs directly map to partitions.
        """
        self._clear_block_matrices()
        self._clear_cross_block_storage()  # Clear triplet storage

        # Pre-reordered mode: vertex IDs are already in METIS partition order
        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True  # Cross-block triplets are now available
        self.triplets_sorted = False  # Reset sorted flag when triplets change

    # ========================================================================
    # Block Inversion (IC(0))
    # ========================================================================

    @ti.kernel
    def _expand_sym_to_full(self):
        """Expand symmetric block matrices to full 48x48 dense matrices."""
        n_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_matrix[block_id, i, j] = 0.0

            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    s_idx = sym_index(row, col)
                    block_3x3 = self.block_matrices[block_id, s_idx]

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.full_block_matrix[block_id, row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                            if row != col:
                                self.full_block_matrix[block_id, col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    @ti.kernel
    def _incomplete_cholesky_invert_blocks(self):
        """IC(0) factorization and approximate inversion."""
        n_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE
        BANDWIDTH = 6

        for block_id in range(n_blocks):
            # IC(0) Factorization
            for i in range(BLOCK_DOF):
                j_start = ti.max(0, i - BANDWIDTH)

                for j in range(j_start, i):
                    if i - j <= BANDWIDTH:
                        sum_val = self.full_block_matrix[block_id, i, j]
                        k_start = ti.max(0, ti.max(i - BANDWIDTH, j - BANDWIDTH))
                        for k in range(k_start, j):
                            if i - k <= BANDWIDTH and j - k <= BANDWIDTH:
                                sum_val -= self.full_block_matrix[block_id, i, k] * \
                                           self.full_block_matrix[block_id, j, k]
                        L_jj = self.full_block_matrix[block_id, j, j]
                        if ti.abs(L_jj) > 1e-12:
                            self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                        else:
                            self.full_block_matrix[block_id, i, j] = 0.0
                    else:
                        self.full_block_matrix[block_id, i, j] = 0.0

                sum_val = self.full_block_matrix[block_id, i, i]
                k_start = ti.max(0, i - BANDWIDTH)
                for k in range(k_start, i):
                    if i - k <= BANDWIDTH:
                        sum_val -= self.full_block_matrix[block_id, i, k] ** 2
                if sum_val > 1e-12:
                    self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                else:
                    self.full_block_matrix[block_id, i, i] = 1e-3

            # Approximate inversion
            for col in range(BLOCK_DOF):
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    k_start = ti.max(0, i - BANDWIDTH)
                    for k in range(k_start, i):
                        if i - k <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, i, k] * \
                                       self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f32(self.full_block_inverse[block_id, i, col])
                    k_end = ti.min(BLOCK_DOF, i + BANDWIDTH + 1)
                    for k in range(i + 1, k_end):
                        if k - i <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, k, i] * \
                                       ti.f32(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _copy_inverse_to_sym(self):
        """Copy inverted full matrix back to symmetric storage."""
        n_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    s_idx = sym_index(row, col)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, s_idx][di, dj] = \
                                self.full_block_inverse[block_id, row * 3 + di, col * 3 + dj]

    def invert_block_matrices(self):
        """Invert all block matrices using IC(0)."""
        self._expand_sym_to_full()
        self._incomplete_cholesky_invert_blocks()
        self._copy_inverse_to_sym()
        self.matrices_inverted = True

    # ========================================================================
    # Apply Preconditioner
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        """Build multi-level residual from gradient (non-METIS)."""
        # Level 0: copy from mesh gradient using MeshTaichi iterator
        for vert in self.mesh.verts:
            idx = vert.id
            self.multi_level_r[idx] = ti.cast(vert.grad, ti.f32)

        # Coarse levels: aggregate from fine via going_next
        for vert in self.mesh.verts:
            idx = vert.id
            r = self.multi_level_r[idx]
            coarse_idx = self.going_next[idx]
            if coarse_idx >= 0:
                for d in ti.static(range(3)):
                    ti.atomic_add(self.multi_level_r[coarse_idx][d], r[d])

        # Propagate to higher coarse levels
        for level in range(1, self.level_num - 1):
            level_offset = self.level_size[level][1]
            level_size_val = self.level_size[level][0]

            for i in range(level_size_val):
                idx = level_offset + i
                r = self.multi_level_r[idx]
                coarse_idx = self.going_next[idx]
                if coarse_idx >= 0:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.multi_level_r[coarse_idx][d], r[d])

    @ti.kernel
    def _schwarz_local_solve_full(self):
        """
        Full block matrix-vector multiplication: z = P^{-1} @ r.

        This implements the CORRECT local Schwarz solve using the FULL 16x16 block
        inverse matrix (stored as 136 3x3 blocks in symmetric format).

        Reference: Stiff-GIPC _schwarzLocalXSym6 uses hessianSize = BANKSIZE * BANKSIZE
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Full block matvec
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

                # Iterate over ALL nodes in the block (full 16x16 block)
                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]

                        # Compute symmetric storage index
                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        s_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, s_idx]

                        if lane_i <= lane_j:
                            # Upper triangle: use directly
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                            z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                            z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                        else:
                            # Lower triangle: use transpose
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                            z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                            z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

        # Coarse levels: Full block matvec
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size_val = self.level_size[level][0]
            n_coarse_blocks = (level_size_val + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size_val:
                    z0 = ti.f32(0.0)
                    z1 = ti.f32(0.0)
                    z2 = ti.f32(0.0)

                    # Full block iteration
                    for lane_j in range(BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size_val:
                            r_j = self.multi_level_r[idx_j]

                            min_lane = ti.min(lane_i, lane_j)
                            max_lane = ti.max(lane_i, lane_j)
                            s_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, s_idx]

                            if lane_i <= lane_j:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                                z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                                z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                            else:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                                z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                                z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

    @ti.kernel
    def _schwarz_local_solve_banded(self):
        """Banded solve for METIS pre-reordered mesh.

        Since mesh is pre-reordered, vertex IDs directly correspond to partitions:
        - block_id = vertex_id // BANKSIZE
        - lane_id = vertex_id % BANKSIZE

        No mapping lookup needed, but can use banded approximation since
        vertices in the same block are topologically adjacent (METIS property).
        """
        NODE_BANDWIDTH = 2
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Banded block matvec (direct indexing)
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

                # Banded iteration: only ±NODE_BANDWIDTH neighbors
                lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
                lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

                for lane_j in range(lane_j_start, lane_j_end):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]

                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        s_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, s_idx]

                        if lane_i <= lane_j:
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                            z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                            z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                        else:
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                            z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                            z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

        # Coarse levels: Banded block matvec
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size_val = self.level_size[level][0]
            n_coarse_blocks = (level_size_val + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size_val:
                    z0 = ti.f32(0.0)
                    z1 = ti.f32(0.0)
                    z2 = ti.f32(0.0)

                    lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
                    lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

                    for lane_j in range(lane_j_start, lane_j_end):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size_val:
                            r_j = self.multi_level_r[idx_j]

                            min_lane = ti.min(lane_i, lane_j)
                            max_lane = ti.max(lane_i, lane_j)
                            s_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, s_idx]

                            if lane_i <= lane_j:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                                z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                                z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]
                            else:
                                z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                                z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                                z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)

    @ti.kernel
    def _collect_final_z(self, level_num: ti.i32):
        """
        Collect z from all levels (prolongation).

        CORRECT implementation: For each fine vertex, traverse the going_next
        chain to collect contributions from all coarse levels.

        z_i = z_i^{(0)} + z_{parent(i)}^{(1)} + z_{grandparent(i)}^{(2)} + ...
        """
        # Use MeshTaichi iterator for natural mesh traversal
        for vert in self.mesh.verts:
            idx = vert.id
            # Start with level 0 contribution
            z_total = ti.cast(self.multi_level_z[idx], ti.f32)

            # Traverse hierarchy to collect coarse contributions
            coarse_idx = self.going_next[idx]
            for _ in range(1, level_num):
                if coarse_idx >= 0:
                    z_coarse = self.multi_level_z[coarse_idx]
                    z_total += ti.cast(z_coarse, ti.f32)
                    coarse_idx = self.going_next[coarse_idx]
                else:
                    break

            # Write directly to mesh vertex field
            vert.z = z_total

    def apply(self, use_multilevel: bool = True):
        """Apply MAS preconditioner: z = P * grad

        Uses banded solve for METIS pre-reordered mesh where vertex IDs
        directly correspond to partitions, enabling fast direct indexing
        with topological locality (METIS property).

        Args:
            use_multilevel: If True (default), use all levels for preconditioning.
                           If False, only use level 0 (block-diagonal only).
                           Set to False for scenes with multiple disconnected objects
                           to avoid incorrect cross-object coupling in coarse levels.
        """
        self._clear_multi_level_buffers()

        # Pre-reordered mode: vertex IDs are already in METIS partition order
        # Use banded solve (METIS locality) + no mapping lookup needed
        self._build_multi_level_r()
        self._schwarz_local_solve_banded()

        if use_multilevel:
            self._collect_final_z(self.level_num)
        else:
            # Only use level 0 (no coarse level contribution)
            self._collect_final_z(1)

    # ========================================================================
    # Block-Diagonal Hessian Matrix-Vector Multiplication
    # ========================================================================
    #
    # IMPORTANT NOTE ON hessian_matvec:
    # The MAS preconditioner's multi-level block matrices are designed for
    # PRECONDITIONING, not for exact Hessian matvec.
    #
    # - Level 0 stores only INTRA-BLOCK coupling (entries within same 16-node block)
    # - Cross-block coupling is stored in coarse levels, but the storage format
    #   is designed for additive Schwarz preconditioning, NOT for exact matvec.
    #
    # The reference implementation (Stiff-GIPC) uses a SEPARATE triplet-format
    # sparse matrix for exact H @ v (see spmv.cu), not the MAS block matrices.
    #
    # The hessian_matvec function below attempts to reconstruct H @ v from the
    # multi-level structure, but this is an APPROXIMATION, not exact.
    # For exact H @ v, use the original triplet-format sparse Hessian.
    # ========================================================================

    @ti.kernel
    def _hessian_matvec_level0_block_diag(self, v: ti.template(), result: ti.template()):
        """
        Compute block-diagonal part of Hessian @ v for level 0 only.

        This only computes the block-diagonal contribution (intra-block coupling).
        Cross-block coupling is stored in coarse levels.
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                r0 = ti.f32(0.0)
                r1 = ti.f32(0.0)
                r2 = ti.f32(0.0)

                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        v_j = v[idx_j]

                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        s_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        H_block = self.block_matrices[block_id, s_idx]

                        if lane_i <= lane_j:
                            r0 += H_block[0, 0] * v_j[0] + H_block[0, 1] * v_j[1] + H_block[0, 2] * v_j[2]
                            r1 += H_block[1, 0] * v_j[0] + H_block[1, 1] * v_j[1] + H_block[1, 2] * v_j[2]
                            r2 += H_block[2, 0] * v_j[0] + H_block[2, 1] * v_j[1] + H_block[2, 2] * v_j[2]
                        else:
                            r0 += H_block[0, 0] * v_j[0] + H_block[1, 0] * v_j[1] + H_block[2, 0] * v_j[2]
                            r1 += H_block[0, 1] * v_j[0] + H_block[1, 1] * v_j[1] + H_block[2, 1] * v_j[2]
                            r2 += H_block[0, 2] * v_j[0] + H_block[1, 2] * v_j[1] + H_block[2, 2] * v_j[2]

                result[idx_i] = ti.Vector([r0, r1, r2], dt=ti.f32)

    def hessian_matvec(self, v: ti.template(), result: ti.template()):
        """
        Compute result = H @ v EXACTLY using block-diagonal + cross-block triplets.

        This function computes the EXACT Hessian matrix-vector product by:
        1. Level 0 block-diagonal contribution (intra-block coupling)
        2. Cross-block coupling from triplet storage

        Args:
            v: Input vector field with 3D vectors (indexed by vertex id)
            result: Output vector field with 3D vectors (indexed by vertex id)

        Example usage:
            v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
            result = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
            preconditioner.hessian_matvec(v, result)
        """
        self.hessian_matvec_exact(v, result)

    @ti.kernel
    def _copy_z_to_buffer(self, z_buffer: ti.template()):
        """Copy mesh.verts.z to a buffer field."""
        for vert in self.mesh.verts:
            z_buffer[vert.id] = vert.z

    @ti.kernel
    def _copy_buffer_to_grad(self, result_buffer: ti.template()):
        """Copy result buffer to mesh.verts.grad."""
        for vert in self.mesh.verts:
            vert.grad = result_buffer[vert.id]

    def hessian_matvec_mesh(self, z_buffer: ti.template(), result_buffer: ti.template()):
        """
        Compute H @ z where z comes from mesh.verts.z, result goes to mesh.verts.grad.

        This is a convenience wrapper that:
        1. Copies mesh.verts.z to z_buffer
        2. Computes H @ z_buffer -> result_buffer
        3. Copies result_buffer to mesh.verts.grad

        Args:
            z_buffer: Temporary ti.Vector.field(3, ti.f32, shape=n_verts)
            result_buffer: Temporary ti.Vector.field(3, ti.f32, shape=n_verts)

        Example usage:
            z_buf = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
            result_buf = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
            preconditioner.hessian_matvec_mesh(z_buf, result_buf)
            # Now mesh.verts.grad contains H @ z
        """
        self._copy_z_to_buffer(z_buffer)
        self.hessian_matvec(z_buffer, result_buffer)
        self._copy_buffer_to_grad(result_buffer)

    # ========================================================================
    # EXACT Hessian Matrix-Vector Multiplication
    # ========================================================================

    @ti.kernel
    def _cross_block_spmv(self, v: ti.template(), result: ti.template(), n_triplets: ti.i32):
        """
        Compute cross-block contribution to H @ v using triplet format.

        For each triplet (row, col, H_block):
          result[row] += H_block @ v[col]
          result[col] += H_block^T @ v[row]  (symmetric matrix)

        The triplets store upper triangle entries only (row <= col).
        """
        for t in range(n_triplets):
            row = self.cross_block_row[t]
            col = self.cross_block_col[t]

            if row < 0 or col < 0:
                continue

            H_block = self.cross_block_val[t]
            v_row = v[row]
            v_col = v[col]

            # H @ v contribution: result[row] += H_block @ v[col]
            r0_row = H_block[0, 0] * v_col[0] + H_block[0, 1] * v_col[1] + H_block[0, 2] * v_col[2]
            r1_row = H_block[1, 0] * v_col[0] + H_block[1, 1] * v_col[1] + H_block[1, 2] * v_col[2]
            r2_row = H_block[2, 0] * v_col[0] + H_block[2, 1] * v_col[1] + H_block[2, 2] * v_col[2]

            ti.atomic_add(result[row][0], r0_row)
            ti.atomic_add(result[row][1], r1_row)
            ti.atomic_add(result[row][2], r2_row)

            # Symmetric contribution: result[col] += H_block^T @ v[row]
            if row != col:
                r0_col = H_block[0, 0] * v_row[0] + H_block[1, 0] * v_row[1] + H_block[2, 0] * v_row[2]
                r1_col = H_block[0, 1] * v_row[0] + H_block[1, 1] * v_row[1] + H_block[2, 1] * v_row[2]
                r2_col = H_block[0, 2] * v_row[0] + H_block[1, 2] * v_row[1] + H_block[2, 2] * v_row[2]

                ti.atomic_add(result[col][0], r0_col)
                ti.atomic_add(result[col][1], r1_col)
                ti.atomic_add(result[col][2], r2_col)

    def hessian_matvec_exact(self, v: ti.template(), result: ti.template()):
        """
        Compute result = H @ v EXACTLY using block-diagonal + cross-block triplets.

        This function computes the EXACT Hessian matrix-vector product by:
        1. Level 0 block-diagonal contribution (intra-block coupling)
        2. Cross-block coupling from triplet storage

        Args:
            v: Input vector field with 3D vectors (indexed by vertex id)
            result: Output vector field with 3D vectors (indexed by vertex id)
        """
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled. Call assemble_block_matrices first.")
        if not self.has_cross_block_data:
            raise RuntimeError("Cross-block data not available. Call assemble_block_matrices first.")

        # Step 1: Compute level 0 block-diagonal contribution
        self._hessian_matvec_level0_block_diag(v, result)

        # Step 2: Add cross-block contributions from triplet storage
        n_triplets = self.cross_block_count[None]
        if n_triplets > 0:
            self._cross_block_spmv(v, result, n_triplets)

    def hessian_matvec_exact_mesh(self, z_buffer: ti.template(), result_buffer: ti.template()):
        """
        Compute EXACT H @ z where z comes from mesh.verts.z, result goes to mesh.verts.grad.

        This is a convenience wrapper that:
        1. Copies mesh.verts.z to z_buffer
        2. Computes EXACT H @ z_buffer -> result_buffer
        3. Copies result_buffer to mesh.verts.grad

        Args:
            z_buffer: Temporary ti.Vector.field(3, ti.f32, shape=n_verts)
            result_buffer: Temporary ti.Vector.field(3, ti.f32, shape=n_verts)
        """
        self._copy_z_to_buffer(z_buffer)
        self.hessian_matvec_exact(z_buffer, result_buffer)
        self._copy_buffer_to_grad(result_buffer)

    def get_cross_block_stats(self):
        """Get statistics about cross-block coupling storage."""
        n_triplets = int(self.cross_block_count[None])
        max_entries = self.max_cross_block_entries
        usage_pct = 100.0 * n_triplets / max_entries if max_entries > 0 else 0.0
        return {
            'n_triplets': n_triplets,
            'max_entries': max_entries,
            'usage_percent': usage_pct,
            'memory_mb': n_triplets * (4 + 4 + 9 * 4) / (1024 * 1024)  # row + col + 3x3 float
        }

    # ========================================================================
    # High-level API
    # ========================================================================

    def rebuild(self, solver):
        """Full rebuild of preconditioner."""
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_block_matrices(solver)
        self.invert_block_matrices()
