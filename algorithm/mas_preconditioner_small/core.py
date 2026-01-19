"""
MAS Preconditioner Small - Core implementation.

Simplified single-file implementation for fast compilation.
"""

import taichi as ti
import numpy as np

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
    Simplified MAS Preconditioner with minimal features for fast compilation.

    Only includes:
    - ARAP elastic Hessian
    - IC(0) inversion
    - Banded local solve
    """

    def __init__(self, mesh, max_verts: int = None):
        """
        Initialize MAS Preconditioner.

        Args:
            mesh: MeshTaichi mesh object
            max_verts: Maximum number of vertices (default: mesh.verts.size)
        """
        self.mesh = mesh
        self.n_verts = mesh.verts.size
        self.n_cells = mesh.cells.size

        if max_verts is None:
            max_verts = self.n_verts

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
        self.full_block_matrix = ti.field(dtype=ti.f64,
                                          shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f32,
                                           shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))

        # Hierarchy mapping
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)

        # Multi-level buffers
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)

        # State flags
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

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
    def _build_going_next(self):
        """Build going_next mapping for hierarchy."""
        # Level 0: fine vertices map to coarse level
        for i in range(self.n_verts):
            coarse_idx = self.level_size[1][1] + i // BANKSIZE
            self.going_next[i] = coarse_idx

        # Higher levels
        for level in range(1, self.level_num - 1):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            next_offset = self.level_size[level + 1][1]

            for i in range(level_size):
                idx = level_offset + i
                coarse_idx = next_offset + i // BANKSIZE
                self.going_next[idx] = coarse_idx

        # Last level: map to -1
        if self.level_num > 1:
            last_offset = self.level_size[self.level_num - 1][1]
            last_size = self.level_size[self.level_num - 1][0]
            for i in range(last_size):
                self.going_next[last_offset + i] = -1

    def build_hierarchy(self):
        """Build the multi-level hierarchy."""
        self._build_going_next()
        self.hierarchy_built = True
        print(f"[MAS-Small] Hierarchy built: {self.level_num} levels")

    # ========================================================================
    # Matrix Assembly (ARAP only)
    # ========================================================================

    @ti.kernel
    def _clear_block_matrices(self):
        """Zero out all block matrices."""
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f32):
        """Add mass matrix to diagonal blocks."""
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            m = self.mesh.verts.m[idx]
            sym_idx = sym_index(lane_id, lane_id)
            mass_val = m / (dt * dt)
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
                        # Cross-warp: propagate to coarse level
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

    def assemble_block_matrices(self, solver):
        """Assemble Hessian contributions into block matrices."""
        self._clear_block_matrices()
        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True

    # ========================================================================
    # Block Inversion (IC(0) only)
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
                    sum_val = ti.f64(self.full_block_inverse[block_id, i, col])
                    k_end = ti.min(BLOCK_DOF, i + BANDWIDTH + 1)
                    for k in range(i + 1, k_end):
                        if k - i <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, k, i] * \
                                       ti.f64(self.full_block_inverse[block_id, k, col])
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
    # Apply Preconditioner (Banded solve only)
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        """Build multi-level residual from gradient."""
        # Level 0: copy from mesh gradient
        for i in range(self.n_verts):
            self.multi_level_r[i] = ti.cast(self.mesh.verts.grad[i], ti.f32)

        # Coarse levels: aggregate from fine
        for level in range(1, self.level_num):
            prev_offset = self.level_size[level - 1][1]
            prev_size = self.level_size[level - 1][0]
            curr_offset = self.level_size[level][1]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                curr_idx = curr_offset + i // BANKSIZE
                r = self.multi_level_r[prev_idx]
                for d in ti.static(range(3)):
                    ti.atomic_add(self.multi_level_r[curr_idx][d], r[d])

    @ti.kernel
    def _schwarz_local_solve_banded(self):
        """Banded sparse matrix-vector multiplication for IC(0)."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        NODE_BANDWIDTH = 2

        # Level 0
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

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

        # Coarse levels
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size:
                    z0 = ti.f32(0.0)
                    z1 = ti.f32(0.0)
                    z2 = ti.f32(0.0)

                    lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
                    lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

                    for lane_j in range(lane_j_start, lane_j_end):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
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
    def _collect_final_z(self):
        """Collect z from all levels to fine level (prolongation)."""
        # Start with level 0 contribution
        for i in range(self.n_verts):
            self.mesh.verts.z[i] = ti.cast(self.multi_level_z[i], ti.f64)

        # Add coarse level contributions
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            prev_offset = self.level_size[level - 1][1]
            prev_size = self.level_size[level - 1][0]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                coarse_idx = level_offset + i // BANKSIZE

                # Trace back to fine level
                fine_idx = prev_idx
                if level > 1:
                    # Need to trace through hierarchy
                    for lv in range(level - 1, 0, -1):
                        lv_offset = self.level_size[lv][1]
                        fine_idx = (fine_idx - lv_offset) * BANKSIZE + self.level_size[lv - 1][1]

                if fine_idx < self.n_verts:
                    z_coarse = self.multi_level_z[coarse_idx]
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.mesh.verts.z[fine_idx][d], ti.f64(z_coarse[d]))

    def apply(self):
        """Apply MAS preconditioner: z = P * grad"""
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._schwarz_local_solve_banded()
        self._collect_final_z()

    # ========================================================================
    # High-level API
    # ========================================================================

    def rebuild(self, solver):
        """Full rebuild of preconditioner."""
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_block_matrices(solver)
        self.invert_block_matrices()
