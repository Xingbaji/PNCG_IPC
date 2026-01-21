"""
MAS Preconditioner Small - Float64 precision version.

This is a copy of core.py with all float32 types changed to float64.
Used for benchmarking precision differences with high stiffness materials.

Key changes from core.py:
- All ti.f32 -> ti.f64
- All np.float32 -> np.float64
- Block matrices, buffers, and triplets use f64
- Inline f64 versions of compute_dFdx and compute_d2PsidF2_ARAP_filter

Usage:
    from algorithm.mas_preconditioner_small.core_f64 import MASPreconditionerSmallF64
    precond = MASPreconditionerSmallF64(mesh)
"""

import taichi as ti
import numpy as np

# Constants
BANKSIZE = 16
SYM_BLOCK_COUNT = 136  # BANKSIZE * (BANKSIZE + 1) // 2
BLOCK_DOF = BANKSIZE * 3  # 48
MAX_LEVELS = 6


# ============================================================================
# Inline f64 versions of math utility functions
# ============================================================================

@ti.func
def ssvd_f64(F):
    """Signed SVD for f64 matrices."""
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def flatten_matrix_f64(A):
    """Flatten 3x3 A to 9x1 vector (column first), f64 version."""
    return ti.Matrix([
        A[0, 0], A[1, 0], A[2, 0],
        A[0, 1], A[1, 1], A[2, 1],
        A[0, 2], A[1, 2], A[2, 2]
    ], ti.f64)


@ti.func
def compute_dFdx_f64(DmInv):
    """Compute dFdx matrix (9x12), f64 version."""
    dFdx = ti.Matrix.zero(ti.f64, 9, 12)
    m = DmInv[0, 0]
    n = DmInv[0, 1]
    o = DmInv[0, 2]
    p = DmInv[1, 0]
    q = DmInv[1, 1]
    r = DmInv[1, 2]
    s = DmInv[2, 0]
    t = DmInv[2, 1]
    u = DmInv[2, 2]
    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    dFdx[0, 0] = t1
    dFdx[0, 3] = m
    dFdx[0, 6] = p
    dFdx[0, 9] = s

    dFdx[1, 1] = t1
    dFdx[1, 4] = m
    dFdx[1, 7] = p
    dFdx[1, 10] = s

    dFdx[2, 2] = t1
    dFdx[2, 5] = m
    dFdx[2, 8] = p
    dFdx[2, 11] = s

    dFdx[3, 0] = t2
    dFdx[3, 3] = n
    dFdx[3, 6] = q
    dFdx[3, 9] = t

    dFdx[4, 1] = t2
    dFdx[4, 4] = n
    dFdx[4, 7] = q
    dFdx[4, 10] = t

    dFdx[5, 2] = t2
    dFdx[5, 5] = n
    dFdx[5, 8] = q
    dFdx[5, 11] = t

    dFdx[6, 0] = t3
    dFdx[6, 3] = o
    dFdx[6, 6] = r
    dFdx[6, 9] = u

    dFdx[7, 1] = t3
    dFdx[7, 4] = o
    dFdx[7, 7] = r
    dFdx[7, 10] = u

    dFdx[8, 2] = t3
    dFdx[8, 5] = o
    dFdx[8, 8] = r
    dFdx[8, 11] = u

    return dFdx


@ti.func
def compute_d2PsidF2_ARAP_filter_f64(F, mu, la):
    """
    Compute ARAP Hessian with eigenvalue filtering, f64 version.
    Returns 9x9 matrix.
    """
    U, sig, V = ssvd_f64(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]

    lambda0 = ti.f64(2.0) / (s1 + s2)
    lambda1 = ti.f64(2.0) / (s0 + s2)
    lambda2 = ti.f64(2.0) / (s0 + s1)

    if s1 + s2 < ti.f64(2.0):
        lambda0 = ti.f64(1.0)
    if s0 + s2 < ti.f64(2.0):
        lambda1 = ti.f64(1.0)
    if s0 + s1 < ti.f64(2.0):
        lambda2 = ti.f64(1.0)

    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]

    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)

    q0 = flatten_matrix_f64(Q0)
    q1 = flatten_matrix_f64(Q1)
    q2 = flatten_matrix_f64(Q2)

    d2PsidF2 = -mu * (lambda0 * q0.outer_product(q0) +
                       lambda1 * q1.outer_product(q1) +
                       lambda2 * q2.outer_product(q2))

    for i in ti.static(range(9)):
        d2PsidF2[i, i] += ti.f64(2.0) * mu

    return d2PsidF2


@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """Compute symmetric storage index for upper triangle."""
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


@ti.data_oriented
class MASPreconditionerSmallF64:
    """
    MAS Preconditioner with Float64 precision.

    Identical to MASPreconditionerSmall but uses float64 for all internal
    computations. Recommended for high stiffness materials (E > 1e5).
    """

    def __init__(self, mesh, metis_reordered: bool = True, max_verts: int = None,
                 metis_n_parts: int = None):
        """
        Initialize MAS Preconditioner with float64 precision.

        Args:
            mesh: MeshTaichi mesh object (must be pre-reordered with METIS)
            metis_reordered: Must be True.
            max_verts: Maximum number of vertices (default: mesh.verts.size)
            metis_n_parts: Actual number of METIS partitions.
        """
        if not metis_reordered:
            raise ValueError("MASPreconditionerSmallF64 requires metis_reordered=True.")

        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        if max_verts is None:
            max_verts = self.n_verts

        self.metis_reordered = True
        self.use_metis = True

        if metis_n_parts is not None:
            self.n_parts = metis_n_parts
            print(f"[MAS-Small-F64] METIS pre-reordered mode: {self.n_parts} partitions (from METIS)")
        else:
            self.n_parts = (self.n_verts + BANKSIZE - 1) // BANKSIZE
            print(f"[MAS-Small-F64] METIS pre-reordered mode: {self.n_parts} partitions (computed)")

        self.level_num = min(MAX_LEVELS, self._compute_level_num_from_parts(self.n_parts))
        self.total_nodes_all_levels = self._compute_total_nodes_from_parts(
            self.n_verts, self.n_parts, self.level_num)
        self.total_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS)
        self._init_level_sizes()

        # Block matrices (symmetric storage) - FLOAT64
        self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f64,
                                              shape=(self.total_blocks, SYM_BLOCK_COUNT))
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f64,
                                                   shape=(self.total_blocks, SYM_BLOCK_COUNT))

        # Full block matrices for inversion - FLOAT64
        self.full_block_matrix = ti.field(dtype=ti.f64,
                                          shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f64,
                                           shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))

        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)

        # Multi-level buffers - FLOAT64
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f64, shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f64, shape=self.total_nodes_all_levels)

        self.partId_map_real = None
        self.real_map_partId = None
        self.sorted_to_partition = None

        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

        # Cross-block coupling storage - FLOAT64
        max_cross_block_entries = self.n_cells * 6

        self.cross_block_row = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_col = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_val = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_cross_block_entries)
        self.cross_block_count = ti.field(dtype=ti.i32, shape=())
        self.max_cross_block_entries = max_cross_block_entries
        self.has_cross_block_data = False

        print(f"[MAS-Small-F64] Initialized: {self.n_verts} verts, {self.level_num} levels, "
              f"{self.total_blocks} blocks, precision=f64")

    def _compute_level_num_from_parts(self, n_parts: int) -> int:
        levels = 2
        size = n_parts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return levels

    def _compute_total_nodes_from_parts(self, n_verts: int, n_parts: int, level_num: int) -> int:
        total = n_verts + n_parts
        size = n_parts
        for _ in range(2, level_num):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return total

    def _init_level_sizes(self):
        sizes = []
        offsets = []

        sizes.append(self.n_verts)
        offsets.append(0)

        if self.level_num >= 2:
            sizes.append(self.n_parts)
            offsets.append(self.n_verts)

        offset = self.n_verts + self.n_parts
        size = self.n_parts
        for level in range(2, self.level_num):
            size = (size + BANKSIZE - 1) // BANKSIZE
            sizes.append(size)
            offsets.append(offset)
            offset += size

        for i in range(len(sizes)):
            self.level_size[i] = ti.Vector([sizes[i], offsets[i]])
        for i in range(len(sizes), MAX_LEVELS):
            self.level_size[i] = ti.Vector([0, offset])

    def _build_going_next(self, level_num: int):
        going_next_np = np.full(self.total_nodes_all_levels, -1, dtype=np.int32)

        if level_num <= 1:
            self.going_next.from_numpy(going_next_np)
            return

        level_1_offset = int(self.level_size[1][1])

        if self.sorted_to_partition is not None:
            for i in range(self.n_verts):
                part_id = self.sorted_to_partition[i]
                coarse_idx = level_1_offset + part_id
                going_next_np[i] = coarse_idx
        else:
            for i in range(self.n_verts):
                part_id = i // BANKSIZE
                coarse_idx = level_1_offset + part_id
                going_next_np[i] = coarse_idx

        for level in range(1, level_num - 1):
            level_offset = int(self.level_size[level][1])
            level_size_val = int(self.level_size[level][0])
            next_offset = int(self.level_size[level + 1][1])

            for i in range(level_size_val):
                idx = level_offset + i
                coarse_idx = next_offset + i // BANKSIZE
                going_next_np[idx] = coarse_idx

        self.going_next.from_numpy(going_next_np)

    def build_hierarchy(self):
        if self.sorted_to_partition is not None:
            actual_n_parts = int(self.sorted_to_partition.max()) + 1
            if actual_n_parts != self.n_parts:
                print(f"[MAS-Small-F64] WARNING: n_parts mismatch")

        self._build_going_next(self.level_num)
        print(f"[MAS-Small-F64] Hierarchy built: {self.level_num} levels")
        self.hierarchy_built = True

    # ========================================================================
    # Matrix Assembly - FLOAT64 version
    # ========================================================================

    @ti.kernel
    def _clear_block_matrices(self):
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f64, 3, 3)

    @ti.kernel
    def _clear_cross_block_storage(self):
        self.cross_block_count[None] = 0

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f64):
        for vert in self.mesh.verts:
            idx = vert.id
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            m = ti.cast(vert.m, ti.f64)
            sym_idx = sym_index(lane_id, lane_id)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], m)

    @ti.kernel
    def _add_elastic_contribution_arap(self, mu: ti.f64, la: ti.f64, dt: ti.f64):
        for c in self.mesh.cells:
            W = ti.cast(c.W, ti.f64)
            para = W * dt * dt

            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                                  v2 // BANKSIZE, v3 // BANKSIZE])
            lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                                  v2 % BANKSIZE, v3 % BANKSIZE])

            # Compute deformation gradient in f64
            x0 = ti.cast(c.verts[0].x, ti.f64)
            x1 = ti.cast(c.verts[1].x, ti.f64)
            x2 = ti.cast(c.verts[2].x, ti.f64)
            x3 = ti.cast(c.verts[3].x, ti.f64)

            Ds = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
            B = ti.cast(c.B, ti.f64)
            F = Ds @ B

            # Compute element Hessian in f64
            dFdx = compute_dFdx_f64(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter_f64(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):
                    warp_i = warp_ids[i]
                    warp_j = warp_ids[j]
                    lane_i = lane_ids[i]
                    lane_j = lane_ids[j]

                    if warp_i == warp_j:
                        sub_block = ti.Matrix.zero(ti.f64, 3, 3)
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
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        sub_block = ti.Matrix.zero(ti.f64, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        orig_i = v_ids[i]
                        orig_j = v_ids[j]
                        triplet_idx = ti.atomic_add(self.cross_block_count[None], 1)
                        if triplet_idx < self.max_cross_block_entries:
                            if orig_i <= orig_j:
                                self.cross_block_row[triplet_idx] = orig_i
                                self.cross_block_col[triplet_idx] = orig_j
                                self.cross_block_val[triplet_idx] = sub_block
                            else:
                                self.cross_block_row[triplet_idx] = orig_j
                                self.cross_block_col[triplet_idx] = orig_i
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        self.cross_block_val[triplet_idx][di, dj] = sub_block[dj, di]

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

                    mat_norm = ti.f64(0.0)
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
        self._clear_block_matrices()
        self._clear_cross_block_storage()

        # Cast solver parameters to f64
        dt = float(solver.dt)
        mu = float(solver.mu)
        la = float(solver.la)

        self._add_inertia_contribution(dt)
        self._add_elastic_contribution_arap(mu, la, dt)

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True
        self.triplets_sorted = False

    # ========================================================================
    # Block Inversion (IC(0)) - FLOAT64 version
    # ========================================================================

    @ti.kernel
    def _expand_sym_to_full(self):
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
                        if ti.abs(L_jj) > 1e-15:  # Tighter tolerance for f64
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
                if sum_val > 1e-15:  # Tighter tolerance for f64
                    self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                else:
                    self.full_block_matrix[block_id, i, i] = 1e-6  # Smaller regularization for f64

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
                    if ti.abs(L_ii) > 1e-15:
                        self.full_block_inverse[block_id, i, col] = ti.f64(sum_val / L_ii)
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
                    if ti.abs(L_ii) > 1e-15:
                        self.full_block_inverse[block_id, i, col] = ti.f64(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _copy_inverse_to_sym(self):
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
        self._expand_sym_to_full()
        self._incomplete_cholesky_invert_blocks()
        self._copy_inverse_to_sym()
        self.matrices_inverted = True

    # ========================================================================
    # Apply Preconditioner - FLOAT64 version
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f64, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        # Level 0: copy from mesh gradient
        for vert in self.mesh.verts:
            idx = vert.id
            self.multi_level_r[idx] = ti.cast(vert.grad, ti.f64)

        # Coarse levels: aggregate from fine
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
    def _schwarz_local_solve_banded(self):
        NODE_BANDWIDTH = 2
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = ti.f64(0.0)
                z1 = ti.f64(0.0)
                z2 = ti.f64(0.0)

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

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f64)

        # Coarse levels
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size_val = self.level_size[level][0]
            n_coarse_blocks = (level_size_val + BANKSIZE - 1) // BANKSIZE

            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size_val:
                    z0 = ti.f64(0.0)
                    z1 = ti.f64(0.0)
                    z2 = ti.f64(0.0)

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

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f64)

    @ti.kernel
    def _collect_final_z(self, level_num: ti.i32):
        for vert in self.mesh.verts:
            idx = vert.id
            z_total = ti.cast(self.multi_level_z[idx], ti.f64)

            coarse_idx = self.going_next[idx]
            for _ in range(1, level_num):
                if coarse_idx >= 0:
                    z_coarse = self.multi_level_z[coarse_idx]
                    z_total += ti.cast(z_coarse, ti.f64)
                    coarse_idx = self.going_next[coarse_idx]
                else:
                    break

            # Write back as f32 to mesh (mesh uses f32)
            vert.z = ti.cast(z_total, ti.f32)

    def apply(self):
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._schwarz_local_solve_banded()
        self._collect_final_z(self.level_num)

    # ========================================================================
    # Hessian Matrix-Vector Multiplication - FLOAT64 version
    # ========================================================================

    @ti.kernel
    def _hessian_matvec_level0_block_diag(self, v: ti.template(), result: ti.template()):
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                r0 = ti.f64(0.0)
                r1 = ti.f64(0.0)
                r2 = ti.f64(0.0)

                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        v_j = ti.cast(v[idx_j], ti.f64)

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

                result[idx_i] = ti.cast(ti.Vector([r0, r1, r2], dt=ti.f64), ti.f32)

    @ti.kernel
    def _cross_block_spmv(self, v: ti.template(), result: ti.template(), n_triplets: ti.i32):
        for t in range(n_triplets):
            row = self.cross_block_row[t]
            col = self.cross_block_col[t]

            if row < 0 or col < 0:
                continue

            H_block = self.cross_block_val[t]
            v_row = ti.cast(v[row], ti.f64)
            v_col = ti.cast(v[col], ti.f64)

            r0_row = H_block[0, 0] * v_col[0] + H_block[0, 1] * v_col[1] + H_block[0, 2] * v_col[2]
            r1_row = H_block[1, 0] * v_col[0] + H_block[1, 1] * v_col[1] + H_block[1, 2] * v_col[2]
            r2_row = H_block[2, 0] * v_col[0] + H_block[2, 1] * v_col[1] + H_block[2, 2] * v_col[2]

            ti.atomic_add(result[row][0], ti.cast(r0_row, ti.f32))
            ti.atomic_add(result[row][1], ti.cast(r1_row, ti.f32))
            ti.atomic_add(result[row][2], ti.cast(r2_row, ti.f32))

            if row != col:
                r0_col = H_block[0, 0] * v_row[0] + H_block[1, 0] * v_row[1] + H_block[2, 0] * v_row[2]
                r1_col = H_block[0, 1] * v_row[0] + H_block[1, 1] * v_row[1] + H_block[2, 1] * v_row[2]
                r2_col = H_block[0, 2] * v_row[0] + H_block[1, 2] * v_row[1] + H_block[2, 2] * v_row[2]

                ti.atomic_add(result[col][0], ti.cast(r0_col, ti.f32))
                ti.atomic_add(result[col][1], ti.cast(r1_col, ti.f32))
                ti.atomic_add(result[col][2], ti.cast(r2_col, ti.f32))

    def hessian_matvec_exact(self, v: ti.template(), result: ti.template()):
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled.")
        if not self.has_cross_block_data:
            raise RuntimeError("Cross-block data not available.")

        self._hessian_matvec_level0_block_diag(v, result)

        n_triplets = self.cross_block_count[None]
        if n_triplets > 0:
            self._cross_block_spmv(v, result, n_triplets)

    def hessian_matvec(self, v: ti.template(), result: ti.template()):
        self.hessian_matvec_exact(v, result)

    # ========================================================================
    # High-level API
    # ========================================================================

    def rebuild(self, solver):
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_block_matrices(solver)
        self.invert_block_matrices()

    def get_cross_block_stats(self):
        n_triplets = int(self.cross_block_count[None])
        max_entries = self.max_cross_block_entries
        usage_pct = 100.0 * n_triplets / max_entries if max_entries > 0 else 0.0
        return {
            'n_triplets': n_triplets,
            'max_entries': max_entries,
            'usage_percent': usage_pct,
            'memory_mb': n_triplets * (4 + 4 + 9 * 8) / (1024 * 1024)  # f64 = 8 bytes
        }
