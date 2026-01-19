"""
Schwarz local solvers for MAS Preconditioner.

This module provides local subdomain solve operations:
- Full block solve (z = B^{-1} * r)
- Diagonal-only approximation
- Conflict-free symmetric SpMV
- Banded sparse MV for IC(0)

Reference: MASPreconditioner.cu (CUDA reference implementation)
"""

import taichi as ti

from .constants import BANKSIZE


# ==============================================================================
# SchwarzMixin Class
# ==============================================================================

class SchwarzMixin:
    """
    Mixin class providing Schwarz local solve methods for MASPreconditioner.
    """

    @ti.kernel
    def _schwarz_local_solve_conflict_free(self):
        """
        Conflict-free symmetric matrix-vector multiplication.

        Optimization over _schwarz_local_solve_full:
        1. Parallel over all (block_id, lane_i) pairs
        2. Unrolled 3x3 matrix-vector multiply
        3. Direct symmetric storage access without branching

        CUDA Reference: _schwarzLocalXSym6() (MASPreconditioner.cu lines 957-1027)
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Parallel over (block_id, lane_i)
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                r_i = self.multi_level_r[idx_i]

                z0 = ti.f32(0.0)
                z1 = ti.f32(0.0)
                z2 = ti.f32(0.0)

                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]

                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, sym_idx]

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

                    for lane_j in range(BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
                            r_j = self.multi_level_r[idx_j]

                            min_lane = ti.min(lane_i, lane_j)
                            max_lane = ti.max(lane_i, lane_j)
                            sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, sym_idx]

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
        """
        Banded sparse matrix-vector multiplication for IC(0).

        Only computes z_i = sum_j M^{-1}[i,j] * r[j] for |node_i - node_j| <= NODE_BANDWIDTH.

        P6 Optimization for use with IC(0).
        """
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
                        sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                        inv_block = self.inv_block_matrices[block_id, sym_idx]

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
                            sym_idx = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane

                            inv_block = self.inv_block_matrices[block_id, sym_idx]

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
    def _schwarz_local_solve_full(self):
        """
        Solve z_d = B_d^{-1} * r_d for each subdomain at Level 0 and coarse levels.

        CUDA Reference: _schwarzLocalXSym6() (MASPreconditioner.cu lines 957-1027)
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0
        for block_id in range(n_blocks):
            for lane_i in range(BANKSIZE):
                idx_i = block_id * BANKSIZE + lane_i
                if idx_i < self.n_verts:
                    z = ti.Vector.zero(ti.f32, 3)

                    for lane_j in range(BANKSIZE):
                        idx_j = block_id * BANKSIZE + lane_j
                        if idx_j < self.n_verts:
                            if lane_i <= lane_j:
                                sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                            else:
                                sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                    self.multi_level_z[idx_i] = z

        # Coarse levels
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                for lane_i in range(BANKSIZE):
                    idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                    if idx_i < level_offset + level_size:
                        z = ti.Vector.zero(ti.f32, 3)

                        for lane_j in range(BANKSIZE):
                            idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                            if idx_j < level_offset + level_size:
                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    inv_block = self.inv_block_matrices[block_id, sym_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                        self.multi_level_z[idx_i] = z

    @ti.kernel
    def _schwarz_local_solve_full_parallel(self):
        """
        Parallelized version of local solve: z_d = B_d^{-1} * r_d

        Key optimization: Parallelize over (block_id, lane_i).
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z = ti.Vector.zero(ti.f32, 3)

                for lane_j in range(BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                        else:
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                self.multi_level_z[idx_i] = z

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
                    z = ti.Vector.zero(ti.f32, 3)

                    for lane_j in range(BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size:
                            if lane_i <= lane_j:
                                sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[di, dj] * self.multi_level_r[idx_j][dj]
                            else:
                                sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                inv_block = self.inv_block_matrices[block_id, sym_idx]
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        z[di] += inv_block[dj, di] * self.multi_level_r[idx_j][dj]

                    self.multi_level_z[idx_i] = z

    @ti.kernel
    def _schwarz_local_solve(self):
        """
        Diagonal block solve (fallback method, faster but less accurate).
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0
        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    sym_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id
                    r = self.multi_level_r[idx]
                    inv_block = self.inv_block_matrices[block_id, sym_idx]

                    z = ti.Vector.zero(ti.f32, 3)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            z[di] += inv_block[di, dj] * r[dj]

                    self.multi_level_z[idx] = z

        # Coarse levels
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]
            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                for lane_id in range(BANKSIZE):
                    idx = level_offset + local_block_id * BANKSIZE + lane_id
                    if idx < level_offset + level_size:
                        sym_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id
                        r = self.multi_level_r[idx]
                        inv_block = self.inv_block_matrices[block_id, sym_idx]

                        z = ti.Vector.zero(ti.f32, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                z[di] += inv_block[di, dj] * r[dj]

                        self.multi_level_z[idx] = z
