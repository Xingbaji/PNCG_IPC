"""
BANKSIZE Performance Benchmark for MAS Preconditioner.

This benchmark compares MAS preconditioner performance with different BANKSIZE values
(4, 8, 16) across different mesh sizes.

The test measures:
1. build_hierarchy() - one-time initialization
2. assemble_block_matrices() - per-iteration Hessian assembly
3. invert_block_matrices() - per-iteration block inversion
4. apply() - per-iteration preconditioner application

Key insight: Different BANKSIZE values require different kernel compilation, so each
BANKSIZE is tested in a separate subprocess to avoid kernel caching issues.

Usage:
    python benchmark_banksize.py                     # Quick test (cube_10)
    python benchmark_banksize.py --size medium       # Medium mesh (cube_20)
    python benchmark_banksize.py --size large        # Large mesh (cube_40)
    python benchmark_banksize.py --iterations 20     # More iterations
    python benchmark_banksize.py --verbose           # Detailed output
"""

import sys
import os
import json
import subprocess
import argparse
import tempfile
import time
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file_path)
unittest_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(unittest_dir)


def generate_banksize_test_script(banksize: int, model_path: str, n_iter: int, warmup: int) -> str:
    """Generate a Python script to test a specific BANKSIZE configuration."""
    return f'''"""
Auto-generated script to benchmark BANKSIZE={banksize}
"""
import sys
import os
import time
import json
import numpy as np

# Setup paths
project_root = "{project_root}"
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(project_root)

import taichi as ti
ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache_banksize_{banksize}")

import meshtaichi_patcher as Patcher

# ============================================================================
# Dynamically define MAS Preconditioner with BANKSIZE={banksize}
# ============================================================================

from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter

BANKSIZE = {banksize}
SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) // 2
BLOCK_DOF = BANKSIZE * 3
MAX_LEVELS = 6

@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c

# Build adjacency and METIS partition functions
def build_adjacency_from_cells(n_verts: int, cells):
    adj_dict = {{i: set() for i in range(n_verts)}}
    for cell in cells:
        v0, v1, v2, v3 = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
        vertices = [v0, v1, v2, v3]
        for i in range(4):
            for j in range(4):
                if i != j:
                    adj_dict[vertices[i]].add(vertices[j])
    return [np.array(list(adj_dict[i]), dtype=np.int32) for i in range(n_verts)]

def metis_partition(n_verts: int, adjacency_list, n_parts: int):
    try:
        import pymetis
        if n_parts <= 1:
            return np.zeros(n_verts, dtype=np.int32)
        n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
        return np.array(membership, dtype=np.int32)
    except Exception:
        partition = np.zeros(n_verts, dtype=np.int32)
        for i in range(n_verts):
            partition[i] = i // BANKSIZE
        return partition

def compute_metis_reorder(n_verts: int, cells):
    adjacency_list = build_adjacency_from_cells(n_verts, cells)
    partition = None
    n_parts = 1

    for metis_offset in range(BANKSIZE):
        n_parts = (n_verts + BANKSIZE - metis_offset - 1) // (BANKSIZE - metis_offset)
        if n_parts <= 1:
            partition = np.zeros(n_verts, dtype=np.int32)
            n_parts = 1
            break
        partition = metis_partition(n_verts, adjacency_list, n_parts)
        partition_sizes = np.bincount(partition)
        if int(np.max(partition_sizes)) <= BANKSIZE:
            break

    indexed = [(i, partition[i]) for i in range(n_verts)]
    indexed.sort(key=lambda x: x[1])
    sort_index = np.array([p[0] for p in indexed], dtype=np.int32)
    old_to_new = np.zeros(n_verts, dtype=np.int32)
    for new_pos in range(n_verts):
        old_to_new[sort_index[new_pos]] = new_pos

    return sort_index, old_to_new, n_parts

# MAS Preconditioner Class
@ti.data_oriented
class MASPreconditionerBenchmark:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        self.n_parts = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.level_num = min(MAX_LEVELS, self._compute_level_num(self.n_verts))
        self.total_nodes_all_levels = self._compute_total_nodes(self.n_verts, self.level_num)
        self.total_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS)
        self._init_level_sizes()

        self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.total_blocks, SYM_BLOCK_COUNT))
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.total_blocks, SYM_BLOCK_COUNT))
        self.full_block_matrix = ti.field(dtype=ti.f32, shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f32, shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32, shape=self.total_nodes_all_levels)

        self.hierarchy_built = False

    def _compute_level_num(self, n_verts):
        levels = 1
        size = n_verts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return levels

    def _compute_total_nodes(self, n_verts, level_num):
        total = n_verts
        size = n_verts
        for _ in range(level_num - 1):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return total

    def _init_level_sizes(self):
        sizes, offsets = [], []
        size, offset = self.n_verts, 0
        for _ in range(self.level_num):
            sizes.append(size)
            offsets.append(offset)
            offset += size
            size = (size + BANKSIZE - 1) // BANKSIZE
        for i in range(self.level_num):
            self.level_size[i] = ti.Vector([sizes[i], offsets[i]])
        for i in range(self.level_num, MAX_LEVELS):
            self.level_size[i] = ti.Vector([0, offset])

    @ti.kernel
    def _build_going_next(self, level_num: ti.i32):
        if level_num == 1:
            for i in range(self.n_verts):
                self.going_next[i] = -1
        else:
            for i in range(self.n_verts):
                self.going_next[i] = self.level_size[1][1] + i // BANKSIZE
            for level in range(1, level_num - 1):
                level_offset = self.level_size[level][1]
                level_size_val = self.level_size[level][0]
                next_offset = self.level_size[level + 1][1]
                for i in range(level_size_val):
                    self.going_next[level_offset + i] = next_offset + i // BANKSIZE
            last_offset = self.level_size[level_num - 1][1]
            last_size = self.level_size[level_num - 1][0]
            for i in range(last_size):
                self.going_next[last_offset + i] = -1

    def build_hierarchy(self):
        level_1_size = self.n_parts
        level_1_offset = self.n_verts
        sizes = [self.n_verts, level_1_size]
        offsets = [0, level_1_offset]
        size, offset = level_1_size, level_1_offset + level_1_size
        for _ in range(2, self.level_num):
            size = (size + BANKSIZE - 1) // BANKSIZE
            sizes.append(size)
            offsets.append(offset)
            offset += size
        for i in range(len(sizes)):
            self.level_size[i] = ti.Vector([sizes[i], offsets[i]])
        self._build_going_next(self.level_num)
        self.hierarchy_built = True

    @ti.kernel
    def _clear_block_matrices(self):
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f32):
        for vert in self.mesh.verts:
            idx = vert.id
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            m = vert.m
            s_idx = sym_index(lane_id, lane_id)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, s_idx][d, d], m)

    @ti.kernel
    def _add_elastic_contribution_arap(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])
            warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE, v2 // BANKSIZE, v3 // BANKSIZE])
            lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE, v2 % BANKSIZE, v3 % BANKSIZE])

            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B, F = c.B, Ds @ c.B
            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            H_e = para * (dFdx.transpose() @ (d2PsidF2 @ dFdx))

            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):
                    warp_i, warp_j = warp_ids[i], warp_ids[j]
                    lane_i, lane_j = lane_ids[i], lane_ids[j]
                    if warp_i == warp_j:
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]
                        if lane_i <= lane_j:
                            s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, s_idx][di, dj], sub_block[di, dj])
                        else:
                            s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, s_idx][di, dj], sub_block[dj, di])
                    else:
                        vert_i, vert_j = v_ids[i], v_ids[j]
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
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj], sub_block[di, dj])
                                            if coarse_lane_i == coarse_lane_j:
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj], sub_block[dj, di])
                                else:
                                    s_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, s_idx][di, dj], sub_block[dj, di])
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
                    mat_norm = ti.f32(0.0)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            mat_norm += ti.abs(mat3[di, dj])
                    if mat_norm < 1e-12:
                        continue
                    rdx, cdx = row_idx, col_idx
                    for _ in range(self.level_num - 1):
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
        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)
        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

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
            for i in range(BLOCK_DOF):
                j_start = ti.max(0, i - BANDWIDTH)
                for j in range(j_start, i):
                    if i - j <= BANDWIDTH:
                        sum_val = self.full_block_matrix[block_id, i, j]
                        k_start = ti.max(0, ti.max(i - BANDWIDTH, j - BANDWIDTH))
                        for k in range(k_start, j):
                            if i - k <= BANDWIDTH and j - k <= BANDWIDTH:
                                sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_matrix[block_id, j, k]
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
            for col in range(BLOCK_DOF):
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    k_start = ti.max(0, i - BANDWIDTH)
                    for k in range(k_start, i):
                        if i - k <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_inverse[block_id, k, col]
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
                            sum_val -= self.full_block_matrix[block_id, k, i] * ti.f32(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
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
                            self.inv_block_matrices[block_id, s_idx][di, dj] = self.full_block_inverse[block_id, row * 3 + di, col * 3 + dj]

    def invert_block_matrices(self):
        self._expand_sym_to_full()
        self._incomplete_cholesky_invert_blocks()
        self._copy_inverse_to_sym()

    @ti.kernel
    def _clear_multi_level_buffers(self):
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        for vert in self.mesh.verts:
            self.multi_level_r[vert.id] = ti.cast(vert.grad, ti.f32)
        for vert in self.mesh.verts:
            r = self.multi_level_r[vert.id]
            coarse_idx = self.going_next[vert.id]
            if coarse_idx >= 0:
                for d in ti.static(range(3)):
                    ti.atomic_add(self.multi_level_r[coarse_idx][d], r[d])
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
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0, z1, z2 = ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)
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
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size_val = self.level_size[level][0]
            n_coarse_blocks = (level_size_val + BANKSIZE - 1) // BANKSIZE
            for local_block_id, lane_i in ti.ndrange(n_coarse_blocks, BANKSIZE):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE
                idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                if idx_i < level_offset + level_size_val:
                    z0, z1, z2 = ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)
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
        for vert in self.mesh.verts:
            idx = vert.id
            z_total = ti.cast(self.multi_level_z[idx], ti.f32)
            coarse_idx = self.going_next[idx]
            for _ in range(1, level_num):
                if coarse_idx >= 0:
                    z_total += ti.cast(self.multi_level_z[coarse_idx], ti.f32)
                    coarse_idx = self.going_next[coarse_idx]
                else:
                    break
            vert.z = z_total

    def apply(self):
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._schwarz_local_solve_banded()
        self._collect_final_z(self.level_num)

# ============================================================================
# Main Test
# ============================================================================

# Load mesh
model_path = "{model_path}"
raw_data = Patcher.load_mesh_rawdata(model_path)
vertices = raw_data[0].astype(np.float32)
cells = raw_data[3].astype(np.int32)
n_verts, n_cells = len(vertices), len(cells)

# METIS reorder
sort_index, old_to_new, n_parts = compute_metis_reorder(n_verts, cells)
reordered_verts = vertices[sort_index]
reordered_cells = old_to_new[cells]

# Create mesh
mesh_dict = {{0: reordered_verts, 3: reordered_cells}}
mesh = Patcher.load_mesh(mesh_dict, relations=['CV'])
mesh.verts.place({{
    'x': ti.types.vector(3, float),
    'v': ti.types.vector(3, float),
    'm': float,
    'x_n': ti.types.vector(3, float),
    'x_hat': ti.types.vector(3, float),
    'grad': ti.types.vector(3, float),
    'z': ti.types.vector(3, float),
}})
mesh.cells.place({{'B': ti.math.mat3, 'W': float}})

mesh.verts.x.from_numpy(reordered_verts)
mesh.verts.v.fill([0.0, 0.0, 0.0])

# Precompute
@ti.kernel
def precompute(mesh: ti.template(), density: ti.f32):
    for c in mesh.cells:
        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        c.B = Ds.inverse()
        c.W = ti.abs(Ds.determinant()) / 6.0
        for i in ti.static(range(4)):
            c.verts[i].m += density * c.W / 4.0

@ti.kernel
def set_xhat(mesh: ti.template(), dt: ti.f32, gravity: ti.f32):
    for vert in mesh.verts:
        vert.x_n = vert.x
        vert.x_hat = vert.x + dt * vert.v
        vert.x_hat[1] += gravity * dt * dt

@ti.kernel
def set_grad(mesh: ti.template()):
    for vert in mesh.verts:
        vert.grad = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5])

precompute(mesh, 1000.0)
set_xhat(mesh, 0.01, -9.8)

class MockSolver:
    def __init__(self):
        self.mu = 384615.4
        self.la = 576923.1
        self.dt = 0.01

solver = MockSolver()

# Create preconditioner
mas = MASPreconditionerBenchmark(mesh)
mas.build_hierarchy()

# Benchmark
n_iter = {n_iter}
warmup = {warmup}

# Assemble
for i in range(warmup):
    mas.assemble_block_matrices(solver)
ti.sync()
t0 = time.perf_counter()
for i in range(n_iter):
    mas.assemble_block_matrices(solver)
    ti.sync()
assemble_time = (time.perf_counter() - t0) / n_iter * 1000

# Invert
for i in range(warmup):
    mas.invert_block_matrices()
ti.sync()
t0 = time.perf_counter()
for i in range(n_iter):
    mas.invert_block_matrices()
    ti.sync()
invert_time = (time.perf_counter() - t0) / n_iter * 1000

# Apply
for i in range(warmup):
    set_grad(mesh)
    mas.apply()
ti.sync()
t0 = time.perf_counter()
for i in range(n_iter):
    set_grad(mesh)
    mas.apply()
    ti.sync()
apply_time = (time.perf_counter() - t0) / n_iter * 1000

# Output results as JSON
results = {{
    'banksize': BANKSIZE,
    'n_verts': n_verts,
    'n_cells': n_cells,
    'levels': mas.level_num,
    'blocks': mas.total_blocks,
    'assemble_ms': assemble_time,
    'invert_ms': invert_time,
    'apply_ms': apply_time,
}}
print('BENCHMARK_RESULT:' + json.dumps(results))
'''


def run_banksize_benchmark(banksize: int, model_path: str, n_iter: int, warmup: int, verbose: bool = False):
    """Run benchmark for a specific BANKSIZE in a subprocess."""
    script_content = generate_banksize_test_script(banksize, model_path, n_iter, warmup)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        if verbose:
            print(f"\n[Running BANKSIZE={banksize} benchmark...]")

        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if verbose:
            print(result.stdout)

        if result.returncode != 0:
            print(f"Error running BANKSIZE={banksize}:")
            print(result.stderr)
            return None

        # Parse JSON result from output
        for line in result.stdout.split('\n'):
            if line.startswith('BENCHMARK_RESULT:'):
                json_str = line[len('BENCHMARK_RESULT:'):]
                return json.loads(json_str)

        print(f"No result found for BANKSIZE={banksize}")
        return None

    finally:
        os.unlink(script_path)


def print_comparison_table(all_results: dict, mesh_name: str = "mesh"):
    """Print a comparison table for all BANKSIZE results."""
    print("\n" + "=" * 90)
    print(f"Performance Comparison Summary - {mesh_name}")
    print("=" * 90)

    if not all_results:
        print("No results to compare.")
        return

    # Get mesh info from first result
    first_result = next(iter(all_results.values()))
    print(f"Mesh: {first_result['n_verts']} vertices, {first_result['n_cells']} cells")

    # Header
    print(f"\n{'BANKSIZE':<12} {'Levels':<8} {'Blocks':<10} {'Assemble':>12} {'Invert':>12} {'Apply':>12} {'Total':>12}")
    print("-" * 90)

    for banksize in sorted(all_results.keys()):
        r = all_results[banksize]
        total_ms = r['assemble_ms'] + r['invert_ms'] + r['apply_ms']

        print(f"{banksize:<12} {r['levels']:<8} {r['blocks']:<10} "
              f"{r['assemble_ms']:>10.3f}ms {r['invert_ms']:>10.3f}ms {r['apply_ms']:>10.3f}ms {total_ms:>10.3f}ms")

    # Speedup comparison (relative to BANKSIZE=16)
    if 16 in all_results:
        base = all_results[16]
        base_total = base['assemble_ms'] + base['invert_ms'] + base['apply_ms']

        print(f"\n{'Speedup vs BANKSIZE=16:':<30}")
        print("-" * 50)

        for banksize in sorted(all_results.keys()):
            r = all_results[banksize]
            total = r['assemble_ms'] + r['invert_ms'] + r['apply_ms']
            speedup = base_total / total if total > 0 else 0
            print(f"  BANKSIZE={banksize:<4}  {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='BANKSIZE Performance Benchmark')
    parser.add_argument('--size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Mesh size: small (cube_10), medium (cube_20), large (cube_40)')
    parser.add_argument('--model', type=str, default=None,
                        help='Custom model path (overrides --size)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to time (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--banksizes', type=str, default='4,8,16',
                        help='Comma-separated list of BANKSIZE values to test (default: 4,8,16)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    # Select model path
    if args.model:
        model_path = args.model
        mesh_name = os.path.basename(os.path.dirname(model_path))
    else:
        size_to_model = {
            'small': os.path.join(project_root, 'model/mesh/cube_10/cube_10.node'),
            'medium': os.path.join(project_root, 'model/mesh/cube_20/cube_20.node'),
            'large': os.path.join(project_root, 'model/mesh/cube_40/cube_40.node'),
        }
        model_path = size_to_model[args.size]
        mesh_name = f"{args.size} ({os.path.basename(os.path.dirname(model_path))})"

    # Parse banksizes
    banksizes = [int(b) for b in args.banksizes.split(',')]

    print("=" * 90)
    print("BANKSIZE Performance Benchmark for MAS Preconditioner")
    print("=" * 90)
    print(f"Model: {model_path}")
    print(f"BANKSIZE values: {banksizes}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

    # Run benchmarks
    all_results = {}
    for banksize in banksizes:
        result = run_banksize_benchmark(
            banksize, model_path, args.iterations, args.warmup, args.verbose
        )
        if result:
            all_results[banksize] = result
            print(f"BANKSIZE={banksize}: assemble={result['assemble_ms']:.3f}ms, "
                  f"invert={result['invert_ms']:.3f}ms, apply={result['apply_ms']:.3f}ms")

    # Print comparison table
    print_comparison_table(all_results, mesh_name)

    print("\n" + "=" * 90)
    print("Benchmark Complete")
    print("=" * 90)


if __name__ == '__main__':
    main()
