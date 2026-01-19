"""
Matrix assembly functions for MAS Preconditioner.

This module handles:
- Elastic Hessian assembly (ARAP, SNH, FCR)
- IPC contact barrier Hessian assembly
- Inertia (mass matrix) contribution
- Fine-to-coarse aggregation
- Regularization for numerical stability

Reference: MASPreconditioner.cu (CUDA reference implementation)
"""

import taichi as ti

from .constants import BANKSIZE, SYM_BLOCK_COUNT

# Import math utilities for elastic Hessian computation
# These must be imported at module level for Taichi kernels
from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import (
    compute_d2PsidF2_ARAP_filter, compute_d2PsidF2_SNH, compute_d2PsidF2_FCR_filter,
    # SPD-projected Hessian functions (eigenanalysis-based)
    compute_d2PsidF2_ARAP_SPD, compute_d2PsidF2_NH_SPD, compute_d2PsidF2_STVK_SPD
)

# Elastic type constants
ELASTIC_ARAP = 0
ELASTIC_SNH = 1
ELASTIC_FCR = 2
ELASTIC_ARAP_SPD = 3
ELASTIC_NH_SPD = 4
ELASTIC_STVK_SPD = 5


# ==============================================================================
# Utility Functions
# ==============================================================================

@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """
    Compute symmetric storage index for upper triangle.

    For row <= col: index = BANKSIZE * row - row*(row+1)/2 + col
    """
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


# ==============================================================================
# AssemblyMixin Class
# ==============================================================================

class AssemblyMixin:
    """
    Mixin class providing matrix assembly methods for MASPreconditioner.

    This class provides kernels for assembling the block Hessian matrices
    from elastic and contact contributions.
    """

    @ti.func
    def _sym_index(self, row: ti.i32, col: ti.i32) -> ti.i32:
        """Compute symmetric storage index for upper triangle."""
        return sym_index(row, col)

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

            # Get mass from mesh
            m = self.mesh.verts.m[idx]

            # Diagonal block index in symmetric storage
            sym_idx = self._sym_index(lane_id, lane_id)

            # Add mass to diagonal (scaled by 1/dt^2 for implicit)
            mass_val = m / (dt * dt)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], mass_val)

    @ti.kernel
    def _add_elastic_contribution_full_optimized(self, mu: ti.f32, la: ti.f32, dt: ti.f32,
                                                   elastic_type: ti.i32):
        """
        Optimized elastic Hessian assembly with reduced branching and local accumulation.

        Optimization strategies:
        1. Pre-compute all 16 vertex-pair interactions for each cell
        2. Use local variables to accumulate before atomic write
        3. Minimize branching by separating same-warp and cross-warp handling
        4. Batch atomic operations where possible
        """
        for c in self.mesh.cells:
            # Get cell volume weight
            W = c.W
            para = W * dt * dt

            # Get vertex IDs
            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            # Pre-compute warp IDs for all 4 vertices
            warp_ids = ti.Vector([v0 // BANKSIZE, v1 // BANKSIZE,
                                  v2 // BANKSIZE, v3 // BANKSIZE])
            lane_ids = ti.Vector([v0 % BANKSIZE, v1 % BANKSIZE,
                                  v2 % BANKSIZE, v3 % BANKSIZE])

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Compute dFdx (9x12 matrix)
            dFdx = compute_dFdx(B)

            # Compute d2PsidF2 (9x9 matrix) based on elastic type
            d2PsidF2 = ti.Matrix.zero(ti.f32, 9, 9)
            if elastic_type == ELASTIC_ARAP:  # ARAP (filtered)
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            elif elastic_type == ELASTIC_SNH:  # SNH
                d2PsidF2 = compute_d2PsidF2_SNH(F, mu, la)
            elif elastic_type == ELASTIC_FCR:  # FCR (filtered)
                d2PsidF2 = compute_d2PsidF2_FCR_filter(F, mu, la)
            elif elastic_type == ELASTIC_ARAP_SPD:  # ARAP with full SPD projection
                d2PsidF2 = compute_d2PsidF2_ARAP_SPD(F, mu, la)
            elif elastic_type == ELASTIC_NH_SPD:  # Neo-Hookean with SPD projection
                d2PsidF2 = compute_d2PsidF2_NH_SPD(F, mu, la)
            elif elastic_type == ELASTIC_STVK_SPD:  # StVK with SPD projection
                d2PsidF2 = compute_d2PsidF2_STVK_SPD(F, mu, la)
            else:
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

            # Compute element Hessian: H_e = dFdx^T @ d2PsidF2 @ dFdx (12x12)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Process vertex pairs for symmetric Hessian storage
            # Only process upper triangle (i <= j) of the element Hessian
            # to avoid double-counting symmetric entries
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):  # j >= i: upper triangle only
                    warp_i = warp_ids[i]
                    warp_j = warp_ids[j]
                    lane_i = lane_ids[i]
                    lane_j = lane_ids[j]

                    if warp_i == warp_j:
                        # Same warp: direct assembly to Level 0 block
                        # Extract 3x3 sub-block from H_e[i,j]
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # Store in symmetric storage at position (min_lane, max_lane)
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj],
                                                  sub_block[di, dj])
                        else:
                            # lane_i > lane_j: need to transpose when storing
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj],
                                                  sub_block[dj, di])
                    else:
                        # Cross-warp: propagate to coarse level via goingNext
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

                                # Handle symmetric storage at coarse level
                                if coarse_lane_i <= coarse_lane_j:
                                    sym_idx = BANKSIZE * coarse_lane_i - coarse_lane_i * (coarse_lane_i + 1) // 2 + coarse_lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj],
                                                          sub_block[di, dj])
                                            # FIX: When mapping to diagonal block (two different fine vertices
                                            # map to same coarse vertex), add transpose for symmetry
                                            if coarse_lane_i == coarse_lane_j:
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj],
                                                              sub_block[dj, di])
                                else:
                                    # Transpose for lower triangle
                                    sym_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj],
                                                          sub_block[dj, di])
                                break

    @ti.kernel
    def _add_elastic_contribution_full(self, mu: ti.f32, la: ti.f32, dt: ti.f32,
                                        elastic_type: ti.i32):
        """
        Add full elastic Hessian contribution with proper off-diagonal coupling.
        (Original implementation - fallback for non-CUDA backends)
        """
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt

            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            dFdx = compute_dFdx(B)

            d2PsidF2 = ti.Matrix.zero(ti.f32, 9, 9)
            if elastic_type == ELASTIC_ARAP:
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            elif elastic_type == ELASTIC_SNH:
                d2PsidF2 = compute_d2PsidF2_SNH(F, mu, la)
            elif elastic_type == ELASTIC_FCR:
                d2PsidF2 = compute_d2PsidF2_FCR_filter(F, mu, la)
            elif elastic_type == ELASTIC_ARAP_SPD:
                d2PsidF2 = compute_d2PsidF2_ARAP_SPD(F, mu, la)
            elif elastic_type == ELASTIC_NH_SPD:
                d2PsidF2 = compute_d2PsidF2_NH_SPD(F, mu, la)
            elif elastic_type == ELASTIC_STVK_SPD:
                d2PsidF2 = compute_d2PsidF2_STVK_SPD(F, mu, la)
            else:
                d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)

            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Process upper triangle of vertex pairs (i <= j) to avoid double-counting
            for i in ti.static(range(4)):
                for j in ti.static(range(i, 4)):  # j >= i: upper triangle only
                    vi = v_ids[i]
                    vj = v_ids[j]
                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    if warp_i == warp_j:
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        # Extract 3x3 sub-block from H_e
                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # For symmetric storage, handle upper/lower triangle
                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], sub_block[di, dj])
                        else:
                            # Transpose for lower triangle
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], sub_block[dj, di])
                    else:
                        # Cross-warp: propagate to coarse level via goingNext
                        vert_i = vi
                        vert_j = vj

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

                                # Handle symmetric storage at coarse level
                                if coarse_lane_i <= coarse_lane_j:
                                    sym_idx = BANKSIZE * coarse_lane_i - coarse_lane_i * (coarse_lane_i + 1) // 2 + coarse_lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], sub_block[di, dj])
                                            # FIX: When mapping to diagonal block, add transpose for symmetry
                                            if coarse_lane_i == coarse_lane_j:
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], sub_block[dj, di])
                                else:
                                    # Transpose for lower triangle
                                    sym_idx = BANKSIZE * coarse_lane_j - coarse_lane_j * (coarse_lane_j + 1) // 2 + coarse_lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], sub_block[dj, di])
                                break

    @ti.kernel
    def _add_elastic_contribution_approx(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """
        Add approximate elastic Hessian contribution using diagonal approximation.
        """
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt

            v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
            v_ids = ti.Vector([v0, v1, v2, v3])

            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            stiffness = para * (2.0 * mu + la)

            for i in ti.static(range(4)):
                vi = v_ids[i]
                warp_id = vi // BANKSIZE
                lane_id = vi % BANKSIZE
                sym_idx = self._sym_index(lane_id, lane_id)

                for d in ti.static(range(3)):
                    ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d],
                                  stiffness * 0.25)

            for i in ti.static(range(4)):
                for j in ti.static(range(i + 1, 4)):
                    vi = v_ids[i]
                    vj = v_ids[j]
                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE

                    if warp_i == warp_j:
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE
                        sym_idx = self._sym_index(lane_i, lane_j)

                        coupling = para * mu * 0.1
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.block_matrices[warp_i, sym_idx][d, d],
                                          coupling)

    @ti.kernel
    def _add_regularization(self, epsilon: ti.f32):
        """Add small regularization to diagonal for numerical stability."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    sym_idx = self._sym_index(lane_id, lane_id)

                    for d in ti.static(range(3)):
                        val = self.block_matrices[block_id, sym_idx][d, d]
                        if val < epsilon:
                            self.block_matrices[block_id, sym_idx][d, d] = epsilon

    # ========================================================================
    # IPC Contact Contribution
    # ========================================================================

    @ti.kernel
    def _add_ipc_contact_contribution(self, cid: ti.template(), dHat: ti.f32, kappa: ti.f32):
        """
        Add IPC barrier Hessian contribution from contact pairs to block matrices.
        (Legacy bitmasked cid version)
        """
        for k, j in cid:
            pair = cid[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            normal = pair.d

            if dist >= dHat:
                continue

            barrier_H = 4.0 * kappa * (1.0 - dist / dHat)

            # Process upper triangle of vertex pairs (i <= jj) to avoid double-counting
            for i in ti.static(range(4)):
                vi = ids[i]
                ci = cord[i]

                if ti.abs(ci) < 1e-10:
                    continue

                for jj in ti.static(range(i, 4)):  # jj >= i: upper triangle only
                    vj = ids[jj]
                    cj = cord[jj]

                    if ti.abs(cj) < 1e-10:
                        continue

                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE
                    scale = barrier_H * ci * cj

                    if warp_i == warp_j:
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[di] * normal[dj]
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
                        else:
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[dj] * normal[di]
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][di, dj], val)
                    else:
                        vert_i = vi
                        vert_j = vj

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                lane_i = vert_i % BANKSIZE
                                lane_j = vert_j % BANKSIZE

                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[di] * normal[dj]
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val)
                                            # FIX: When mapping to diagonal block, add transpose for symmetry
                                            if lane_i == lane_j:
                                                val_t = scale * normal[dj] * normal[di]
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val_t)
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[dj] * normal[di]
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][di, dj], val)
                                break

    def _add_ipc_contact_contribution_compact(self, solver, n_contacts: int):
        """
        Add IPC barrier Hessian contribution using compact array storage (P0 optimization).
        """
        self._add_ipc_contact_contribution_compact_kernel(
            solver.contact_pairs, n_contacts, solver.dHat, solver.kappa)

    @ti.kernel
    def _add_ipc_contact_contribution_compact_kernel(self, contact_pairs: ti.template(),
                                                      n_contacts: ti.i32,
                                                      dHat: ti.f32, kappa: ti.f32):
        """
        Kernel for IPC contact Hessian using compact array (P0 optimization).
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            normal = pair.d

            if dist >= dHat:
                continue

            barrier_H = 4.0 * kappa * (1.0 - dist / dHat)

            # Process upper triangle of vertex pairs (i <= jj) to avoid double-counting
            for i in ti.static(range(4)):
                vi = ids[i]
                ci = cord[i]

                if ti.abs(ci) < 1e-10:
                    continue

                for jj in ti.static(range(i, 4)):  # jj >= i: upper triangle only
                    vj = ids[jj]
                    cj = cord[jj]

                    if ti.abs(cj) < 1e-10:
                        continue

                    warp_i = vi // BANKSIZE
                    warp_j = vj // BANKSIZE
                    scale = barrier_H * ci * cj

                    if warp_i == warp_j:
                        lane_i = vi % BANKSIZE
                        lane_j = vj % BANKSIZE

                        if lane_i <= lane_j:
                            sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[di] * normal[dj]
                                    ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
                        else:
                            sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    val = scale * normal[dj] * normal[di]
                                    ti.atomic_add(self.block_matrices[warp_j, sym_idx][di, dj], val)
                    else:
                        vert_i = vi
                        vert_j = vj

                        for _ in range(self.level_num - 1):
                            vert_i = self.going_next[vert_i]
                            vert_j = self.going_next[vert_j]

                            if vert_i < 0 or vert_j < 0:
                                break

                            coarse_warp_i = vert_i // BANKSIZE
                            coarse_warp_j = vert_j // BANKSIZE

                            if coarse_warp_i == coarse_warp_j:
                                lane_i = vert_i % BANKSIZE
                                lane_j = vert_j % BANKSIZE

                                if lane_i <= lane_j:
                                    sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[di] * normal[dj]
                                            ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val)
                                            # FIX: When mapping to diagonal block, add transpose for symmetry
                                            if lane_i == lane_j:
                                                val_t = scale * normal[dj] * normal[di]
                                                ti.atomic_add(self.block_matrices[coarse_warp_i, sym_idx][di, dj], val_t)
                                else:
                                    sym_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            val = scale * normal[dj] * normal[di]
                                            ti.atomic_add(self.block_matrices[coarse_warp_j, sym_idx][di, dj], val)
                                break

    # ========================================================================
    # Fine-to-Coarse Aggregation
    # ========================================================================

    @ti.kernel
    def _aggregate_fine_to_coarse(self, level_num: ti.i32):
        """
        Aggregate fine-level block entries to coarse levels.

        CUDA Reference: PrepareHessian_bcoo second pass (lines 1933-2062)
        """
        n_fine_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_fine_blocks):
            for lane_row in range(BANKSIZE):
                for lane_col in range(lane_row, BANKSIZE):
                    row_idx = block_id * BANKSIZE + lane_row
                    col_idx = block_id * BANKSIZE + lane_col

                    if row_idx >= self.n_verts or col_idx >= self.n_verts:
                        continue

                    sym_idx = BANKSIZE * lane_row - lane_row * (lane_row + 1) // 2 + lane_col
                    mat3 = self.block_matrices[block_id, sym_idx]

                    mat_norm = ti.f32(0.0)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            mat_norm += ti.abs(mat3[di, dj])
                    if mat_norm < 1e-12:
                        continue

                    rdx = row_idx
                    cdx = col_idx

                    for level in range(level_num - 1):
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
                                coarse_sym_idx = BANKSIZE * coarse_lane_r - coarse_lane_r * (coarse_lane_r + 1) // 2 + coarse_lane_c
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_r, coarse_sym_idx][di, dj], mat3[di, dj])
                                        # When rdx == cdx (coarse diagonal), both (row,col) and (col,row)
                                        # map to same position, so add transpose contribution
                                        if coarse_lane_r == coarse_lane_c and row_idx != col_idx:
                                            ti.atomic_add(self.block_matrices[coarse_block_r, coarse_sym_idx][di, dj], mat3[dj, di])
                            else:
                                coarse_sym_idx = BANKSIZE * coarse_lane_c - coarse_lane_c * (coarse_lane_c + 1) // 2 + coarse_lane_r
                                for di in ti.static(range(3)):
                                    for dj in ti.static(range(3)):
                                        ti.atomic_add(self.block_matrices[coarse_block_c, coarse_sym_idx][di, dj], mat3[dj, di])

    @ti.kernel
    def _add_regularization_coarse(self, epsilon: ti.f32, level_num: ti.i32):
        """Add regularization to coarse-level diagonal blocks."""
        for level in range(1, level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                for lane_id in range(BANKSIZE):
                    local_idx = local_block_id * BANKSIZE + lane_id
                    if local_idx < level_size:
                        global_idx = level_offset + local_idx
                        block_id = global_idx // BANKSIZE
                        lane_in_block = global_idx % BANKSIZE

                        sym_idx = self._sym_index(lane_in_block, lane_in_block)

                        for d in ti.static(range(3)):
                            val = self.block_matrices[block_id, sym_idx][d, d]
                            if val < epsilon:
                                self.block_matrices[block_id, sym_idx][d, d] = epsilon

    # ========================================================================
    # Main Assembly Entry Point
    # ========================================================================

    def assemble_block_matrices(self, solver, use_full_hessian: bool = True,
                                use_optimized_kernel: bool = True):
        """
        Assemble Hessian contributions into block matrices.

        Args:
            solver: The PNCG solver containing material parameters
            use_full_hessian: If True, compute full element Hessian with coupling.
            use_optimized_kernel: If True, use optimized kernel with reduced branching.
        """
        print("[MAS] Assembling block matrices...")

        # Clear matrices
        self._clear_block_matrices()

        # Add inertia contribution (mass matrix)
        self._add_inertia_contribution(solver.dt)

        # Add elastic contribution
        if use_full_hessian:
            use_opt = use_optimized_kernel and self._is_cuda_backend()

            if use_opt:
                self._add_elastic_contribution_full_optimized(
                    solver.mu, solver.la, solver.dt, self.elastic_type)
                print("[MAS] Full elastic Hessian assembled (optimized kernel)")
            else:
                self._add_elastic_contribution_full(solver.mu, solver.la, solver.dt,
                                                     self.elastic_type)
                print("[MAS] Full elastic Hessian assembled")
        else:
            self._add_elastic_contribution_approx(solver.mu, solver.la, solver.dt)
            print("[MAS] Approximate elastic Hessian assembled")

        # Add IPC barrier Hessian from contact pairs
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                self._add_ipc_contact_contribution_compact(solver, n_contacts)
                print(f"[MAS] IPC contact Hessian assembled ({n_contacts} contacts, compact)")
        elif hasattr(solver, 'cid') and solver.cid is not None:
            try:
                n_contacts = len(solver.cid)
                if n_contacts > 0:
                    self._add_ipc_contact_contribution(solver.cid, solver.dHat, solver.kappa)
                    print(f"[MAS] IPC contact Hessian assembled ({n_contacts} contacts)")
            except Exception:
                pass

        # Note: regularization disabled for debugging - enable if needed
        # self._add_regularization(1e-3)

        # Aggregate fine-level block entries to coarse levels
        if self.hierarchy_built and self.actual_levels > 1:
            self._aggregate_fine_to_coarse(self.actual_levels)
            print(f"[MAS] Fine-to-coarse aggregation complete ({self.actual_levels} levels)")

            # self._add_regularization_coarse(1e-3, self.actual_levels)

        self.matrices_assembled = True
        print("[MAS] Block matrices assembled")
