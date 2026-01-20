"""
Contact Hessian assembly utilities for MAS preconditioner.

Provides functions to compute contact Hessian subblocks and assemble
them into the block matrix structure.
"""

import taichi as ti
from typing import Optional

from ...core.precision import PrecisionType, PrecisionMixin


@ti.func
def compute_contact_subblock(para0, para, t, coeff):
    """
    Compute 3x3 contact Hessian subblock.

    H_ij = coeff * (para0 * (t @ t^T) + para * I_3x3)

    where:
        coeff = scale * cord[i] * cord[j] (barycentric weighting)
        para = barrier_g / dist
        para0 = (barrier_H - para) / dist^2
        t = contact direction (normalized)

    Args:
        para0: Curvature-related term
        para: Gradient-related term
        t: Contact direction (3D)
        coeff: Scaling coefficient

    Returns:
        3x3 Hessian subblock
    """
    H_ij = ti.Matrix.zero(t.dtype, 3, 3)
    for di in ti.static(range(3)):
        for dj in ti.static(range(3)):
            H_ij[di, dj] = coeff * para0 * t[di] * t[dj]
            if di == dj:
                H_ij[di, dj] += coeff * para
    return H_ij


@ti.func
def compute_contact_subblock_spd(e, curvature, coeff):
    """
    Compute SPD-guaranteed 3x3 contact Hessian subblock.

    Uses the rank-1 formulation that guarantees positive semi-definiteness:
    H = coeff * curvature * (e @ e^T) / ||e||²

    This has eigenvalues [coeff*curvature, 0, 0], always non-negative.

    Args:
        e: Contact edge vector (not normalized)
        curvature: Barrier second derivative (must be non-negative)
        coeff: Scaling coefficient (cord[i] * cord[j])

    Returns:
        3x3 SPD Hessian subblock
    """
    H_ij = ti.Matrix.zero(e.dtype, 3, 3)
    e_sqr = e.norm_sqr()
    if e_sqr > 1e-12 and curvature > 0.0:
        scale = coeff * curvature / e_sqr
        for di in ti.static(range(3)):
            for dj in ti.static(range(3)):
                H_ij[di, dj] = scale * e[di] * e[dj]
    return H_ij


@ti.data_oriented
class ContactAssembler(PrecisionMixin):
    """
    Contact Hessian assembler for MAS preconditioner.

    Handles the assembly of contact Hessian contributions into the
    block matrix structure, including cross-block triplet storage.
    """

    def __init__(
        self,
        max_contacts: int = 2**18,
        max_triplets: int = 2**20,
        banksize: int = 16,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize contact assembler.

        Args:
            max_contacts: Maximum number of contact pairs
            max_triplets: Maximum cross-block triplet entries
            banksize: Nodes per block (typically 16)
            precision: Float precision
        """
        self.init_precision(precision)
        self.MAX_CONTACTS = max_contacts
        self.MAX_TRIPLETS = max_triplets
        self.BANKSIZE = banksize

        float_type = self.cfg.float_type

        # Cross-block triplet storage (COO format)
        self._triplet_row = ti.field(dtype=ti.i32, shape=max_triplets)
        self._triplet_col = ti.field(dtype=ti.i32, shape=max_triplets)
        self._triplet_val = ti.Matrix.field(3, 3, dtype=float_type, shape=max_triplets)
        self._n_triplets = ti.field(dtype=ti.i32, shape=())

        # Reference to block matrices (set by preconditioner)
        self._block_matrices = None
        self._n_blocks = 0

    def set_block_matrices(self, block_matrices, n_blocks: int):
        """
        Set reference to preconditioner's block matrices.

        Args:
            block_matrices: Block matrix field from MAS preconditioner
            n_blocks: Number of blocks at level 0
        """
        self._block_matrices = block_matrices
        self._n_blocks = n_blocks

    @ti.kernel
    def reset(self):
        """Reset triplet storage for new assembly."""
        self._n_triplets[None] = 0

    @property
    def n_triplets(self) -> int:
        """Get current number of triplets."""
        return self._n_triplets[None]

    @ti.kernel
    def assemble_contact_hessian(
        self,
        contact_pairs: ti.template(),
        n_contacts: ti.i32,
        dt_sq: ti.template(),
        kappa: ti.template(),
        dHat: ti.template(),
    ):
        """
        Assemble contact Hessian into block matrices.

        For each contact with 4 vertices, this computes 16 subblock
        contributions (4×4 vertex pairs) and adds them to:
        - Block matrices (same-block contributions)
        - Triplet storage (cross-block contributions)

        Args:
            contact_pairs: Contact pair storage
            n_contacts: Number of contacts
            dt_sq: Time step squared (scaling factor)
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist > 1e-10 and dist < dHat:
                dist2 = dist * dist

                # Barrier gradient and hessian (log barrier)
                bg = kappa * ((dist - dHat) * ti.log(dist / dHat) * (-2.0) -
                             (dist - dHat) ** 2 / dist)
                bH = kappa * ((-2.0) * ti.log(dist / dHat) - 4.0 +
                             4.0 * dHat / dist + (dist - dHat) ** 2 / dist2)

                para = bg / dist
                para0 = (bH - para) / dist2
                scale = dt_sq

                # Normalize contact direction
                t_norm = t / dist if dist > 1e-10 else t

                # Add 16 vertex-pair contributions
                for i in ti.static(range(4)):
                    for j in ti.static(range(4)):
                        vi = ti.cast(ids[i], ti.i32)
                        vj = ti.cast(ids[j], ti.i32)
                        coeff = scale * cord[i] * cord[j]
                        H_ij = compute_contact_subblock(para0, para, t_norm, coeff)

                        # Determine block IDs
                        block_i = vi // self.BANKSIZE
                        block_j = vj // self.BANKSIZE
                        lane_i = vi % self.BANKSIZE
                        lane_j = vj % self.BANKSIZE

                        if block_i == block_j:
                            # Same-block: add to block matrix
                            self._add_to_block(block_i, lane_i, lane_j, H_ij)
                        else:
                            # Cross-block: store as triplet
                            self._add_triplet(vi, vj, H_ij)

    @ti.func
    def _add_to_block(self, block_id, lane_i, lane_j, H_ij):
        """
        Add contribution to a block matrix.

        Uses symmetric storage: only upper triangle is stored.

        Args:
            block_id: Block index
            lane_i: Row lane within block
            lane_j: Column lane within block
            H_ij: 3×3 subblock to add
        """
        if lane_i <= lane_j:
            # Upper triangle: compute symmetric storage index
            s_idx = self.BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    ti.atomic_add(self._block_matrices[block_id, s_idx][di, dj], H_ij[di, dj])
        else:
            # Lower triangle: transpose and add to upper
            s_idx = self.BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    ti.atomic_add(self._block_matrices[block_id, s_idx][di, dj], H_ij[dj, di])

    @ti.func
    def _add_triplet(self, vi, vj, H_ij):
        """
        Add cross-block contribution to triplet storage.

        Stores upper triangle only (vi <= vj).

        Args:
            vi: Row vertex ID
            vj: Column vertex ID
            H_ij: 3×3 subblock
        """
        triplet_idx = ti.atomic_add(self._n_triplets[None], 1)
        if triplet_idx < self.MAX_TRIPLETS:
            if vi <= vj:
                self._triplet_row[triplet_idx] = vi
                self._triplet_col[triplet_idx] = vj
                self._triplet_val[triplet_idx] = H_ij
            else:
                # Store transpose
                self._triplet_row[triplet_idx] = vj
                self._triplet_col[triplet_idx] = vi
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        self._triplet_val[triplet_idx][di, dj] = H_ij[dj, di]

    @ti.kernel
    def apply_cross_block_spmv(
        self,
        x: ti.template(),
        y: ti.template(),
    ):
        """
        Apply cross-block sparse matrix-vector product: y += H_cross @ x

        Args:
            x: Input vector field
            y: Output vector field (accumulated)
        """
        for idx in range(self._n_triplets[None]):
            vi = self._triplet_row[idx]
            vj = self._triplet_col[idx]
            H_ij = self._triplet_val[idx]

            # y[vi] += H_ij @ x[vj]
            Hx = H_ij @ x[vj]
            for d in ti.static(range(3)):
                ti.atomic_add(y[vi][d], Hx[d])

            # y[vj] += H_ij^T @ x[vi] (symmetric)
            if vi != vj:
                Hx_t = H_ij.transpose() @ x[vi]
                for d in ti.static(range(3)):
                    ti.atomic_add(y[vj][d], Hx_t[d])
