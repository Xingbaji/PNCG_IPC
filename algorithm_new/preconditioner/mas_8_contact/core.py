"""
MAS Preconditioner with BANKSIZE=8 and Contact Support - Core Implementation.

Combines the bank-8 optimizations from mas_preconditioner_8 with
the contact functions from mas_preconditioner_contact.

Key optimizations for BANKSIZE=8 (adapted from Wu-2022-GBM):
1. One-way Gauss-Jordan elimination for matrix inverse
   - Sequential elimination: O(n³/2) + parallel L⁻ᵀD⁻¹L⁻¹: O(n³/6)
   - Avoids difficult-to-parallelize triangular solves
   - ~3x faster than traditional Cholesky + forward/backward substitution
2. Optimized symmetric matvec with reduced branching
   - Two-pass structure: diagonal pass + upper-triangular pass
   - Pre-computed storage indices
3. Compact block storage - 36 sym blocks (vs 136 for BANKSIZE=16)

Contact support features:
- SPD Contact Hessian (PPF-Contact-Solver style)
- Log and Cubic barrier functions
- Friction support
- Cross-block contact triplet storage
- Woodbury low-rank updates for incremental contact changes

BANKSIZE=8 characteristics:
- SYM_BLOCK_COUNT = 36 (vs 136 for BANKSIZE=16) - 3.8x less storage
- BLOCK_DOF = 24 (vs 48 for BANKSIZE=16)
- Better for small/medium meshes (<15K vertices)
- 24 DOF = 3/4 warp, good GPU alignment
"""

import taichi as ti
import numpy as np

from ...core.precision import PrecisionType, PrecisionMixin, get_precision_config

# Constants - BANKSIZE=8 specific
BANKSIZE = 8
SYM_BLOCK_COUNT = 36  # 8 * (8 + 1) // 2
BLOCK_DOF = 24  # 8 * 3
MAX_LEVELS = 6

# Import barrier functions from contact assembly module
from .contact_assembly import (
    barrier_g_log,
    barrier_H_log,
    barrier_g_cubic,
    barrier_H_cubic,
    # SPD barrier/friction functions (PPF-Contact-Solver style)
    barrier_curvature_cubic,
    barrier_curvature_log,
    compute_spd_contact_hessian_3x3,
    compute_spd_edge_hessian,
    compute_spd_edge_gradient,
    compute_friction_projection_matrix,
    compute_friction_lambda,
    compute_friction_hessian,
    compute_spd_contact_friction_hessian,
)


# ============================================================================
# Helper functions
# ============================================================================

@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """Compute symmetric storage index for BANKSIZE=8."""
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


# ============================================================================
# Main class
# ============================================================================

@ti.data_oriented
class MASPreconditioner8Contact(PrecisionMixin):
    """
    MAS Preconditioner with BANKSIZE=8 and IPC Contact Hessian support.

    Combines:
    - BANKSIZE=8 optimizations from MASPreconditioner8
    - Contact Hessian support from MASPreconditionerContact

    Key features:
    1. One-way Gauss-Jordan elimination for matrix inverse (~3x faster)
    2. Optimized symmetric matvec with two-pass structure
    3. Compact block storage (36 sym blocks vs 136 for BANKSIZE=16)
    4. SPD Contact Hessian with barrier and friction support
    5. Woodbury low-rank updates for incremental contact changes

    Parameters:
        mesh: MeshTaichi mesh object (must be METIS pre-reordered with BANKSIZE=8)
        max_contacts: Maximum number of contact pairs (default: 2^18)
        metis_reordered: Must be True
        max_verts: Maximum number of vertices
        metis_n_parts: Actual number of METIS partitions
        precision: Float precision ('f32' or 'f64')
    """

    def __init__(self, mesh, max_contacts: int = 2**18, metis_reordered: bool = True,
                 max_verts: int = None, metis_n_parts: int = None,
                 precision: PrecisionType = 'f32'):
        """
        Initialize MAS Preconditioner with BANKSIZE=8 and Contact Support.

        Args:
            mesh: MeshTaichi mesh object (must be pre-reordered with METIS-8)
            max_contacts: Maximum number of contact pairs
            metis_reordered: Must be True
            max_verts: Maximum number of vertices
            metis_n_parts: Actual number of METIS partitions
            precision: Float precision ('f32' or 'f64')
        """
        if not metis_reordered:
            raise ValueError("MASPreconditioner8Contact requires metis_reordered=True")

        # Initialize precision
        self.init_precision(precision)
        self.float_type = self.cfg.float_type

        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        if max_verts is None:
            max_verts = self.n_verts

        self.metis_reordered = True
        self.use_metis = True

        # BANKSIZE=8: use actual METIS partition count if provided
        if metis_n_parts is not None:
            self.n_parts = metis_n_parts
            print(f"[MAS-8-Contact] METIS pre-reordered mode: {self.n_parts} partitions (from METIS)")
        else:
            self.n_parts = (self.n_verts + BANKSIZE - 1) // BANKSIZE
            print(f"[MAS-8-Contact] METIS pre-reordered mode: {self.n_parts} partitions (computed)")

        # Compute hierarchy using actual n_parts for coarse levels
        self.level_num = min(MAX_LEVELS, self._compute_level_num_from_parts(self.n_parts))
        self.total_nodes_all_levels = self._compute_total_nodes_from_parts(
            self.n_verts, self.n_parts, self.level_num)
        self.total_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        # Level info
        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS)
        self._init_level_sizes()

        # Block matrices - much smaller than BANKSIZE=16
        # SYM_BLOCK_COUNT = 36 vs 136
        self.block_matrices = ti.Matrix.field(3, 3, dtype=self.float_type,
                                              shape=(self.total_blocks, SYM_BLOCK_COUNT))
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=self.float_type,
                                                   shape=(self.total_blocks, SYM_BLOCK_COUNT))

        # Full block matrices - 24x24 vs 48x48
        self.full_block_matrix = ti.field(dtype=self.float_type,
                                          shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=self.float_type,
                                           shape=(self.total_blocks, BLOCK_DOF, BLOCK_DOF))

        # Hierarchy mapping
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_nodes_all_levels)

        # Multi-level buffers
        self.multi_level_r = ti.Vector.field(3, dtype=self.float_type, shape=self.total_nodes_all_levels)
        self.multi_level_z = ti.Vector.field(3, dtype=self.float_type, shape=self.total_nodes_all_levels)

        # Cross-block storage for elastic terms
        max_cross_block_entries = self.n_cells * 6
        self.cross_block_row = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_col = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_val = ti.Matrix.field(3, 3, dtype=self.float_type, shape=max_cross_block_entries)
        self.cross_block_count = ti.field(dtype=ti.i32, shape=())
        self.max_cross_block_entries = max_cross_block_entries
        self.has_cross_block_data = False

        # Partition mapping for going_next
        self.sorted_to_partition = None

        # State flags
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

        # Initialize contact-specific storage
        self._init_contact_storage(max_contacts)

        print(f"[MAS-8-Contact] Initialized: {self.n_verts} verts, {self.level_num} levels, "
              f"{self.total_blocks} blocks, max_contacts={max_contacts}, precision={precision}")

    def _init_contact_storage(self, max_contacts: int):
        """Initialize contact-specific data structures."""
        self.max_contacts = max_contacts

        # Each contact pair has 4 vertices, which gives 16 vertex pairs (4x4).
        # For cross-block triplets, we need up to 16 entries per contact.
        max_contact_triplets = max_contacts * 8  # Upper bound estimate

        # Contact triplet storage for cross-block contacts (separate from elastic)
        self.contact_triplet_row = ti.field(dtype=ti.i32, shape=max_contact_triplets)
        self.contact_triplet_col = ti.field(dtype=ti.i32, shape=max_contact_triplets)
        self.contact_triplet_val = ti.Matrix.field(3, 3, dtype=self.float_type, shape=max_contact_triplets)
        self.contact_triplet_count = ti.field(dtype=ti.i32, shape=())
        self.max_contact_triplets = max_contact_triplets

        # Track whether contact data has been assembled
        self.has_contact_data = False

    # ========================================================================
    # Hierarchy level computation
    # ========================================================================

    def _compute_level_num_from_parts(self, n_parts: int) -> int:
        """Compute number of hierarchy levels using actual partition count."""
        levels = 2  # At least 2 levels (verts and partitions)
        size = n_parts
        while size > BANKSIZE and levels < MAX_LEVELS:
            size = (size + BANKSIZE - 1) // BANKSIZE
            levels += 1
        return levels

    def _compute_total_nodes_from_parts(self, n_verts: int, n_parts: int, level_num: int) -> int:
        """Compute total nodes using actual partition count."""
        total = n_verts + n_parts
        size = n_parts
        for _ in range(2, level_num):
            size = (size + BANKSIZE - 1) // BANKSIZE
            total += size
        return total

    def _init_level_sizes(self):
        """Initialize level size and offset arrays using actual n_parts."""
        sizes = []
        offsets = []

        # Level 0: n_verts
        sizes.append(self.n_verts)
        offsets.append(0)

        # Level 1: n_parts (from METIS)
        if self.level_num >= 2:
            sizes.append(self.n_parts)
            offsets.append(self.n_verts)

        # Level 2+: ceil(prev_level / BANKSIZE)
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

    # ========================================================================
    # Hierarchy Building
    # ========================================================================

    def _build_going_next(self, level_num: int):
        """Build going_next mapping for METIS pre-reordered mesh."""
        going_next_np = np.full(self.total_nodes_all_levels, -1, dtype=np.int32)

        if level_num <= 1:
            self.going_next.from_numpy(going_next_np)
            return

        # Level 0: vertices map to coarse based on their partition ID
        level_1_offset = int(self.level_size[1][1])

        if self.sorted_to_partition is not None:
            # Use actual partition mapping
            for i in range(self.n_verts):
                part_id = self.sorted_to_partition[i]
                coarse_idx = level_1_offset + part_id
                going_next_np[i] = coarse_idx
        else:
            # Fallback: assume partition_id = vertex_id // BANKSIZE
            for i in range(self.n_verts):
                part_id = i // BANKSIZE
                coarse_idx = level_1_offset + part_id
                going_next_np[i] = coarse_idx

        # Higher levels: sequential mapping
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
        """Build the multi-level hierarchy for METIS pre-reordered mesh."""
        # Sanity check
        if self.sorted_to_partition is not None:
            actual_n_parts = int(self.sorted_to_partition.max()) + 1
            if actual_n_parts != self.n_parts:
                print(f"[MAS-8-Contact] WARNING: n_parts mismatch: {self.n_parts} != {actual_n_parts}")

        self._build_going_next(self.level_num)
        print(f"[MAS-8-Contact] Hierarchy built: {self.level_num} levels, "
              f"L0={self.n_verts}, L1={self.n_parts}")

        self.hierarchy_built = True

    # ========================================================================
    # Matrix Assembly - Basic Operations
    # ========================================================================

    @ti.kernel
    def _clear_block_matrices(self):
        """Zero out all block matrices."""
        for block_id, sym_idx in ti.ndrange(self.total_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(self.float_type, 3, 3)

    @ti.kernel
    def _clear_cross_block_storage(self):
        """Clear cross-block storage counter."""
        self.cross_block_count[None] = 0

    @ti.kernel
    def _clear_contact_triplets(self):
        """Clear contact triplet storage."""
        self.contact_triplet_count[None] = 0

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.template()):
        """Add mass matrix to diagonal blocks."""
        for vert in self.mesh.verts:
            idx = vert.id
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            m = vert.m
            sym_idx = sym_index(lane_id, lane_id)
            for d in ti.static(range(3)):
                ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], m)

    @ti.kernel
    def _add_elastic_contribution_arap(self, mu: ti.template(), la: ti.template(), dt: ti.template(),
                                       dFdx_func: ti.template(), d2PsidF2_func: ti.template()):
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
            dFdx = dFdx_func(B)
            d2PsidF2 = d2PsidF2_func(F, mu, la)
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
                        # Same block: direct assembly
                        sub_block = ti.Matrix.zero(self.float_type, 3, 3)
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
                        # Cross-block: store in triplet format and propagate to coarse
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        sub_block = ti.Matrix.zero(self.float_type, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # Store cross-block entry
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

                        # Propagate to coarse levels
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

                    mat_norm = 0.0
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
        """Assemble Hessian contributions into block matrices (without contacts)."""
        from math_utils.matrix_util import compute_dFdx
        from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter

        self._clear_block_matrices()
        self._clear_cross_block_storage()

        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt,
                                            compute_dFdx, compute_d2PsidF2_ARAP_filter)

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True

    # ========================================================================
    # Contact Hessian Assembly
    # ========================================================================

    @ti.kernel
    def _add_contact_contribution(
        self,
        contact_pairs: ti.template(),
        n_contacts: ti.i32,
        dHat: ti.template(),
        kappa: ti.template(),
        dt: ti.template(),
        use_cubic_barrier: ti.template(),
    ):
        """
        Add contact Hessian contributions to block matrices.

        For each contact pair with 4 vertices:
        1. Compute barrier Hessian coefficients (para, para0)
        2. For each of 16 vertex pairs (i,j):
           - If same block: direct assembly to level-0 block
           - If cross block: store in triplet + propagate to coarse levels
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            ids_raw = pair.a
            ids = ti.Vector([ti.i32(ids_raw[0]), ti.i32(ids_raw[1]),
                            ti.i32(ids_raw[2]), ti.i32(ids_raw[3])])
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist >= dHat or dist < 1e-10:
                continue

            # Compute barrier Hessian coefficients
            bg = 0.0
            bH = 0.0
            if ti.static(use_cubic_barrier):
                bg = barrier_g_cubic(dist, dHat, kappa)
                bH = barrier_H_cubic(dist, dHat, kappa)
            else:
                bg = barrier_g_log(dist, dHat, kappa)
                bH = barrier_H_log(dist, dHat, kappa)

            dist2 = dist * dist
            para = bg / dist
            para0 = (bH - para) / dist2

            # Scale by dt^2 to match elastic Hessian scaling
            scale = dt * dt

            # Iterate over all 16 vertex pairs (4x4)
            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    vi = ids[i]
                    vj = ids[j]

                    coeff = scale * cord[i] * cord[j]

                    if vi >= 0 and vj >= 0 and vi < self.n_verts and vj < self.n_verts:
                        if ti.abs(coeff) >= 1e-12:
                            block_i = vi // BANKSIZE
                            block_j = vj // BANKSIZE
                            lane_i = vi % BANKSIZE
                            lane_j = vj % BANKSIZE

                            # Compute 3x3 sub-block: H_ij = coeff * (para0 * t @ t^T + para * I)
                            H_ij = ti.Matrix.zero(self.float_type, 3, 3)
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    H_ij[di, dj] = coeff * para0 * t[di] * t[dj]
                                    if di == dj:
                                        H_ij[di, dj] += coeff * para

                            if block_i == block_j:
                                # Same block: direct assembly
                                if lane_i <= lane_j:
                                    s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[di, dj])
                                else:
                                    s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[dj, di])
                            else:
                                # Cross-block: store in contact triplet format
                                triplet_idx = ti.atomic_add(self.contact_triplet_count[None], 1)
                                if triplet_idx < self.max_contact_triplets:
                                    if vi <= vj:
                                        self.contact_triplet_row[triplet_idx] = vi
                                        self.contact_triplet_col[triplet_idx] = vj
                                        self.contact_triplet_val[triplet_idx] = H_ij
                                    else:
                                        self.contact_triplet_row[triplet_idx] = vj
                                        self.contact_triplet_col[triplet_idx] = vi
                                        for di in ti.static(range(3)):
                                            for dj in ti.static(range(3)):
                                                self.contact_triplet_val[triplet_idx][di, dj] = H_ij[dj, di]

                                # Propagate to coarse levels
                                self._propagate_contact_to_coarse(vi, vj, H_ij)

    @ti.kernel
    def _add_contact_contribution_spd(
        self,
        contact_pairs: ti.template(),
        n_contacts: ti.i32,
        dHat: ti.template(),
        kappa: ti.template(),
        dt: ti.template(),
        mu: ti.template(),
        friction_eps: ti.template(),
        use_cubic_barrier: ti.template(),
        x0: ti.template(),
        x: ti.template(),
    ):
        """
        Add contact Hessian contributions using SPD formulation (PPF-Contact-Solver style).

        This version uses the guaranteed-PSD Hessian formula:
            H_contact = curvature * (e ⊗ e^T) / ||e||²
            H_friction = λ * P  (P = I - n⊗n^T projection matrix)

        Both components are PSD by construction, no eigenvalue clamping needed.
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            ids_raw = pair.a
            ids = ti.Vector([ti.i32(ids_raw[0]), ti.i32(ids_raw[1]),
                            ti.i32(ids_raw[2]), ti.i32(ids_raw[3])])
            dist = pair.b
            cord = pair.c
            t = pair.d

            if dist >= dHat or dist < 1e-10:
                continue

            # Compute contact edge vector
            e = ti.Vector.zero(self.float_type, 3)
            e0 = ti.Vector.zero(self.float_type, 3)
            for i in ti.static(range(4)):
                e += cord[i] * x[ids[i]]
                e0 += cord[i] * x0[ids[i]]

            # Relative displacement for friction
            dx = e - e0

            # Scale factor
            scale = dt * dt

            # Compute SPD 3x3 Hessian (barrier + optional friction)
            H_3x3 = ti.Matrix.zero(self.float_type, 3, 3)
            if ti.static(use_cubic_barrier):
                H_3x3 = compute_spd_contact_friction_hessian(
                    e, dx, dHat, kappa, mu, friction_eps, True, self.float_type
                )
            else:
                H_3x3 = compute_spd_contact_friction_hessian(
                    e, dx, dHat, kappa, mu, friction_eps, False, self.float_type
                )

            # Scale by dt^2
            H_3x3 = scale * H_3x3

            # Assemble to block matrices (16 vertex pairs)
            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    vi = ids[i]
                    vj = ids[j]

                    coeff = cord[i] * cord[j]

                    if vi >= 0 and vj >= 0 and vi < self.n_verts and vj < self.n_verts:
                        if ti.abs(coeff) >= 1e-12:
                            block_i = vi // BANKSIZE
                            block_j = vj // BANKSIZE
                            lane_i = vi % BANKSIZE
                            lane_j = vj % BANKSIZE

                            H_ij = coeff * H_3x3

                            if block_i == block_j:
                                if lane_i <= lane_j:
                                    s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[di, dj])
                                else:
                                    s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[dj, di])
                            else:
                                triplet_idx = ti.atomic_add(self.contact_triplet_count[None], 1)
                                if triplet_idx < self.max_contact_triplets:
                                    if vi <= vj:
                                        self.contact_triplet_row[triplet_idx] = vi
                                        self.contact_triplet_col[triplet_idx] = vj
                                        self.contact_triplet_val[triplet_idx] = H_ij
                                    else:
                                        self.contact_triplet_row[triplet_idx] = vj
                                        self.contact_triplet_col[triplet_idx] = vi
                                        for di in ti.static(range(3)):
                                            for dj in ti.static(range(3)):
                                                self.contact_triplet_val[triplet_idx][di, dj] = H_ij[dj, di]

                                self._propagate_contact_to_coarse(vi, vj, H_ij)

    @ti.func
    def _propagate_contact_to_coarse(self, vi: ti.i32, vj: ti.i32, H_ij: ti.template()):
        """Propagate cross-block contact term to coarse hierarchy levels."""
        curr_i = vi
        curr_j = vj

        for level in range(1, self.level_num):
            curr_i = self.going_next[curr_i]
            curr_j = self.going_next[curr_j]

            if curr_i < 0 or curr_j < 0:
                break

            block_i = curr_i // BANKSIZE
            block_j = curr_j // BANKSIZE

            if block_i == block_j:
                lane_i = curr_i % BANKSIZE
                lane_j = curr_j % BANKSIZE

                if lane_i <= lane_j:
                    s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                          H_ij[di, dj])
                else:
                    s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                          H_ij[dj, di])
                break

    def assemble_with_contacts(self, solver):
        """
        Assemble block matrices including contact Hessians.

        Args:
            solver: The IPC solver object with contact_pairs, dHat, kappa, dt
        """
        from math_utils.matrix_util import compute_dFdx
        from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter

        self._clear_block_matrices()
        self._clear_cross_block_storage()
        self._clear_contact_triplets()

        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt,
                                            compute_dFdx, compute_d2PsidF2_ARAP_filter)

        n_contacts = solver.n_contacts[None]
        if n_contacts > 0:
            use_cubic = getattr(solver, 'barrier_type', 'log') == 'cubic'
            self._add_contact_contribution(
                solver.contact_pairs,
                n_contacts,
                solver.dHat,
                solver.kappa,
                solver.dt,
                use_cubic
            )

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True
        self.has_contact_data = n_contacts > 0

        n_contact_triplets = int(self.contact_triplet_count[None])
        n_elastic_triplets = int(self.cross_block_count[None])
        if n_contacts > 0:
            print(f"[MAS-8-Contact] Assembled: {n_contacts} contacts, "
                  f"{n_contact_triplets} contact triplets, {n_elastic_triplets} elastic triplets")

    def assemble_with_contacts_spd(self, solver, mu: float = 0.0, friction_eps: float = 1e-4):
        """
        Assemble block matrices using SPD contact Hessian formulation.

        Args:
            solver: The IPC solver object with contact_pairs, dHat, kappa, dt
            mu: Friction coefficient (0 to disable friction)
            friction_eps: Minimum displacement for friction regularization
        """
        from math_utils.matrix_util import compute_dFdx
        from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter

        self._clear_block_matrices()
        self._clear_cross_block_storage()
        self._clear_contact_triplets()

        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt,
                                            compute_dFdx, compute_d2PsidF2_ARAP_filter)

        n_contacts = solver.n_contacts[None]
        if n_contacts > 0:
            use_cubic = getattr(solver, 'barrier_type', 'log') == 'cubic'
            self._add_contact_contribution_spd(
                solver.contact_pairs,
                n_contacts,
                solver.dHat,
                solver.kappa,
                solver.dt,
                mu,
                friction_eps,
                use_cubic,
                solver.mesh.verts.x_n,
                solver.mesh.verts.x,
            )

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True
        self.has_contact_data = n_contacts > 0

        n_contact_triplets = int(self.contact_triplet_count[None])
        n_elastic_triplets = int(self.cross_block_count[None])
        if n_contacts > 0:
            friction_str = f", μ={mu}" if mu > 0 else ""
            print(f"[MAS-8-Contact-SPD] Assembled: {n_contacts} contacts, "
                  f"{n_contact_triplets} contact triplets, {n_elastic_triplets} elastic triplets{friction_str}")

    # ========================================================================
    # Block Inversion - One-way Gauss-Jordan Elimination (Wu-2022-GBM)
    # ========================================================================

    @ti.kernel
    def _expand_sym_to_full(self):
        """Expand symmetric to full 24x24 matrices."""
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
    def _oneway_gauss_jordan_invert_blocks(self):
        """
        One-way Gauss-Jordan elimination for 24x24 block inversion.

        Adapted from Wu-2022-GBM paper for BANKSIZE=8.
        Produces L⁻¹ and D⁻¹ directly, then computes A⁻¹ = L⁻ᵀ D⁻¹ L⁻¹.
        """
        n_blocks = (self.total_nodes_all_levels + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Initialize L⁻¹ as identity
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    if i == j:
                        self.full_block_inverse[block_id, i, j] = 1.0
                    else:
                        self.full_block_inverse[block_id, i, j] = 0.0

            # Step 1: One-way Gauss-Jordan elimination
            for i in range(BLOCK_DOF):
                D_i = self.full_block_matrix[block_id, i, i]

                if ti.abs(D_i) < 1e-10:
                    D_i = 1e-6
                    self.full_block_matrix[block_id, i, i] = D_i

                D_inv_i = 1.0 / D_i

                for j in range(i + 1, BLOCK_DOF):
                    L_ji = self.full_block_matrix[block_id, j, i] * D_inv_i
                    self.full_block_matrix[block_id, j, i] = L_ji

                    for k in range(i + 1):
                        self.full_block_inverse[block_id, j, k] -= L_ji * self.full_block_inverse[block_id, i, k]

                for j in range(i + 1, BLOCK_DOF):
                    L_ji = self.full_block_matrix[block_id, j, i]
                    for k in range(i + 1, j + 1):
                        L_ki = self.full_block_matrix[block_id, k, i]
                        self.full_block_matrix[block_id, j, k] -= L_ji * L_ki * D_i

            # Step 2: Compute A⁻¹ = L⁻ᵀ D⁻¹ L⁻¹
            for i in range(BLOCK_DOF):
                D_inv_i = 1.0 / self.full_block_matrix[block_id, i, i]
                for j in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, i, j] *= D_inv_i

            # Compute L⁻ᵀ (D⁻¹ L⁻¹)
            for i in range(BLOCK_DOF):
                for j in range(i, BLOCK_DOF):
                    sum_val = 0.0
                    for k in range(j, BLOCK_DOF):
                        sum_val += self.full_block_inverse[block_id, k, i] * \
                                   self.full_block_inverse[block_id, k, j]
                    self.full_block_matrix[block_id, i, j] = sum_val
                    if i != j:
                        self.full_block_matrix[block_id, j, i] = sum_val

            # Copy result back to full_block_inverse
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, i, j] = self.full_block_matrix[block_id, i, j]

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
        """Invert all block matrices using one-way Gauss-Jordan elimination."""
        self._expand_sym_to_full()
        self._oneway_gauss_jordan_invert_blocks()
        self._copy_inverse_to_sym()
        self.matrices_inverted = True

    # ========================================================================
    # Apply Preconditioner - Optimized for BANKSIZE=8
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(self.float_type, 3)
            self.multi_level_z[i] = ti.Vector.zero(self.float_type, 3)

    @ti.kernel
    def _build_multi_level_r(self):
        """Build multi-level residual from gradient."""
        for vert in self.mesh.verts:
            idx = vert.id
            self.multi_level_r[idx] = vert.grad

        for vert in self.mesh.verts:
            idx = vert.id
            r = self.multi_level_r[idx]
            coarse_idx = self.going_next[idx]
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
    def _schwarz_local_solve_full(self):
        """
        Optimized symmetric matvec for BANKSIZE=8.

        Two-pass structure adapted from Wu-2022-GBM:
        Pass 1: Diagonal blocks
        Pass 2: Upper-triangular blocks with symmetric contribution
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Optimized symmetric matvec
        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                z0 = self.float_type(0.0)
                z1 = self.float_type(0.0)
                z2 = self.float_type(0.0)

                r_i = self.multi_level_r[idx_i]
                s_idx_diag = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_i
                inv_diag = self.inv_block_matrices[block_id, s_idx_diag]
                z0 += inv_diag[0, 0] * r_i[0] + inv_diag[0, 1] * r_i[1] + inv_diag[0, 2] * r_i[2]
                z1 += inv_diag[1, 0] * r_i[0] + inv_diag[1, 1] * r_i[1] + inv_diag[1, 2] * r_i[2]
                z2 += inv_diag[2, 0] * r_i[0] + inv_diag[2, 1] * r_i[1] + inv_diag[2, 2] * r_i[2]

                for lane_j in range(lane_i + 1, BANKSIZE):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]
                        s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                        inv_block = self.inv_block_matrices[block_id, s_idx]
                        z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                        z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                        z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]

                for lane_j in range(lane_i):
                    idx_j = block_id * BANKSIZE + lane_j
                    if idx_j < self.n_verts:
                        r_j = self.multi_level_r[idx_j]
                        s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                        inv_block = self.inv_block_matrices[block_id, s_idx]
                        z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                        z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                        z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2])

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
                    z0 = self.float_type(0.0)
                    z1 = self.float_type(0.0)
                    z2 = self.float_type(0.0)

                    r_i = self.multi_level_r[idx_i]
                    s_idx_diag = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_i
                    inv_diag = self.inv_block_matrices[block_id, s_idx_diag]
                    z0 += inv_diag[0, 0] * r_i[0] + inv_diag[0, 1] * r_i[1] + inv_diag[0, 2] * r_i[2]
                    z1 += inv_diag[1, 0] * r_i[0] + inv_diag[1, 1] * r_i[1] + inv_diag[1, 2] * r_i[2]
                    z2 += inv_diag[2, 0] * r_i[0] + inv_diag[2, 1] * r_i[1] + inv_diag[2, 2] * r_i[2]

                    for lane_j in range(lane_i + 1, BANKSIZE):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size_val:
                            r_j = self.multi_level_r[idx_j]
                            s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                            inv_block = self.inv_block_matrices[block_id, s_idx]
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[0, 1] * r_j[1] + inv_block[0, 2] * r_j[2]
                            z1 += inv_block[1, 0] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[1, 2] * r_j[2]
                            z2 += inv_block[2, 0] * r_j[0] + inv_block[2, 1] * r_j[1] + inv_block[2, 2] * r_j[2]

                    for lane_j in range(lane_i):
                        idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                        if idx_j < level_offset + level_size_val:
                            r_j = self.multi_level_r[idx_j]
                            s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                            inv_block = self.inv_block_matrices[block_id, s_idx]
                            z0 += inv_block[0, 0] * r_j[0] + inv_block[1, 0] * r_j[1] + inv_block[2, 0] * r_j[2]
                            z1 += inv_block[0, 1] * r_j[0] + inv_block[1, 1] * r_j[1] + inv_block[2, 1] * r_j[2]
                            z2 += inv_block[0, 2] * r_j[0] + inv_block[1, 2] * r_j[1] + inv_block[2, 2] * r_j[2]

                    self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2])

    @ti.kernel
    def _collect_final_z(self, level_num: ti.i32):
        """Collect z from all levels (prolongation)."""
        for vert in self.mesh.verts:
            idx = vert.id
            z_total = self.multi_level_z[idx]

            coarse_idx = self.going_next[idx]
            for _ in range(1, level_num):
                if coarse_idx >= 0:
                    z_coarse = self.multi_level_z[coarse_idx]
                    z_total += z_coarse
                    coarse_idx = self.going_next[coarse_idx]
                else:
                    break

            vert.z = z_total

    def apply(self):
        """Apply MAS preconditioner: z = P^{-1} @ grad"""
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._schwarz_local_solve_full()
        self._collect_final_z(self.level_num)

    # ========================================================================
    # Hessian Matrix-Vector Multiplication
    # ========================================================================

    @ti.kernel
    def _hessian_matvec_level0_block_diag(self, v: ti.template(), result: ti.template()):
        """Compute block-diagonal part of Hessian @ v."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
            idx_i = block_id * BANKSIZE + lane_i
            if idx_i < self.n_verts:
                r0 = self.float_type(0.0)
                r1 = self.float_type(0.0)
                r2 = self.float_type(0.0)

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

                result[idx_i] = ti.Vector([r0, r1, r2])

    @ti.kernel
    def _cross_block_spmv(self, v: ti.template(), result: ti.template(), n_triplets: ti.i32):
        """Compute cross-block contribution to H @ v."""
        for t in range(n_triplets):
            row = self.cross_block_row[t]
            col = self.cross_block_col[t]

            if row < 0 or col < 0:
                continue

            H_block = self.cross_block_val[t]
            v_row = v[row]
            v_col = v[col]

            r0_row = H_block[0, 0] * v_col[0] + H_block[0, 1] * v_col[1] + H_block[0, 2] * v_col[2]
            r1_row = H_block[1, 0] * v_col[0] + H_block[1, 1] * v_col[1] + H_block[1, 2] * v_col[2]
            r2_row = H_block[2, 0] * v_col[0] + H_block[2, 1] * v_col[1] + H_block[2, 2] * v_col[2]

            ti.atomic_add(result[row][0], r0_row)
            ti.atomic_add(result[row][1], r1_row)
            ti.atomic_add(result[row][2], r2_row)

            if row != col:
                r0_col = H_block[0, 0] * v_row[0] + H_block[1, 0] * v_row[1] + H_block[2, 0] * v_row[2]
                r1_col = H_block[0, 1] * v_row[0] + H_block[1, 1] * v_row[1] + H_block[2, 1] * v_row[2]
                r2_col = H_block[0, 2] * v_row[0] + H_block[1, 2] * v_row[1] + H_block[2, 2] * v_row[2]

                ti.atomic_add(result[col][0], r0_col)
                ti.atomic_add(result[col][1], r1_col)
                ti.atomic_add(result[col][2], r2_col)

    @ti.kernel
    def _contact_cross_block_spmv(self, v: ti.template(), result: ti.template(), n_triplets: ti.i32):
        """Compute contact cross-block contribution to H @ v."""
        for t in range(n_triplets):
            row = self.contact_triplet_row[t]
            col = self.contact_triplet_col[t]

            if row < 0 or col < 0:
                continue

            H_block = self.contact_triplet_val[t]
            v_row = v[row]
            v_col = v[col]

            r0_row = H_block[0, 0] * v_col[0] + H_block[0, 1] * v_col[1] + H_block[0, 2] * v_col[2]
            r1_row = H_block[1, 0] * v_col[0] + H_block[1, 1] * v_col[1] + H_block[1, 2] * v_col[2]
            r2_row = H_block[2, 0] * v_col[0] + H_block[2, 1] * v_col[1] + H_block[2, 2] * v_col[2]

            ti.atomic_add(result[row][0], r0_row)
            ti.atomic_add(result[row][1], r1_row)
            ti.atomic_add(result[row][2], r2_row)

            if row != col:
                r0_col = H_block[0, 0] * v_row[0] + H_block[1, 0] * v_row[1] + H_block[2, 0] * v_row[2]
                r1_col = H_block[0, 1] * v_row[0] + H_block[1, 1] * v_row[1] + H_block[2, 1] * v_row[2]
                r2_col = H_block[0, 2] * v_row[0] + H_block[1, 2] * v_row[1] + H_block[2, 2] * v_row[2]

                ti.atomic_add(result[col][0], r0_col)
                ti.atomic_add(result[col][1], r1_col)
                ti.atomic_add(result[col][2], r2_col)

    def hessian_matvec(self, v, result):
        """Compute result = H @ v exactly."""
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled")

        self._hessian_matvec_level0_block_diag(v, result)

        n_elastic_triplets = self.cross_block_count[None]
        if n_elastic_triplets > 0:
            self._cross_block_spmv(v, result, n_elastic_triplets)

        n_contact_triplets = self.contact_triplet_count[None]
        if n_contact_triplets > 0:
            self._contact_cross_block_spmv(v, result, n_contact_triplets)

    def hessian_matvec_exact(self, v, result):
        """Alias for hessian_matvec."""
        self.hessian_matvec(v, result)

    def hessian_matvec_with_contacts(self, v, result):
        """Alias for hessian_matvec."""
        self.hessian_matvec(v, result)

    # ========================================================================
    # High-level API
    # ========================================================================

    def rebuild(self, solver):
        """Full rebuild of preconditioner (without contacts)."""
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_block_matrices(solver)
        self.invert_block_matrices()

    def rebuild_with_contacts(self, solver):
        """Full rebuild of preconditioner with contact support."""
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_with_contacts(solver)
        self.invert_block_matrices()

    def rebuild_with_contacts_spd(self, solver, mu: float = 0.0, friction_eps: float = 1e-4):
        """Full rebuild using SPD contact Hessian formulation."""
        if not self.hierarchy_built:
            self.build_hierarchy()
        self.assemble_with_contacts_spd(solver, mu, friction_eps)
        self.invert_block_matrices()

    def get_cross_block_stats(self):
        """Get statistics about cross-block coupling storage."""
        n_triplets = int(self.cross_block_count[None])
        max_entries = self.max_cross_block_entries
        usage_pct = 100.0 * n_triplets / max_entries if max_entries > 0 else 0.0
        return {
            'n_triplets': n_triplets,
            'max_entries': max_entries,
            'usage_percent': usage_pct,
            'memory_mb': n_triplets * (4 + 4 + 9 * 4) / (1024 * 1024)
        }

    def get_contact_stats(self):
        """Get statistics about contact storage."""
        n_contact_triplets = int(self.contact_triplet_count[None])
        max_triplets = self.max_contact_triplets
        usage_pct = 100.0 * n_contact_triplets / max_triplets if max_triplets > 0 else 0.0
        return {
            'n_contact_triplets': n_contact_triplets,
            'max_contact_triplets': max_triplets,
            'usage_percent': usage_pct,
            'memory_mb': n_contact_triplets * (4 + 4 + 9 * 4) / (1024 * 1024)
        }

    # =========================================================================
    # Woodbury Update Support
    # =========================================================================

    def init_woodbury(self):
        """Initialize Woodbury update structures."""
        from .woodbury import WoodburySupport8
        self._woodbury = WoodburySupport8(self)
        self._woodbury.init_woodbury_structures()

    def save_base_state(self, solver):
        """Save current contact state as base for Woodbury updates."""
        if not hasattr(self, '_woodbury'):
            self.init_woodbury()
        self._woodbury.save_base_contact_state(solver)

    def woodbury_update(self, solver):
        """Compute Woodbury updates from contact changes."""
        if not hasattr(self, '_woodbury'):
            self.init_woodbury()
        self._woodbury.compute_woodbury_updates(solver)

    def apply_with_woodbury(self):
        """Apply preconditioner with Woodbury corrections."""
        if not hasattr(self, '_woodbury') or not self._woodbury.initialized:
            raise RuntimeError("Woodbury not initialized. Call init_woodbury() first.")
        self._woodbury.apply_with_woodbury()

    def should_use_woodbury(self, solver) -> bool:
        """Determine if Woodbury update is appropriate."""
        if not hasattr(self, '_woodbury') or not self._woodbury.initialized:
            return False

        n_base = len(self._woodbury.base_contacts)
        if n_base == 0:
            return False

        n_curr = solver.n_contacts[None]
        change_ratio = abs(n_curr - n_base) / max(n_base, 1)
        return change_ratio < 0.5

    def get_woodbury_stats(self):
        """Get statistics about Woodbury updates."""
        if not hasattr(self, '_woodbury') or not self._woodbury.initialized:
            return {
                'initialized': False,
                'n_base_contacts': 0,
                'n_updates_total': 0
            }

        return {
            'initialized': True,
            'n_base_contacts': len(self._woodbury.base_contacts),
            'n_updates_total': self._woodbury.get_num_updates_total(),
            'top_k': self._woodbury.top_k,
            'n_blocks': self._woodbury.n_blocks
        }
