"""
MAS Preconditioner ABD - Core implementation with ABD system integration.

This module extends MASPreconditionerSmall to support hybrid FEM-ABD simulation.
ABD bodies are treated as additional blocks with 12x12 Hessians.

Key design:
- FEM: 3D DOFs per vertex, 16 vertices per block (BANKSIZE), 48x48 block matrices
- ABD: 12D DOFs per body, 1 body per ABD block, 12x12 block matrices

The hybrid system's Hessian has block structure:
    H = [H_FEM    H_coupling^T]
        [H_coupling  H_ABD    ]

where:
- H_FEM: FEM elastic + inertia Hessian (n_fem_verts x 3) DOFs
- H_ABD: ABD inertia + shape Hessian (n_abd_bodies x 12) DOFs
- H_coupling: FEM-ABD coupling through contact (stored separately)

For MAS preconditioning, we treat ABD bodies as separate blocks and
use additive Schwarz at the ABD level.
"""

import taichi as ti
import numpy as np

# Constants
BANKSIZE = 16
SYM_BLOCK_COUNT = 136  # BANKSIZE * (BANKSIZE + 1) // 2
BLOCK_DOF = BANKSIZE * 3  # 48
ABD_DOF = 12  # ABD body has 12 DOFs
MAX_LEVELS = 6

# Import utilities
from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter
from ..abd_system import ABDSystem, ABDShapeEnergy, BodyBoundaryType


@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """Compute symmetric storage index for upper triangle."""
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c


@ti.data_oriented
class MASPreconditionerABD:
    """
    MAS Preconditioner with ABD system integration.

    This extends the standard FEM MAS preconditioner to handle hybrid
    FEM-ABD simulation. ABD bodies are treated as additional 12x12 blocks
    appended after the FEM blocks.

    Memory layout:
    - FEM blocks: indices 0 to n_fem_blocks-1 (48x48 each)
    - ABD blocks: indices n_fem_blocks to n_fem_blocks+n_abd_bodies-1 (12x12 each)

    For the hierarchy:
    - Level 0: FEM vertices + ABD bodies
    - Level 1+: Coarsened FEM + ABD (ABD doesn't coarsen further)
    """

    def __init__(self, mesh, abd_system: ABDSystem = None, max_verts: int = None):
        """
        Initialize MAS Preconditioner with ABD support.

        Args:
            mesh: MeshTaichi mesh object (must be created with METIS-reordered data)
            abd_system: ABDSystem instance (can be None for pure FEM)
            max_verts: Maximum number of vertices (default: mesh.verts.size)
        """
        self.mesh = mesh
        self.abd_system = abd_system
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        if max_verts is None:
            max_verts = self.n_verts

        # ABD info
        self.n_abd_bodies = abd_system.n_bodies if abd_system else 0
        self.max_abd_bodies = abd_system.max_bodies if abd_system else 64

        # FEM partition info
        self.n_fem_parts = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        self.use_metis = False
        self.metis_reordered = False

        # Compute hierarchy sizes
        self.level_num = min(MAX_LEVELS, self._compute_level_num(self.n_verts))
        self.total_fem_nodes = self._compute_total_nodes(self.n_verts, self.level_num)
        self.total_fem_blocks = (self.total_fem_nodes + BANKSIZE - 1) // BANKSIZE

        # Total blocks = FEM blocks + ABD bodies
        self.total_blocks = self.total_fem_blocks + self.max_abd_bodies

        # Level info: [size, offset] for each level
        self.level_size = ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS)
        self._init_level_sizes()

        # =======================================================================
        # FEM block matrices (48x48 symmetric storage)
        # =======================================================================
        self.block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                              shape=(self.total_fem_blocks, SYM_BLOCK_COUNT))
        self.inv_block_matrices = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                   shape=(self.total_fem_blocks, SYM_BLOCK_COUNT))

        # Full block matrices for IC(0) inversion
        self.full_block_matrix = ti.field(dtype=ti.f32,
                                          shape=(self.total_fem_blocks, BLOCK_DOF, BLOCK_DOF))
        self.full_block_inverse = ti.field(dtype=ti.f32,
                                           shape=(self.total_fem_blocks, BLOCK_DOF, BLOCK_DOF))

        # =======================================================================
        # ABD block matrices (12x12)
        # =======================================================================
        self.abd_block_matrices = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                   shape=self.max_abd_bodies)
        self.abd_block_inverse = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                  shape=self.max_abd_bodies)

        # =======================================================================
        # FEM-ABD coupling (for contact between FEM vertices and ABD bodies)
        # Stored as triplets: (fem_vert_id, abd_body_id, 3x12 block)
        # =======================================================================
        max_coupling_entries = self.n_verts * 2  # Upper bound
        self.coupling_fem_vert = ti.field(dtype=ti.i32, shape=max_coupling_entries)
        self.coupling_abd_body = ti.field(dtype=ti.i32, shape=max_coupling_entries)
        self.coupling_block = ti.Matrix.field(3, ABD_DOF, dtype=ti.f32,
                                               shape=max_coupling_entries)
        self.coupling_count = ti.field(dtype=ti.i32, shape=())
        self.max_coupling_entries = max_coupling_entries

        # =======================================================================
        # Hierarchy mapping
        # =======================================================================
        self.going_next = ti.field(dtype=ti.i32, shape=self.total_fem_nodes)

        # Multi-level buffers for FEM
        self.multi_level_r = ti.Vector.field(3, dtype=ti.f32, shape=self.total_fem_nodes)
        self.multi_level_z = ti.Vector.field(3, dtype=ti.f32, shape=self.total_fem_nodes)

        # ABD residual and preconditioned vectors (12D per body)
        self.abd_r = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=self.max_abd_bodies)
        self.abd_z = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=self.max_abd_bodies)

        # =======================================================================
        # Cross-block coupling storage (triplet format for exact Hessian matvec)
        # =======================================================================
        max_cross_block_entries = self.n_cells * 6  # Upper bound
        self.cross_block_row = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_col = ti.field(dtype=ti.i32, shape=max_cross_block_entries)
        self.cross_block_val = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_cross_block_entries)
        self.cross_block_count = ti.field(dtype=ti.i32, shape=())
        self.max_cross_block_entries = max_cross_block_entries
        self.has_cross_block_data = False

        # State flags
        self.hierarchy_built = False
        self.matrices_assembled = False
        self.matrices_inverted = False

        print(f"[MAS-ABD] Initialized: {self.n_verts} FEM verts, {self.n_abd_bodies} ABD bodies, "
              f"{self.level_num} levels, {self.total_fem_blocks} FEM blocks")

    def update_abd_system(self, abd_system: ABDSystem):
        """Update the ABD system reference (e.g., when bodies are added)."""
        self.abd_system = abd_system
        self.n_abd_bodies = abd_system.n_bodies if abd_system else 0
        print(f"[MAS-ABD] Updated ABD system: {self.n_abd_bodies} bodies")

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
        """Build going_next mapping for FEM hierarchy."""
        if level_num == 1:
            for i in range(self.n_verts):
                self.going_next[i] = -1
        else:
            # Level 0: fine vertices map to coarse level
            for i in range(self.n_verts):
                coarse_idx = self.level_size[1][1] + i // BANKSIZE
                self.going_next[i] = coarse_idx

            # Higher levels
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
        """Build the multi-level hierarchy for FEM."""
        self._build_going_next(self.level_num)
        print(f"[MAS-ABD] Hierarchy built: {self.level_num} levels")
        self.hierarchy_built = True

    # ========================================================================
    # Matrix Assembly
    # ========================================================================

    @ti.kernel
    def _clear_block_matrices(self):
        """Zero out all FEM block matrices."""
        for block_id, sym_idx in ti.ndrange(self.total_fem_blocks, SYM_BLOCK_COUNT):
            self.block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _clear_abd_block_matrices(self):
        """Zero out all ABD block matrices."""
        for body_id in range(self.max_abd_bodies):
            self.abd_block_matrices[body_id] = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

    @ti.kernel
    def _clear_coupling_storage(self):
        """Clear coupling triplet storage."""
        self.coupling_count[None] = 0

    @ti.kernel
    def _clear_cross_block_storage(self):
        """Clear cross-block triplet storage counter."""
        self.cross_block_count[None] = 0

    @ti.kernel
    def _add_inertia_contribution(self, dt: ti.f32):
        """Add FEM mass matrix to diagonal blocks."""
        for vert in self.mesh.verts:
            idx = vert.id
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            m = vert.m
            sym_idx = sym_index(lane_id, lane_id)
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
                        # Cross-warp: propagate to coarse level and store triplet
                        vert_i = v_ids[i]
                        vert_j = v_ids[j]

                        sub_block = ti.Matrix.zero(ti.f32, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                sub_block[di, dj] = H_e[i * 3 + di, j * 3 + dj]

                        # Store cross-block entry
                        triplet_idx = ti.atomic_add(self.cross_block_count[None], 1)
                        if triplet_idx < self.max_cross_block_entries:
                            if v_ids[i] <= v_ids[j]:
                                self.cross_block_row[triplet_idx] = v_ids[i]
                                self.cross_block_col[triplet_idx] = v_ids[j]
                                self.cross_block_val[triplet_idx] = sub_block
                            else:
                                self.cross_block_row[triplet_idx] = v_ids[j]
                                self.cross_block_col[triplet_idx] = v_ids[i]
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

    def _assemble_abd_blocks(self, dt: float):
        """
        Assemble ABD body Hessian blocks (12x12).

        ABD Hessian = M_abd + dt^2 * H_shape

        where:
        - M_abd: 12x12 mass matrix
        - H_shape: 12x12 shape energy Hessian (affects only affine DOFs 3-11)
        """
        if self.abd_system is None or self.n_abd_bodies == 0:
            return

        self._clear_abd_block_matrices()
        self._assemble_abd_inertia_and_shape(dt)

    @ti.kernel
    def _assemble_abd_inertia_and_shape(self, dt: ti.f32):
        """Assemble ABD inertia and shape energy Hessian."""
        for body_id in range(self.n_abd_bodies):
            if self.abd_system.boundary_type[body_id] == BodyBoundaryType.FIXED:
                # Fixed body: set identity block
                for i in ti.static(range(ABD_DOF)):
                    self.abd_block_matrices[body_id][i, i] = 1.0
            else:
                # Add mass matrix
                M = self.abd_system.abd_mass[body_id]
                for i in ti.static(range(ABD_DOF)):
                    for j in ti.static(range(ABD_DOF)):
                        self.abd_block_matrices[body_id][i, j] = M[i, j]

                # Add shape energy Hessian (dt^2 * kappa * v * H_shape)
                q = self.abd_system.q[body_id]
                kappa = self.abd_system.body_kappa[body_id]
                volume = self.abd_system.body_volume[body_id]
                scale = dt * dt * kappa * volume

                # Compute 9x9 shape Hessian
                H_shape = ABDShapeEnergy.compute_hessian(q)
                H_shape = ABDShapeEnergy.make_positive_definite(H_shape)

                # Add to affine DOFs (3-11)
                for i in ti.static(range(9)):
                    for j in ti.static(range(9)):
                        self.abd_block_matrices[body_id][3 + i, 3 + j] += scale * H_shape[i, j]

    def assemble_block_matrices(self, solver):
        """Assemble all Hessian contributions into block matrices."""
        # FEM blocks
        self._clear_block_matrices()
        self._clear_cross_block_storage()

        self._add_inertia_contribution(solver.dt)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)

        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        # ABD blocks
        self._assemble_abd_blocks(solver.dt)

        self.matrices_assembled = True
        self.has_cross_block_data = True

    # ========================================================================
    # Block Inversion
    # ========================================================================

    @ti.kernel
    def _expand_sym_to_full(self):
        """Expand symmetric FEM block matrices to full 48x48 dense matrices."""
        n_blocks = (self.total_fem_nodes + BANKSIZE - 1) // BANKSIZE

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
        """IC(0) factorization and approximate inversion for FEM blocks."""
        n_blocks = (self.total_fem_nodes + BANKSIZE - 1) // BANKSIZE
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
        n_blocks = (self.total_fem_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    s_idx = sym_index(row, col)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, s_idx][di, dj] = \
                                self.full_block_inverse[block_id, row * 3 + di, col * 3 + dj]

    @ti.kernel
    def _invert_abd_blocks(self):
        """Invert ABD 12x12 blocks directly using Cholesky."""
        for body_id in range(self.n_abd_bodies):
            H = self.abd_block_matrices[body_id]

            # Simple Cholesky decomposition for 12x12 SPD matrix
            L = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

            for i in ti.static(range(ABD_DOF)):
                for j in range(i + 1):
                    sum_val = H[i, j]
                    for k in range(j):
                        sum_val -= L[i, k] * L[j, k]

                    if i == j:
                        if sum_val > 1e-12:
                            L[i, j] = ti.sqrt(sum_val)
                        else:
                            L[i, j] = 1e-3
                    else:
                        if ti.abs(L[j, j]) > 1e-12:
                            L[i, j] = sum_val / L[j, j]
                        else:
                            L[i, j] = 0.0

            # Inversion via forward/backward substitution
            H_inv = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

            for col in ti.static(range(ABD_DOF)):
                # Forward substitution: L @ y = e_col
                y = ti.Vector.zero(ti.f32, ABD_DOF)
                for i in ti.static(range(ABD_DOF)):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= L[i, k] * y[k]
                    if ti.abs(L[i, i]) > 1e-12:
                        y[i] = sum_val / L[i, i]

                # Backward substitution: L^T @ x = y
                x = ti.Vector.zero(ti.f32, ABD_DOF)
                for i_rev in ti.static(range(ABD_DOF)):
                    i = ABD_DOF - 1 - i_rev
                    sum_val = y[i]
                    for k in range(i + 1, ABD_DOF):
                        sum_val -= L[k, i] * x[k]
                    if ti.abs(L[i, i]) > 1e-12:
                        x[i] = sum_val / L[i, i]

                for i in ti.static(range(ABD_DOF)):
                    H_inv[i, col] = x[i]

            self.abd_block_inverse[body_id] = H_inv

    def invert_block_matrices(self):
        """Invert all block matrices (FEM + ABD)."""
        # FEM: IC(0)
        self._expand_sym_to_full()
        self._incomplete_cholesky_invert_blocks()
        self._copy_inverse_to_sym()

        # ABD: Direct Cholesky
        if self.n_abd_bodies > 0:
            self._invert_abd_blocks()

        self.matrices_inverted = True

    # ========================================================================
    # Apply Preconditioner
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level buffers."""
        for i in range(self.total_fem_nodes):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _clear_abd_buffers(self):
        """Clear ABD buffers."""
        for i in range(self.max_abd_bodies):
            self.abd_r[i] = ti.Vector.zero(ti.f32, ABD_DOF)
            self.abd_z[i] = ti.Vector.zero(ti.f32, ABD_DOF)

    @ti.kernel
    def _build_multi_level_r(self):
        """Build multi-level residual from gradient."""
        # Level 0: copy from mesh gradient
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
    def _build_abd_r(self):
        """Build ABD residual from ABD gradient."""
        for body_id in range(self.n_abd_bodies):
            self.abd_r[body_id] = self.abd_system.grad_q[body_id]

    @ti.kernel
    def _schwarz_local_solve_banded(self):
        """Banded solve for FEM blocks."""
        NODE_BANDWIDTH = 2
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Banded block matvec
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
    def _abd_local_solve(self):
        """Apply ABD block inverse: z = H^{-1} @ r."""
        for body_id in range(self.n_abd_bodies):
            r = self.abd_r[body_id]
            H_inv = self.abd_block_inverse[body_id]

            z = ti.Vector.zero(ti.f32, ABD_DOF)
            for i in ti.static(range(ABD_DOF)):
                for j in ti.static(range(ABD_DOF)):
                    z[i] += H_inv[i, j] * r[j]

            self.abd_z[body_id] = z

    @ti.kernel
    def _collect_final_z(self, level_num: ti.i32):
        """Collect z from all FEM levels (prolongation)."""
        for vert in self.mesh.verts:
            idx = vert.id
            z_total = ti.cast(self.multi_level_z[idx], ti.f32)

            coarse_idx = self.going_next[idx]
            for _ in range(1, level_num):
                if coarse_idx >= 0:
                    z_coarse = self.multi_level_z[coarse_idx]
                    z_total += ti.cast(z_coarse, ti.f32)
                    coarse_idx = self.going_next[coarse_idx]
                else:
                    break

            vert.z = z_total

    @ti.kernel
    def _write_abd_z_to_system(self):
        """Write ABD preconditioned result back to ABD system."""
        for body_id in range(self.n_abd_bodies):
            self.abd_system.dq[body_id] = self.abd_z[body_id]

    def apply(self):
        """
        Apply MAS preconditioner to both FEM and ABD systems.

        For FEM: z = P_FEM @ grad (multi-level additive Schwarz)
        For ABD: dq = H_ABD^{-1} @ grad_q (direct block solve)
        """
        # FEM part
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._schwarz_local_solve_banded()
        self._collect_final_z(self.level_num)

        # ABD part
        if self.n_abd_bodies > 0:
            self._clear_abd_buffers()
            self._build_abd_r()
            self._abd_local_solve()
            self._write_abd_z_to_system()

    # ========================================================================
    # Hessian Matrix-Vector Product
    # ========================================================================

    @ti.kernel
    def _hessian_matvec_level0_block_diag(self, v: ti.template(), result: ti.template()):
        """Compute block-diagonal part of FEM Hessian @ v."""
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

            # H @ v contribution
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

    def hessian_matvec(self, v: ti.template(), result: ti.template()):
        """
        Compute FEM Hessian @ v exactly.

        Args:
            v: Input vector field (n_verts x 3)
            result: Output vector field (n_verts x 3)
        """
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled. Call assemble_block_matrices first.")

        self._hessian_matvec_level0_block_diag(v, result)

        n_triplets = self.cross_block_count[None]
        if n_triplets > 0:
            self._cross_block_spmv(v, result, n_triplets)

    @ti.kernel
    def _abd_hessian_matvec(self, v: ti.template(), result: ti.template()):
        """Compute ABD Hessian @ v: result = H_ABD @ v."""
        for body_id in range(self.n_abd_bodies):
            H = self.abd_block_matrices[body_id]
            v_body = v[body_id]

            r = ti.Vector.zero(ti.f32, ABD_DOF)
            for i in ti.static(range(ABD_DOF)):
                for j in ti.static(range(ABD_DOF)):
                    r[i] += H[i, j] * v_body[j]

            result[body_id] = r

    def abd_hessian_matvec(self, v: ti.template(), result: ti.template()):
        """
        Compute ABD Hessian @ v.

        Args:
            v: Input vector field (max_abd_bodies x 12)
            result: Output vector field (max_abd_bodies x 12)
        """
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled. Call assemble_block_matrices first.")

        self._abd_hessian_matvec(v, result)

    # ========================================================================
    # High-level API
    # ========================================================================

    def rebuild(self, solver):
        """Full rebuild of preconditioner."""
        if not self.hierarchy_built:
            self.build_hierarchy()

        # Update ABD body count
        if self.abd_system:
            self.n_abd_bodies = self.abd_system.n_bodies

        self.assemble_block_matrices(solver)
        self.invert_block_matrices()

    def get_stats(self):
        """Get preconditioner statistics."""
        return {
            'n_fem_verts': self.n_verts,
            'n_abd_bodies': self.n_abd_bodies,
            'n_fem_blocks': self.total_fem_blocks,
            'level_num': self.level_num,
            'cross_block_triplets': int(self.cross_block_count[None]),
        }
