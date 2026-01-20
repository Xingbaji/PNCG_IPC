"""
MAS Preconditioner Contact - Core implementation with IPC contact support.

Extends MASPreconditionerSmall to integrate contact Hessian into block matrices.
Follows the pattern from Stiff-GIPC reference implementation.

Contact Hessian formula:
    H_contact = para0 * (dtdx @ dtdx^T) + para * I_12x12

where:
    para = barrier_g / dist
    para0 = (barrier_H - para) / dist^2
    dtdx[i*3 + j] = cord[i] * t[j]  (12-vector for 4 vertices x 3D)
"""

import taichi as ti

# Import base class
from algorithm.mas_preconditioner_small.core import (
    MASPreconditionerSmall,
    BANKSIZE,
    SYM_BLOCK_COUNT,
    sym_index,
)

# Import barrier functions from contact assembly module
from .contact_assembly import (
    barrier_g_log,
    barrier_H_log,
    barrier_g_cubic,
    barrier_H_cubic,
)


@ti.data_oriented
class MASPreconditionerContact(MASPreconditionerSmall):
    """
    MAS Preconditioner with IPC Contact Hessian support.

    Extends MASPreconditionerSmall to handle dynamic contact Hessians.
    Contact pairs change every frame/iteration, requiring rebuild.

    Key features:
    - Same-block contacts: Direct assembly to level-0 blocks
    - Cross-block contacts: Triplet storage for exact matvec + coarse propagation
    - SPD preservation: Contact Hessian structure is naturally PSD when properly scaled

    Parameters:
        mesh: MeshTaichi mesh object (must be METIS pre-reordered)
        max_contacts: Maximum number of contact pairs (default: 2^18)
        metis_reordered: Must be True (inherited from parent)
    """

    def __init__(self, mesh, max_contacts: int = 2**18, **kwargs):
        # Initialize base class
        super().__init__(mesh, **kwargs)

        # Initialize contact-specific storage
        self._init_contact_storage(max_contacts)

        print(f"[MAS-Contact] Initialized with max_contacts={max_contacts}")

    def _init_contact_storage(self, max_contacts: int):
        """Initialize contact-specific data structures."""
        self.max_contacts = max_contacts

        # Each contact pair has 4 vertices, which gives 16 vertex pairs (4x4).
        # For cross-block triplets, we need up to 16 entries per contact.
        # However, many contacts will have same-block pairs, so estimate lower.
        max_contact_triplets = max_contacts * 8  # Upper bound estimate

        # Contact triplet storage for cross-block contacts (separate from elastic)
        self.contact_triplet_row = ti.field(dtype=ti.i32, shape=max_contact_triplets)
        self.contact_triplet_col = ti.field(dtype=ti.i32, shape=max_contact_triplets)
        self.contact_triplet_val = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_contact_triplets)
        self.contact_triplet_count = ti.field(dtype=ti.i32, shape=())
        self.max_contact_triplets = max_contact_triplets

        # Track whether contact data has been assembled
        self.has_contact_data = False

    @ti.kernel
    def _clear_contact_triplets(self):
        """Clear contact triplet storage."""
        self.contact_triplet_count[None] = 0

    @ti.kernel
    def _add_contact_contribution(
        self,
        contact_pairs: ti.template(),
        n_contacts: ti.i32,
        dHat: ti.f32,
        kappa: ti.f32,
        dt: ti.f32,
        use_cubic_barrier: ti.template(),
    ):
        """
        Add contact Hessian contributions to block matrices.

        For each contact pair with 4 vertices:
        1. Compute barrier Hessian coefficients (para, para0)
        2. For each of 16 vertex pairs (i,j):
           - If same block: direct assembly to level-0 block
           - If cross block: store in triplet + propagate to coarse levels

        Contact Hessian structure:
            H_ij = cord[i] * cord[j] * (para0 * t @ t^T + para * I_3x3)

        where:
            para = barrier_g / dist
            para0 = (barrier_H - para) / dist^2
            t = contact direction (normalized)
        """
        for idx in range(n_contacts):
            pair = contact_pairs[idx]
            # ids: 4 vertex IDs as uint32 vector
            ids_raw = pair.a
            ids = ti.Vector([ti.i32(ids_raw[0]), ti.i32(ids_raw[1]),
                            ti.i32(ids_raw[2]), ti.i32(ids_raw[3])])
            dist = pair.b      # distance (scalar)
            cord = pair.c      # barycentric coords (float4)
            t = pair.d         # direction vector (float3)

            # Skip if distance is beyond threshold
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

                    # Compute coefficient early to avoid branching
                    coeff = scale * cord[i] * cord[j]

                    # Process valid vertex pairs with non-negligible coefficient
                    # Using nested if instead of continue for Taichi static loop compatibility
                    if vi >= 0 and vj >= 0 and vi < self.n_verts and vj < self.n_verts:
                        if ti.abs(coeff) >= 1e-12:
                            block_i = vi // BANKSIZE
                            block_j = vj // BANKSIZE
                            lane_i = vi % BANKSIZE
                            lane_j = vj % BANKSIZE

                            # Compute 3x3 sub-block: H_ij = coeff * (para0 * t @ t^T + para * I)
                            H_ij = ti.Matrix.zero(ti.f32, 3, 3)
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    H_ij[di, dj] = coeff * para0 * t[di] * t[dj]
                                    if di == dj:
                                        H_ij[di, dj] += coeff * para

                            if block_i == block_j:
                                # Same block: direct assembly to level-0 block matrix
                                # Use symmetric storage (upper triangle only)
                                if lane_i <= lane_j:
                                    s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[di, dj])
                                else:
                                    # Transpose for lower triangle
                                    s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            ti.atomic_add(self.block_matrices[block_i, s_idx][di, dj],
                                                          H_ij[dj, di])
                            else:
                                # Cross-block: store in contact triplet format
                                # Store upper triangle only (vi <= vj)
                                triplet_idx = ti.atomic_add(self.contact_triplet_count[None], 1)
                                if triplet_idx < self.max_contact_triplets:
                                    if vi <= vj:
                                        self.contact_triplet_row[triplet_idx] = vi
                                        self.contact_triplet_col[triplet_idx] = vj
                                        self.contact_triplet_val[triplet_idx] = H_ij
                                    else:
                                        # Transpose for lower triangle
                                        self.contact_triplet_row[triplet_idx] = vj
                                        self.contact_triplet_col[triplet_idx] = vi
                                        for di in ti.static(range(3)):
                                            for dj in ti.static(range(3)):
                                                self.contact_triplet_val[triplet_idx][di, dj] = H_ij[dj, di]

                                # Propagate to coarse levels for preconditioning
                                self._propagate_contact_to_coarse(vi, vj, H_ij)

    @ti.func
    def _propagate_contact_to_coarse(self, vi: ti.i32, vj: ti.i32, H_ij: ti.template()):
        """
        Propagate cross-block contact term to coarse hierarchy levels.

        Following Stiff-GIPC pattern: traverse hierarchy via going_next
        until both vertices are in the same block, then aggregate there.
        """
        curr_i = vi
        curr_j = vj

        for level in range(1, self.level_num):
            # Move to coarse level via going_next
            curr_i = self.going_next[curr_i]
            curr_j = self.going_next[curr_j]

            if curr_i < 0 or curr_j < 0:
                break

            block_i = curr_i // BANKSIZE
            block_j = curr_j // BANKSIZE

            if block_i == block_j:
                # Found common block at this level - aggregate here
                lane_i = curr_i % BANKSIZE
                lane_j = curr_j % BANKSIZE

                # Add to coarse block (symmetric storage)
                # Note: H_ij = para0 * t⊗t^T + para * I is already symmetric,
                # so no need for separate transpose handling on diagonal entries
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

        This is the main assembly entry point when contacts are present.
        Handles both elastic and contact contributions.

        Args:
            solver: The IPC solver object with contact_pairs, dHat, kappa, dt
        """
        # 1. Clear all storage
        self._clear_block_matrices()
        self._clear_cross_block_storage()
        self._clear_contact_triplets()

        # 2. Add inertia contribution
        self._add_inertia_contribution(solver.dt)

        # 3. Add elastic contribution (from parent class)
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)

        # 4. Add contact contributions
        n_contacts = solver.n_contacts[None]
        if n_contacts > 0:
            # Determine barrier type
            use_cubic = getattr(solver, 'barrier_type', 'log') == 'cubic'
            self._add_contact_contribution(
                solver.contact_pairs,
                n_contacts,
                solver.dHat,
                solver.kappa,
                solver.dt,
                use_cubic
            )

        # 5. Aggregate to coarse levels (for elastic cross-block terms)
        # Note: Contact cross-block terms already propagated in _add_contact_contribution
        if self.hierarchy_built and self.level_num > 1:
            self._aggregate_fine_to_coarse()

        self.matrices_assembled = True
        self.has_cross_block_data = True
        self.has_contact_data = n_contacts > 0

        # Print statistics
        n_contact_triplets = int(self.contact_triplet_count[None])
        n_elastic_triplets = int(self.cross_block_count[None])
        if n_contacts > 0:
            print(f"[MAS-Contact] Assembled: {n_contacts} contacts, "
                  f"{n_contact_triplets} contact triplets, {n_elastic_triplets} elastic triplets")

    @ti.kernel
    def _contact_cross_block_spmv(self, v: ti.template(), result: ti.template(), n_triplets: ti.i32):
        """
        Compute contact cross-block contribution to H @ v.

        For each triplet (row, col, H_block):
          result[row] += H_block @ v[col]
          result[col] += H_block^T @ v[row]  (symmetric matrix)
        """
        for t in range(n_triplets):
            row = self.contact_triplet_row[t]
            col = self.contact_triplet_col[t]

            if row < 0 or col < 0:
                continue

            H_block = self.contact_triplet_val[t]
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

    def hessian_matvec_with_contacts(self, v: ti.template(), result: ti.template()):
        """
        Compute exact H @ v including contact contributions.

        result = (H_elastic + H_inertia + H_contact) @ v

        Args:
            v: Input vector field (n_verts, 3)
            result: Output vector field (n_verts, 3)
        """
        if not self.matrices_assembled:
            raise RuntimeError("Matrices not assembled. Call assemble_with_contacts first.")

        # Step 1: Compute level 0 block-diagonal contribution
        self._hessian_matvec_level0_block_diag(v, result)

        # Step 2: Add elastic cross-block contributions from triplet storage
        n_elastic_triplets = self.cross_block_count[None]
        if n_elastic_triplets > 0:
            self._cross_block_spmv(v, result, n_elastic_triplets)

        # Step 3: Add contact cross-block contributions
        n_contact_triplets = self.contact_triplet_count[None]
        if n_contact_triplets > 0:
            self._contact_cross_block_spmv(v, result, n_contact_triplets)

    def rebuild_with_contacts(self, solver):
        """
        Full rebuild of preconditioner with contact support.

        Call this at each Newton iteration when contacts change.

        Args:
            solver: The IPC solver object with contact data
        """
        if not self.hierarchy_built:
            self.build_hierarchy()

        self.assemble_with_contacts(solver)
        self.invert_block_matrices()

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
