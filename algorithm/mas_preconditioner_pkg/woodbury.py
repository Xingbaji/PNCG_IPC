"""
Woodbury Module: Sparse-Input Woodbury low-rank updates.

This module implements the Woodbury identity for efficiently updating
the preconditioner when contact conditions change slightly.

Reference: Section 3.1 of MAS-PNCG paper
"""

import taichi as ti
from .constants import BANKSIZE


class WoodburyMixin:
    """
    Mixin class providing Sparse-Input Woodbury update functionality.

    The Woodbury formula allows efficient rank-k updates to the inverse:
    (A + U*S*U^T)^{-1} = A^{-1} - A^{-1}*U*(S^{-1} + U^T*A^{-1}*U)^{-1}*U^T*A^{-1}

    This is used to update the preconditioner when contact conditions change
    without full recomputation.

    Required attributes from main class:
        - n_verts: int
        - level_num: int
        - level_size: ti.Vector.field(2, i32)
        - multi_level_r: ti.Vector.field(3, f32)
        - multi_level_z: ti.Vector.field(3, f32)
        - inv_block_matrices: ti.field(mat3x3, f32)
        - _sym_index: function for symmetric indexing
    """

    def init_woodbury_structures(self):
        """Initialize data structures for Woodbury updates."""
        self.top_k = 8  # Maximum number of rank-1 updates per subdomain
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # U matrix: update vectors (n_blocks, top_k, BANKSIZE*3)
        self.woodbury_U = ti.field(dtype=ti.f32,
                                    shape=(n_blocks, self.top_k, BANKSIZE * 3))
        # Delta S: stiffness changes (n_blocks, top_k)
        self.woodbury_delta_S = ti.field(dtype=ti.f32,
                                          shape=(n_blocks, self.top_k))
        # Number of updates per subdomain
        self.woodbury_num_updates = ti.field(dtype=ti.i32, shape=n_blocks)
        # B*U precomputed product
        self.BU = ti.field(dtype=ti.f32,
                           shape=(n_blocks, BANKSIZE * 3, self.top_k))
        # Capacitance matrix: I + U^T * B * U
        self.capacitance_matrix = ti.field(dtype=ti.f32,
                                            shape=(n_blocks, self.top_k, self.top_k))
        # Base contact state dictionary
        self.base_contacts = {}
        self.woodbury_initialized = True
        print(f"[Woodbury] Initialized for {n_blocks} subdomains")

    def save_base_contact_state(self, solver):
        """
        Save current contact state as base for Woodbury updates.
        Uses compact array storage (P0 optimization) when available.

        Args:
            solver: The PNCG solver containing contact information
        """
        self.base_contacts = {}

        # Try compact array first (P0 optimization)
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                contact_pairs_np = solver.contact_pairs.to_numpy()[:n_contacts]
                for i in range(n_contacts):
                    pair = contact_pairs_np[i]
                    ids = tuple(int(x) for x in pair['a'])
                    dist = float(pair['b'])
                    normal = tuple(float(x) for x in pair['d'])
                    cord = tuple(float(x) for x in pair['c'])
                    stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
                    contact_key = tuple(sorted(ids))
                    self.base_contacts[contact_key] = {
                        'stiffness': stiffness, 'normal': normal,
                        'dist': dist, 'ids': ids, 'cord': cord
                    }
                return

        # Fallback to bitmasked cid (legacy)
        try:
            cid_keys = solver.cid.keys_numpy()
        except (AttributeError, TypeError, RuntimeError):
            return

        for key in cid_keys:
            k, j = key
            pair_data = solver.cid[k, j]
            ids = tuple(int(x) for x in pair_data.a)
            dist = float(pair_data.b)
            normal = tuple(float(x) for x in pair_data.d)
            cord = tuple(float(x) for x in pair_data.c)
            stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
            contact_key = tuple(sorted(ids))
            self.base_contacts[contact_key] = {
                'stiffness': stiffness, 'normal': normal,
                'dist': dist, 'ids': ids, 'cord': cord
            }

    def _compute_barrier_stiffness(self, d, dHat, kappa):
        """
        Compute barrier Hessian k = b''(d) for IPC log barrier.

        IPC uses: b(d) = -kappa * (d - dHat)^2 * log(d / dHat)  for d < dHat

        First derivative:
        b'(d) = -kappa * [2(d - dHat) * log(d/dHat) + (d - dHat)^2 / d]

        Second derivative:
        b''(d) = -kappa * [2*log(d/dHat) + 2(d-dHat)/d + 2(d-dHat)/d - (d-dHat)^2/d^2]
               = -kappa * [2*log(d/dHat) + 4(d-dHat)/d - (d-dHat)^2/d^2]

        For Gauss-Newton approximation, we use the positive semi-definite part:
        k = kappa * [(d-dHat)/d]^2 * (2*d/dHat - 2 + dHat/d)

        Simplified approximation (matching reference):
        k = kappa * (dHat - d)^2 / (d^2 * dHat) * (2*d + dHat)
        """
        if d >= dHat or d <= 0:
            return 0.0

        # Use the Gauss-Newton approximation for SPD guarantee
        # This is the second derivative of the barrier, taking only the PSD part
        ratio = (dHat - d) / d
        # Simplified form that's always positive
        k = kappa * ratio * ratio * (2.0 + dHat / d) / dHat
        return max(k, 0.0)

    def compute_woodbury_updates(self, solver):
        """
        Compute low-rank update vectors from contact changes.

        Detects differences between current contacts and base contacts,
        computes update vectors U and stiffness changes delta_S.

        Args:
            solver: The PNCG solver containing current contact information
        """
        import numpy as np
        ROTATION_THRESHOLD = 0.9  # Threshold for detecting normal rotation

        self._clear_woodbury_updates()
        current_contacts = self._get_current_contacts(solver)
        if not current_contacts:
            return

        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        subdomain_updates = {d: [] for d in range(n_blocks)}

        for contact_key, curr_data in current_contacts.items():
            curr_stiffness = curr_data['stiffness']
            if curr_stiffness < 1e-10:
                continue

            curr_normal = np.array(curr_data['normal'])
            curr_ids = curr_data['ids']
            curr_cord = np.array(curr_data['cord'])

            if contact_key in self.base_contacts:
                base_data = self.base_contacts[contact_key]
                base_stiffness = base_data['stiffness']
                base_normal = np.array(base_data['normal'])
                n1_norm = np.linalg.norm(curr_normal)
                n2_norm = np.linalg.norm(base_normal)
                dot_product = np.dot(curr_normal, base_normal) / (n1_norm * n2_norm + 1e-10)

                if dot_product < ROTATION_THRESHOLD:
                    # Normal rotated significantly - treat as new contact
                    delta_S = curr_stiffness
                else:
                    # Same direction - compute stiffness change
                    if curr_stiffness > base_stiffness:
                        delta_S = curr_stiffness - base_stiffness
                    else:
                        continue
            else:
                # New contact
                delta_S = curr_stiffness

            if delta_S < 1e-10:
                continue

            # Build update vector contribution
            u_scale = np.sqrt(max(delta_S, 0.0))
            for i, vid in enumerate(curr_ids):
                subdomain_id = vid // BANKSIZE
                lane_id = vid % BANKSIZE
                u_contribution = u_scale * curr_cord[i] * curr_normal
                subdomain_updates[subdomain_id].append({
                    'delta_S': delta_S, 'lane_id': lane_id, 'u_vec': u_contribution
                })

        # Select top-k updates per subdomain (sorted by importance)
        for d in range(n_blocks):
            updates = subdomain_updates[d]
            if not updates:
                continue
            updates.sort(key=lambda x: -x['delta_S'])
            top_updates = updates[:self.top_k]

            for k, upd in enumerate(top_updates):
                self.woodbury_delta_S[d, k] = upd['delta_S']
                lane_id = upd['lane_id']
                u_vec = upd['u_vec']
                for dim in range(3):
                    self.woodbury_U[d, k, lane_id * 3 + dim] = float(u_vec[dim])
            self.woodbury_num_updates[d] = len(top_updates)

    def _get_current_contacts(self, solver):
        """
        Extract current contact state from solver.
        Uses compact array storage (P0 optimization) when available.

        Args:
            solver: The PNCG solver containing contact information

        Returns:
            dict: Contact dictionary mapping contact_key -> contact_data
        """
        current_contacts = {}

        # Try compact array first (P0 optimization)
        if hasattr(solver, 'n_contacts') and hasattr(solver, 'contact_pairs'):
            n_contacts = solver.n_contacts[None]
            if n_contacts > 0:
                contact_pairs_np = solver.contact_pairs.to_numpy()[:n_contacts]
                for i in range(n_contacts):
                    pair = contact_pairs_np[i]
                    ids = tuple(int(x) for x in pair['a'])
                    dist = float(pair['b'])
                    normal = tuple(float(x) for x in pair['d'])
                    cord = tuple(float(x) for x in pair['c'])
                    stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
                    current_contacts[tuple(sorted(ids))] = {
                        'stiffness': stiffness, 'normal': normal,
                        'dist': dist, 'ids': ids, 'cord': cord
                    }
                return current_contacts

        # Fallback to bitmasked cid (legacy)
        try:
            cid_keys = solver.cid.keys_numpy()
        except (AttributeError, TypeError, RuntimeError):
            return current_contacts

        for key in cid_keys:
            k, j = key
            pair_data = solver.cid[k, j]
            ids = tuple(int(x) for x in pair_data.a)
            dist = float(pair_data.b)
            normal = tuple(float(x) for x in pair_data.d)
            cord = tuple(float(x) for x in pair_data.c)
            stiffness = self._compute_barrier_stiffness(dist, solver.dHat, solver.kappa)
            current_contacts[tuple(sorted(ids))] = {
                'stiffness': stiffness, 'normal': normal,
                'dist': dist, 'ids': ids, 'cord': cord
            }
        return current_contacts

    @ti.kernel
    def _clear_woodbury_updates(self):
        """Clear Woodbury update structures."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for d in range(n_blocks):
            self.woodbury_num_updates[d] = 0
            for k in range(8):
                self.woodbury_delta_S[d, k] = 0.0
                for i in range(BANKSIZE * 3):
                    self.woodbury_U[d, k, i] = 0.0

    @ti.kernel
    def _compute_BU_and_capacitance(self):
        """
        Compute B*U and capacitance matrix (I + U^T B U).

        B is the cached inverse of the base subdomain Hessian (stored in symmetric format).
        U is the update matrix with shape (BANKSIZE*3, num_updates).

        BU[block_id, row*3+di, k] = sum_col sum_dj B[row,col][di,dj] * U[col*3+dj, k]

        The capacitance matrix is: C = I + U^T B U
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            num_updates = self.woodbury_num_updates[block_id]
            if num_updates == 0:
                continue

            # Compute BU = B @ U for each update vector k
            # B is stored in symmetric format: inv_block_matrices[block_id, sym_idx]
            # where sym_idx = _sym_index(min(row,col), max(row,col))
            for k in range(num_updates):
                for row in range(BANKSIZE):
                    idx_row = block_id * BANKSIZE + row
                    if idx_row >= self.n_verts:
                        continue

                    for di in ti.static(range(3)):
                        bu_val = 0.0

                        for col in range(BANKSIZE):
                            idx_col = block_id * BANKSIZE + col
                            if idx_col >= self.n_verts:
                                continue

                            # Get the 3x3 block from symmetric storage
                            # Symmetric storage: stores upper triangle (row <= col)
                            min_idx = ti.min(row, col)
                            max_idx = ti.max(row, col)
                            sym_idx = self._sym_index(min_idx, max_idx)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]

                            for dj in ti.static(range(3)):
                                u_val = self.woodbury_U[block_id, k, col * 3 + dj]
                                if ti.abs(u_val) > 1e-15:  # Skip zero entries
                                    # Handle symmetric access:
                                    # If row <= col: B[row,col] = inv_block
                                    # If row > col: B[row,col] = inv_block^T
                                    if row <= col:
                                        bu_val += inv_block[di, dj] * u_val
                                    else:
                                        bu_val += inv_block[dj, di] * u_val

                        self.BU[block_id, row * 3 + di, k] = bu_val

            # Compute capacitance matrix: C = I + U^T @ B @ U
            for i in range(num_updates):
                for j in range(num_updates):
                    # Start with identity on diagonal
                    cap_val = 1.0 if i == j else 0.0

                    # Add U_i^T @ BU_j = sum over all DOFs
                    for lane in range(BANKSIZE):
                        idx = block_id * BANKSIZE + lane
                        if idx < self.n_verts:
                            for d in ti.static(range(3)):
                                dof_idx = lane * 3 + d
                                cap_val += self.woodbury_U[block_id, i, dof_idx] * self.BU[block_id, dof_idx, j]

                    self.capacitance_matrix[block_id, i, j] = cap_val

    @ti.kernel
    def _schwarz_local_solve_woodbury(self):
        """
        Solve z_d = B̂_d * r_d using Woodbury formula.

        For subdomains with updates:
        z = B*r - B*U * (I + U^T*B*U)^{-1} * U^T*B*r

        For subdomains without updates (or coarse levels):
        z = B*r (standard solve)
        """
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Level 0: Apply Woodbury updates
        for block_id in range(n_blocks):
            num_updates = self.woodbury_num_updates[block_id]

            # First compute base solve: z = B * r
            for lane_i in range(BANKSIZE):
                idx_i = block_id * BANKSIZE + lane_i
                if idx_i < self.n_verts:
                    z = ti.Vector.zero(ti.f32, 3)
                    for lane_j in range(BANKSIZE):
                        idx_j = block_id * BANKSIZE + lane_j
                        if idx_j < self.n_verts:
                            sym_idx = self._sym_index(lane_i, lane_j)
                            inv_block = self.inv_block_matrices[block_id, sym_idx]
                            r_j = self.multi_level_r[idx_j]
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    if lane_i <= lane_j:
                                        z[di] += inv_block[di, dj] * r_j[dj]
                                    else:
                                        z[di] += inv_block[dj, di] * r_j[dj]
                    self.multi_level_z[idx_i] = z

            if num_updates == 0:
                continue

            # Compute r_vec = U^T * z_base
            r_vec = ti.Vector.zero(ti.f32, 8)
            for k in range(num_updates):
                r_val = 0.0
                for lane_id in range(BANKSIZE):
                    idx = block_id * BANKSIZE + lane_id
                    if idx < self.n_verts:
                        z_base = self.multi_level_z[idx]
                        for di in ti.static(range(3)):
                            r_val += self.woodbury_U[block_id, k, lane_id * 3 + di] * z_base[di]
                r_vec[k] = r_val

            # Load capacitance matrix to local
            cap_local = ti.Matrix.zero(ti.f32, 8, 8)
            for i in range(num_updates):
                for j in range(num_updates):
                    cap_local[i, j] = self.capacitance_matrix[block_id, i, j]

            # Solve capacitance system: C * lambda = r_vec using Gaussian elimination
            lambda_vec = ti.Vector.zero(ti.f32, 8)

            # Forward elimination with partial pivoting
            for pivot in range(num_updates):
                max_val = ti.abs(cap_local[pivot, pivot])
                max_row = pivot
                for r in range(pivot + 1, num_updates):
                    if ti.abs(cap_local[r, pivot]) > max_val:
                        max_val = ti.abs(cap_local[r, pivot])
                        max_row = r
                if max_row != pivot:
                    for c in range(8):
                        tmp = cap_local[pivot, c]
                        cap_local[pivot, c] = cap_local[max_row, c]
                        cap_local[max_row, c] = tmp
                    tmp_r = r_vec[pivot]
                    r_vec[pivot] = r_vec[max_row]
                    r_vec[max_row] = tmp_r
                if ti.abs(cap_local[pivot, pivot]) > 1e-12:
                    for r in range(pivot + 1, num_updates):
                        factor = cap_local[r, pivot] / cap_local[pivot, pivot]
                        for c in range(pivot, 8):
                            cap_local[r, c] -= factor * cap_local[pivot, c]
                        r_vec[r] -= factor * r_vec[pivot]

            # Back substitution (reverse iteration emulation)
            for rev_idx in range(num_updates):
                i = num_updates - 1 - rev_idx
                if ti.abs(cap_local[i, i]) > 1e-12:
                    lambda_vec[i] = r_vec[i]
                    for j in range(i + 1, num_updates):
                        lambda_vec[i] -= cap_local[i, j] * lambda_vec[j]
                    lambda_vec[i] /= cap_local[i, i]

            # Apply correction: z -= B*U * lambda
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < self.n_verts:
                    correction = ti.Vector.zero(ti.f32, 3)
                    for k in range(num_updates):
                        for di in ti.static(range(3)):
                            correction[di] += self.BU[block_id, lane_id * 3 + di, k] * lambda_vec[k]
                    self.multi_level_z[idx] -= correction

        # Coarse levels: Use frozen cached inverses (no Woodbury updates)
        # This matches the paper: "coarse-level components primarily capture
        # low-frequency error modes, which evolve relatively slowly"
        for level in range(1, self.level_num):
            level_offset = self.level_size[level][1]
            level_size = self.level_size[level][0]

            if level_size <= 0:
                continue

            n_coarse_blocks = (level_size + BANKSIZE - 1) // BANKSIZE

            for local_block_id in range(n_coarse_blocks):
                first_node_in_block = level_offset + local_block_id * BANKSIZE
                block_id = first_node_in_block // BANKSIZE

                # Full block solve using cached inverse
                for lane_i in range(BANKSIZE):
                    idx_i = level_offset + local_block_id * BANKSIZE + lane_i
                    if idx_i < level_offset + level_size:
                        z = ti.Vector.zero(ti.f32, 3)

                        # Multiply by full inverse block: z_i = sum_j (inv_M[i,j] @ r_j)
                        for lane_j in range(BANKSIZE):
                            idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                            if idx_j < level_offset + level_size:
                                # Get the inverse 3x3 block (symmetric storage)
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

    def apply_with_woodbury(self):
        """
        Apply MAS preconditioner with Woodbury updates.

        This is an alternative to the standard apply() that uses
        precomputed Woodbury updates for efficiency when contact
        conditions change incrementally.
        """
        self._clear_multi_level_buffers()
        self._build_multi_level_r()
        self._compute_BU_and_capacitance()
        self._schwarz_local_solve_woodbury()
        self._collect_final_z()

    def woodbury_update(self, solver):
        """
        Perform Sparse-Input Woodbury update.

        This should be called after contact detection to compute
        the low-rank updates before applying the preconditioner.

        Args:
            solver: The PNCG solver containing current contact information
        """
        if not hasattr(self, 'woodbury_initialized') or not self.woodbury_initialized:
            self.init_woodbury_structures()
        self.compute_woodbury_updates(solver)
