"""
Woodbury Update Module for MAS Preconditioner with Contacts.

Implements Sherman-Morrison-Woodbury formula for efficient preconditioner updates:
    (A + U*S*U^T)^{-1} = A^{-1} - A^{-1}*U*(S^{-1} + U^T*A^{-1}*U)^{-1}*U^T*A^{-1}

Key insight: Only contact stiffness changes between iterations, elastic part is constant.
We cache the base inverse and apply low-rank corrections for contact changes.

Reference: Section 3.1 of MAS-PNCG paper
"""

import taichi as ti
import numpy as np

from algorithm.mas_preconditioner_small.core import BANKSIZE, sym_index


# Constants
TOP_K = 8  # Maximum number of rank-1 updates per subdomain
ROTATION_THRESHOLD = 0.9  # Threshold for detecting normal rotation (cos(~26°))


@ti.data_oriented
class WoodburySupport:
    """
    Woodbury low-rank update support for MAS Preconditioner.

    This class provides sparse-input Woodbury updates for efficiently updating
    the preconditioner when contact conditions change incrementally.

    The Woodbury formula is applied per subdomain at level 0:
        z = B*r - B*U * (I + U^T*B*U)^{-1} * U^T*B*r

    where:
        B = cached inverse (inv_block_matrices)
        U = update vectors (from contact stiffness changes)

    Coarse levels use frozen cached inverses (no Woodbury updates).
    """

    def __init__(self, preconditioner):
        """
        Initialize Woodbury support.

        Args:
            preconditioner: MASPreconditionerContact instance
        """
        self.precond = preconditioner
        self.n_verts = preconditioner.n_verts
        self.level_num = preconditioner.level_num
        self.top_k = TOP_K
        self.initialized = False

        # Base contact state (Python dict for flexible key-value lookups)
        self.base_contacts = {}

    def init_woodbury_structures(self):
        """Initialize Taichi fields for Woodbury updates."""
        n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # U matrix: update vectors (n_blocks, top_k, BANKSIZE*3)
        # Each column of U is a sparse update vector for the subdomain
        self.woodbury_U = ti.field(
            dtype=ti.f32,
            shape=(n_blocks, self.top_k, BANKSIZE * 3)
        )

        # Delta S: stiffness changes (n_blocks, top_k)
        self.woodbury_delta_S = ti.field(
            dtype=ti.f32,
            shape=(n_blocks, self.top_k)
        )

        # Number of active updates per subdomain
        self.woodbury_num_updates = ti.field(dtype=ti.i32, shape=n_blocks)

        # B*U precomputed product (n_blocks, BANKSIZE*3, top_k)
        self.BU = ti.field(
            dtype=ti.f32,
            shape=(n_blocks, BANKSIZE * 3, self.top_k)
        )

        # Capacitance matrix: C = I + U^T * B * U (n_blocks, top_k, top_k)
        self.capacitance_matrix = ti.field(
            dtype=ti.f32,
            shape=(n_blocks, self.top_k, self.top_k)
        )

        self.n_blocks = n_blocks
        self.initialized = True
        print(f"[Woodbury] Initialized for {n_blocks} subdomains, top_k={self.top_k}")

    def save_base_contact_state(self, solver):
        """
        Save current contact state as base for Woodbury updates.

        Call this at the start of each frame or after a full rebuild.

        Args:
            solver: The IPC solver containing contact information
        """
        self.base_contacts = {}

        # Use compact array storage (solver.contact_pairs)
        if not hasattr(solver, 'n_contacts') or not hasattr(solver, 'contact_pairs'):
            return

        n_contacts = solver.n_contacts[None]
        if n_contacts == 0:
            return

        # Transfer contact data from GPU to CPU for processing
        # Taichi Struct.field.to_numpy() returns a dict: {'a': array, 'b': array, ...}
        contact_pairs_np = solver.contact_pairs.to_numpy()
        ids_arr = contact_pairs_np['a']      # shape (N, 4), uint32
        dist_arr = contact_pairs_np['b']     # shape (N,), float32
        cord_arr = contact_pairs_np['c']     # shape (N, 4), float32
        normal_arr = contact_pairs_np['d']   # shape (N, 3), float32

        for i in range(n_contacts):
            ids = tuple(int(x) for x in ids_arr[i])  # 4 vertex IDs
            dist = float(dist_arr[i])
            normal = tuple(float(x) for x in normal_arr[i])  # Contact direction
            cord = tuple(float(x) for x in cord_arr[i])  # Barycentric coords

            # Compute barrier stiffness
            stiffness = self._compute_barrier_stiffness(
                dist, solver.dHat, solver.kappa
            )

            # Use sorted vertex IDs as key for consistent lookup
            contact_key = tuple(sorted(ids))
            self.base_contacts[contact_key] = {
                'stiffness': stiffness,
                'normal': normal,
                'dist': dist,
                'ids': ids,
                'cord': cord
            }

    def _compute_barrier_stiffness(self, d, dHat, kappa):
        """
        Compute barrier Hessian stiffness k = b''(d) for IPC log barrier.

        IPC log barrier: b(d) = -kappa * (d - dHat)^2 * log(d / dHat)

        For Gauss-Newton approximation, we use the positive semi-definite part:
            k = kappa * ((dHat - d) / d)^2 * (2 + dHat/d) / dHat

        Args:
            d: Current distance
            dHat: Distance threshold
            kappa: Barrier stiffness parameter

        Returns:
            Non-negative barrier stiffness
        """
        if d >= dHat or d <= 0:
            return 0.0

        # Gauss-Newton approximation for SPD guarantee
        ratio = (dHat - d) / d
        k = kappa * ratio * ratio * (2.0 + dHat / d) / dHat
        return max(k, 0.0)

    def _get_current_contacts(self, solver):
        """
        Extract current contact state from solver.

        Args:
            solver: The IPC solver containing contact information

        Returns:
            dict: Contact dictionary mapping contact_key -> contact_data
        """
        current_contacts = {}

        if not hasattr(solver, 'n_contacts') or not hasattr(solver, 'contact_pairs'):
            return current_contacts

        n_contacts = solver.n_contacts[None]
        if n_contacts == 0:
            return current_contacts

        # Taichi Struct.field.to_numpy() returns a dict: {'a': array, 'b': array, ...}
        contact_pairs_np = solver.contact_pairs.to_numpy()
        ids_arr = contact_pairs_np['a']      # shape (N, 4), uint32
        dist_arr = contact_pairs_np['b']     # shape (N,), float32
        cord_arr = contact_pairs_np['c']     # shape (N, 4), float32
        normal_arr = contact_pairs_np['d']   # shape (N, 3), float32

        for i in range(n_contacts):
            ids = tuple(int(x) for x in ids_arr[i])
            dist = float(dist_arr[i])
            normal = tuple(float(x) for x in normal_arr[i])
            cord = tuple(float(x) for x in cord_arr[i])

            stiffness = self._compute_barrier_stiffness(
                dist, solver.dHat, solver.kappa
            )

            contact_key = tuple(sorted(ids))
            current_contacts[contact_key] = {
                'stiffness': stiffness,
                'normal': normal,
                'dist': dist,
                'ids': ids,
                'cord': cord
            }

        return current_contacts

    def compute_woodbury_updates(self, solver):
        """
        Compute low-rank update vectors from contact changes.

        Detects differences between current contacts and base contacts,
        computes update vectors U and stiffness changes delta_S.

        Args:
            solver: The IPC solver containing current contact information
        """
        if not self.initialized:
            self.init_woodbury_structures()

        # Clear previous updates
        self._clear_woodbury_updates()

        current_contacts = self._get_current_contacts(solver)
        if not current_contacts:
            return

        # Collect updates per subdomain
        subdomain_updates = {d: [] for d in range(self.n_blocks)}

        for contact_key, curr_data in current_contacts.items():
            curr_stiffness = curr_data['stiffness']
            if curr_stiffness < 1e-10:
                continue

            curr_normal = np.array(curr_data['normal'])
            curr_ids = curr_data['ids']
            curr_cord = np.array(curr_data['cord'])

            delta_S = 0.0

            if contact_key in self.base_contacts:
                base_data = self.base_contacts[contact_key]
                base_stiffness = base_data['stiffness']
                base_normal = np.array(base_data['normal'])

                # Check for normal rotation
                n1_norm = np.linalg.norm(curr_normal)
                n2_norm = np.linalg.norm(base_normal)
                if n1_norm > 1e-10 and n2_norm > 1e-10:
                    dot_product = np.dot(curr_normal, base_normal) / (n1_norm * n2_norm)
                else:
                    dot_product = 1.0

                if dot_product < ROTATION_THRESHOLD:
                    # Normal rotated significantly - treat as new contact
                    delta_S = curr_stiffness
                elif curr_stiffness > base_stiffness:
                    # Same direction - compute stiffness increase
                    delta_S = curr_stiffness - base_stiffness
                else:
                    # Stiffness decreased or unchanged - skip
                    continue
            else:
                # New contact
                delta_S = curr_stiffness

            if delta_S < 1e-10:
                continue

            # Build update vector contribution for each vertex in contact
            # u = sqrt(delta_S) * coord * normal
            u_scale = np.sqrt(max(delta_S, 0.0))

            for i, vid in enumerate(curr_ids):
                if vid < 0 or vid >= self.n_verts:
                    continue

                subdomain_id = vid // BANKSIZE
                lane_id = vid % BANKSIZE
                u_contribution = u_scale * curr_cord[i] * curr_normal

                subdomain_updates[subdomain_id].append({
                    'delta_S': delta_S,
                    'lane_id': lane_id,
                    'u_vec': u_contribution
                })

        # Select top-k updates per subdomain (sorted by importance)
        for d in range(self.n_blocks):
            updates = subdomain_updates[d]
            if not updates:
                continue

            # Sort by delta_S (descending) - prioritize largest changes
            updates.sort(key=lambda x: -x['delta_S'])
            top_updates = updates[:self.top_k]

            for k, upd in enumerate(top_updates):
                self.woodbury_delta_S[d, k] = upd['delta_S']
                lane_id = upd['lane_id']
                u_vec = upd['u_vec']
                for dim in range(3):
                    self.woodbury_U[d, k, lane_id * 3 + dim] = float(u_vec[dim])

            self.woodbury_num_updates[d] = len(top_updates)

    @ti.kernel
    def _clear_woodbury_updates(self):
        """Clear Woodbury update structures."""
        for d in range(self.n_blocks):
            self.woodbury_num_updates[d] = 0
            for k in range(TOP_K):
                self.woodbury_delta_S[d, k] = 0.0
                for i in range(BANKSIZE * 3):
                    self.woodbury_U[d, k, i] = 0.0
                for j in range(TOP_K):
                    self.capacitance_matrix[d, k, j] = 0.0
            for row in range(BANKSIZE * 3):
                for k in range(TOP_K):
                    self.BU[d, row, k] = 0.0

    @ti.kernel
    def _compute_BU_and_capacitance(self):
        """
        Compute B*U and capacitance matrix (I + U^T B U).

        B is the cached inverse of the subdomain Hessian (stored in symmetric format).
        U is the update matrix with shape (BANKSIZE*3, num_updates).

        BU[block_id, row*3+di, k] = sum_col sum_dj B[row,col][di,dj] * U[col*3+dj, k]

        The capacitance matrix is: C = I + U^T B U
        """
        for block_id in range(self.n_blocks):
            num_updates = self.woodbury_num_updates[block_id]
            if num_updates == 0:
                continue

            # Compute BU = B @ U for each update vector k
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
                            min_idx = ti.min(row, col)
                            max_idx = ti.max(row, col)
                            s_idx = sym_index(min_idx, max_idx)
                            inv_block = self.precond.inv_block_matrices[block_id, s_idx]

                            for dj in ti.static(range(3)):
                                u_val = self.woodbury_U[block_id, k, col * 3 + dj]
                                if ti.abs(u_val) > 1e-15:
                                    # Handle symmetric access
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
                                cap_val += (self.woodbury_U[block_id, i, dof_idx] *
                                            self.BU[block_id, dof_idx, j])

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
        # Level 0: Apply Woodbury updates
        for block_id in range(self.n_blocks):
            num_updates = self.woodbury_num_updates[block_id]

            # First compute base solve: z = B * r
            for lane_i in range(BANKSIZE):
                idx_i = block_id * BANKSIZE + lane_i
                if idx_i < self.n_verts:
                    z = ti.Vector.zero(ti.f32, 3)
                    for lane_j in range(BANKSIZE):
                        idx_j = block_id * BANKSIZE + lane_j
                        if idx_j < self.n_verts:
                            s_idx = sym_index(ti.min(lane_i, lane_j), ti.max(lane_i, lane_j))
                            inv_block = self.precond.inv_block_matrices[block_id, s_idx]
                            r_j = self.precond.multi_level_r[idx_j]
                            for di in ti.static(range(3)):
                                for dj in ti.static(range(3)):
                                    if lane_i <= lane_j:
                                        z[di] += inv_block[di, dj] * r_j[dj]
                                    else:
                                        z[di] += inv_block[dj, di] * r_j[dj]
                    self.precond.multi_level_z[idx_i] = z

            if num_updates == 0:
                continue

            # Compute r_vec = U^T * z_base
            r_vec = ti.Vector.zero(ti.f32, TOP_K)
            for k in range(num_updates):
                r_val = 0.0
                for lane_id in range(BANKSIZE):
                    idx = block_id * BANKSIZE + lane_id
                    if idx < self.n_verts:
                        z_base = self.precond.multi_level_z[idx]
                        for di in ti.static(range(3)):
                            r_val += self.woodbury_U[block_id, k, lane_id * 3 + di] * z_base[di]
                r_vec[k] = r_val

            # Load capacitance matrix to local
            cap_local = ti.Matrix.zero(ti.f32, TOP_K, TOP_K)
            for i in range(num_updates):
                for j in range(num_updates):
                    cap_local[i, j] = self.capacitance_matrix[block_id, i, j]

            # Solve capacitance system: C * lambda = r_vec using Gaussian elimination
            lambda_vec = ti.Vector.zero(ti.f32, TOP_K)

            # Forward elimination with partial pivoting
            for pivot in range(num_updates):
                # Find max pivot
                max_val = ti.abs(cap_local[pivot, pivot])
                max_row = pivot
                for r in range(pivot + 1, num_updates):
                    if ti.abs(cap_local[r, pivot]) > max_val:
                        max_val = ti.abs(cap_local[r, pivot])
                        max_row = r

                # Swap rows if needed
                if max_row != pivot:
                    for c in range(TOP_K):
                        tmp = cap_local[pivot, c]
                        cap_local[pivot, c] = cap_local[max_row, c]
                        cap_local[max_row, c] = tmp
                    tmp_r = r_vec[pivot]
                    r_vec[pivot] = r_vec[max_row]
                    r_vec[max_row] = tmp_r

                # Eliminate
                if ti.abs(cap_local[pivot, pivot]) > 1e-12:
                    for r in range(pivot + 1, num_updates):
                        factor = cap_local[r, pivot] / cap_local[pivot, pivot]
                        for c in range(pivot, TOP_K):
                            cap_local[r, c] -= factor * cap_local[pivot, c]
                        r_vec[r] -= factor * r_vec[pivot]

            # Back substitution
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
                    self.precond.multi_level_z[idx] -= correction

        # Coarse levels: Use frozen cached inverses (no Woodbury updates)
        # Following the paper: "coarse-level components primarily capture
        # low-frequency error modes, which evolve relatively slowly"
        for level in range(1, self.level_num):
            level_offset = self.precond.level_size[level, 1]
            level_size = self.precond.level_size[level, 0]

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

                        for lane_j in range(BANKSIZE):
                            idx_j = level_offset + local_block_id * BANKSIZE + lane_j
                            if idx_j < level_offset + level_size:
                                # Symmetric storage
                                if lane_i <= lane_j:
                                    s_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_j
                                    inv_block = self.precond.inv_block_matrices[block_id, s_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[di, dj] * self.precond.multi_level_r[idx_j][dj]
                                else:
                                    s_idx = BANKSIZE * lane_j - lane_j * (lane_j + 1) // 2 + lane_i
                                    inv_block = self.precond.inv_block_matrices[block_id, s_idx]
                                    for di in ti.static(range(3)):
                                        for dj in ti.static(range(3)):
                                            z[di] += inv_block[dj, di] * self.precond.multi_level_r[idx_j][dj]

                        self.precond.multi_level_z[idx_i] = z

    def apply_with_woodbury(self):
        """
        Apply MAS preconditioner with Woodbury updates.

        This is an alternative to the standard apply() that uses
        precomputed Woodbury updates for efficiency when contact
        conditions change incrementally.
        """
        self.precond._clear_multi_level_buffers()
        self.precond._build_multi_level_r()
        self._compute_BU_and_capacitance()
        self._schwarz_local_solve_woodbury()
        self.precond._collect_final_z(self.level_num)

    def get_num_updates_total(self) -> int:
        """Get total number of active Woodbury updates across all subdomains."""
        total = 0
        num_updates_np = self.woodbury_num_updates.to_numpy()
        for d in range(self.n_blocks):
            total += num_updates_np[d]
        return total
