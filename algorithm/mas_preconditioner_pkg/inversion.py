"""
Block matrix inversion algorithms for MAS Preconditioner.

This module provides multiple inversion methods:
- Gauss-Jordan elimination (general)
- Cholesky decomposition (SPD matrices)
- Blocked Cholesky (better GPU parallelism)
- Incomplete Cholesky IC(0) (approximate)
- One-way Gauss-Jordan (P4 optimization)

Reference: MASPreconditioner.cu (CUDA reference implementation)
"""

import taichi as ti

from .constants import BANKSIZE, SYM_BLOCK_COUNT, BLOCK_DOF


# ==============================================================================
# InversionMixin Class
# ==============================================================================

class InversionMixin:
    """
    Mixin class providing block matrix inversion methods for MASPreconditioner.
    """

    @ti.func
    def _invert_3x3(self, m: ti.template()) -> ti.Matrix:
        """Invert a 3x3 matrix."""
        det = m.determinant()

        # Use safe division to avoid division by zero
        # If det is too small, use identity matrix
        safe_det = ti.max(ti.abs(det), ti.f32(1e-12))
        inv_det = 1.0 / safe_det

        # Compute cofactor matrix and scale by inv_det
        inv = ti.Matrix([
            [(m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det,
             (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det,
             (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det],
            [(m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det,
             (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det,
             (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv_det],
            [(m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * inv_det,
             (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv_det,
             (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det]
        ], dt=ti.f32)

        # If determinant was too small, return identity instead
        result = inv
        if ti.abs(det) < 1e-12:
            result = ti.Matrix.identity(ti.f32, 3)

        return result

    @ti.kernel
    def _expand_sym_to_full(self):
        """
        Expand symmetric block matrices to full 48x48 dense matrices for inversion.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Clear full matrix
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_matrix[block_id, i, j] = 0.0

            # Copy from symmetric storage to full matrix
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    sym_idx = self._sym_index(row, col)
                    block_3x3 = self.block_matrices[block_id, sym_idx]

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            # Upper triangle: block(row, col) at position [row*3+di, col*3+dj]
                            self.full_block_matrix[block_id, row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                            # Lower triangle: block(col, row) = block(row, col).T
                            # So: [col*3+di, row*3+dj] = block_3x3[dj, di]
                            if row != col:
                                self.full_block_matrix[block_id, col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

    @ti.kernel
    def _symmetrize_full_block_matrices(self):
        """
        Enforce symmetry on full block matrices: A = (A + A^T) / 2.
        This is a safety net to ensure matrices are symmetric before IC/Cholesky.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for i in range(BLOCK_DOF):
                for j in range(i + 1, BLOCK_DOF):
                    avg = 0.5 * (self.full_block_matrix[block_id, i, j] +
                                 self.full_block_matrix[block_id, j, i])
                    self.full_block_matrix[block_id, i, j] = avg
                    self.full_block_matrix[block_id, j, i] = avg

    @ti.kernel
    def _add_diagonal_regularization(self, epsilon: ti.f32):
        """
        Add diagonal regularization to shift eigenvalues positive.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for i in range(BLOCK_DOF):
                self.full_block_matrix[block_id, i, i] += epsilon

    @ti.kernel
    def _gauss_jordan_invert_blocks(self):
        """
        Invert full 48x48 block matrices using Gauss-Jordan elimination.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Initialize identity in the inverse storage
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    if i == j:
                        self.full_block_inverse[block_id, i, j] = 1.0
                    else:
                        self.full_block_inverse[block_id, i, j] = 0.0

            # Gauss-Jordan elimination
            for pivot in range(BLOCK_DOF):
                # Find pivot (partial pivoting)
                max_val = ti.abs(self.full_block_matrix[block_id, pivot, pivot])
                max_row = pivot

                for r in range(pivot + 1, BLOCK_DOF):
                    val = ti.abs(self.full_block_matrix[block_id, r, pivot])
                    if val > max_val:
                        max_val = val
                        max_row = r

                # Swap rows if needed
                if max_row != pivot:
                    for c in range(BLOCK_DOF):
                        tmp_m = self.full_block_matrix[block_id, pivot, c]
                        self.full_block_matrix[block_id, pivot, c] = self.full_block_matrix[block_id, max_row, c]
                        self.full_block_matrix[block_id, max_row, c] = tmp_m
                        tmp_i = self.full_block_inverse[block_id, pivot, c]
                        self.full_block_inverse[block_id, pivot, c] = self.full_block_inverse[block_id, max_row, c]
                        self.full_block_inverse[block_id, max_row, c] = tmp_i

                # Scale pivot row
                pivot_val = self.full_block_matrix[block_id, pivot, pivot]
                if ti.abs(pivot_val) > 1e-12:
                    scale = 1.0 / pivot_val
                    for c in range(BLOCK_DOF):
                        self.full_block_matrix[block_id, pivot, c] *= scale
                        self.full_block_inverse[block_id, pivot, c] *= ti.f32(scale)
                else:
                    self.full_block_matrix[block_id, pivot, pivot] = 1e-6
                    for c in range(BLOCK_DOF):
                        if c == pivot:
                            self.full_block_inverse[block_id, pivot, c] = 1e6
                        else:
                            self.full_block_inverse[block_id, pivot, c] = 0.0

                # Eliminate in all other rows
                for r in range(BLOCK_DOF):
                    if r != pivot:
                        factor = self.full_block_matrix[block_id, r, pivot]
                        for c in range(BLOCK_DOF):
                            self.full_block_matrix[block_id, r, c] -= factor * self.full_block_matrix[block_id, pivot, c]
                            self.full_block_inverse[block_id, r, c] -= ti.f32(factor) * self.full_block_inverse[block_id, pivot, c]

    @ti.kernel
    def _cholesky_invert_blocks(self):
        """
        Invert full 48x48 SPD block matrices using Cholesky decomposition.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Step 1: Cholesky factorization A = L * L^T
            for i in range(BLOCK_DOF):
                for j in range(i):
                    sum_val = self.full_block_matrix[block_id, i, j]
                    for k in range(j):
                        sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_matrix[block_id, j, k]
                    L_jj = self.full_block_matrix[block_id, j, j]
                    if ti.abs(L_jj) > 1e-12:
                        self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                    else:
                        self.full_block_matrix[block_id, i, j] = 0.0

                # Diagonal element
                sum_val = self.full_block_matrix[block_id, i, i]
                for k in range(i):
                    sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_matrix[block_id, i, k]
                if sum_val > 1e-12:
                    self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                else:
                    self.full_block_matrix[block_id, i, i] = 1e-3

            # Step 2 & 3: Solve L * L^T * X = I for X = A^{-1}
            for col in range(BLOCK_DOF):
                # Forward substitution
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= self.full_block_matrix[block_id, i, k] * self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                # Backward substitution
                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f32(self.full_block_inverse[block_id, i, col])
                    for k in range(i + 1, BLOCK_DOF):
                        sum_val -= self.full_block_matrix[block_id, k, i] * ti.f32(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _copy_inverse_to_sym(self):
        """Copy inverted full matrix back to symmetric storage format."""
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for row in range(BANKSIZE):
                for col in range(row, BANKSIZE):
                    sym_idx = self._sym_index(row, col)

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, sym_idx][di, dj] = \
                                self.full_block_inverse[block_id, row * 3 + di, col * 3 + dj]

    @ti.kernel
    def _invert_diagonal_blocks(self):
        """
        Invert diagonal 3x3 blocks as a simple approximation.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            for lane_id in range(BANKSIZE):
                idx = block_id * BANKSIZE + lane_id
                if idx < total_nodes:
                    sym_idx = self._sym_index(lane_id, lane_id)
                    diag_block = self.block_matrices[block_id, sym_idx]
                    inv_block = self._invert_3x3(diag_block)

                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            self.inv_block_matrices[block_id, sym_idx][di, dj] = \
                                ti.cast(inv_block[di, dj], ti.f32)

            # Set off-diagonal inverses to zero
            for row in range(BANKSIZE):
                for col in range(row + 1, BANKSIZE):
                    sym_idx = self._sym_index(row, col)
                    self.inv_block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _blocked_cholesky_invert_blocks(self):
        """
        Blocked Cholesky decomposition for better GPU parallelism.
        Uses block size of 12 (4 nodes * 3 DOF).
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        BSIZE = 12
        N_SUB = BLOCK_DOF // BSIZE

        for block_id in range(n_blocks):
            # Blocked Cholesky factorization
            for kb in range(N_SUB):
                k_start = kb * BSIZE
                k_end = k_start + BSIZE

                # Cholesky on diagonal block
                for i in range(k_start, k_end):
                    for j in range(k_start, i):
                        sum_val = self.full_block_matrix[block_id, i, j]
                        for kk in range(k_start, j):
                            sum_val -= self.full_block_matrix[block_id, i, kk] * \
                                       self.full_block_matrix[block_id, j, kk]
                        L_jj = self.full_block_matrix[block_id, j, j]
                        if ti.abs(L_jj) > 1e-12:
                            self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                        else:
                            self.full_block_matrix[block_id, i, j] = 0.0

                    sum_val = self.full_block_matrix[block_id, i, i]
                    for kk in range(k_start, i):
                        sum_val -= self.full_block_matrix[block_id, i, kk] ** 2
                    if sum_val > 1e-12:
                        self.full_block_matrix[block_id, i, i] = ti.sqrt(sum_val)
                    else:
                        self.full_block_matrix[block_id, i, i] = 1e-3

                # Solve for off-diagonal blocks
                for ib in range(kb + 1, N_SUB):
                    i_start = ib * BSIZE
                    i_end = i_start + BSIZE

                    for i in range(i_start, i_end):
                        for j in range(k_start, k_end):
                            sum_val = self.full_block_matrix[block_id, i, j]
                            for kk in range(k_start, j):
                                sum_val -= self.full_block_matrix[block_id, i, kk] * \
                                           self.full_block_matrix[block_id, j, kk]
                            L_jj = self.full_block_matrix[block_id, j, j]
                            if ti.abs(L_jj) > 1e-12:
                                self.full_block_matrix[block_id, i, j] = sum_val / L_jj
                            else:
                                self.full_block_matrix[block_id, i, j] = 0.0

                # Update remaining blocks
                for ib in range(kb + 1, N_SUB):
                    i_start = ib * BSIZE
                    i_end = i_start + BSIZE

                    for jb in range(kb + 1, ib + 1):
                        j_start = jb * BSIZE
                        j_end = j_start + BSIZE

                        for i in range(i_start, i_end):
                            j_limit = j_end if jb < ib else i + 1
                            for j in range(j_start, j_limit):
                                sum_val = 0.0
                                for kk in range(k_start, k_end):
                                    sum_val += self.full_block_matrix[block_id, i, kk] * \
                                               self.full_block_matrix[block_id, j, kk]
                                self.full_block_matrix[block_id, i, j] -= sum_val

            # Inversion via forward/backward substitution
            for col in range(BLOCK_DOF):
                for i in range(BLOCK_DOF):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= self.full_block_matrix[block_id, i, k] * \
                                   self.full_block_inverse[block_id, k, col]
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

                for i_rev in range(BLOCK_DOF):
                    i = BLOCK_DOF - 1 - i_rev
                    sum_val = ti.f64(self.full_block_inverse[block_id, i, col])
                    for k in range(i + 1, BLOCK_DOF):
                        sum_val -= self.full_block_matrix[block_id, k, i] * \
                                   ti.f64(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _incomplete_cholesky_invert_blocks(self):
        """
        Incomplete Cholesky IC(0) factorization for approximate inversion.
        Uses banded pattern for efficiency.
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

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
                    sum_val = ti.f64(self.full_block_inverse[block_id, i, col])
                    k_end = ti.min(BLOCK_DOF, i + BANDWIDTH + 1)
                    for k in range(i + 1, k_end):
                        if k - i <= BANDWIDTH:
                            sum_val -= self.full_block_matrix[block_id, k, i] * \
                                       ti.f64(self.full_block_inverse[block_id, k, col])
                    L_ii = self.full_block_matrix[block_id, i, i]
                    if ti.abs(L_ii) > 1e-12:
                        self.full_block_inverse[block_id, i, col] = ti.f32(sum_val / L_ii)
                    else:
                        self.full_block_inverse[block_id, i, col] = 0.0

    @ti.kernel
    def _oneway_gauss_jordan_invert_blocks(self):
        """
        One-way Gauss-Jordan inversion without pivoting (P4 optimization).
        Optimized for SPD matrices.

        CUDA Reference: __inverse6_P96x96() (MASPreconditioner.cu lines 630-726)
        """
        total_nodes = self.total_nodes_all_levels
        n_blocks = (total_nodes + BANKSIZE - 1) // BANKSIZE

        for block_id in range(n_blocks):
            # Initialize as copy of the matrix
            for i in range(BLOCK_DOF):
                for j in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, i, j] = \
                        self.full_block_matrix[block_id, i, j]

            # One-way Gauss-Jordan elimination
            for j in range(BLOCK_DOF):
                rt = self.full_block_inverse[block_id, j, j]

                if ti.abs(rt) < 1e-10:
                    rt = 1e-6
                    self.full_block_inverse[block_id, j, j] = rt

                inv_rt = 1.0 / rt

                for i in range(BLOCK_DOF):
                    self.full_block_inverse[block_id, j, i] *= inv_rt

                for k in range(BLOCK_DOF):
                    if k != j:
                        factor = self.full_block_inverse[block_id, k, j]
                        for i in range(BLOCK_DOF):
                            self.full_block_inverse[block_id, k, i] -= \
                                factor * self.full_block_inverse[block_id, j, i]

            # Symmetry recovery
            for node_i in range(BANKSIZE):
                for node_j in range(node_i + 1, BANKSIZE):
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            row = node_i * 3 + di
                            col = node_j * 3 + dj
                            row_t = node_j * 3 + dj
                            col_t = node_i * 3 + di
                            avg = 0.5 * (self.full_block_inverse[block_id, row, col] +
                                        self.full_block_inverse[block_id, row_t, col_t])
                            self.full_block_inverse[block_id, row, col] = avg
                            self.full_block_inverse[block_id, row_t, col_t] = avg

    def invert_block_matrices(self, use_full_inversion: bool = True,
                              use_cholesky: bool = True,
                              use_blocked: bool = False,
                              use_incomplete: bool = False,
                              use_oneway_gj: bool = False,
                              force_symmetry: bool = True,
                              regularization_epsilon: float = 0.0):
        """
        Invert all block matrices on GPU.

        Args:
            use_full_inversion: If True, use full 48x48 block inversion.
            use_cholesky: If True, use Cholesky decomposition.
            use_blocked: If True, use blocked Cholesky.
            use_incomplete: If True, use Incomplete Cholesky IC(0).
            use_oneway_gj: If True, use One-way Gauss-Jordan (P4 optimization).
            force_symmetry: If True, symmetrize matrices before inversion.
            regularization_epsilon: If > 0, add diagonal regularization.
        """
        print("[MAS] Inverting block matrices...")

        if use_full_inversion:
            self._expand_sym_to_full()

            # Apply symmetrization as safety net
            if force_symmetry:
                self._symmetrize_full_block_matrices()

            # Apply diagonal regularization if requested
            if regularization_epsilon > 0:
                self._add_diagonal_regularization(regularization_epsilon)

            if use_incomplete:
                self._incomplete_cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Incomplete Cholesky IC(0))")
            elif use_oneway_gj:
                self._oneway_gauss_jordan_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (One-way Gauss-Jordan)")
            elif use_blocked and use_cholesky:
                self._blocked_cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Blocked Cholesky)")
            elif use_cholesky:
                self._cholesky_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Cholesky)")
            else:
                self._gauss_jordan_invert_blocks()
                self._copy_inverse_to_sym()
                print("[MAS] Full block inversion complete (Gauss-Jordan)")
        else:
            self._invert_diagonal_blocks()
            print("[MAS] Diagonal block inversion complete")

        self.matrices_inverted = True
