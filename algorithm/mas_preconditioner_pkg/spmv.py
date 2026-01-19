"""
SRBK SpMV: Symmetric Reduce-By-Key Sparse Matrix-Vector Multiplication.

Reference: /root/Stiff-GIPC_init/StiffGIPC/linear_system/utils/spmv.cu

This implements a GPU-optimized symmetric SpMV that:
1. Only stores upper triangle of symmetric matrix
2. Uses reduce-by-key pattern for efficient accumulation
3. Processes both A[i,j] and A[j,i]^T simultaneously
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class SRBKSpMV:
    """
    Symmetric Reduce-By-Key Sparse Matrix-Vector Multiplication.

    This class implements the SRBK SpMV algorithm from StiffGIPC for computing
    y = A * x where A is a symmetric sparse matrix stored in block-triplet format.

    Key optimizations:
    1. Upper triangle storage: Only store (i,j) where i <= j
    2. Symmetric processing: Compute both upper and lower contributions simultaneously
    3. Row-grouped accumulation: Triplets sorted by row for efficient reduction
    4. Warp-level parallelism: Multiple threads per row reduce before atomic write
    """

    def __init__(self, max_triplets: int, n_dofs: int):
        """
        Initialize SRBK SpMV structures.

        Args:
            max_triplets: Maximum number of 3x3 block triplets
            n_dofs: Number of degrees of freedom (3 * n_verts)
        """
        self.max_triplets = max_triplets
        self.n_dofs = n_dofs
        self.n_verts = n_dofs // 3

        # Triplet storage: (row_id, col_id, value) where value is 3x3 block
        # Row and col are block (vertex) indices, not DOF indices
        self.triplet_row = ti.field(dtype=ti.i32, shape=max_triplets)
        self.triplet_col = ti.field(dtype=ti.i32, shape=max_triplets)
        self.triplet_val = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_triplets)

        # Triplet count
        self.n_triplets = ti.field(dtype=ti.i32, shape=())

        # Row segment information for reduce-by-key
        # row_starts[i] = first triplet index for row i
        self.row_starts = ti.field(dtype=ti.i32, shape=self.n_verts + 1)

        # Work buffers for reduction
        self.y_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)

        # Flag for whether triplets are sorted
        self.sorted = False

    def clear(self):
        """Clear all triplets."""
        self.n_triplets[None] = 0
        self.sorted = False

    @ti.kernel
    def add_triplet(self, row: ti.i32, col: ti.i32, val: ti.types.matrix(3, 3, ti.f64)):
        """
        Add a single 3x3 block triplet.

        For symmetric matrices, only add upper triangle (row <= col).
        The SpMV will automatically handle the symmetric contribution.
        """
        idx = ti.atomic_add(self.n_triplets[None], 1)
        if idx < self.max_triplets:
            # Ensure upper triangle storage
            if row <= col:
                self.triplet_row[idx] = row
                self.triplet_col[idx] = col
                self.triplet_val[idx] = val
            else:
                self.triplet_row[idx] = col
                self.triplet_col[idx] = row
                self.triplet_val[idx] = val.transpose()

    def sort_by_row(self):
        """
        Sort triplets by row index for efficient reduce-by-key.

        Uses numpy for sorting since Taichi doesn't have efficient parallel sort.
        """
        n = self.n_triplets[None]
        if n == 0:
            self.sorted = True
            return

        # Extract to numpy
        rows_np = self.triplet_row.to_numpy()[:n]
        cols_np = self.triplet_col.to_numpy()[:n]
        vals_np = self.triplet_val.to_numpy()[:n]

        # Sort by row, then by col for stability
        sort_idx = np.lexsort((cols_np, rows_np))

        # Apply sorting
        rows_sorted = rows_np[sort_idx]
        cols_sorted = cols_np[sort_idx]
        vals_sorted = vals_np[sort_idx]

        # Write back
        temp_rows = np.zeros(self.max_triplets, dtype=np.int32)
        temp_cols = np.zeros(self.max_triplets, dtype=np.int32)
        temp_vals = np.zeros((self.max_triplets, 3, 3), dtype=np.float64)
        temp_rows[:n] = rows_sorted
        temp_cols[:n] = cols_sorted
        temp_vals[:n] = vals_sorted

        self.triplet_row.from_numpy(temp_rows)
        self.triplet_col.from_numpy(temp_cols)
        self.triplet_val.from_numpy(temp_vals)

        # Compute row_starts using CSR-style computation
        # row_starts[i] = first triplet index for row i
        # row_starts[n_verts] = total number of triplets
        row_starts_np = np.zeros(self.n_verts + 1, dtype=np.int32)

        # Count elements per row
        row_counts = np.zeros(self.n_verts, dtype=np.int32)
        for i in range(n):
            row = rows_sorted[i]
            row_counts[row] += 1

        # Cumulative sum to get row starts
        row_starts_np[0] = 0
        for i in range(self.n_verts):
            row_starts_np[i + 1] = row_starts_np[i] + row_counts[i]

        self.row_starts.from_numpy(row_starts_np)
        self.sorted = True

    @ti.kernel
    def spmv_naive(self, x: ti.template(), y: ti.template(), alpha: ti.f64, beta: ti.f64):
        """
        Naive symmetric SpMV: y = alpha * A * x + beta * y

        This is the reference implementation without reduce-by-key optimization.
        """
        n = self.n_triplets[None]
        n_verts = self.n_verts

        # Scale existing y
        if beta != 0.0:
            for i in range(n_verts):
                y[i] = beta * y[i]
        else:
            for i in range(n_verts):
                y[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Process each triplet
        for tid in range(n):
            i = self.triplet_row[tid]
            j = self.triplet_col[tid]
            mat = self.triplet_val[tid]

            # Upper triangle contribution: y[i] += A[i,j] * x[j]
            contrib = mat @ x[j]
            for d in ti.static(range(3)):
                ti.atomic_add(y[i][d], alpha * contrib[d])

            # Lower triangle contribution: y[j] += A[j,i] * x[i] = A[i,j]^T * x[i]
            if i != j:
                contrib_t = mat.transpose() @ x[i]
                for d in ti.static(range(3)):
                    ti.atomic_add(y[j][d], alpha * contrib_t[d])

    @ti.kernel
    def spmv_row_parallel(self, x: ti.template(), y: ti.template(), alpha: ti.f64, beta: ti.f64):
        """
        Row-parallel symmetric SpMV: y = alpha * A * x + beta * y

        This version processes each row in parallel, reducing atomic conflicts
        by accumulating row contributions locally before writing.

        Reference: _schwarzLocalXSym9 in MASPreconditioner.cu (lines 1045-1129)
        """
        n_verts = self.n_verts

        # Initialize output
        if beta != 0.0:
            for i in range(n_verts):
                y[i] = beta * y[i]
        else:
            for i in range(n_verts):
                y[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Clear work buffer
        for i in range(n_verts):
            self.y_buffer[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        # Process each row in parallel
        for row in range(n_verts):
            start = self.row_starts[row]
            end = self.row_starts[row + 1]

            # Accumulate row contribution locally
            row_sum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

            for tid in range(start, end):
                j = self.triplet_col[tid]
                mat = self.triplet_val[tid]

                # Upper triangle contribution: y[row] += A[row,j] * x[j]
                row_sum += mat @ x[j]

                # Lower triangle contribution: y[j] += A[j,row] * x[row]
                if row != j:
                    contrib_t = mat.transpose() @ x[row]
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.y_buffer[j][d], contrib_t[d])

            # Write row sum (no atomics needed for diagonal)
            y[row] += alpha * row_sum

        # Add lower triangle contributions
        for i in range(n_verts):
            y[i] += alpha * self.y_buffer[i]

    def spmv(self, x, y, alpha: float = 1.0, beta: float = 0.0):
        """
        Compute y = alpha * A * x + beta * y

        Args:
            x: Input vector, ti.Vector.field(3, f64, shape=n_verts)
            y: Output vector, ti.Vector.field(3, f64, shape=n_verts)
            alpha: Scalar multiplier for A*x
            beta: Scalar multiplier for existing y
        """
        if not self.sorted:
            self.sort_by_row()

        self.spmv_row_parallel(x, y, alpha, beta)
