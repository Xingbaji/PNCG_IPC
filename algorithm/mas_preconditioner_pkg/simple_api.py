"""
Simple API Module: Simplified interface without meshtaichi dependency.

This module provides a simplified interface to the MAS preconditioner
that works with raw numpy/taichi arrays instead of meshtaichi meshes.
Useful for testing, benchmarking, and integration with non-meshtaichi solvers.
"""

import taichi as ti
import numpy as np
from .constants import BANKSIZE, MAX_LEVELS, WARP_REDUCTION_ENABLED


class SimpleAPIMixin:
    """
    Mixin class providing simplified API without meshtaichi dependency.

    This allows using the MAS preconditioner with direct array access,
    which is useful for:
    - Testing and benchmarking
    - Integration with non-meshtaichi solvers
    - Simpler debugging and profiling

    Required attributes from main class:
        - n_verts: int
        - n_cells: int
        - actual_levels: int
        - level_size: ti.Vector.field(2, i32)
        - neighbor_list, neighbor_start, neighbor_num: neighbor fields
        - fine_connect_mask: ti.field(u32)
        - going_next: ti.field(i32)
        - multi_level_r, multi_level_z: ti.Vector.field(3, f32)
        - block_matrices: matrix field
        - warp_prefix_cache: ti.field(i32)
        - warp_sum_buffer: ti.field(f32)
    """

    # ========================================================================
    # Hierarchy Building from Arrays
    # ========================================================================

    def _build_hierarchy_from_adjacency(self, neighbor_list_np: np.ndarray,
                                        neighbor_starts_np: np.ndarray):
        """
        Build hierarchy from pre-computed adjacency arrays.

        This provides a meshtaichi-free interface for building the hierarchy.

        Args:
            neighbor_list_np: Flat array of neighbor indices (int32)
            neighbor_starts_np: CSR-style starts array (int32), length n_verts+1
        """
        # Copy neighbor data to fields
        total_len = min(len(neighbor_list_np), self.neighbor_list.shape[0])
        temp_neighbor_list = np.zeros(self.neighbor_list.shape[0], dtype=np.int32)
        temp_neighbor_list[:total_len] = neighbor_list_np[:total_len]
        self.neighbor_list.from_numpy(temp_neighbor_list)
        self.neighbor_start.from_numpy(neighbor_starts_np)
        self.total_neighbors = len(neighbor_list_np)

        # Compute neighbor counts
        neighbor_num_np = np.diff(neighbor_starts_np)
        self.neighbor_num.from_numpy(neighbor_num_np)

        # Build connectivity mask
        self._build_connect_mask_from_csr()

        # Build hierarchy levels
        self._build_coarse_levels_from_scratch()

        # Cache warp prefix for P1 optimization
        self._cache_warp_prefix()

        self.hierarchy_built = True
        print(f"[MAS] Hierarchy built: {self.actual_levels} levels")

    @ti.kernel
    def _build_connect_mask_from_csr(self):
        """Build connectivity bitmask from CSR neighbor format."""
        for i in range(self.n_verts):
            warp_id = i // BANKSIZE
            lane_id = i % BANKSIZE
            mask = ti.u32(0)

            # Self-connection
            mask |= ti.u32(1) << lane_id

            # Add neighbors in same warp
            start = self.neighbor_start[i]
            end = self.neighbor_start[i + 1]
            for k in range(start, end):
                neighbor = self.neighbor_list[k]
                if neighbor // BANKSIZE == warp_id:
                    neighbor_lane = neighbor % BANKSIZE
                    mask |= ti.u32(1) << neighbor_lane

            self.fine_connect_mask[i] = mask

    def _build_coarse_levels_from_scratch(self):
        """Build coarse levels without mesh dependency."""
        # Initialize level 0
        self.level_size[0] = ti.Vector([self.n_verts, 0], dt=ti.i32)

        current_size = self.n_verts
        current_offset = 0
        actual_levels = 1

        for level in range(1, MAX_LEVELS):
            # Compute next level size
            next_offset = current_offset + current_size
            next_size = (current_size + BANKSIZE - 1) // BANKSIZE

            if next_size < 1:
                break

            self.level_size[level] = ti.Vector([next_size, next_offset], dt=ti.i32)

            # Build going_next mapping
            self._build_going_next_level(current_offset, current_size, next_offset)

            current_offset = next_offset
            current_size = next_size
            actual_levels += 1

            if next_size <= BANKSIZE:
                break

        self.actual_levels = actual_levels
        self.level_size[actual_levels] = ti.Vector([0, current_offset + current_size], dt=ti.i32)

    @ti.kernel
    def _build_going_next_level(self, current_offset: ti.i32, current_size: ti.i32, next_offset: ti.i32):
        """Build going_next mapping for one level."""
        for i in range(current_size):
            global_idx = current_offset + i
            parent_idx = next_offset + i // BANKSIZE
            self.going_next[global_idx] = parent_idx

    # ========================================================================
    # Matrix Assembly from Arrays
    # ========================================================================

    def assemble_block_matrices_simple(self, solver, x_field, cell_verts_field,
                                       cell_B_field, cell_W_field):
        """
        Assemble block matrices using direct field access (no meshtaichi).

        Args:
            solver: Object with mu, la, dt attributes
            x_field: Vertex positions, ti.Vector.field(3, float, shape=n_verts)
            cell_verts_field: Cell vertex indices, ti.field(int, shape=(n_cells, 4))
            cell_B_field: Cell inverse rest matrices, ti.Matrix.field(3,3, shape=n_cells)
            cell_W_field: Cell volumes, ti.field(float, shape=n_cells)
        """
        print("[MAS] Assembling block matrices (simple interface)...")

        # Clear block matrices
        self._clear_block_matrices()

        # Assemble using simple kernel
        self._assemble_simple_kernel(x_field, cell_verts_field, cell_B_field, cell_W_field,
                                     solver.mu, solver.la, solver.dt)

        # Add mass/regularization
        self._add_mass_term_simple(solver.dt)

        self.matrices_assembled = True

    @ti.kernel
    def _assemble_simple_kernel(self, x: ti.template(), cell_verts: ti.template(),
                                cell_B: ti.template(), cell_W: ti.template(),
                                mu: ti.f32, la: ti.f32, dt: ti.f32):
        """Kernel for assembling block matrices from separate fields."""
        W_scale = dt * dt

        for c in range(self.n_cells):
            # Get vertex indices
            v0 = cell_verts[c, 0]
            v1 = cell_verts[c, 1]
            v2 = cell_verts[c, 2]
            v3 = cell_verts[c, 3]

            vids = ti.Vector([v0, v1, v2, v3])

            # Compute deformation gradient
            x0 = x[v0]
            Ds = ti.Matrix.cols([x[v1] - x0, x[v2] - x0, x[v3] - x0])
            B = cell_B[c]
            F = Ds @ B
            W = cell_W[c]

            # Compute Hessian (ARAP for simplicity)
            # Full Hessian requires SVD and material-specific d2PsidF2
            # For benchmark, use simplified diagonal approximation
            diag_val = 2.0 * mu * W * W_scale

            # Add to block matrices
            for i in ti.static(range(4)):
                vi = vids[i]
                block_i = vi // BANKSIZE
                lane_i = vi % BANKSIZE

                # Diagonal entry: sym_idx = BANKSIZE * lane - lane * (lane + 1) / 2 + lane
                sym_idx = BANKSIZE * lane_i - lane_i * (lane_i + 1) // 2 + lane_i
                diag_mat = ti.Matrix.identity(ti.f32, 3) * diag_val
                ti.atomic_add(self.block_matrices[block_i, sym_idx], diag_mat)

                # Off-diagonal within same block
                for j in ti.static(range(i)):
                    vj = vids[j]
                    block_j = vj // BANKSIZE
                    if block_j == block_i:
                        lane_j = vj % BANKSIZE
                        # Symmetric storage: use upper triangle (min_lane, max_lane)
                        min_lane = ti.min(lane_i, lane_j)
                        max_lane = ti.max(lane_i, lane_j)
                        sym_idx_ij = BANKSIZE * min_lane - min_lane * (min_lane + 1) // 2 + max_lane
                        # Coupling term (simplified)
                        coupling = ti.Matrix.identity(ti.f32, 3) * (diag_val * 0.1)
                        ti.atomic_add(self.block_matrices[block_i, sym_idx_ij], coupling)

    @ti.kernel
    def _add_mass_term_simple(self, dt: ti.f32):
        """Add mass/inertia term to diagonal."""
        mass_scale = 1.0 / (dt * dt)
        for i in range(self.n_verts):
            block_id = i // BANKSIZE
            lane_id = i % BANKSIZE
            # Correct symmetric index formula
            sym_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id
            self.block_matrices[block_id, sym_idx] += ti.Matrix.identity(ti.f32, 3) * mass_scale

    # ========================================================================
    # Apply Preconditioner from Arrays
    # ========================================================================

    def apply_simple(self, grad_field, z_field, use_warp_reduction: bool = True):
        """
        Apply preconditioner using direct field access.

        Args:
            grad_field: Input gradient, ti.Vector.field(3, float, shape=n_verts)
            z_field: Output preconditioned direction, ti.Vector.field(3, float, shape=n_verts)
            use_warp_reduction: Whether to use P1 warp reduction optimization
        """
        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction (copy gradient to multi_level_r)
        if use_warp_reduction and WARP_REDUCTION_ENABLED:
            self._clear_warp_sum_buffer()
            self._restrict_simple_optimized(grad_field)
        else:
            self._restrict_simple(grad_field)

        # Phase 2: Local solve
        self._schwarz_local_solve_full()

        # Phase 3: Prolongation
        self._prolong_simple(z_field)

    @ti.kernel
    def _restrict_simple(self, grad_field: ti.template()):
        """Restrict gradient to multi-level residual (level 0)."""
        for i in range(self.n_verts):
            self.multi_level_r[i] = ti.cast(grad_field[i], ti.f32)

        # Propagate to coarse levels
        for level in range(1, self.actual_levels):
            level_info = self.level_size[level - 1]
            prev_size = level_info[0]
            prev_offset = level_info[1]

            next_info = self.level_size[level]
            next_offset = next_info[1]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                next_idx = next_offset + i // BANKSIZE
                ti.atomic_add(self.multi_level_r[next_idx], self.multi_level_r[prev_idx])

    @ti.kernel
    def _restrict_simple_optimized(self, grad_field: ti.template()):
        """Optimized restriction using warp-level patterns."""
        # Phase 1: Copy gradient to level 0
        for i in range(self.n_verts):
            self.multi_level_r[i] = ti.cast(grad_field[i], ti.f32)

        # Phase 2 & 3: Use existing optimized kernels
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id in range(n_warps):
            prefix = self.warp_prefix_cache[warp_id]

            if prefix == 1:
                # Fully connected: sum all nodes
                warp_sum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                warp_start = warp_id * BANKSIZE
                warp_end = ti.min(warp_start + BANKSIZE, self.n_verts)

                for i in range(warp_start, warp_end):
                    warp_sum += self.multi_level_r[i]

                # Store to coarse level
                if self.actual_levels > 1:
                    next_offset = self.level_size[1][1]
                    self.multi_level_r[next_offset + warp_id] = warp_sum
            else:
                # Multi-component: use atomic
                if self.actual_levels > 1:
                    next_offset = self.level_size[1][1]
                    warp_start = warp_id * BANKSIZE
                    warp_end = ti.min(warp_start + BANKSIZE, self.n_verts)

                    for i in range(warp_start, warp_end):
                        ti.atomic_add(self.multi_level_r[next_offset + warp_id], self.multi_level_r[i])

        # Higher levels
        for level in range(2, self.actual_levels):
            prev_info = self.level_size[level - 1]
            prev_size = prev_info[0]
            prev_offset = prev_info[1]

            next_info = self.level_size[level]
            next_offset = next_info[1]

            for i in range(prev_size):
                prev_idx = prev_offset + i
                next_idx = next_offset + i // BANKSIZE
                ti.atomic_add(self.multi_level_r[next_idx], self.multi_level_r[prev_idx])

    @ti.kernel
    def _prolong_simple(self, z_field: ti.template()):
        """Prolong solution back to z_field."""
        for i in range(self.n_verts):
            # Collect from all levels
            z_total = self.multi_level_z[i]

            # Add coarse level contributions
            for level in ti.static(range(1, MAX_LEVELS)):
                if level < self.actual_levels:
                    # Find coarse index
                    coarse_idx = i
                    for l in range(level):
                        level_info = self.level_size[l]
                        offset = level_info[1]
                        coarse_idx = self.level_size[l + 1][1] + (coarse_idx - offset) // BANKSIZE
                    z_total += self.multi_level_z[coarse_idx]

            z_field[i] = ti.cast(z_total, ti.f32)
