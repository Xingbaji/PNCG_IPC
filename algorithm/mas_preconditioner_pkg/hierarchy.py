"""
Hierarchy Module: Multi-level restriction and prolongation operations.

This module implements the hierarchical operations for the MAS preconditioner:
1. Restriction: gradient -> multi_level_r (fine to coarse)
2. Prolongation: multi_level_z -> z (coarse to fine)

Reference: MASPreconditioner.cu lines 729-879
"""

import taichi as ti
from .constants import BANKSIZE, WARP_REDUCTION_ENABLED


class HierarchyMixin:
    """
    Mixin class providing multi-level restriction and prolongation operations.

    This class should be mixed into the main MASPreconditioner class.

    Required attributes from main class:
        - n_verts: int
        - level_num: int
        - actual_levels: int
        - total_nodes_all_levels: int
        - multi_level_r: ti.Vector.field(3, f32)
        - multi_level_z: ti.Vector.field(3, f32)
        - mesh: meshtaichi mesh with verts.grad, verts.z
        - fine_connect_mask: ti.field(u32)
        - going_next: ti.field(i32)
        - aggregation_table: ti.field(i32)
        - level_size: ti.Vector.field(2, i32)
        - prefix_original: ti.field(i32)
        - warp_prefix_cache: ti.field(i32)
        - warp_sum_buffer: ti.field(f32)
    """

    # ========================================================================
    # Buffer Management
    # ========================================================================

    @ti.kernel
    def _clear_multi_level_buffers(self):
        """Clear multi-level residual and solution buffers."""
        for i in range(self.total_nodes_all_levels):
            self.multi_level_r[i] = ti.Vector.zero(ti.f32, 3)
            self.multi_level_z[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _clear_warp_sum_buffer(self):
        """Clear warp sum buffer."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id, lane_id, d in ti.ndrange(n_warps, BANKSIZE, 3):
            self.warp_sum_buffer[warp_id, lane_id, d] = 0.0

    @ti.kernel
    def _cache_warp_prefix(self):
        """Cache prefix_original values for each warp for fast access."""
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE
        for warp_id in range(n_warps):
            self.warp_prefix_cache[warp_id] = self.prefix_original[warp_id]

    # ========================================================================
    # Restriction: Fine to Coarse (Original Version)
    # ========================================================================

    @ti.kernel
    def _build_multi_level_r(self):
        """
        Hierarchically restrict gradient to coarse levels (original version).

        CUDA Reference: __buildMultiLevelR_optimized_new() (MASPreconditioner.cu lines 729-847)

        This implements a multi-level restriction with optimization for
        fully-connected blocks (prefix == 1) using reduction, and shared
        memory accumulation for multi-component blocks.

        Level 0: r = g (gradient)
        Level l: r_l = sum of r_{l-1} within cluster, propagated through hierarchy
        """
        # Copy gradient to level 0
        for idx in range(self.n_verts):
            self.multi_level_r[idx] = ti.cast(self.mesh.verts.grad[idx], ti.f32)

        # Restrict to all coarse levels through hierarchy
        for idx in range(self.n_verts):
            r = self.multi_level_r[idx]
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Get connectivity info
            connect_mask = self.fine_connect_mask[idx]

            # Check if this vertex is an elected representative
            elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))

            if elected_prefix == 0:
                # This is a representative - propagate to all coarse levels
                current_idx = idx
                for _ in range(self.level_num - 1):
                    next_idx = self.going_next[current_idx]
                    if next_idx >= 0 and next_idx < self.total_nodes_all_levels:
                        # Accumulate to coarse level
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                        current_idx = next_idx
                    else:
                        break
            else:
                # Non-representative: find elected lane and add to its accumulator
                elected_lane = self._find_first_set(connect_mask)
                elected_idx = warp_id * BANKSIZE + elected_lane

                # Contribute to the elected node's value (already in multi_level_r)
                # The elected node will propagate the sum to coarse levels
                # We accumulate locally first
                if elected_idx < self.n_verts and elected_idx != idx:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.multi_level_r[elected_idx][d], r[d])

    # ========================================================================
    # P1 Optimization: Warp-level Parallel Reduction for Restriction
    # ========================================================================
    #
    # CUDA reference uses __shfl_down_sync for warp reduction when prefix==1
    # (fully connected warp). Taichi 1.7.4 does not support ti.simt warp
    # primitives, so we use a 3-phase approach:
    #
    # Phase 1: Copy gradients to Level 0 and warp_sum_buffer
    # Phase 2: Parallel tree reduction within each warp using warp_sum_buffer
    # Phase 3: Elected nodes propagate reduced sums to coarse levels
    #
    # For multi-component warps (prefix > 1), we fall back to atomic accumulation
    # to the elected node of each component.

    @ti.kernel
    def _build_multi_level_r_phase1(self):
        """
        Phase 1: Copy gradients to Level 0 and initialize warp_sum_buffer.

        For each vertex, copy gradient to multi_level_r and warp_sum_buffer.
        """
        for idx in range(self.n_verts):
            r = ti.cast(self.mesh.verts.grad[idx], ti.f32)
            self.multi_level_r[idx] = r

            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE

            # Copy to warp_sum_buffer for reduction
            for d in ti.static(range(3)):
                self.warp_sum_buffer[warp_id, lane_id, d] = r[d]

    @ti.kernel
    def _build_multi_level_r_phase2_tree_reduce(self):
        """
        Phase 2: Tree reduction within fully-connected warps (prefix == 1).

        For warps where all vertices are connected (prefix_original == 1),
        perform parallel tree reduction: O(log BANKSIZE) steps instead of
        O(BANKSIZE) atomic operations.

        Tree reduction pattern (BANKSIZE=16):
            Step 0: lane 0 += lane 8,  lane 1 += lane 9,  ..., lane 7 += lane 15
            Step 1: lane 0 += lane 4,  lane 1 += lane 5,  ..., lane 3 += lane 7
            Step 2: lane 0 += lane 2,  lane 1 += lane 3
            Step 3: lane 0 += lane 1

        After reduction, lane 0 holds the sum of all 16 vertices.
        """
        n_warps = (self.n_verts + BANKSIZE - 1) // BANKSIZE

        # Process each warp
        for warp_id in range(n_warps):
            prefix = self.warp_prefix_cache[warp_id]

            if prefix == 1:
                # Fully connected warp: use tree reduction
                # Step 0: stride = 8
                for lane_id in range(8):
                    src_lane = lane_id + 8
                    src_idx = warp_id * BANKSIZE + src_lane
                    if src_idx < self.n_verts:
                        for d in ti.static(range(3)):
                            self.warp_sum_buffer[warp_id, lane_id, d] += \
                                self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 1: stride = 4
                for lane_id in range(4):
                    src_lane = lane_id + 4
                    for d in ti.static(range(3)):
                        self.warp_sum_buffer[warp_id, lane_id, d] += \
                            self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 2: stride = 2
                for lane_id in range(2):
                    src_lane = lane_id + 2
                    for d in ti.static(range(3)):
                        self.warp_sum_buffer[warp_id, lane_id, d] += \
                            self.warp_sum_buffer[warp_id, src_lane, d]

                # Step 3: stride = 1
                for d in ti.static(range(3)):
                    self.warp_sum_buffer[warp_id, 0, d] += \
                        self.warp_sum_buffer[warp_id, 1, d]

    @ti.kernel
    def _build_multi_level_r_phase2_multi_component(self):
        """
        Phase 2b: Accumulate to elected nodes for multi-component warps (prefix > 1).

        For warps with multiple connected components, each vertex atomically
        adds to its elected representative in warp_sum_buffer.
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            prefix = self.warp_prefix_cache[warp_id]

            if prefix > 1:
                # Multi-component warp: use atomic accumulation to elected node
                connect_mask = self.fine_connect_mask[idx]
                elected_lane = self._find_first_set(connect_mask)

                # Only non-elected nodes accumulate to elected node
                if elected_lane != lane_id:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.warp_sum_buffer[warp_id, elected_lane, d],
                                      self.warp_sum_buffer[warp_id, lane_id, d])

    @ti.kernel
    def _build_multi_level_r_phase3_propagate(self):
        """
        Phase 3: Propagate reduced sums from elected nodes to coarse levels.

        For fully-connected warps (prefix==1), only lane 0 propagates.
        For multi-component warps, each elected node propagates its component sum.
        """
        for idx in range(self.n_verts):
            warp_id = idx // BANKSIZE
            lane_id = idx % BANKSIZE
            prefix = self.warp_prefix_cache[warp_id]

            # Determine if this vertex should propagate to coarse levels
            should_propagate = False

            if prefix == 1:
                # Fully connected warp: only lane 0 propagates (holds total sum)
                should_propagate = (lane_id == 0)
            else:
                # Multi-component warp: elected nodes propagate
                connect_mask = self.fine_connect_mask[idx]
                elected_prefix = self._popcount(connect_mask & self._lanemask_lt(lane_id))
                should_propagate = (elected_prefix == 0)

            if should_propagate:
                # Get the reduced sum from warp_sum_buffer
                r = ti.Vector([
                    self.warp_sum_buffer[warp_id, lane_id, 0],
                    self.warp_sum_buffer[warp_id, lane_id, 1],
                    self.warp_sum_buffer[warp_id, lane_id, 2]
                ], dt=ti.f32)

                # Propagate to all coarse levels
                current_idx = idx
                for _ in range(self.level_num - 1):
                    next_idx = self.going_next[current_idx]
                    if next_idx >= 0 and next_idx < self.total_nodes_all_levels:
                        for d in ti.static(range(3)):
                            ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                        current_idx = next_idx
                    else:
                        break

    def _build_multi_level_r_optimized(self):
        """
        P1 Optimized version of multi-level restriction.

        Uses 3-phase approach for warp-level parallelism:
        - Phase 1: Copy gradients to buffers
        - Phase 2a: Tree reduction for fully-connected warps (prefix==1)
        - Phase 2b: Atomic accumulation for multi-component warps
        - Phase 3: Elected nodes propagate to coarse levels

        This reduces atomic operations from O(n_verts) to O(n_warps * avg_components).
        """
        # Phase 1: Initialize
        self._build_multi_level_r_phase1()

        # Phase 2a: Tree reduction for fully-connected warps
        self._build_multi_level_r_phase2_tree_reduce()

        # Phase 2b: Atomic accumulation for multi-component warps
        self._build_multi_level_r_phase2_multi_component()

        # Phase 3: Propagate to coarse levels
        self._build_multi_level_r_phase3_propagate()

    # ========================================================================
    # Prolongation: Coarse to Fine
    # ========================================================================

    def _collect_final_z(self):
        """
        Prolongation: aggregate solutions from all levels.
        z = z_0 + C_1^T * z_1 + C_2^T * z_2 + ...

        CUDA Reference: __collectFinalZ_new() (MASPreconditioner.cu lines 850-879)

        The aggregation table stores the path through hierarchy for each vertex:
            aggregation_table[idx][level] = coarse node index at (level+1)

        CUDA loop: for(int i = 1; i < levelnum; i++) { now = tablePtr[i-1]; }
        Python equivalent: for level in range(level_num - 1): coarse_idx = table[level]
        This gives tablePtr[0], tablePtr[1], ..., tablePtr[levelnum-2]
        """
        self._collect_final_z_kernel(self.actual_levels)

    @ti.kernel
    def _collect_final_z_kernel(self, level_num: ti.i32):
        """
        Kernel for prolongation phase.

        For each fine vertex, sum up solutions from all levels:
        - Level 0: direct solution at vertex index
        - Level 1..N: solutions at coarse nodes via aggregation_table
        """
        for idx in range(self.n_verts):
            # Start with Level 0 solution
            z_total = self.multi_level_z[idx]

            # Add contributions from all coarse levels using aggregation table
            # CUDA: for(int i = 1; i < levelnum; i++) { now = tablePtr[i-1]; ... }
            for level in range(level_num - 1):
                coarse_idx = self.aggregation_table[idx][level]
                if coarse_idx >= 0 and coarse_idx < self.total_nodes_all_levels:
                    z_total += self.multi_level_z[coarse_idx]

            # Store result (cast to f32 for mesh storage)
            self.mesh.verts.z[idx] = ti.cast(z_total, ti.f32)

    # ========================================================================
    # Simple API Prolongation (without mesh)
    # ========================================================================

    @ti.kernel
    def _collect_final_z_simple_kernel(self, z_out: ti.template(), level_num: ti.i32):
        """
        Prolongation kernel for simple API (without mesh).

        Args:
            z_out: Output vector field, ti.Vector.field(3, dtype=ti.f32/f64, shape=n_verts)
            level_num: Number of levels in hierarchy
        """
        for idx in range(self.n_verts):
            # Start with Level 0 solution
            z_total = self.multi_level_z[idx]

            # Add contributions from all coarse levels using aggregation table
            for level in range(level_num - 1):
                coarse_idx = self.aggregation_table[idx][level]
                if coarse_idx >= 0 and coarse_idx < self.total_nodes_all_levels:
                    z_total += self.multi_level_z[coarse_idx]

            # Store result
            z_out[idx] = z_total

    def _collect_final_z_simple(self, z_out):
        """
        Prolongation for simple API (without mesh).

        Args:
            z_out: Output vector field, ti.Vector.field(3, dtype, shape=n_verts)
        """
        self._collect_final_z_simple_kernel(z_out, self.actual_levels)

    # ========================================================================
    # Main Apply Method (Restriction + Local Solve + Prolongation)
    # ========================================================================

    def apply(self, use_full_solve: bool = True, use_parallel_solve: bool = False,
              use_warp_reduction: bool = True, use_conflict_free: bool = False,
              use_banded: bool = False):
        """
        Apply MAS preconditioner: z = P * grad

        Three phases:
        1. Restriction: gradient -> multi_level_r
        2. Local solve: multi_level_r -> multi_level_z
        3. Prolongation: multi_level_z -> z

        Args:
            use_full_solve: If True, use full block inverse in local solve.
                           If False, use diagonal-only approximation.
            use_parallel_solve: If True, use parallelized local solve with atomics.
                               This can be faster on GPUs with many cores.
            use_warp_reduction: If True, use P1 optimized warp-level reduction
                               for restriction phase. Default True.
            use_conflict_free: If True, use P5 conflict-free SpMV for local solve.
                              This uses unrolled matrix-vector multiply for better
                              instruction-level parallelism. Default False.
            use_banded: If True, use P6 banded sparse MV for local solve.
                       This is optimized for IC(0) which produces banded inverse.
                       Only accesses node pairs within NODE_BANDWIDTH=2.
                       Should be used together with IC(0) inversion for best results.
        """
        # Clear buffers
        self._clear_multi_level_buffers()

        # Phase 1: Restriction
        if use_warp_reduction and WARP_REDUCTION_ENABLED:
            self._clear_warp_sum_buffer()
            self._build_multi_level_r_optimized()
        else:
            self._build_multi_level_r()

        # Phase 2: Local solve (delegated to SchwarzMixin)
        if use_full_solve:
            if use_banded:
                # P6 optimization: banded sparse MV for IC(0)
                self._schwarz_local_solve_banded()
            elif use_conflict_free:
                # P5 optimization: conflict-free SpMV with unrolled MatVec
                self._schwarz_local_solve_conflict_free()
            elif use_parallel_solve:
                self._schwarz_local_solve_full_parallel()
            else:
                self._schwarz_local_solve_full()
        else:
            self._schwarz_local_solve()

        # Phase 3: Prolongation
        self._collect_final_z()
