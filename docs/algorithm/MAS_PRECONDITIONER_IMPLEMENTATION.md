# MAS Preconditioner: Detailed Implementation Guide

This document provides a comprehensive guide to implementing the **Multilevel Additive Schwarz (MAS) Preconditioner** based on the StiffGIPC paper and the CUDA reference implementation at `/root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu`.

## Table of Contents
1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Data Structures](#3-data-structures)
4. [Hierarchy Construction](#4-hierarchy-construction)
5. [Matrix Assembly](#5-matrix-assembly)
6. [Block Matrix Inversion](#6-block-matrix-inversion)
7. [Preconditioning Operation](#7-preconditioning-operation)
8. [GPU Optimization Techniques](#8-gpu-optimization-techniques)
9. [Key Parameters](#9-key-parameters)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Mathematical Foundation

### 1.1 MAS Preconditioner Formula

The MAS preconditioner is defined as:

```
M_MAS^{-1} = M_{(0)}^{-1} + Σ_{l=1}^{L} C_{(l)}^T M_{(l)}^{-1} C_{(l)}
```

Where:
- **M_{(0)}^{-1}**: Level-0 Additive Schwarz (fine-level local solves)
- **C_{(l)}**: Coarsening/restriction matrix at level l (binary aggregation)
- **M_{(l)}^{-1}**: Single-level AS preconditioner for coarse level l
- **L**: Total number of coarse levels

### 1.2 Level-0 Additive Schwarz

```
M_{(0)}^{-1} = Σ_d S_d^T M_{d,(0)}^{-1} S_d
```

Where:
- **S_d ∈ R^{3N_d × 3N}**: Selection matrix extracting N_d nodes from domain d
- **M_{d,(0)} = S_d A S_d^T**: Local Hessian within domain d (BANKSIZE × BANKSIZE nodes)
- Domains are **non-overlapping**: Ω = ∪ Ω_d

### 1.3 Coarse Level Construction

The coarsening matrix C_{(l)} maps fine-level nodes to coarse supernodes:
- Connected nodes within a subdomain merge into one supernode
- Binary aggregation: C_{(l)}[i,j] = 1 if fine node i maps to coarse node j

---

## 2. Algorithm Overview

### 2.1 Three Main Phases

1. **Hierarchy Construction** (per frame, when restart)
   - Build mesh connectivity graph
   - Partition nodes into BANKSIZE subdomains
   - Find connected components within each subdomain
   - Create coarse levels by merging connected nodes

2. **Matrix Assembly** (per Newton iteration)
   - Assemble local Hessian blocks for each subdomain
   - Include: mass matrix, elastic Hessian, barrier Hessian
   - Invert local blocks (48×48 Gauss-Jordan for BANKSIZE=16)

3. **Preconditioning** (per PCG iteration)
   - **Restriction**: g → multi_level_r (hierarchically restrict gradient)
   - **Local Solve**: z_l = M_l^{-1} r_l at each level
   - **Prolongation**: Aggregate solutions from all levels → z

### 2.2 Connectivity-Enhanced MAS (CEMAS)

The StiffGIPC paper proposes using **METIS graph partitioning** instead of Morton code sorting:

1. Build mesh adjacency graph from element connectivity
2. Run METIS to partition nodes into M groups (minimizing inter-partition edges)
3. Map METIS partitions to MAS subdomains with padding
4. Result: Dense connectivity within each subdomain → better aggregation

---

## 3. Data Structures

### 3.1 Core Arrays

```python
# Constants
BANKSIZE = 16          # Nodes per subdomain (matches warp size for GPU)
MAX_LEVELS = 6         # Maximum hierarchy depth
SYM_BLOCK_COUNT = 136  # BANKSIZE * (BANKSIZE + 1) / 2 for symmetric storage

# Hierarchy Management
level_size: [MAX_LEVELS, 2]         # (num_nodes, offset) at each level
coarse_space_tables: [MAX_LEVELS, n_verts]  # Fine→coarse node mapping
going_next: [total_nodes_all_levels]         # Node→parent mapping
aggregation_table: [n_verts, MAX_LEVELS-1]   # Path through hierarchy

# Connectivity
neighbor_list: [max_edges]          # CSR neighbor indices
neighbor_start: [n_verts + 1]       # CSR row pointers
neighbor_num: [n_verts]             # Neighbor count per vertex
fine_connect_mask: [n_verts]        # 32-bit connectivity bitmask (u32)
next_connect_mask: [n_verts]        # Bitmask for next level

# Matrix Storage (Symmetric Upper Triangle)
block_matrices: [n_blocks, SYM_BLOCK_COUNT, 3, 3]      # Original blocks (f64)
inv_block_matrices: [n_blocks, SYM_BLOCK_COUNT, 3, 3]  # Inverted blocks (f32)

# Preconditioning Buffers
multi_level_r: [total_nodes_all_levels, 3]  # Restricted residuals
multi_level_z: [total_nodes_all_levels, 3]  # Solutions at each level
```

### 3.2 Symmetric Block Storage

For a BANKSIZE×BANKSIZE block matrix with 3×3 sub-blocks:
- Total: 16×16 = 256 sub-blocks
- Upper triangle: 16×17/2 = 136 sub-blocks
- Index formula: `sym_idx(row, col) = BANKSIZE * min(row,col) - min(row,col)*(min(row,col)+1)/2 + max(row,col)`

### 3.3 CUDA Reference Data Structures

```cpp
// From MASPreconditioner.cuh
struct MasMatrixSymf {
    Eigen::Matrix3f M[BANKSIZE * (BANKSIZE + 1) / 2];  // 136 × 3×3 blocks
};

struct itable {
    int index[6];  // Path through 6-level hierarchy for prolongation
};
```

---

## 4. Hierarchy Construction

### 4.1 Level 0: Build Connectivity Mask

For each vertex idx with warp_id = idx // BANKSIZE, lane_id = idx % BANKSIZE:

```python
@ti.kernel
def _build_connect_mask_l0():
    for idx in range(n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE

        # Start with self-connectivity
        connect_mask = ti.u32(1) << ti.u32(lane_id)

        # Add neighbors in same warp
        for i in range(neighbor_num[idx]):
            neighbor_id = neighbor_list[neighbor_start[idx] + i]
            neighbor_warp = neighbor_id // BANKSIZE

            if warp_id == neighbor_warp:
                neighbor_lane = neighbor_id % BANKSIZE
                connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))

        fine_connect_mask[idx] = connect_mask
```

### 4.2 Connectivity Propagation (Transitive Closure)

Find connected components by iteratively merging connectivity masks:

```python
@ti.kernel
def _propagate_connectivity():
    for idx in range(n_verts):
        warp_id = idx // BANKSIZE
        connect_mask = fine_connect_mask[idx]
        visited = ti.u32(1) << ti.u32(idx % BANKSIZE)

        # Iterate until no new connections found
        while connect_mask != ti.u32(0xFFFFFFFF):  # Not all bits set
            todo = visited ^ connect_mask
            if todo == 0:
                break

            # Find first unvisited connected node
            next_visit = find_first_set(todo)  # __ffs(todo) - 1
            visited |= (ti.u32(1) << ti.u32(next_visit))

            # Merge that node's connections
            other_idx = warp_id * BANKSIZE + next_visit
            if other_idx < n_verts:
                connect_mask |= fine_connect_mask[other_idx]

        fine_connect_mask[idx] = connect_mask
```

### 4.3 Find Cluster Representatives

A node is elected as representative if it has the lowest lane_id in its connected component:

```python
@ti.kernel
def _find_cluster_representatives():
    for idx in range(n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE
        connect_mask = fine_connect_mask[idx]

        # Count connected nodes with lower lane_id
        prefix = popcount(connect_mask & lanemask_lt(lane_id))

        if prefix == 0:
            # This is the representative of its cluster
            atomic_add(prefix_count[warp_id], 1)
            atomic_or(elected_mask[warp_id], ti.u32(1) << ti.u32(lane_id))
```

### 4.4 Assign Cluster IDs

```python
@ti.kernel
def _assign_cluster_ids(level_offset: ti.i32):
    for idx in range(n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE
        connect_mask = fine_connect_mask[idx]

        prefix = popcount(connect_mask & lanemask_lt(lane_id))

        if prefix == 0:
            # Elected node: compute global cluster ID
            base_id = prefix_sum[warp_id]  # From exclusive scan
            local_offset = popcount(elected_mask[warp_id] & lanemask_lt(lane_id))
            cluster_id = base_id + local_offset

            coarse_space_tables[0, idx] = cluster_id
            going_next[idx] = level_offset + cluster_id
        else:
            # Non-elected: find representative and copy its cluster ID
            rep_lane = find_first_set(connect_mask)
            rep_idx = warp_id * BANKSIZE + rep_lane
            # Copy in second pass after all representatives assigned
```

### 4.5 Multi-Level Construction (Levels 2+)

Recursively apply the same process:

```python
def build_hierarchy():
    # Level 0 → Level 1
    _build_connect_mask_l0()
    _propagate_connectivity()
    _find_cluster_representatives()
    prefix_sum = exclusive_scan(prefix_count)
    level_1_size = prefix_sum[-1] + prefix_count[-1]
    _assign_cluster_ids(level_1_offset)

    # For each subsequent level
    current_size = level_1_size
    for level in range(2, MAX_LEVELS):
        if current_size <= BANKSIZE:
            break

        # Build connectivity at this level
        _build_connect_mask_lx(level)
        _propagate_connectivity_lx(level)
        _find_cluster_representatives_lx(level)

        # Compute next level size
        prefix_sum = exclusive_scan(prefix_count)
        next_level_size = prefix_sum[-1] + prefix_count[-1]

        _assign_cluster_ids_lx(level, level_offset)
        current_size = next_level_size

    # Build aggregation table for fast prolongation
    _build_aggregation_table()
```

### 4.6 Contact-Aware Hierarchy Update

Include collision pairs in connectivity:

```python
@ti.kernel
def _update_collision_connectivity(collision_pairs, level: ti.i32):
    for cp_idx in range(num_collision_pairs):
        # Get nodes in collision stencil (PP: 2, PE: 3, PT: 4, EE: 4)
        nodes = collision_pairs[cp_idx]

        for i in range(num_nodes_in_stencil):
            for j in range(i + 1, num_nodes_in_stencil):
                if level == 0:
                    ni = map_array[nodes[i]]
                    nj = map_array[nodes[j]]
                else:
                    ni = coarse_space_tables[level-1, nodes[i]]
                    nj = coarse_space_tables[level-1, nodes[j]]

                # If in same subdomain, update connectivity
                if ni // BANKSIZE == nj // BANKSIZE:
                    lane_i = ni % BANKSIZE
                    lane_j = nj % BANKSIZE
                    atomic_or(connect_mask[nodes[i]], ti.u32(1) << ti.u32(lane_j))
                    atomic_or(connect_mask[nodes[j]], ti.u32(1) << ti.u32(lane_i))
```

---

## 5. Matrix Assembly

### 5.1 Hessian Block Assembly

For each subdomain (warp), assemble a BANKSIZE×BANKSIZE block matrix:

```python
@ti.kernel
def _assemble_block_matrices(dt: ti.f64):
    # Clear matrices
    for block_id, sym_idx in ti.ndrange(n_blocks, SYM_BLOCK_COUNT):
        block_matrices[block_id, sym_idx] = ti.Matrix.zero(ti.f64, 3, 3)

    # Add mass matrix contribution (diagonal)
    for idx in range(n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE
        sym_idx = sym_index(lane_id, lane_id)

        mass = mesh.verts.m[idx]
        for d in ti.static(range(3)):
            ti.atomic_add(block_matrices[warp_id, sym_idx][d, d], mass)

    # Add elastic Hessian contribution
    for c in mesh.cells:
        # Get vertex IDs
        v_ids = [c.verts[i].id for i in range(4)]

        # Compute local 12×12 Hessian from elastic energy
        local_H = compute_d2Psidx2(c, dt)  # Full 12×12 Hessian

        # Scatter to block matrices
        for i in range(4):
            for j in range(4):
                vi, vj = v_ids[i], v_ids[j]
                warp_i, warp_j = vi // BANKSIZE, vj // BANKSIZE

                if warp_i == warp_j:  # Same block
                    lane_i = vi % BANKSIZE
                    lane_j = vj % BANKSIZE
                    sym_idx = sym_index(lane_i, lane_j)

                    # Add 3×3 sub-block
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            ti.atomic_add(
                                block_matrices[warp_i, sym_idx][di, dj],
                                local_H[i*3 + di, j*3 + dj] * dt * dt
                            )

    # Add IPC barrier Hessian contribution (similar pattern)
    # ... (iterate over contact pairs)
```

### 5.2 Coarse Level Matrix Assembly

Use warp reduction to aggregate fine-level contributions:

```python
@ti.kernel
def _assemble_coarse_matrices():
    for block_id in range(n_blocks_l0):
        # Sum all 3×3 blocks in this subdomain
        sum_block = ti.Matrix.zero(ti.f64, 3, 3)

        for sym_idx in range(SYM_BLOCK_COUNT):
            sum_block += block_matrices[block_id, sym_idx]

        # This summed block contributes to coarse level
        # Add to all coarse levels this subdomain maps to
        for level in range(1, actual_levels):
            coarse_block_id = ...  # Determined by hierarchy mapping
            ti.atomic_add(coarse_block_matrices[level, coarse_block_id], sum_block)
```

---

## 6. Block Matrix Inversion

### 6.1 Gauss-Jordan Elimination for 48×48 Block

The CUDA reference inverts the full BANKSIZE×BANKSIZE block (48×48 DOFs for BANKSIZE=16, 3 DOFs per node):

```cpp
// From MASPreconditioner.cu - Gauss-Jordan inversion
__global__ void _GaussJordan(MasMatrixSymT* d_MatMas,
                              MasMatrixSymf* d_precondMatMas,
                              int n_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    // Load 48×48 matrix into shared memory (symmetric → full)
    __shared__ double sharedMat[48][49];  // Extra column for pivot

    // Each thread handles one row/column
    int tid = threadIdx.x;  // 0-47

    // Gauss-Jordan elimination: 48 steps
    for (int pivot = 0; pivot < 48; pivot++) {
        // 1. Normalize pivot row
        double pivot_val = sharedMat[pivot][pivot];
        if (tid == pivot) {
            for (int j = 0; j < 48; j++) {
                sharedMat[pivot][j] /= pivot_val;
            }
        }
        __syncthreads();

        // 2. Eliminate in all other rows
        if (tid != pivot) {
            double factor = sharedMat[tid][pivot];
            for (int j = 0; j < 48; j++) {
                sharedMat[tid][j] -= factor * sharedMat[pivot][j];
            }
        }
        __syncthreads();
    }

    // Store result back (in single precision for efficiency)
    // ...
}
```

### 6.2 Taichi Implementation

```python
@ti.kernel
def _invert_block_matrices():
    for block_id in range(n_blocks):
        # Load symmetric block into full 48×48 matrix
        full_mat = ti.Matrix.zero(ti.f64, BANKSIZE * 3, BANKSIZE * 3)

        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                sym_idx = sym_index(row, col)
                block_3x3 = block_matrices[block_id, sym_idx]

                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        full_mat[row*3 + di, col*3 + dj] = block_3x3[di, dj]
                        full_mat[col*3 + dj, row*3 + di] = block_3x3[dj, di]  # Symmetric

        # Add regularization for numerical stability
        for i in range(BANKSIZE * 3):
            if full_mat[i, i] < 1e-10:
                full_mat[i, i] = 1e-10

        # Augment with identity for inverse
        aug_mat = ti.Matrix.zero(ti.f64, BANKSIZE * 3, BANKSIZE * 6)
        for i in range(BANKSIZE * 3):
            for j in range(BANKSIZE * 3):
                aug_mat[i, j] = full_mat[i, j]
            aug_mat[i, BANKSIZE * 3 + i] = 1.0

        # Gauss-Jordan elimination
        for pivot in range(BANKSIZE * 3):
            # Find pivot (partial pivoting for stability)
            max_val = ti.abs(aug_mat[pivot, pivot])
            max_row = pivot
            for r in range(pivot + 1, BANKSIZE * 3):
                if ti.abs(aug_mat[r, pivot]) > max_val:
                    max_val = ti.abs(aug_mat[r, pivot])
                    max_row = r

            # Swap rows if needed
            if max_row != pivot:
                for c in range(BANKSIZE * 6):
                    aug_mat[pivot, c], aug_mat[max_row, c] = aug_mat[max_row, c], aug_mat[pivot, c]

            # Scale pivot row
            pivot_val = aug_mat[pivot, pivot]
            if ti.abs(pivot_val) > 1e-12:
                for c in range(BANKSIZE * 6):
                    aug_mat[pivot, c] /= pivot_val

            # Eliminate in other rows
            for r in range(BANKSIZE * 3):
                if r != pivot:
                    factor = aug_mat[r, pivot]
                    for c in range(BANKSIZE * 6):
                        aug_mat[r, c] -= factor * aug_mat[pivot, c]

        # Extract inverse and store in symmetric format
        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                sym_idx = sym_index(row, col)
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        inv_block_matrices[block_id, sym_idx][di, dj] = ti.cast(
                            aug_mat[row*3 + di, BANKSIZE*3 + col*3 + dj], ti.f32
                        )
```

---

## 7. Preconditioning Operation

### 7.1 Overview: z = P * g

```python
def apply(gradient):
    # Phase 1: Restriction
    _build_multi_level_r(gradient)

    # Phase 2: Local solve at each level
    _schwarz_local_solve()

    # Phase 3: Prolongation
    _collect_final_z()
```

### 7.2 Restriction Phase

Hierarchically restrict gradient to coarse levels:

```python
@ti.kernel
def _build_multi_level_r():
    # Copy gradient to level 0
    for idx in range(n_verts):
        multi_level_r[idx] = mesh.verts.grad[idx]

    # Restrict to coarse levels
    for idx in range(n_verts):
        r = multi_level_r[idx]

        # Traverse hierarchy
        current_idx = idx
        for level in range(actual_levels - 1):
            next_idx = aggregation_table[idx][level]

            # Accumulate to coarse level
            for d in ti.static(range(3)):
                ti.atomic_add(multi_level_r[next_idx][d], r[d])
```

### 7.3 Local Solve Phase

Apply inverted block matrices at each level:

```python
@ti.kernel
def _schwarz_local_solve():
    # Level 0: Full block solve
    for block_id in range(n_blocks_l0):
        # Load residuals for this block
        r_block = ti.Matrix.zero(ti.f64, BANKSIZE * 3, 1)
        for lane_id in range(BANKSIZE):
            idx = block_id * BANKSIZE + lane_id
            if idx < n_verts:
                for d in ti.static(range(3)):
                    r_block[lane_id * 3 + d, 0] = multi_level_r[idx][d]

        # Apply inverted block: z = inv_M * r
        z_block = ti.Matrix.zero(ti.f64, BANKSIZE * 3, 1)

        for row in range(BANKSIZE):
            for col in range(BANKSIZE):
                sym_idx = sym_index(row, col)
                inv_block = inv_block_matrices[block_id, sym_idx]

                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        z_block[row * 3 + di, 0] += ti.f64(inv_block[di, dj]) * r_block[col * 3 + dj, 0]

                # Symmetric: also add transposed contribution
                if row != col:
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            z_block[col * 3 + di, 0] += ti.f64(inv_block[dj, di]) * r_block[row * 3 + dj, 0]

        # Store result
        for lane_id in range(BANKSIZE):
            idx = block_id * BANKSIZE + lane_id
            if idx < n_verts:
                for d in ti.static(range(3)):
                    multi_level_z[idx][d] = z_block[lane_id * 3 + d, 0]

    # Coarse levels: Similar process with coarse blocks
    for level in range(1, actual_levels):
        # ... (apply coarse-level inverted blocks)
```

### 7.4 Prolongation Phase

Aggregate solutions from all levels:

```python
@ti.kernel
def _collect_final_z():
    for idx in range(n_verts):
        # Start with level 0 solution
        z_total = multi_level_z[idx]

        # Add contributions from coarse levels
        for level in range(actual_levels - 1):
            coarse_idx = aggregation_table[idx][level]
            z_total += multi_level_z[coarse_idx]

        # Store final result
        mesh.verts.z[idx] = z_total
```

---

## 8. GPU Optimization Techniques

### 8.1 Warp-Level Operations

The CUDA reference uses BANKSIZE=16 to match warp half-size for efficient operations:

- **__ffs(x)**: Find first set bit position
- **__popc(x)**: Population count (number of set bits)
- **__shfl_down_sync()**: Warp shuffle for reduction
- **__ballot_sync()**: Gather binary votes across warp

### 8.2 Taichi Equivalents

```python
@ti.func
def popcount(x: ti.u32) -> ti.i32:
    count = 0
    while x:
        count += ti.i32(x & 1)
        x >>= 1
    return count

@ti.func
def find_first_set(x: ti.u32) -> ti.i32:
    if x == 0:
        return -1
    pos = 0
    while (x & 1) == 0:
        x >>= 1
        pos += 1
    return pos

@ti.func
def lanemask_lt(lane_id: ti.i32) -> ti.u32:
    return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)
```

### 8.3 Two-Level Reduction for Matrix Assembly

For assembling coarse-level matrices efficiently:

1. **Level 0 blocks**: Use warp reduction (no atomics)
2. **Higher levels**: Accumulate level-0 results with atomicAdd

This significantly reduces atomic conflicts compared to pure atomic accumulation.

---

## 9. Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain (GPU warp half-size) |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| SYM_BLOCK_COUNT | 136 | Symmetric storage: 16×17/2 |
| Regularization | 1e-10 | Added to diagonal for numerical stability |

### 9.1 Level Size Estimation

```python
def estimate_total_nodes():
    total = n_verts
    size = n_verts
    for level in range(MAX_LEVELS - 1):
        size = (size + BANKSIZE - 1) // BANKSIZE  # Reduction factor ~16
        total += size
    return int(total * 1.5)  # Safety buffer
```

---

## 10. Implementation Checklist

### ✅ Currently Implemented in mas_preconditioner_pkg/

> **Note:** The monolithic `mas_preconditioner.py` has been deprecated and moved to `tmp/`. Use the modular `mas_preconditioner_pkg/` package instead.
- [x] Class structure and initialization
- [x] Neighbor list from mesh topology
- [x] Level 0 connectivity mask
- [x] Connectivity propagation (transitive closure)
- [x] Cluster representative finding
- [x] **Multi-level hierarchy (Levels 0 to MAX_LEVELS)** ✅ IMPLEMENTED
- [x] **Full BANKSIZE×BANKSIZE block assembly** (with off-diagonal coupling) ✅ IMPLEMENTED
- [x] **Full 48×48 block inversion via Gauss-Jordan** ✅ IMPLEMENTED
- [x] **Proper elastic Hessian integration** (compute_d2PsidF2 for ARAP/SNH/FCR) ✅ IMPLEMENTED
- [x] Three-phase preconditioning (restrict, local solve, prolong)
- [x] Full block solve using inverted symmetric matrices ✅ IMPLEMENTED

### ❌ Remaining Improvements (Future Work)
- [ ] **Contact-aware hierarchy update** (dynamic collision connectivity)
- [ ] **Warp reduction optimization** for matrix assembly

### ✅ Recently Implemented (2025-01)
- [x] **IPC barrier Hessian contribution** (contact stiffness in block assembly) ✅ IMPLEMENTED
  - `_add_ipc_contact_contribution()` - Adds barrier Hessian from contact pairs
  - Computes H_ij = b''(d) * cord[i] * cord[j] * outer(n, n) for each contact
  - Cross-subdomain entries propagate to coarse levels via hierarchy
  - Called automatically in `assemble_block_matrices()` when contacts exist

### ✅ Recently Added
- [x] **Sparse-Input Woodbury Update** (Section 3.1 of paper) ✅ IMPLEMENTED
  - `init_woodbury_structures()` - Initialize low-rank update storage
  - `save_base_contact_state()` - Save base preconditioner contact state
  - `compute_woodbury_updates()` - Extract top-k contact stiffness changes
  - `apply_with_woodbury()` - Apply preconditioner with Woodbury correction
  - Uses Sherman-Morrison-Woodbury formula: `B̂⁻¹ = B⁻¹ - B⁻¹U(I + UᵀB⁻¹U)⁻¹UᵀB⁻¹`

- [x] **METIS-based node reordering** (CEMAS - Connectivity-Enhanced MAS) ✅ IMPLEMENTED
  - `metis_reorder.py` - Standalone module for METIS-based mesh reordering
    - `build_adjacency_from_cells()` - Build CSR graph from mesh topology
    - `metis_partition()` - K-way partitioning using pymetis
    - `compute_sort_index()` - Sort vertices by partition for contiguous blocks
    - `build_partition_mappings()` - Create bidirectional GPU mappings
    - `metis_reorder_mesh()` - Main entry point for complete reordering
  - MAS preconditioner integration:
    - `init_metis_reordering()` - Initialize METIS partitioning
    - `_build_connect_mask_l0_metis()` - Build connectivity using METIS partitions
    - `_schwarz_local_solve_full_metis()` - Local solve with METIS mapping
    - `build_hierarchy_metis()` - Build hierarchy using METIS structure
    - `rebuild_with_metis()` - Full rebuild with METIS partitioning
    - `apply_metis()` - Apply preconditioner using METIS structure
  - Follows Stiff-GIPC reference: `/root/Stiff-GIPC_init/MeshProcess/metis_partition/`
  - Requires: `pip install pymetis`

---

## 11. Modular Package Structure

MAS Preconditioner已被重构为模块化包结构 (`algorithm/mas_preconditioner_pkg/`):

```
mas_preconditioner_pkg/  (6092 lines total)
├── __init__.py          # 模块导出 (109 lines)
├── constants.py         # 核心常量 (79 lines)
├── core.py              # 主MASPreconditioner类 (360 lines)
├── topology.py          # 网格拓扑和邻居构建 (718 lines)
├── assembly.py          # 矩阵组装 (弹性+接触Hessian) (719 lines)
├── inversion.py         # 块矩阵求逆算法 (554 lines)
├── schwarz.py           # Schwarz局部求解器 (374 lines)
├── hierarchy.py         # 多级限制与延拓 (433 lines)
├── woodbury.py          # Woodbury低秩更新 (520 lines)
├── metis_integration.py # METIS重排序集成 (1273 lines)
├── simple_api.py        # 简化API接口 (349 lines)
├── spmv.py              # 稀疏矩阵向量乘 (242 lines)
└── warp_utils.py        # Warp工具函数 (362 lines)
```

### 使用方法

```python
# 从模块化包导入
from algorithm.mas_preconditioner_pkg import MASPreconditioner

# 使用方式与原单文件版本相同
mas = MASPreconditioner(n_verts, n_cells, mesh, use_metis=False)
mas.build_hierarchy()
mas.assemble_block_matrices(solver)
mas.invert_block_matrices(use_full_inversion=True, use_oneway_gj=True)
mas.apply()
```

### 测试框架

完整的单元测试框架位于 `unittest/` 目录:

```bash
# 运行所有单元测试
cd unittest
python run_all_tests.py

# 运行特定测试
python -m pytest tests/test_mas_ground_truth.py -v

# 测试覆盖率: 20个测试文件, 44.4%通过率
# 核心测试全部通过: ground_truth, multilevel, simple
```

详见: `docs/algorithm/MAS_PRECONDITIONER_PKG_TESTING.md`

---

## 12. Known Issues

### Lane 0 对称性问题 (Critical, 调查中)
- **现象**: Block(0,0) 对称性误差 ~1.65e+04，其他Lane误差 < 1e-06
- **影响**: Incomplete Cholesky 分解失败
- **临时方案**: 使用 Gauss-Jordan 或 One-way GJ 求逆
- **详情**: `experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md`

### Taichi 编译器兼容性
- 部分 mesh kernel 触发 `codegen_llvm.cpp` 断言失败
- 影响测试: test_He_symmetry, test_level0_only, test_diagonal_contrib

### 推荐求逆方法
| 方法 | 推荐 | 说明 |
|------|------|------|
| One-way GJ | ✅ 推荐 | 快速且稳定 |
| Gauss-Jordan | ✅ 备选 | 更稳定但较慢 |
| Cholesky | ❌ 不推荐 | 块矩阵非SPD |
| Incomplete Cholesky | ❌ 不推荐 | 产生NaN |

---

## References

1. **StiffGIPC Paper**: "StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation"
2. **CUDA Reference**: `/root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu`
3. **MAS Original Paper**: Wu et al. 2022, "A GPU-based multilevel additive schwarz preconditioner"
4. **Testing Documentation**: `docs/algorithm/MAS_PRECONDITIONER_PKG_TESTING.md`
5. **Bug Analysis**: `experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md`
