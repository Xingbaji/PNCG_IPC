# MAS Preconditioner Implementation Details

This document provides a comprehensive technical description of the Multilevel Additive Schwarz (MAS) preconditioner implementation in PNCG_IPC. The implementation is based on the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact" and the CUDA reference implementation in Stiff-GIPC.

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Constants and Parameters](#3-constants-and-parameters)
4. [Data Structures](#4-data-structures)
5. [Hierarchy Construction](#5-hierarchy-construction)
6. [Matrix Assembly](#6-matrix-assembly)
7. [Block Matrix Inversion](#7-block-matrix-inversion)
8. [Preconditioning Application](#8-preconditioning-application)
9. [Woodbury Updates](#9-woodbury-updates)
10. [METIS Integration](#10-metis-integration)
11. [Performance Optimizations](#11-performance-optimizations)
12. [Algorithm Details](#12-algorithm-details)

---

## 1. Overview

### 1.1 Mathematical Formulation

The MAS preconditioner approximates the inverse of the Hessian matrix using the additive Schwarz formula:

$$
P = M_{(0)}^{-1} + \sum_{l=1}^{L} C_{(l)}^T M_{(l)}^{-1} C_{(l)}
$$

Where:
- $M_{(0)}^{-1}$ = Level-0 local block inverses (Schwarz blocks)
- $C_{(l)}$ = Coarsening/restriction operators (binary aggregation)
- $M_{(l)}^{-1}$ = Coarse-level preconditioners
- $L$ = Number of hierarchy levels

### 1.2 Key Features

- **Multilevel hierarchy**: Captures both local and global coupling
- **BANKSIZE=16**: Optimal warp subdivision for GPU parallelism
- **Symmetric block storage**: Reduces memory by storing only upper triangle
- **Connectivity-aware coarsening**: Groups connected vertices for better quality
- **Sparse-Input Woodbury updates**: Efficient incremental updates

### 1.3 File Organization

The implementation is organized in a modular package structure:

```
algorithm/mas_preconditioner_pkg/
├── __init__.py              # Package exports
├── constants.py             # Core constants (80 lines)
├── core.py                  # Main MASPreconditioner class (360 lines)
├── topology.py              # Mesh topology and neighbors (718 lines)
├── assembly.py              # Matrix assembly (677 lines)
├── inversion.py             # Block inversion algorithms (505 lines)
├── schwarz.py               # Local subdomain solvers (375 lines)
├── hierarchy.py             # Restriction/prolongation (433 lines)
├── woodbury.py              # Woodbury updates (520 lines)
├── metis_integration.py     # METIS partitioning (595 lines)
├── simple_api.py            # Simplified API (348 lines)
├── spmv.py                  # SRBK SpMV (242 lines)
└── warp_utils.py            # Bit manipulation utilities (362 lines)
```

---

## 2. Architecture

### 2.1 Mixin-Based Design

The `MASPreconditioner` class combines multiple mixin classes:

```python
@ti.data_oriented
class MASPreconditioner(
    TopologyMixin,      # Mesh connectivity and neighbor building
    AssemblyMixin,      # Elastic and contact Hessian assembly
    InversionMixin,     # Block matrix inversion algorithms
    SchwarzMixin,       # Local subdomain solvers
    HierarchyMixin,     # Multi-level restriction/prolongation
    WoodburyMixin,      # Sparse-Input Woodbury updates
    METISMixin,         # METIS-based reordering
    SimpleAPIMixin      # Simplified API without meshtaichi
):
```

### 2.2 Class Hierarchy

```
MASPreconditioner
├── _allocate_neighbor_structures()     # CSR neighbor list
├── _allocate_hierarchy_structures()    # Multi-level arrays
├── _allocate_matrix_structures()       # Block matrices
├── _allocate_preconditioning_buffers() # Restrict/prolong buffers
│
├── build_hierarchy()                   # Construct multilevel hierarchy
├── assemble_block_matrices()           # Assemble Hessian blocks
├── invert_block_matrices()             # Invert all blocks
└── apply()                             # Apply preconditioner: z = P * grad
```

### 2.3 Execution Flow

```
1. Initialization
   └── Allocate all Taichi fields
   └── Build neighbor list from mesh
   └── (Optional) Initialize METIS reordering

2. Rebuild (called on first iteration or restart)
   ├── build_hierarchy()
   │   ├── _build_connect_mask_l0()        # Level 0 connectivity
   │   ├── _propagate_connectivity()        # Transitive closure
   │   ├── _find_cluster_representatives()  # Elected nodes
   │   └── Build coarse levels recursively
   │
   ├── assemble_block_matrices()
   │   ├── _clear_block_matrices()
   │   ├── _add_inertia_contribution()
   │   ├── _add_elastic_contribution_full()
   │   ├── _add_ipc_contact_contribution()
   │   └── _add_regularization()
   │
   └── invert_block_matrices()
       ├── _expand_sym_to_full()
       ├── _gauss_jordan_invert_blocks()
       └── _copy_inverse_to_sym()

3. Apply (each CG iteration)
   ├── _clear_multi_level_buffers()
   ├── Restriction: _build_multi_level_r()
   ├── Local Solve: _schwarz_local_solve_conflict_free()
   └── Prolongation: _collect_final_z()
```

---

## 3. Constants and Parameters

### 3.1 Core MAS Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `BANKSIZE` | 16 | Nodes per subdomain (matches GPU warp subdivision) |
| `MAX_LEVELS` | 6 | Maximum hierarchy depth |
| `SYM_BLOCK_COUNT` | 136 | Symmetric storage size: $\frac{16 \times 17}{2}$ |
| `BLOCK_DOF` | 48 | DOFs per block: $16 \times 3$ |
| `MAX_NEIGHBORS_PER_VERTEX` | 64 | Conservative upper bound for tetrahedral meshes |

### 3.2 Optimization Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WARP_REDUCTION_ENABLED` | True | Enable P1 warp-level reduction |
| `NODE_BANDWIDTH` | 2 | Bandwidth for IC(0) banded inversion |
| `TOP_K_UPDATES` | 8 | Rank-1 updates per subdomain for Woodbury |

### 3.3 Solver Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `RESTART_THRESHOLD` | 0.3 | Powell's restart criterion $\delta$ |
| `CCD_ALPHA_MIN` | 1e-6 | Minimum step size for CCD |
| `DEFAULT_REGULARIZATION` | 1e-6 | Numerical stability regularization |
| `EPS` | 1e-10 | Small epsilon for comparisons |

---

## 4. Data Structures

### 4.1 Neighbor/Topology Fields

```python
# CSR-format neighbor list
neighbor_list: ti.field(dtype=ti.i32, shape=n_verts * MAX_NEIGHBORS_PER_VERTEX)
neighbor_start: ti.field(dtype=ti.i32, shape=n_verts + 1)  # Prefix sum
neighbor_num: ti.field(dtype=ti.i32, shape=n_verts)        # Count per vertex

# Connectivity bitmask (Level 0)
fine_connect_mask: ti.field(dtype=ti.u32, shape=n_verts)   # 32-bit per vertex
```

**Connectivity Mask Format**: Each vertex stores a 32-bit mask where bit $i$ is set if the vertex is connected to lane $i$ within its BANKSIZE group.

### 4.2 Hierarchy Fields

```python
# Level sizes: [level] -> (num_nodes, offset_in_global_array)
level_size: ti.Vector.field(2, dtype=ti.i32, shape=MAX_LEVELS + 1)

# Coarse space tables: maps vertex to coarse ID at each level
coarse_space_tables: ti.field(dtype=ti.i32, shape=(MAX_LEVELS, n_verts))

# Going next: for each node, its parent in next level
going_next: ti.field(dtype=ti.i32, shape=total_nodes_all_levels)

# Aggregation table: stores path through hierarchy for prolongation
aggregation_table: ti.Vector.field(MAX_LEVELS - 1, dtype=ti.i32, shape=n_verts)

# Elected mask: bitmask of cluster representatives per warp
elected_mask: ti.field(dtype=ti.u32, shape=n_warps)

# Prefix sum arrays for hierarchy construction
prefix_original: ti.field(dtype=ti.i32, shape=n_warps + 1)
prefix_sum_original: ti.field(dtype=ti.i32, shape=n_warps + 1)
```

### 4.3 Matrix Fields

```python
# Block matrices: symmetric storage (upper triangle)
# Each block is BANKSIZE×BANKSIZE vertices = 136 entries of 3×3 matrices
block_matrices: ti.Matrix.field(3, 3, dtype=ti.f32, shape=(total_blocks, SYM_BLOCK_COUNT))

# Inverted block matrices (single precision)
inv_block_matrices: ti.Matrix.field(3, 3, dtype=ti.f32, shape=(total_blocks, SYM_BLOCK_COUNT))

# Full 48×48 block matrices for inversion workspace
full_block_matrix: ti.field(dtype=ti.f32, shape=(total_blocks, BLOCK_DOF, BLOCK_DOF))
full_block_inverse: ti.field(dtype=ti.f32, shape=(total_blocks, BLOCK_DOF, BLOCK_DOF))
```

### 4.4 Preconditioning Buffers

```python
# Multi-level residual (restricted gradient at each level)
multi_level_r: ti.Vector.field(3, dtype=ti.f32, shape=total_nodes_all_levels)

# Multi-level solution (z at each level before prolongation)
multi_level_z: ti.Vector.field(3, dtype=ti.f32, shape=total_nodes_all_levels)

# P1 Optimization: Warp-level reduction buffers
warp_sum_buffer: ti.field(dtype=ti.f32, shape=(n_warps, BANKSIZE, 3))
warp_prefix_cache: ti.field(dtype=ti.i32, shape=n_warps)
```

### 4.5 Symmetric Storage Index Formula

For a BANKSIZE×BANKSIZE matrix stored in upper triangular form:

```python
@ti.func
def sym_index(row: ti.i32, col: ti.i32) -> ti.i32:
    """
    Compute symmetric storage index for (row, col) where row <= col.
    Formula: BANKSIZE * row - row*(row+1)/2 + col
    """
    r = ti.min(row, col)
    c = ti.max(row, col)
    return BANKSIZE * r - r * (r + 1) // 2 + c
```

**Example for BANKSIZE=16**:
- (0,0) → 0, (0,1) → 1, ..., (0,15) → 15
- (1,1) → 16, (1,2) → 17, ..., (1,15) → 30
- (2,2) → 31, ...
- Total: 136 entries

---

## 5. Hierarchy Construction

### 5.1 Level 0: Connectivity Mask Building

**Purpose**: Identify which vertices within each BANKSIZE group are connected via mesh edges.

**CUDA Reference**: `_buildCML0_new()` (MASPreconditioner.cu lines 63-103)

```python
@ti.kernel
def _build_connect_mask_l0(self):
    """Build connectivity bitmask at Level 0."""
    for idx in range(self.n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE

        # Start with self-connectivity
        connect_mask = ti.u32(1) << ti.u32(lane_id)
        nk = 0  # Count of inter-warp neighbors to keep

        # Check neighbors
        num_neighbor = self.neighbor_num[idx]
        start_id = self.neighbor_start[idx]

        for i in range(num_neighbor):
            neighbor_id = self.neighbor_list[start_id + i]
            neighbor_warp = neighbor_id // BANKSIZE

            if warp_id == neighbor_warp:
                # Same warp: add to connectivity mask
                neighbor_lane = neighbor_id % BANKSIZE
                connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))
            else:
                # Different warp: keep for higher-level aggregation
                self.neighbor_list[start_id + nk] = neighbor_id
                nk += 1

        self.neighbor_num[idx] = nk  # Only inter-warp neighbors
        self.fine_connect_mask[idx] = connect_mask
```

**Key Points**:
1. Only same-warp neighbors are added to connectivity mask
2. Inter-warp neighbors are kept in neighbor list for coarse-level aggregation
3. Each vertex's mask includes self (bit at lane_id is set)

### 5.2 Transitive Closure (BFS-style Propagation)

**Purpose**: Find all vertices reachable within the same BANKSIZE group through any path of edges.

**CUDA Reference**: `_preparePrefixSumL0_new()` (MASPreconditioner.cu lines 157-214)

```python
@ti.kernel
def _propagate_connectivity(self):
    """Propagate connectivity to find connected components."""
    for idx in range(self.n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE

        connect_mask = self.fine_connect_mask[idx]
        visited = ti.u32(1) << ti.u32(lane_id)

        # BFS-style expansion
        for _ in range(BANKSIZE):
            todo = visited ^ connect_mask
            if todo == 0:
                break

            # Find first unvisited connected node
            next_visit = find_first_set(todo)
            if next_visit < 0:
                break

            # Mark as visited and merge its connections
            visited |= ti.u32(1) << ti.u32(next_visit)
            other_idx = warp_id * BANKSIZE + next_visit
            if other_idx < self.n_verts:
                connect_mask |= self.fine_connect_mask[other_idx]

        self.fine_connect_mask[idx] = connect_mask

        # Count elected representatives
        elected_prefix = popcount(connect_mask & lanemask_lt(lane_id))
        if elected_prefix == 0:
            ti.atomic_add(self.prefix_original[warp_id], 1)
```

**Algorithm**:
1. Start with direct neighbors in `connect_mask`
2. Iteratively expand by visiting unvisited connected nodes
3. Merge each visited node's connections into the mask
4. Continue until no new nodes are reachable

### 5.3 Cluster Representative Election

**Rule**: A vertex is elected as cluster representative if it has the lowest lane ID among all connected vertices in its component.

```python
elected_prefix = popcount(connect_mask & lanemask_lt(lane_id))
is_elected = (elected_prefix == 0)
```

**Explanation**:
- `lanemask_lt(lane_id)` creates a mask with bits set for lanes < lane_id
- `connect_mask & lanemask_lt(lane_id)` gives connected nodes with lower lane ID
- `popcount(...) == 0` means no connected node has lower lane ID → this node is elected

### 5.4 Cluster ID Assignment

```python
@ti.kernel
def _assign_cluster_ids(self, level_offset: ti.i32):
    """Assign coarse-level cluster IDs."""
    for idx in range(self.n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE

        connect_mask = self.fine_connect_mask[idx]
        elected_prefix = popcount(connect_mask & lanemask_lt(lane_id))

        if elected_prefix == 0:
            # Elected: compute unique cluster ID
            base_id = self.prefix_sum_original[warp_id]
            elected_in_warp = self.elected_mask[warp_id]
            local_offset = popcount(elected_in_warp & lanemask_lt(lane_id))

            cluster_id = base_id + local_offset
            self.going_next[idx] = level_offset + cluster_id
```

### 5.5 Aggregation Table for Prolongation

**Purpose**: Pre-compute the path through hierarchy for each fine vertex to enable O(1) prolongation lookup.

```python
@ti.kernel
def _build_aggregation_table(self):
    """Build aggregation table for fast prolongation."""
    for idx in range(self.n_verts):
        current_idx = idx
        for level in range(self.level_num - 1):
            next_idx = self.going_next[current_idx]
            self.aggregation_table[idx][level] = next_idx
            current_idx = next_idx
```

### 5.6 Coarse Level Construction

For levels 2 and above, the process repeats on the coarse level:

1. Build connectivity mask for coarse nodes
2. Propagate to find connected components
3. Assign cluster IDs for next level
4. Stop when level size ≤ BANKSIZE or reduction ratio < 0.9

---

## 6. Matrix Assembly

### 6.1 Overview

The block Hessian matrices are assembled from three contributions:

$$
H_{block} = H_{inertia} + H_{elastic} + H_{contact} + H_{regularization}
$$

### 6.2 Inertia Contribution

```python
@ti.kernel
def _add_inertia_contribution(self, dt: ti.f32):
    """Add mass matrix: H_inertia = M / dt^2"""
    for idx in range(self.n_verts):
        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE

        m = self.mesh.verts.m[idx]
        sym_idx = sym_index(lane_id, lane_id)

        mass_val = m / (dt * dt)
        for d in ti.static(range(3)):
            ti.atomic_add(self.block_matrices[warp_id, sym_idx][d, d], mass_val)
```

### 6.3 Elastic Hessian Assembly

The elastic Hessian is computed per cell and distributed to vertex pairs:

**Element Hessian Computation**:
```
For each tetrahedral cell:
1. Compute deformation gradient: F = Ds @ B
   - Ds = [x1-x0, x2-x0, x3-x0] (deformed edge matrix)
   - B = inverse of rest edge matrix (precomputed)

2. Compute dFdx (9×12 matrix):
   - Relates vertex DOF to vectorized F
   - Depends only on B (rest configuration)

3. Compute d²Ψ/dF² (9×9 material Hessian):
   - ARAP: compute_d2PsidF2_ARAP_filter(F, mu, la)
   - SNH:  compute_d2PsidF2_SNH(F, mu, la)
   - FCR:  compute_d2PsidF2_FCR_filter(F, mu, la)

4. Element Hessian: H_e = dFdx^T @ d²Ψ/dF² @ dFdx (12×12)

5. Scale by volume and timestep: H_e *= W * dt²
```

**Distribution to Blocks**:

```python
# For each vertex pair (i, j) in the cell:
for i in range(4):
    for j in range(4):
        vi, vj = v_ids[i], v_ids[j]
        warp_i, warp_j = vi // BANKSIZE, vj // BANKSIZE

        # Extract 3×3 sub-block from element Hessian
        sub_block = H_e[i*3:(i+1)*3, j*3:(j+1)*3]

        if warp_i == warp_j:
            # Same warp: direct assembly to Level 0 block
            lane_i, lane_j = vi % BANKSIZE, vj % BANKSIZE
            if lane_i <= lane_j:
                sym_idx = sym_index(lane_i, lane_j)
                atomic_add(block_matrices[warp_i, sym_idx], sub_block)
            else:
                sym_idx = sym_index(lane_j, lane_i)
                atomic_add(block_matrices[warp_j, sym_idx], sub_block.T)
        else:
            # Cross-warp: propagate to coarse level
            # Follow going_next until both vertices are in same warp
            ...
```

### 6.4 IPC Contact Hessian

**Barrier Function** (IPC log barrier):
$$
b(d) = -\kappa (d - \hat{d})^2 \ln(d/\hat{d}) \quad \text{for } d < \hat{d}
$$

**Barrier Hessian** (Gauss-Newton approximation for SPD guarantee):
$$
H_{barrier} = \kappa \cdot \frac{(\hat{d} - d)^2}{d^2 \hat{d}} (2d + \hat{d}) \cdot n n^T
$$

**Distribution to Blocks**:
```python
for each contact pair (4 vertices with barycentric coords):
    for i, j in [(0,0), (0,1), ..., (3,3)]:
        vi, vj = ids[i], ids[j]
        ci, cj = coords[i], coords[j]  # Barycentric

        scale = barrier_H * ci * cj
        H_ij = scale * outer(normal, normal)

        # Same warp vs cross-warp handling as elastic...
```

### 6.5 Cross-Warp Propagation

When vertices $i$ and $j$ are in different warps, their coupling is propagated to the first coarse level where they share the same block:

```python
# Cross-warp: propagate through hierarchy
vert_i, vert_j = vi, vj
for _ in range(level_num - 1):
    vert_i = going_next[vert_i]
    vert_j = going_next[vert_j]

    if vert_i < 0 or vert_j < 0:
        break

    coarse_warp_i = vert_i // BANKSIZE
    coarse_warp_j = vert_j // BANKSIZE

    if coarse_warp_i == coarse_warp_j:
        # Now in same block: assemble here
        assemble_to_block(coarse_warp_i, vert_i % BANKSIZE, vert_j % BANKSIZE, sub_block)
        break
```

### 6.6 Regularization

Small diagonal regularization for numerical stability:

```python
@ti.kernel
def _add_regularization(self, epsilon: ti.f32):
    for block_id in range(n_blocks):
        for lane_id in range(BANKSIZE):
            sym_idx = sym_index(lane_id, lane_id)
            for d in ti.static(range(3)):
                val = self.block_matrices[block_id, sym_idx][d, d]
                if val < epsilon:
                    self.block_matrices[block_id, sym_idx][d, d] = epsilon
```

---

## 7. Block Matrix Inversion

### 7.1 Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Gauss-Jordan | Full 48×48 elimination with partial pivoting | General, most robust |
| Cholesky | LLᵀ factorization for SPD matrices | When matrix is SPD |
| Blocked Cholesky | 12×12 sub-block decomposition | Better GPU parallelism |
| Incomplete Cholesky IC(0) | Banded approximation | Fast approximate |
| One-way Gauss-Jordan | No pivoting (P4 optimization) | SPD matrices, faster |

### 7.2 Expansion to Full Matrix

```python
@ti.kernel
def _expand_sym_to_full(self):
    """Expand symmetric blocks to full 48×48 dense matrices."""
    for block_id in range(n_blocks):
        # Clear full matrix
        for i, j in ti.ndrange(BLOCK_DOF, BLOCK_DOF):
            self.full_block_matrix[block_id, i, j] = 0.0

        # Copy from symmetric storage
        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                sym_idx = sym_index(row, col)
                block_3x3 = self.block_matrices[block_id, sym_idx]

                for di, dj in ti.static(ti.ndrange(3, 3)):
                    # Upper triangle
                    self.full_block_matrix[block_id, row*3+di, col*3+dj] = block_3x3[di, dj]
                    # Lower triangle (symmetric)
                    if row != col:
                        self.full_block_matrix[block_id, col*3+dj, row*3+di] = block_3x3[dj, di]
```

### 7.3 Gauss-Jordan Elimination

```python
@ti.kernel
def _gauss_jordan_invert_blocks(self):
    for block_id in range(n_blocks):
        # Initialize inverse to identity
        for i, j in ti.ndrange(BLOCK_DOF, BLOCK_DOF):
            self.full_block_inverse[block_id, i, j] = 1.0 if i == j else 0.0

        # Forward elimination with partial pivoting
        for pivot in range(BLOCK_DOF):
            # Find pivot row (largest absolute value)
            max_val = ti.abs(full_block_matrix[block_id, pivot, pivot])
            max_row = pivot
            for r in range(pivot + 1, BLOCK_DOF):
                val = ti.abs(full_block_matrix[block_id, r, pivot])
                if val > max_val:
                    max_val = val
                    max_row = r

            # Swap rows if needed
            if max_row != pivot:
                for c in range(BLOCK_DOF):
                    swap(full_block_matrix[block_id, pivot, c],
                         full_block_matrix[block_id, max_row, c])
                    swap(full_block_inverse[block_id, pivot, c],
                         full_block_inverse[block_id, max_row, c])

            # Scale pivot row
            pivot_val = full_block_matrix[block_id, pivot, pivot]
            if ti.abs(pivot_val) > 1e-12:
                scale = 1.0 / pivot_val
                for c in range(BLOCK_DOF):
                    full_block_matrix[block_id, pivot, c] *= scale
                    full_block_inverse[block_id, pivot, c] *= scale

            # Eliminate in all other rows
            for r in range(BLOCK_DOF):
                if r != pivot:
                    factor = full_block_matrix[block_id, r, pivot]
                    for c in range(BLOCK_DOF):
                        full_block_matrix[block_id, r, c] -= factor * full_block_matrix[block_id, pivot, c]
                        full_block_inverse[block_id, r, c] -= factor * full_block_inverse[block_id, pivot, c]
```

### 7.4 Copy Back to Symmetric Storage

```python
@ti.kernel
def _copy_inverse_to_sym(self):
    for block_id in range(n_blocks):
        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                sym_idx = sym_index(row, col)
                for di, dj in ti.static(ti.ndrange(3, 3)):
                    self.inv_block_matrices[block_id, sym_idx][di, dj] = \
                        self.full_block_inverse[block_id, row*3+di, col*3+dj]
```

---

## 8. Preconditioning Application

### 8.1 Three-Phase Apply

```
Apply: z = P * grad

Phase 1: Restriction (fine to coarse)
  - Copy gradient to Level 0
  - Elected nodes accumulate component sums
  - Propagate to all coarse levels

Phase 2: Local Solve
  - z_d = B_d^{-1} * r_d for each subdomain at each level

Phase 3: Prolongation (coarse to fine)
  - z = z_0 + C_1^T * z_1 + C_2^T * z_2 + ...
```

### 8.2 Restriction Phase

**Standard Version**:
```python
@ti.kernel
def _build_multi_level_r(self):
    # Copy gradient to level 0
    for idx in range(self.n_verts):
        self.multi_level_r[idx] = self.mesh.verts.grad[idx]

    # Restrict to coarse levels
    for idx in range(self.n_verts):
        r = self.multi_level_r[idx]
        connect_mask = self.fine_connect_mask[idx]
        lane_id = idx % BANKSIZE

        elected_prefix = popcount(connect_mask & lanemask_lt(lane_id))

        if elected_prefix == 0:
            # Elected: propagate sum to coarse levels
            current_idx = idx
            for _ in range(self.level_num - 1):
                next_idx = self.going_next[current_idx]
                if next_idx >= 0:
                    for d in ti.static(range(3)):
                        ti.atomic_add(self.multi_level_r[next_idx][d], r[d])
                    current_idx = next_idx
        else:
            # Non-elected: accumulate to elected node
            warp_id = idx // BANKSIZE
            elected_lane = find_first_set(connect_mask)
            elected_idx = warp_id * BANKSIZE + elected_lane
            for d in ti.static(range(3)):
                ti.atomic_add(self.multi_level_r[elected_idx][d], r[d])
```

### 8.3 P1 Optimized Restriction (Warp Reduction)

For fully-connected warps (prefix==1), use tree reduction:

**Phase 1**: Copy to buffers
```python
@ti.kernel
def _build_multi_level_r_phase1(self):
    for idx in range(self.n_verts):
        r = self.mesh.verts.grad[idx]
        self.multi_level_r[idx] = r

        warp_id = idx // BANKSIZE
        lane_id = idx % BANKSIZE
        for d in ti.static(range(3)):
            self.warp_sum_buffer[warp_id, lane_id, d] = r[d]
```

**Phase 2a**: Tree reduction for fully-connected warps
```python
@ti.kernel
def _build_multi_level_r_phase2_tree_reduce(self):
    for warp_id in range(n_warps):
        if self.warp_prefix_cache[warp_id] == 1:  # Fully connected
            # Tree reduction pattern (BANKSIZE=16):
            # Step 0: stride=8, lanes 0-7 += lanes 8-15
            # Step 1: stride=4, lanes 0-3 += lanes 4-7
            # Step 2: stride=2, lanes 0-1 += lanes 2-3
            # Step 3: stride=1, lane 0 += lane 1

            for step, stride in enumerate([8, 4, 2, 1]):
                for lane_id in range(stride):
                    src_lane = lane_id + stride
                    for d in range(3):
                        warp_sum_buffer[warp_id, lane_id, d] += \
                            warp_sum_buffer[warp_id, src_lane, d]
            # Result: lane 0 holds sum of all 16 vertices
```

**Phase 2b**: Atomic accumulation for multi-component warps

**Phase 3**: Propagate reduced sums to coarse levels

### 8.4 Local Solve (Schwarz)

**Conflict-Free Symmetric SpMV** (P5 Optimization):

```python
@ti.kernel
def _schwarz_local_solve_conflict_free(self):
    """z_d = B_d^{-1} * r_d using symmetric storage."""
    for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
        idx_i = block_id * BANKSIZE + lane_i
        if idx_i < self.n_verts:
            z0, z1, z2 = 0.0, 0.0, 0.0

            for lane_j in range(BANKSIZE):
                idx_j = block_id * BANKSIZE + lane_j
                if idx_j < self.n_verts:
                    r_j = self.multi_level_r[idx_j]

                    # Symmetric access
                    min_lane = ti.min(lane_i, lane_j)
                    max_lane = ti.max(lane_i, lane_j)
                    sym_idx = sym_index(min_lane, max_lane)

                    inv_block = self.inv_block_matrices[block_id, sym_idx]

                    if lane_i <= lane_j:
                        # Upper triangle: direct access
                        z0 += inv_block[0,0]*r_j[0] + inv_block[0,1]*r_j[1] + inv_block[0,2]*r_j[2]
                        z1 += inv_block[1,0]*r_j[0] + inv_block[1,1]*r_j[1] + inv_block[1,2]*r_j[2]
                        z2 += inv_block[2,0]*r_j[0] + inv_block[2,1]*r_j[1] + inv_block[2,2]*r_j[2]
                    else:
                        # Lower triangle: transposed access
                        z0 += inv_block[0,0]*r_j[0] + inv_block[1,0]*r_j[1] + inv_block[2,0]*r_j[2]
                        z1 += inv_block[0,1]*r_j[0] + inv_block[1,1]*r_j[1] + inv_block[2,1]*r_j[2]
                        z2 += inv_block[0,2]*r_j[0] + inv_block[1,2]*r_j[1] + inv_block[2,2]*r_j[2]

            self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2])
```

### 8.5 Prolongation Phase

```python
@ti.kernel
def _collect_final_z_kernel(self, level_num: ti.i32):
    """Aggregate solutions from all levels."""
    for idx in range(self.n_verts):
        # Start with Level 0 solution
        z_total = self.multi_level_z[idx]

        # Add contributions from coarse levels
        for level in range(level_num - 1):
            coarse_idx = self.aggregation_table[idx][level]
            if coarse_idx >= 0:
                z_total += self.multi_level_z[coarse_idx]

        self.mesh.verts.z[idx] = z_total
```

---

## 9. Woodbury Updates

### 9.1 Mathematical Background

The Woodbury matrix identity allows efficient rank-k updates to the inverse:

$$
(A + US U^T)^{-1} = A^{-1} - A^{-1} U (S^{-1} + U^T A^{-1} U)^{-1} U^T A^{-1}
$$

Where:
- $A$ = base Hessian (cached inverse available)
- $U$ = update vectors (sparse, contact-related)
- $S$ = stiffness changes (diagonal)

### 9.2 Data Structures

```python
# Update vectors: (n_blocks, top_k, BANKSIZE*3)
woodbury_U: ti.field(dtype=ti.f32)

# Stiffness changes: (n_blocks, top_k)
woodbury_delta_S: ti.field(dtype=ti.f32)

# Precomputed B*U product
BU: ti.field(dtype=ti.f32, shape=(n_blocks, BANKSIZE*3, top_k))

# Capacitance matrix: I + U^T B U
capacitance_matrix: ti.field(dtype=ti.f32, shape=(n_blocks, top_k, top_k))
```

### 9.3 Contact Change Detection

```python
def compute_woodbury_updates(self, solver):
    ROTATION_THRESHOLD = 0.9  # cos(25°)

    for contact_key, curr_data in current_contacts.items():
        if contact_key in self.base_contacts:
            base_data = self.base_contacts[contact_key]

            # Check normal rotation
            dot_product = np.dot(curr_normal, base_normal)
            if dot_product < ROTATION_THRESHOLD:
                # Normal rotated significantly: treat as new contact
                delta_S = curr_stiffness
            else:
                # Same direction: compute stiffness change
                delta_S = curr_stiffness - base_stiffness
        else:
            # New contact
            delta_S = curr_stiffness

        # Build update vector
        u_scale = sqrt(delta_S)
        for i, vid in enumerate(contact_ids):
            subdomain_id = vid // BANKSIZE
            u_contribution = u_scale * barycentric[i] * normal
            # Add to subdomain updates...
```

### 9.4 Woodbury Solve

```python
@ti.kernel
def _schwarz_local_solve_woodbury(self):
    for block_id in range(n_blocks):
        num_updates = self.woodbury_num_updates[block_id]

        # Base solve: z = B * r
        for lane_i in range(BANKSIZE):
            z = base_solve(block_id, lane_i)
            self.multi_level_z[idx_i] = z

        if num_updates == 0:
            continue

        # Compute r_vec = U^T * z_base
        r_vec = U.T @ z_base

        # Solve capacitance system: (I + U^T B U) * lambda = r_vec
        # Using Gaussian elimination
        lambda_vec = solve_capacitance(capacitance_matrix, r_vec)

        # Apply correction: z -= B*U * lambda
        for lane_id in range(BANKSIZE):
            correction = BU @ lambda_vec
            self.multi_level_z[idx] -= correction
```

---

## 10. METIS Integration

### 10.1 Purpose

METIS partitioning groups topologically connected vertices together in blocks, improving preconditioner quality by ensuring subdomains contain geometrically meaningful vertex groups.

### 10.2 Initialization

```python
def init_metis_reordering(self, cells_np, vertices_np=None):
    from algorithm.metis_reorder import metis_reorder_mesh

    reorder_result = metis_reorder_mesh(
        n_verts=self.n_verts,
        cells=cells_np,
        vertices=vertices_np,
        block_size=BANKSIZE
    )

    # Store mappings
    self.metis_sort_index = reorder_result['sort_index']      # sorted_idx -> original_idx
    self.metis_old_to_new = reorder_result['old_to_new']      # original_idx -> sorted_idx
    self.metis_partition = reorder_result['partition']         # vertex -> partition_id

    # Taichi fields for GPU access
    self.partId_map_real: partId -> original_vertex_id
    self.real_map_partId: original_vertex_id -> partId
```

### 10.3 METIS-Specific Kernels

**Connectivity Building**:
```python
@ti.kernel
def _build_connect_mask_l0_metis(self):
    for idx in range(self.n_verts):
        # Use METIS partition info instead of sequential warp
        part_info = self.real_map_partId[idx]
        block_id = part_info // BANKSIZE
        lane_id = part_info % BANKSIZE

        connect_mask = ti.u32(1) << ti.u32(lane_id)

        # Only mark connected if in same METIS partition
        for neighbor_id in neighbors(idx):
            neighbor_part_info = self.real_map_partId[neighbor_id]
            if neighbor_part_info // BANKSIZE == block_id:
                neighbor_lane = neighbor_part_info % BANKSIZE
                connect_mask |= (ti.u32(1) << ti.u32(neighbor_lane))
```

**Local Solve with METIS Mapping**:
```python
@ti.kernel
def _schwarz_local_solve_full_metis(self):
    for block_id in range(self.metis_n_parts):
        for lane_i in range(BANKSIZE):
            # Get original vertex ID from partition mapping
            part_idx_i = block_id * BANKSIZE + lane_i
            idx_i = self.partId_map_real[part_idx_i]

            if idx_i >= 0 and idx_i < self.n_verts:
                # Apply inverse using METIS-reordered indices
                z = compute_local_solve(block_id, lane_i)
                self.multi_level_z[idx_i] = z
```

---

## 11. Performance Optimizations

### 11.1 Summary Table

| ID | Optimization | Benefit | Module |
|----|--------------|---------|--------|
| P0 | Compact contact arrays | Reduce memory for contacts | `assembly.py` |
| P1 | Warp reduction | O(log n) vs O(n) restriction | `hierarchy.py` |
| P2 | Blocked Cholesky | Better GPU parallelism | `inversion.py` |
| P3 | Incomplete Cholesky IC(0) | Fast approximate inversion | `inversion.py` |
| P4 | One-way Gauss-Jordan | Skip pivoting, faster | `inversion.py` |
| P5 | Conflict-free SpMV | No race conditions in solve | `schwarz.py` |
| P6 | Banded sparse MV | IC(0) bandwidth=2 | `schwarz.py` |

### 11.2 P0: Compact Contact Arrays

Instead of bitmasked sparse storage, use contiguous arrays:
```python
# Old (bitmasked):
cid[k, j] = contact_data  # Sparse, slow iteration

# New (compact):
contact_pairs[i] = contact_data  # Dense, fast iteration
n_contacts[None] = num_active_contacts
```

### 11.3 P1: Warp-Level Reduction

For fully-connected warps, replace O(n) atomic operations with O(log n) tree reduction:
```
Standard: 16 vertices × atomic_add = 16 atomics per warp
Optimized: 4 reduction steps × 1 write = 1 write per warp
```

### 11.4 P5: Conflict-Free SpMV

Parallelize over (block_id, lane_i) instead of just block_id:
- Each (block, lane) pair writes to exactly one output location
- No race conditions, no atomic operations needed
- Better GPU occupancy

---

## 12. Algorithm Details

### 12.1 Bit Manipulation Functions

**Popcount (SWAR Algorithm)**:
```python
@ti.func
def popcount(x: ti.u32) -> ti.i32:
    """Count set bits using parallel bit counting."""
    temp = x
    temp = temp - ((temp >> 1) & ti.u32(0x55555555))
    temp = (temp & ti.u32(0x33333333)) + ((temp >> 2) & ti.u32(0x33333333))
    temp = (temp + (temp >> 4)) & ti.u32(0x0F0F0F0F)
    return ti.i32((temp * ti.u32(0x01010101)) >> 24)
```

**Find First Set (De Bruijn)**:
```python
@ti.func
def find_first_set(x: ti.u32) -> ti.i32:
    """Find position of first set bit using De Bruijn sequence."""
    if x == 0:
        return -1
    # Isolate lowest set bit
    isolated = x & (~x + ti.u32(1))
    # De Bruijn sequence lookup
    debruijn = ti.u32(0x077CB531)
    index = (isolated * debruijn) >> 27
    lookup = [0, 1, 28, 2, 29, 14, 24, 3, ...]  # 32 entries
    return lookup[index]
```

**Lane Mask Less Than**:
```python
@ti.func
def lanemask_lt(lane_id: ti.i32) -> ti.u32:
    """Mask with bits set for lanes < lane_id."""
    return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)
```

### 12.2 CUDA Reference Correspondence

| Taichi Function | CUDA Reference | Lines |
|-----------------|----------------|-------|
| `_build_connect_mask_l0()` | `_buildCML0_new()` | 63-103 |
| `_propagate_connectivity()` | `_preparePrefixSumL0_new()` | 157-214 |
| `_build_multi_level_r_optimized()` | `__buildMultiLevelR_optimized_new()` | 729-847 |
| `_schwarz_local_solve_conflict_free()` | `_schwarzLocalXSym6()` | 957-1027 |
| `_collect_final_z_kernel()` | `__collectFinalZ_new()` | 850-879 |
| `_add_elastic_contribution_full()` | `PrepareHessian_bcoo` | 1933-2062 |

### 12.3 Memory Layout

**Block Matrix Storage**:
```
block_matrices[block_id, sym_idx] -> 3×3 matrix
  block_id = global_vertex_id // BANKSIZE
  sym_idx  = sym_index(lane_i, lane_j) for lane_i <= lane_j

Total entries per block: 136 (upper triangular 16×16)
Memory per block: 136 × 9 × 4 = 4,896 bytes (float32)
```

**Hierarchy Layout**:
```
Level 0: vertices [0, n_verts)           offset=0
Level 1: clusters [n_verts, n_verts+L1)  offset=n_verts
Level 2: clusters [L1_end, L1_end+L2)    offset=L1_end
...
```

---

## References

1. "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact"
2. `/root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu` - CUDA reference implementation
3. `ref_doc/MAS_PNCG_clean.tex` - Paper LaTeX source
4. `ref_doc/supplementary.tex` - Detailed derivations
