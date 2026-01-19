# StiffGIPC Implementation Details

This document provides comprehensive implementation details extracted from the StiffGIPC paper and CUDA reference implementation at `/root/Stiff-GIPC_init/`.

## Table of Contents
1. [Overview](#1-overview)
2. [Connectivity-Enhanced MAS Preconditioner](#2-connectivity-enhanced-mas-preconditioner)
3. [CUDA Implementation Details](#3-cuda-implementation-details)
4. [Cubic Inexact Strain-Limiting Energy](#4-cubic-inexact-strain-limiting-energy)
5. [Fast Global Hessian Assembly](#5-fast-global-hessian-assembly)
6. [Symmetric Reduce-By-Key SpMV](#6-symmetric-reduce-by-key-spmv)
7. [Affine Body Dynamics Integration](#7-affine-body-dynamics-integration)
8. [Key Parameters](#8-key-parameters)
9. [Performance Benchmarks](#9-performance-benchmarks)

---

## 1. Overview

StiffGIPC is a GPU-optimized IPC simulation framework achieving up to **10× speedup** over GIPC through three key innovations:

1. **Connectivity-Enhanced MAS (CEMAS) Preconditioner** - Uses METIS partitioning instead of Morton code for better node aggregation
2. **Cubic Inexact Strain-Limiting Energy** - C²-continuous with analytic eigensystem, avoiding backtracking line search
3. **Two-Level Reduction for Hessian Assembly** - Efficient hash-based parallel reduction for ABD coupling

### Newton Iteration Cost Breakdown

```
T^nt = T^ga + T^pa + N^pcg × T^pcg
```

Where:
- `T^ga` = Linear system assembly time
- `T^pa` = Preconditioner construction time
- `N^pcg` = PCG iteration count
- `T^pcg = T^spmv + T^ap + T^dot + T^axpby`

---

## 2. Connectivity-Enhanced MAS Preconditioner

### 2.1 Mathematical Foundation

**Additive Schwarz Preconditioner:**
```
M^(-1)_(0) = Σ_d S_d^T M^(-1)_{d,(0)} S_d
```

Where:
- `Ω_d` = Non-overlapping domains covering all nodes
- `S_d ∈ ℝ^{3N_d × 3N}` = Selection matrix for domain d
- `M_{d,(0)} = S_d A S_d^T` = Subdomain Hessian

**Two-Level MAS:**
```
M^(-1)_{2,MAS} = M^(-1)_(0) + C^T_(1) M^(-1)_(1) C_(1)
```

**General Multilevel MAS:**
```
M^(-1)_{MAS} = M^(-1)_(0) + Σ_l C^T_(l) M^(-1)_(l) C_(l)
```

### 2.2 METIS Partition Count Formula

```
M = ⌈V / (N - N_o)⌉
```

Where:
- `V` = Total mesh nodes
- `N` = Nodes per MAS subdomain (16)
- `N_o` = Slack variable (typically 1)

### 2.3 Key Improvement Over Original MAS

| Aspect | Original MAS (Wu et al. 2022) | CEMAS (StiffGIPC) |
|--------|------------------------------|-------------------|
| Node Reordering | Morton code (spatial) | METIS (connectivity) |
| Subdomain Size | 32 nodes | 16 nodes |
| Aggregation Quality | Variable | Consistently dense |
| GPU Optimization | Standard atomics | Warp reduction |

### 2.4 Mapping and Remapping Arrays

The METIS partitions may have different sizes, requiring padding:

```
// Mapping array: maps each node to MAS subdomain slot
mapping_array[N × M]  // Size: subdomain_size × num_partitions

// Remapping array: maps MAS slots back to actual nodes
// -1 indicates padding (no node mapped)
remapping_array[i] = actual_node_index or -1
```

---

## 3. CUDA Implementation Details

### 3.1 Core Constants (from `eigen_data.h`)

```cpp
#define BANKSIZE 16                  // Nodes per subdomain
#define DEFAULT_BLOCKSIZE 256        // CUDA block size
#define DEFAULT_WARPNUM 16           // Warps per block (256/16)
#define SYM_BLOCK_COUNT 136          // BANKSIZE*(BANKSIZE+1)/2

// Block matrix storage structures
struct MasMatrixSymT {
    Eigen::Matrix3d M[136];  // Double precision for assembly
};

struct MasMatrixSymf {
    Eigen::Matrix3f M[136];  // Single precision for application
};

struct itable {
    int index[6];  // Path through 6-level hierarchy
};
```

### 3.2 Symmetric Storage Index Formula

```cpp
// For row <= col (upper triangle):
int sym_idx(int row, int col) {
    int r = min(row, col);
    int c = max(row, col);
    return BANKSIZE * r - r * (r + 1) / 2 + c;
}
// Total entries: 16 × 17 / 2 = 136 (instead of 256)
```

### 3.3 Hierarchy Construction Pipeline

**Phase 1: Level-0 Connectivity Mask (`_buildCML0_new`)**

```cuda
__global__ void _buildCML0_new() {
    int warpId = threadIdx.x / BANKSIZE;
    int laneId = threadIdx.x % BANKSIZE;

    // Initialize self-connectivity
    unsigned int connectMsk = (1U << laneId);

    // Iterate through neighbors
    for (int i = 0; i < numNeighbor; i++) {
        int vConnected = neighborList[startId + i];
        int warpConnected = vConnected / BANKSIZE;

        // Keep only intra-warp neighbors
        if (warpId == warpConnected) {
            unsigned int laneConnected = vConnected % BANKSIZE;
            connectMsk |= (1U << laneConnected);
        } else {
            // Out-of-warp neighbors retained for higher levels
            neighborList[startId + nk++] = vConnected;
        }
    }
    _fineConnectedMsk[idx] = connectMsk;
}
```

**Phase 2: Transitive Closure (`_preparePrefixSumL0_new`)**

```cuda
__shared__ unsigned int cacheMask[DEFAULT_BLOCKSIZE];
cacheMask[threadIdx.x] = connectMsk;
unsigned int visited = (1U << laneId);

// BFS-style bit merging
while (connectMsk != -1) {
    unsigned int todo = visited ^ connectMsk;
    if (!todo) break;

    unsigned int nextVist = __ffs(todo) - 1;  // Find first unvisited
    visited |= (1U << nextVist);

    // Merge neighbor's connections via shared cache
    connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];
}
_fineConnectedMsk[idx] = connectMsk;
```

**Phase 3: Cluster Representative Election (`_buildLevel1_new`)**

```cuda
// Node is representative if it has lowest lane_id in component
unsigned int electedPrefix = __popc(connMsk & lanemask_lt(laneId));

if (electedPrefix == 0) {
    atomicOr(electedMask[localWarpId], (1U << laneId));
}

// Compute global cluster ID
unsigned int elected_lane = __ffs(connMsk) - 1;
_coarseSpaceTable[idx] = lanePrefix[elected_lane + BANKSIZE * localWarpId];
```

**Phase 4: Multi-Level Construction**

```cpp
for (int level = 1; level < levelnum; level++) {
    BuildConnectMaskLx(level);           // Use level-1 mapping
    if (cpNum)
        BuildCollisionConnection(...);   // Add contact connectivity

    NextLevelCluster(level);             // Find connected components
    PrefixSumLx(level);                  // Compute cluster IDs
    ComputeNextLevel(level);             // Map level l-1 → level l
}
```

### 3.4 Block Matrix Inversion (Gauss-Jordan)

```cuda
__global__ void __inverse6_P96x96(
    __GEIGEN__::MasMatrixSymf* _preMatrix,
    __GEIGEN__::MasMatrixSymT* _invMatrix,
    int numbers
) {
    __shared__ double sPMas[32/BANKSIZE][BANKSIZE*3][BANKSIZE*3];

    // 1. Expand symmetric → full (with symmetry handling)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        if (colId >= rowId) {
            index = SYM_INDEX(rowId, colId);
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](j%3, i%3);
        } else {
            index = SYM_INDEX(colId, rowId);
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](i%3, j%3);  // Transpose
        }
    }

    // 2. Gauss-Jordan elimination (48 pivots)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        __syncthreads();
        double rt = sPMas[block_matId][j][j];
        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        if (i == j) sPMas[block_matId][i][j] = 1;
        else sPMas[block_matId][i][j] = 0;

        __syncthreads();
        sPMas[block_matId][j][i] /= rt;

        // Eliminate column
        for (int k = 0; k < BANKSIZE * 3; k++) {
            if (k != j) {
                double rate = -colm[block_matId][k];
                __syncthreads();
                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }
    }

    // 3. Store back as single precision (f32)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        if (colId >= rowId) {
            _preMatrix[matId].M[SYM_INDEX(rowId, colId)](j%3, i%3) = sPMas[block_matId][j][i];
        }
    }
}
```

### 3.5 Preconditioning Application

**Restriction (Build Multi-Level R):**

```cuda
__global__ void __buildMultiLevelR_optimized_new() {
    Eigen::Vector3f r;
    int idx = partId_map_real[pdx];

    if (idx >= 0) {
        r = R[idx];
    }
    _multiLR[pdx] = r;

    // Warp shuffle reduction for boundary nodes
    if (prefixSum[localWarpId] == 1) {
        for (int iter = 1; iter < BANKSIZE; iter <<= 1) {
            r[0] += __shfl_down_sync(mask_val, r[0], iter);
            r[1] += __shfl_down_sync(mask_val, r[1], iter);
            r[2] += __shfl_down_sync(mask_val, r[2], iter);
        }

        // Propagate to coarse levels
        if (bBoundary) {
            while (level < levelNum - 1) {
                level++;
                idx = goingNext[idx];
                atomicAdd(&_multiLR[idx], r);
            }
        }
    }
}
```

**Local Solve:**

```cuda
__global__ void _schwarzLocalXSym3() {
    // Thread per element of result
    int Hid = idx / hessianSize;
    int MRid = (idx % hessianSize) / BANKSIZE;
    int MCid = (idx % hessianSize) % BANKSIZE;

    // Load residuals into shared memory
    __shared__ Eigen::Vector3f smR[BANKSIZE];
    if (threadIdx.x < BANKSIZE) smR[threadIdx.x] = mR[vcid];
    __syncthreads();

    // Matrix-vector product with symmetric handling
    FloatP rdata = 0;
    if (lvcid >= lvrid) {
        int index = SYM_INDEX(lvrid, lvcid);
        rdata = Pred[Hid].M[index].row(r3id).dot(smR[lvcid]);
    } else {
        int index = SYM_INDEX(lvcid, lvrid);
        rdata = Pred[Hid].M[index].col(r3id).dot(smR[lvcid]);  // Transpose
    }

    // Warp reduction
    for (int iter = 1; iter < maxSize; iter <<= 1) {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if (interval >= iter) rdata += tmpx;
    }

    if (bBoundary) {
        atomicAdd(&mZ[vrid][MRid % 3], rdata);
    }
}
```

**Prolongation (Collect Final Z):**

```cuda
__global__ void __collectFinalZ_new() {
    Precision_T3 cz;
    int rdx = real_map_partId[idx];
    cz = d_multiLevelZ[rdx];

    // Retrieve aggregation table (path through hierarchy)
    __GEIGEN__::itable table = _coarseTable[idx];

    // Accumulate from all coarse levels
    for (int i = 1; i < levelnum; i++) {
        int now = table.index[i - 1];
        cz += d_multiLevelZ[now];
    }

    _Z[idx] = cz;
}
```

### 3.6 Contact-Aware Hierarchy Update

```cuda
__global__ void _buildCollisionConnection_new() {
    // Process collision pair (4 vertices)
    for (int i = 0; i < 4; i++) {
        unsigned int myId = cpVid[i];

        for (int j = i + 1; j < 4; j++) {
            unsigned int otId = cpVid[j];

            if (myId / BANKSIZE == otId / BANKSIZE) {
                // Same block: add edge to connectivity
                atomicOr(_pConnect + myId, (1U << (otId % BANKSIZE)));
                atomicOr(_pConnect + otId, (1U << (myId % BANKSIZE)));
            }
        }
    }
}
```

---

## 4. Cubic Inexact Strain-Limiting Energy

### 4.1 Formulation

**FBW Membrane Energy:**
```
Ψ_memb = Ψ_stretch + Ψ_shear
```

**Stretching Term:**
```
Ψ_stretch = λ·a_t·(√I₅(F,n_u) - 1)² + λ·a_t·(√I₅(F,n_v) - 1)²
```

Where `I₅(F,n) = n^T F^T F n` (squared stretch in direction n)

**Cubic Inexact Strain-Limiting:**
```
Ψ^u_SL = λ'·a_t·max(√I₅(F,n_u) - 1, 0)³
Ψ^v_SL = λ'·a_t·max(√I₅(F,n_v) - 1, 0)³
Ψ_SL = Ψ^u_SL + Ψ^v_SL
```

**Default Parameters:**
- `λ = 0.05 MPa` (stretching stiffness - low to avoid membrane locking)
- `λ' = 5 MPa` (strain-limiting stiffness)
- Shear stiffness = 30% of λ

### 4.2 Analytic Eigensystem

**Eigenvalues for I₅-based energies:**
```
e₁(I₅) = 4·I₅·(∂²Ψ/∂I₅²) + 2·(∂Ψ/∂I₅)
e₂,₃(I₅) = 2·(∂Ψ/∂I₅)
```

**For Cubic Strain-Limiting (I₅ > 1):**
```
e₁(I₅(F,n_u)) = 6·(√I₅(F,n_u) - 1)
e₂,₃(I₅(F,n_u)) = 3·(1/√I₅(F,n_u) + √I₅(F,n_u) - 2)
```

All eigenvalues are positive when I₅ > 1 → **Convex with respect to F**

### 4.3 PSD Hessian Computation

```
H_SL,* = 3·(1 - 1/√I₅(F,n_*))·(√I₅(F,n_*) - 1)·H_*
       + 3·(I₅(F,n_*) - 1) / I₅(F,n_*)^(3/2) · (f_* f_*^T)
```

Where:
- `f_* = vec(F·L_*)`
- `L_* = n_*·(n_*)^T`
- `H_* = L_* ⊗ I_{3×3}`

**Advantages:**
- No numerical eigendecomposition needed
- No backtracking line search filtering required
- Fully GPU-parallelizable
- Maintains convexity

---

## 5. Fast Global Hessian Assembly

### 5.1 Two-Level Reduction Strategy

**Problem:** Accumulating contact Hessians causes excessive write conflicts.

**Solution:** Two-pass approach:

1. **Pass 1:** Reduce all 3×3 blocks with same node pair (r,c) → single 3×3 block
2. **Pass 2:** Apply Jacobian transform once, then reduce 12×12 affine body blocks

**Speedup:** ~5× over single-level reduction

### 5.2 Hash-Based Parallel Reduction

**Hash Key Structure:**
- Higher 32 bits: first index
- Lower 32 bits: second index
- Sorting enables consecutive memory layout

**Algorithm: FastHashReduction**

```
Input: K (sorted hash keys), V (sorted values)
Output: AG (global Hessian)

1. UK ← RunLengthEncode(K)           // Get unique keys
2. P[i] ← K[i] ≠ K[i+1]              // Detect boundaries
3. O ← ExclusiveSum(P)               // Map each K to unique key
4. R ← FastSegmentReduction(O, V)    // Warp-level reduction
5. AG ← ConstructGlobalHessian(UK, R)
```

**FastSegmentReduction (warp-level):**

```cuda
// Tags determine segment boundaries
if (LaneId == 0 || b⁻ ≠ b) {
    IsHead = 1;
}
Value ← HeadSegmentedReduce(Value, IsHead);

if (IsHead && GlobalThreadId < len(O)) {
    AtomicAdd(R[b], Value);
}
```

**Speedup vs CUB ReduceByKey:**
- Peak: 1.55× at 2¹¹ duplicates
- Typical FEM range (2³-2⁶ blocks/row): 1.2×-1.4×

---

## 6. Symmetric Reduce-By-Key (SRBK) SpMV

### 6.1 Algorithm

```cuda
// Only store upper triangular + diagonal blocks
for GlobalThreadId in parallel:
    b ← Rid[GlobalThreadId]
    j ← Cid[GlobalThreadId]
    H ← AU[GlobalThreadId]

    // Upper triangular multiplication
    Value ← H · V_input[j]

    // Lower triangular (transposed) multiplication
    if b ≠ j:  // Not diagonal
        AtomicAdd(V_output[j], H^T · V_input[b])

    // Warp reduction for upper part
    Value ← HeadSegmentedReduce(Value, IsHead)

    if IsHead:
        AtomicAdd(V_output[b], Value)
```

### 6.2 Performance

| Method | Relative Speed |
|--------|---------------|
| Triplet SpMV | 0.16× |
| MatrixFree SpMV | 0.18× |
| CSR SpMV | 0.75× |
| BCOO SpMV | 0.84× |
| BSR SpMV (baseline) | 1.0× |
| RBK SpMV | 0.94× |
| **SRBK SpMV** | **1.85×** |

---

## 7. Affine Body Dynamics Integration

### 7.1 Unified Incremental Potential

**Deformable Bodies:**
```
E_s(x) = ½(x - x̂)^T M_s (x - x̂) + Δt²·Ψ_s(x) + B(x) + D(x)
```

**Affine Bodies (12D DOF):**
```
E_r(q) = ½(q - q̂)^T M_r (q - q̂) + Δt²·Ψ_r(q) + B(x(q)) + D(x(q))
```

Where `q_j = [p_j^T, A_{j1}^T, A_{j2}^T, A_{j3}^T]^T` with:
- `p_j ∈ ℝ³` = translation vector
- `A_j ∈ ℝ^{3×3}` = affine deformation matrix

**Full Space Mapping:**
```
x_i(q_j) = A_j · x̄_i + p_j
```

**Jacobian:**
```
J_{ij} = ∂x_i/∂q_j ∈ ℝ^{3×12}
```

### 7.2 Contact Gradient Transformation

```
∇_{q_j} B = Σ_i J_{ij}^T · ∇_{x_i} B
```

---

## 8. Key Parameters

### 8.1 MAS Preconditioner

| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain (warp subdivision) |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| SYM_BLOCK_COUNT | 136 | Symmetric storage for 16×16 blocks |
| BLOCK_DOF | 48 | 16 nodes × 3 DOF per node |

### 8.2 Simulation Parameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| PCG Tolerance | 10⁻⁴ | Relative error threshold |
| Newton Tolerance | 10⁻²·l·Δt | Displacement convergence |
| Barrier ĥ | 10⁻³·l | Distance threshold |
| λ (stretch) | 0.05 MPa | Membrane stretching stiffness |
| λ' (SL) | 5 MPa | Strain-limiting stiffness |

---

## 9. Performance Benchmarks

### 9.1 Overall Speedups

| Optimization | Speedup | Component |
|--------------|---------|-----------|
| CEMAS + SRBK + Hessian | **2.33×–4.46×** | Overall vs GIPC |
| CEMAS alone | 1.24×–2.4× | PCG convergence |
| Two-level Hessian reduction | ~5× | vs single-level |
| SRBK SpMV | 1.85× | vs cuSPARSE BSR |
| FastHashReduction | 1.2×–1.55× | vs CUB ReduceByKey |
| ABD coupling | 2.6×–10× | vs FEM-only |

### 9.2 Scene Statistics (from paper)

| Scene | Performance | Details |
|-------|-------------|---------|
| Octopus Stack | 1.8× speedup | Validates CEMAS |
| Stiff/Soft Bunnies | 3.57×–10× | Hybrid ABD+FEM |
| Mat-Cloth Twist | 6.93× speedup | Extreme deformation |
| London Bus | 1.56 s/frame | 265K triangles, 118K contacts |
| Box Pile | 3.45 s/frame | 1920 boxes, 100K cloth vertices |

---

## References

1. Huang et al. "StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation" (2024)
2. Wu et al. "A GPU-based multilevel additive schwarz preconditioner" (2022)
3. CUDA Reference Implementation: `/root/Stiff-GIPC_init/StiffGIPC/`
