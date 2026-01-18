# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNCG_IPC is implementing the **MAS-PNCG** framework from the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact". The goal is to replace the current diagonal preconditioner with a GPU-accelerated **Multilevel Additive Schwarz (MAS) preconditioner** to achieve faster convergence for stiff materials and complex contact scenarios.

### Reference Documents
- `ref_doc/MAS_PNCG_clean.tex` - Main paper with algorithm overview
- `ref_doc/supplementary.tex` - Detailed derivations for MAS, 2x2 subspace minimization, and ACCD bounds

### Reference Implementation
- `/root/Stiff-GIPC_init/` - CUDA-based MAS preconditioner reference (C++/CUDA)

## Current Implementation Status

The codebase currently uses a **diagonal (Jacobi) preconditioner**. The goal is to implement:

1. **MAS Preconditioner** (Priority 1 - Current Focus)
2. Sparse-Input Woodbury Level-0 Update
3. Optimal 2D Subspace Minimization
4. Conservative CCD with Improved Lower Bounds
5. Powell's Restart Criterion

## MAS Preconditioner Implementation Plan

### Algorithm Overview (from paper Section 2.2 and Supplementary)

The MAS preconditioner combines local domain decomposition with hierarchical coarsening:

```
P = M_{(0)}^{-1} + Σ_{l=1}^{L} C_{(l)}^T M_{(l)}^{-1} C_{(l)}
```

Where:
- `M_{(0)}^{-1}` = Level-0 Additive Schwarz (fine-level local solves)
- `C_{(l)}` = Coarsening/restriction matrix at level l
- `M_{(l)}^{-1}` = Coarse-level preconditioner

### Implementation Steps (Following Stiff-GIPC Pattern)

#### Step 1: Subdomain Decomposition (Level 0)
- Partition mesh into D non-overlapping subdomains (BANKSIZE=16 nodes each)
- Define selection matrices `S_d` to extract local DOFs
- Compute local Hessian: `M_{(0)}^d = S_d * A * S_d^T`
- Invert small local matrices in parallel on GPU

#### Step 2: Connectivity-Aware Hierarchy Construction
- Build adjacency graph based on mesh topology (not just spatial proximity)
- Aggregate nodes into coarse-level clusters using graph partitioning
- Compute coarsening matrices `C_{(l)}` as binary aggregation operators
- Typical hierarchy: 4-6 levels, each level reduces DOFs by factor of ~BANKSIZE

#### Step 3: GPU Data Structures (Reference: Stiff-GIPC)
```python
# Key data structures to implement in Taichi
# Hierarchy management
level_sizes: ti.field(dtype=ti.i32)           # Size of each level
coarse_space_tables: ti.field(dtype=ti.i32)   # Coarse space mapping
aggregation_table: ti.field(dtype=ti.i32)     # Path through hierarchy

# Connectivity
neighbor_list: ti.field(dtype=ti.i32)         # Neighbor indices
neighbor_start: ti.field(dtype=ti.i32)        # Start index per node
connect_mask: ti.field(dtype=ti.u32)          # Connectivity bitmask

# Matrix storage (3x3 blocks)
inverse_matrices: ti.Matrix.field(3, 3, dtype=ti.f64)  # Inverted subdomain blocks
multi_level_r: ti.Vector.field(3, dtype=ti.f64)        # Restricted residual
multi_level_z: ti.Vector.field(3, dtype=ti.f64)        # Solution at each level
```

#### Step 4: Preconditioning Operation (z = P * g)
1. **Restriction Phase** (`BuildMultiLevelR`):
   - Hierarchically restrict gradient g to coarse levels
   - Use shared memory for warp-level summation

2. **Local Solve Phase** (`SchwarzLocalXSym`):
   - Solve 3x3 block systems locally on GPU
   - Process BANKSIZE x BANKSIZE matrix blocks in parallel

3. **Prolongation Phase** (`CollectFinalZ`):
   - Aggregate solutions from all levels
   - Use aggregation table to reconstruct fine solution

### GPU Parallelization Strategy (from Stiff-GIPC)

| Optimization | Description |
|--------------|-------------|
| Warp-level ops | BANKSIZE=16, use warp shuffles and ballot |
| Shared memory | Reduce global memory access in reduction |
| Atomic operations | `atomicAdd` for multi-level accumulation |
| Block structure | 256 threads/block, multiple warps |
| Symmetric storage | Compact storage for symmetric 3x3 blocks |

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Warp subdivision size |
| BLOCKSIZE | 256 | CUDA thread block size |
| max_levels | 6 | Maximum hierarchy depth |
| subdomain_size | 16-64 | Nodes per subdomain |

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Demos
```bash
cd PNCG_IPC/demo
python cubic_demos.py           # Interactive mode with visualization
python cubic_demos.py --headless --frames 300  # Headless mode for batch processing
```

### Testing Demos
Use a 20 second timeout when verifying demo correctness:
```bash
timeout 20s python cubic_demos.py --headless --frames 10
```

### Available Demos
- `cubic_demos.py` - Cube squeeze/stretch/rotate
- `drag_armadillo_demo.py` - Armadillo dragging simulation
- `squeeze_armadillo_demo.py` - Armadillo compression
- `twist_demo.py` - Twisting deformation
- `n_E_demo.py` - Multi-object collision demo
- `unittest_demos.py` - Unit tests

## Architecture

### Class Hierarchy
```
base_deformer (FEM foundation, vertex/cell management)
    ↓
collision_detection (spatial hashing, PP/PE/PT/EE constraints)
    ↓
pncg_base_ipc (PNCG solver with IPC barrier functions)
    ↓
Demo classes (application-specific boundary conditions)
```

### Key Modules

**algorithm/**
- `base_deformer.py` - Base FEM class with mesh data, precomputation, and elastic type assignment
- `pncg_base_ipc.py` - Main solver combining PNCG optimization with IPC contact handling
- `pncg_base_collision_free.py` - PNCG solver without collision constraints
- `collision_detection.py` - Collision detection (PP, PE, PT, EE)
- `mas_preconditioner.py` - **[TO IMPLEMENT]** MAS preconditioner

**math_utils/**
- `elastic_util.py` - Constitutive models (ARAP, SNH, FCR, NH) with energy, gradient, and Hessian
- `matrix_util.py` - SVD, deformation gradient derivatives, matrix operations

**util/**
- `model_loading.py` - Demo configurations (material params, solver params, mesh paths)

### Data Structures (MeshTaichi)

**Vertex attributes:** `x`, `v`, `m`, `x_n`, `x_hat`, `x_prev`, `x_init`, `grad`, `grad_prev`, `p`, `diagH`

**Cell attributes:** `B` (inverse rest deformation gradient), `W` (cell volume weight)

### Key Parameters

**Simulation:** `dt` (timestep), `E` (Young's modulus), `nu` (Poisson's ratio), `density`, `gravity`

**Solver:** `iter_max` (max CG iterations), `epsilon` (convergence tolerance)

**Contact:** `dHat` (detection threshold), `kappa` (barrier stiffness), `adj` (adjacency filtering)

## MAS-PNCG Algorithm (from Paper Algorithm 1)

```
Input: x^t, v^t, M, d_hat, epsilon, delta
Output: x^{t+1}

x_0 = x^t
x_tilde = x^t + h*v^t + h^2*M^{-1}*f_ext
Restart = True

for k = 0 to IterMax:
    C = ComputeConstraintSet(x_k, d_hat)

    if Restart:
        P_base, H_base = RebuildMASPreconditioner(x_k, C)
        P_{k+1} = P_base
    else:
        P_{k+1} = SparseInputWoodburyUpdate(P_base, C)

    g_{k+1} = grad E(x_k)
    z_{k+1} = P_{k+1} * g_{k+1}
    v = H_tilde * z_{k+1}

    if Restart:
        mu = (z_{k+1}^T * g_{k+1}) / (z_{k+1}^T * v)
        nu = 0
    else:
        Solve 2x2 system for (mu, nu)  # Eq. 6 in paper

    p_{k+1} = -mu * z_{k+1} + nu * p_k
    w_{k+1} = -mu * v + nu * w_k

    alpha_d = ConservativeCCD(x_k, p_{k+1})
    x_{k+1} = x_k + sum_d(S_d^T * alpha_d * S_d * p_{k+1})

    if ||alpha * p_{k+1}|| <= epsilon:
        break
    else:
        r_k = |g_{k+1}^T * z_k| / (g_{k+1}^T * z_{k+1})
        Restart = (r_k > delta)  # Powell's restart criterion

    z_k = z_{k+1}

return x_{k+1}
```

## Known Issues

- **NumPy compatibility:** Use numpy 1.26 or older due to meshtaichi compatibility issues
- **MeshTaichi .face files:** Required even if empty (one face entry needed)
- **Platform:** Linux recommended; Windows performance is significantly worse

## Reference Code Location

The CUDA reference implementation is at `/root/Stiff-GIPC_init/StiffGIPC/`:
- `MASPreconditioner.cu` (2361 lines) - Main MAS implementation
- `MASPreconditioner.cuh` - Header with class definition
- `linear_system/preconditioner/fem_mas_preconditioner.cu` - FEM integration
