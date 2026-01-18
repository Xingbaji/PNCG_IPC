# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNCG_IPC implements the **MAS-PNCG** framework from the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact". The project features a GPU-accelerated **Multilevel Additive Schwarz (MAS) preconditioner** using Taichi for faster convergence with stiff materials and complex contact scenarios.

### Reference Documents
- `ref_doc/MAS_PNCG_clean.tex` - Main paper with algorithm overview
- `ref_doc/supplementary.tex` - Detailed derivations for MAS, 2x2 subspace minimization, and ACCD bounds
- `ref_doc/MAS_PRECONDITIONER_IMPLEMENTATION.md` - Implementation documentation

### Reference Implementation
- `/root/Stiff-GIPC_init/` - CUDA-based MAS preconditioner reference (C++/CUDA)

## Implementation Status

### Completed Features

| Feature | File | Status |
|---------|------|--------|
| MAS Preconditioner | `mas_preconditioner.py` | ✅ Complete (2,123 lines) |
| MAS-PNCG Solver | `mas_pncg_solver.py` | ✅ Complete (854 lines) |
| 2D Subspace Minimization | `mas_pncg_solver.py` | ✅ Complete |
| Powell's Restart Criterion | `mas_pncg_solver.py` | ✅ Complete |
| Sparse-Input Woodbury Updates | `mas_preconditioner.py` | ✅ Complete |
| BVH Collision Detection | `collision_detection_bvh.py` | ✅ Complete |
| LBVH Construction | `lbvh.py` | ✅ Complete |
| ABD System (Rigid Bodies) | `abd_system.py` | ✅ Complete (1,629 lines) |
| ABD-FEM Hybrid Solver | `pncg_abd_ipc.py` | ✅ Complete |
| METIS Graph Partitioning | `metis_reorder.py` | ✅ Complete |
| Configuration System | `config/` | ✅ Complete |

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Demos
```bash
cd PNCG_IPC/demo
python mas_pncg_demo.py           # MAS-PNCG solver demo
python abd_demo.py                # ABD system demo
python cubic_demos.py             # Interactive cube demo
python cubic_demos.py --headless --frames 300  # Headless mode
```

### Testing Demos
Use a 20 second timeout when verifying demo correctness:
```bash
timeout 20s python cubic_demos.py --headless --frames 10
timeout 20s python mas_pncg_demo.py --headless --frames 10
```

### Available Demos
**Main demos (`demo/`):**
- `mas_pncg_demo.py` - Full MAS-PNCG solver demonstration
- `abd_demo.py` - ABD system testing
- `demo_runner.py` - Unified demo framework with RunConfig
- `cubic_demos.py` - Cube squeeze/stretch/rotate
- `drag_armadillo_demo.py` - Armadillo dragging simulation
- `squeeze_armadillo_demo.py` - Armadillo compression
- `twist_demo.py` - Twisting deformation
- `n_E_demo.py` - Multi-object collision demo

**Refactored demos (`demo_new/`):**
- `base_demo.py` - Refactored base demo class
- `cubic_demos.py`, `n_E_demo.py`, `twist_demo.py` - Refactored versions

### Testing
```bash
cd demo
python test_mas_preconditioner.py    # MAS preconditioner tests
python test_metis_reorder.py         # METIS partitioning tests
python test_bvh_performance.py       # BVH benchmark
python compare_collision_detection.py # Collision algorithm comparison
```

## Architecture

### Directory Structure
```
PNCG_IPC/
├── algorithm/          # Core solver implementations (~10,500 lines)
│   ├── mas_preconditioner.py    # MAS preconditioner (2,123 lines)
│   ├── mas_pncg_solver.py       # MAS-PNCG solver (854 lines)
│   ├── abd_system.py            # ABD rigid body system (1,629 lines)
│   ├── pncg_abd_ipc.py          # Hybrid ABD-FEM solver (537 lines)
│   ├── collision_detection_bvh.py # BVH collision detection (699 lines)
│   ├── lbvh.py                  # Linear BVH implementation (754 lines)
│   ├── metis_reorder.py         # METIS graph partitioning (457 lines)
│   ├── pncg_base_ipc.py         # Base PNCG solver with IPC (704 lines)
│   └── base_deformer.py         # FEM foundation (271 lines)
├── config/             # Configuration system
│   ├── base.py         # Config classes (Simulation, Material, Solver, Scene)
│   ├── registry.py     # Decorator-based demo registration
│   └── demos/examples.py # Pre-configured demo setups
├── demo/               # Main demo scripts
├── demo_new/           # Refactored demo framework
├── math_utils/         # Elastic energy and matrix utilities
├── util/               # Model loading and utilities
├── ref_doc/            # Reference documentation
└── model/              # 3D mesh models
```

### Class Hierarchy
```
base_deformer (FEM foundation, vertex/cell management)
    ↓
collision_detection_bvh (BVH-based PT/EE collision detection)
    ↓
pncg_base_ipc (PNCG solver with IPC barrier functions)
    ↓
pncg_abd_ipc (Hybrid ABD-FEM support)
    ↓
Demo classes (application-specific boundary conditions)

MASPreconditioner (standalone, used by mas_pncg_solver)
ABDSystem (standalone, used by pncg_abd_ipc)
```

### Key Modules

**algorithm/**
- `mas_preconditioner.py` - Multilevel Additive Schwarz preconditioner with:
  - Full 48x48 block inversion (BANKSIZE=16 nodes × 3 DOF)
  - Multi-level hierarchy construction (up to 6 levels)
  - Connectivity-aware coarsening with topology-based adjacency
  - Sparse-Input Woodbury rank-1 updates (TOP_K_UPDATES=8)
  - METIS integration for optimal vertex ordering
- `mas_pncg_solver.py` - Complete MAS-PNCG algorithm with:
  - 2D subspace minimization (Eq. 6 from paper)
  - Powell's restart criterion (RESTART_THRESHOLD=0.3)
  - Per-subdomain conservative CCD
  - Line search integration
- `abd_system.py` - Affine Body Dynamics system with:
  - 12D state representation: q = [p; a1; a2; a3]^T
  - Jacobian operations (J, J^T transformations)
  - Motor constraints with angular velocity
  - Boundary conditions (FREE, FIXED, MOTOR)
- `pncg_abd_ipc.py` - Hybrid solver supporting:
  - Mixed ABD-FEM simulation
  - Three contact types: FEM-FEM, ABD-FEM, ABD-ABD
  - Gradient/Hessian transformation for ABD bodies
- `collision_detection_bvh.py` - BVH-based collision with:
  - LBVH construction with Morton codes
  - Point-Triangle (PT) and Edge-Edge (EE) detection
  - Adjacency filtering and BVH refitting
- `lbvh.py` - Linear Bounding Volume Hierarchy with:
  - Morton code computation (3D expansion)
  - Parallel tree building
  - AABB operations and overlap tests
- `metis_reorder.py` - Graph partitioning utilities:
  - k-way mesh partitioning
  - Vertex reordering for block structure optimization
  - Bidirectional partition mappings

**math_utils/**
- `elastic_util.py` - Constitutive models (ARAP, SNH, FCR, NH) with energy, gradient, and Hessian
- `matrix_util.py` - SVD, deformation gradient derivatives, matrix operations

**util/**
- `model_loading.py` - Demo configurations (material params, solver params, mesh paths)

**config/**
- `base.py` - Config classes: SimulationConfig, MaterialConfig, SolverConfig, SceneConfig
- `registry.py` - Decorator-based demo registration pattern
- `demos/examples.py` - Pre-configured demos: cube, cube_10, cube_20, cube_50, eight_E_drop_contact, etc.

### Data Structures (Taichi)

**Vertex attributes:** `x`, `v`, `m`, `x_n`, `x_hat`, `x_prev`, `x_init`, `grad`, `grad_prev`, `p`, `z`, `z_prev`, `w`, `Hv`, `diagH`

**Cell attributes:** `B` (inverse rest deformation gradient), `W` (cell volume weight)

**MAS Preconditioner fields:**
```python
level_sizes: ti.field(dtype=ti.i32)           # Size of each hierarchy level
neighbor_list: ti.field(dtype=ti.i32)         # Topology-based adjacency
connect_mask: ti.field(dtype=ti.u32)          # Connectivity bitmask (BANKSIZE=16)
block_matrices: ti.field(dtype=ti.f64)        # Symmetric block storage
inv_block_matrices: ti.field(dtype=ti.f64)    # Inverted blocks
multi_level_r: ti.Vector.field(3, dtype=ti.f64)  # Restricted residual
multi_level_z: ti.Vector.field(3, dtype=ti.f64)  # Solution at each level
```

### Key Parameters

**MAS Preconditioner:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain (warp subdivision) |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| SYM_BLOCK_COUNT | 136 | Symmetric storage for 16×16 blocks |
| BLOCK_DOF | 48 | 16 nodes × 3 DOF per node |
| MAX_NEIGHBORS_PER_VERTEX | 64 | Max adjacency per vertex |

**MAS-PNCG Solver:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| RESTART_THRESHOLD | 0.3 | Powell's restart threshold (delta) |
| TOP_K_UPDATES | 8 | Rank-1 updates per subdomain |
| CCD_ALPHA_MIN | 1e-6 | Minimum step size |

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

## Version Control

**Important:** After modifying code, commit and push changes promptly to avoid losing work and keep the repository up to date.

```bash
git add -A
git commit -m "Your commit message"
git push
```

## Known Issues

- **NumPy compatibility:** Use numpy 1.26 or older due to meshtaichi compatibility issues
- **MeshTaichi .face files:** Required even if empty (one face entry needed)
- **Platform:** Linux recommended; Windows performance is significantly worse
- **pymetis:** Optional dependency; METIS features disabled if not installed

## Reference Code Location

The CUDA reference implementation is at `/root/Stiff-GIPC_init/StiffGIPC/`:
- `MASPreconditioner.cu` (2361 lines) - Main MAS implementation
- `MASPreconditioner.cuh` - Header with class definition
- `linear_system/preconditioner/fem_mas_preconditioner.cu` - FEM integration
