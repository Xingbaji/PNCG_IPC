# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNCG_IPC implements the **MAS-PNCG** framework from the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact". The project features a GPU-accelerated **Multilevel Additive Schwarz (MAS) preconditioner** using Taichi for faster convergence with stiff materials and complex contact scenarios.

### Reference Documents
- `ref_doc/MAS_PNCG_clean.tex` - Main paper with algorithm overview
- `ref_doc/supplementary.tex` - Detailed derivations for MAS, 2x2 subspace minimization, and ACCD bounds
- `ref_doc/MAS_PRECONDITIONER_IMPLEMENTATION.md` - Implementation documentation
- `ref_doc/MAS_PRECONDITIONER_PKG_TESTING.md` - MAS模块化包测试框架文档
- `ref_doc/MAS_PRECONDITIONER_OPTIMIZATION_TODO.md` - Active optimization roadmap
- `ref_doc/STIFFGIPC_IMPLEMENTATION_DETAILS.md` - StiffGIPC reference implementation details
- `ref_doc/GCP_IMPLEMENTATION.md` - Geometric Contact Potential implementation guide
- `ref_doc/CUBIC_BARRIER_IMPLEMENTATION.md` - Cubic barrier function documentation

### Reference Implementation
- `/root/Stiff-GIPC_init/` - CUDA-based MAS preconditioner reference (C++/CUDA)

## Implementation Status

### Completed Features

| Feature | File | Status |
|---------|------|--------|
| MAS Preconditioner | `mas_preconditioner_pkg/` | ✅ Complete (5,300+ lines, 13 modules) |
| MAS-PNCG Solver | `mas_pncg_solver.py` | ✅ Complete (882 lines) |
| 2D Subspace Minimization | `mas_pncg_solver.py` | ✅ Complete |
| Powell's Restart Criterion | `mas_pncg_solver.py` | ✅ Complete |
| Sparse-Input Woodbury Updates | `mas_preconditioner_pkg/woodbury.py` | ✅ Complete |
| BVH Collision Detection | `collision_detection_bvh.py` | ✅ Complete (806 lines) |
| LBVH Construction | `lbvh.py` | ✅ Complete (754 lines) |
| METIS Graph Partitioning | `metis_reorder.py` | ✅ Complete (662 lines) |
| GCP Contact Potential | `gcp_contact_potential.py` | ✅ Complete (1,054 lines) |
| Hierarchical Partition | `hierarchical_partition.py` | ✅ Complete (545 lines) |
| HUP MAS Preconditioner | `hup_mas_preconditioner.py` | ✅ Complete (766 lines) |
| Configuration System | `config/` | ✅ Complete |

> **Note:** The monolithic `mas_preconditioner.py` has been deprecated and moved to `tmp/`. Use the modular `mas_preconditioner_pkg/` instead.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Demos
```bash
cd PNCG_IPC/demo
python mas_pncg_demo.py           # MAS-PNCG solver demo
python cubic_demos.py             # Interactive cube demo
python cubic_demos.py --headless --frames 300  # Headless mode
python gcp_demo.py                # GCP contact potential demo
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
- `demo_runner.py` - Unified demo framework with RunConfig
- `cubic_demos.py` - Cube squeeze/stretch/rotate
- `gcp_demo.py` - GCP contact potential demo
- `abd_demo.py` - ABD solver demonstration
- `drag_armadillo_demo.py` - Armadillo dragging simulation
- `squeeze_armadillo_demo.py` - Armadillo compression
- `twist_demo.py` - Twisting deformation
- `n_E_demo.py` - Multi-object collision demo
- `mas_stiffness_test_demo.py` - Stiffness testing with MAS

**N-body demos (`n_E_demos/`):**
- `bvh_n_E_demo.py` - BVH-based N-body collision demo
- `gcp_n_E_demo_v2.py` - GCP N-body demo
- `mas_pncg_n_E_demo.py` - MAS-PNCG N-body demo
- `cubic_barrier_n_E_demo.py` - Cubic barrier N-body demo
- `initial_n_E_demo.py` - Basic N-body collision demo
- `spatial_hash_n_E_demo.py` - Spatial hash N-body demo
- `run_mas_benchmark.py` - MAS benchmark runner
- `run_n_E_mas_benchmark.py` - N-body MAS benchmark
- `run_mas_benchmark_v2.py` - MAS benchmark v2
- `run_mas_benchmark_v3.py` - MAS benchmark v3

### Testing
```bash
cd demo
python test_mas_preconditioner.py    # MAS preconditioner tests
python test_metis_reorder.py         # METIS partitioning tests
python test_bvh_performance.py       # BVH benchmark
python compare_collision_detection.py # Collision algorithm comparison
python test_hup_mas.py               # HUP MAS preconditioner tests
python test_penetration_detection.py # Penetration detection tests

cd ../n_E_demos
python test_mas_pkg_unittest.py      # MAS package unit tests (37 tests)
python test_mas_pkg_unittest.py -v   # Verbose output
python test_mas_pkg_unittest.py --benchmark  # Performance benchmarks
python test_srbk_spmv.py             # SRBK SpMV tests
python test_mas_simple.py            # Simple MAS tests
```

### MAS Package Unit Tests Details
MAS模块化包包含完整的单元测试框架 (`n_E_demos/test_mas_pkg_unittest.py`):

| 测试类 | 测试数 | 描述 |
|--------|--------|------|
| TestConstants | 8 | 常量模块测试 |
| TestWarpUtils | 14 | 位操作工具测试 |
| TestSRBKSpMV | 5 | SpMV实现测试 |
| TestInversion | 2 | 对称索引测试 |
| TestHierarchy | 3 | 层次结构测试 |
| TestMASPreconditionerIntegration | 5 | 集成测试 |

详见: `ref_doc/MAS_PRECONDITIONER_PKG_TESTING.md`

## Architecture

### Directory Structure
```
PNCG_IPC/
├── algorithm/              # Core solver implementations (~20,000 lines)
│   ├── mas_preconditioner_pkg/     # MAS preconditioner modular (5,300+ lines)
│   │   ├── __init__.py             # Package initialization
│   │   ├── core.py                 # Main MASPreconditioner class (359 lines)
│   │   ├── assembly.py             # Matrix assembly (700+ lines)
│   │   ├── metis_integration.py    # METIS partitioning (595 lines)
│   │   ├── topology.py             # Mesh topology (587 lines)
│   │   ├── woodbury.py             # Woodbury updates (520 lines)
│   │   ├── inversion.py            # Block inversion (505 lines)
│   │   ├── hierarchy.py            # Multi-level hierarchy (433 lines)
│   │   ├── schwarz.py              # Schwarz local solvers (374 lines)
│   │   ├── warp_utils.py           # Bit manipulation (362 lines)
│   │   ├── simple_api.py           # Simplified API (348 lines)
│   │   ├── spmv.py                 # SRBK SpMV (242 lines)
│   │   └── constants.py            # Core constants (79 lines)
│   ├── mas_pncg_solver.py          # MAS-PNCG solver (882 lines)
│   ├── gcp_contact_potential.py    # Geometric contact potential (1,054 lines)
│   ├── collision_detection_bvh.py  # BVH collision detection (806 lines)
│   ├── lbvh.py                     # Linear BVH implementation (754 lines)
│   ├── pncg_base_ipc.py            # Base PNCG solver with IPC (708 lines)
│   ├── hup_mas_preconditioner.py   # HUP MAS preconditioner (766 lines)
│   ├── metis_reorder.py            # METIS graph partitioning (662 lines)
│   ├── hierarchical_partition.py   # Hierarchical partitioning (545 lines)
│   ├── collision_detection.py      # Base collision detection (752 lines)
│   ├── pncg_abd_ipc.py             # ABD variant (537 lines)
│   └── base_deformer.py            # FEM foundation (277 lines)
├── config/                 # Configuration system
│   ├── base.py             # Config classes (171 lines)
│   ├── registry.py         # Decorator-based demo registration (121 lines)
│   └── demos/examples.py   # Pre-configured demo setups (227 lines)
├── demo/                   # Main demo scripts (~5,000 lines)
├── n_E_demos/              # N-body demos and benchmarks (~8,500 lines)
├── math_utils/             # Elastic energy and matrix utilities
│   ├── graphic_util.py     # Visualization utilities (1,065 lines)
│   ├── elastic_util.py     # Constitutive models (540 lines)
│   ├── cubic_roots.py      # Cubic polynomial roots (457 lines)
│   └── matrix_util.py      # Matrix operations (236 lines)
├── util/                   # Model loading and utilities
│   ├── model_loading.py    # Demo configurations (884 lines)
│   ├── sympy_dfdx.py       # Symbolic differentiation (413 lines)
│   └── msh_to_tetgen.py    # Mesh conversion (332 lines)
├── ref_doc/                # Reference documentation and papers
└── model/                  # 3D mesh models
```

### Class Hierarchy
```
base_deformer (FEM foundation, vertex/cell management)
    ↓
collision_detection_bvh (BVH-based PT/EE collision detection)
    ↓
pncg_base_ipc (PNCG solver with IPC barrier functions)
    ↓
Demo classes (application-specific boundary conditions)

MASPreconditioner (standalone, used by mas_pncg_solver)
  - Location: algorithm/mas_preconditioner_pkg/
  - Mixin-based architecture with 13 specialized modules
HUPMASPreconditioner (standalone, hierarchical update variant)
GCPContactPotential (standalone, geometric contact potential)
```

### Key Modules

**algorithm/**
- `mas_preconditioner_pkg/` - Multilevel Additive Schwarz preconditioner (modular):
  - `core.py` - Main MASPreconditioner class and orchestration
  - `assembly.py` - Elastic + contact Hessian matrix assembly
  - `topology.py` - Mesh topology, neighbor list, hierarchy construction
  - `inversion.py` - Block matrix inversion (5 algorithms)
  - `schwarz.py` - Schwarz local solvers (6 variants)
  - `hierarchy.py` - Multi-level restriction/prolongation with P1 optimization
  - `woodbury.py` - Sparse-Input Woodbury rank-1 updates (TOP_K_UPDATES=8)
  - `metis_integration.py` - METIS-based graph partitioning
  - `warp_utils.py` - Bit manipulation and warp-level operations
  - `simple_api.py` - Simplified API without MeshTaichi
  - `spmv.py` - SRBK SpMV implementation
  - `constants.py` - Core constants (BANKSIZE=16, MAX_LEVELS=6, etc.)
- `mas_pncg_solver.py` - Complete MAS-PNCG algorithm with:
  - 2D subspace minimization (Eq. 6 from paper)
  - Powell's restart criterion (RESTART_THRESHOLD=0.3)
  - Per-subdomain conservative CCD
  - Line search integration
- `gcp_contact_potential.py` - Geometric Contact Potential with:
  - Smooth C2 contact energy formulation
  - Point-triangle and edge-edge primitives
  - Analytic gradient and Hessian computation
- `collision_detection_bvh.py` - BVH-based collision with:
  - LBVH construction with Morton codes
  - Point-Triangle (PT) and Edge-Edge (EE) detection
  - Adjacency filtering and BVH refitting
- `lbvh.py` - Linear Bounding Volume Hierarchy with:
  - Morton code computation (3D expansion)
  - Parallel tree building
  - AABB operations and overlap tests
- `hierarchical_partition.py` - Hierarchical mesh partitioning:
  - Multi-level domain decomposition
  - Coarsening strategies for MAS
- `metis_reorder.py` - Graph partitioning utilities:
  - k-way mesh partitioning
  - Vertex reordering for block structure optimization
  - Bidirectional partition mappings

**math_utils/**
- `elastic_util.py` - Constitutive models (ARAP, SNH, FCR, NH) with energy, gradient, and Hessian
- `matrix_util.py` - SVD, deformation gradient derivatives, matrix operations
- `cubic_roots.py` - Cubic polynomial root finding
- `graphic_util.py` - Visualization and graphics utilities

**util/**
- `model_loading.py` - Demo configurations (material params, solver params, mesh paths)
- `sympy_dfdx.py` - SymPy-based symbolic differentiation utilities
- `msh_to_tetgen.py` - Mesh format conversion utilities

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

**CRITICAL:** After EVERY code modification, you MUST immediately commit and push changes. Do not wait or batch multiple changes.

### Auto-commit Rule
After completing any code edit (using Edit or Write tools), immediately run:

```bash
git add -A && git commit -m "描述性提交信息" && git push
```

### Commit Message Guidelines
- Use clear, descriptive messages in Chinese or English
- Format: `[模块] 简短描述`
- Examples:
  - `[assembly] Fix symmetric storage for elastic Hessian`
  - `[CLAUDE.md] Update documentation for modular MAS`
  - `[mas_pncg_solver] Update import to use mas_preconditioner_pkg`

### Why This Matters
- Prevents loss of work if session disconnects
- Keeps repository synchronized
- Provides clear history of changes
- Allows easy rollback if needed

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
