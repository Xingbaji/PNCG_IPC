# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNCG_IPC implements the **MAS-PNCG** framework from the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact". The project features a GPU-accelerated **Multilevel Additive Schwarz (MAS) preconditioner** using Taichi.

### Reference Documents
- `ref_doc/MAS_PNCG_clean.tex` - Main paper with algorithm overview
- `ref_doc/supplementary.tex` - Detailed derivations
- `ref_doc/MAS_PRECONDITIONER_IMPLEMENTATION.md` - Implementation details
- `ref_doc/MAS_PRECONDITIONER_PKG_TESTING.md` - Testing framework

### Reference Implementation
- `/root/Stiff-GIPC_init/` - CUDA reference (C++/CUDA)

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Run demo
cd PNCG_IPC/demo
python mas_pncg_demo.py

# Test (20s timeout)
timeout 20s python cubic_demos.py --headless --frames 10
```

## Architecture

```
PNCG_IPC/
├── algorithm/              # Core solvers
│   ├── mas_preconditioner_pkg/  # MAS preconditioner (modular)
│   ├── mas_pncg_solver.py       # MAS-PNCG solver
│   ├── collision_detection_bvh.py
│   ├── gcp_contact_potential.py
│   └── pncg_base_ipc.py
├── math_utils/             # Elastic energy, matrix utilities
├── config/                 # Configuration system
├── demo/                   # Main demos
├── n_E_demos/              # N-body demos and benchmarks
├── util/                   # Model loading utilities
├── ref_doc/                # Documentation
└── model/                  # 3D mesh models
```

### Class Hierarchy
```
base_deformer → collision_detection_bvh → pncg_base_ipc → Demo classes

MASPreconditioner (algorithm/mas_preconditioner_pkg/)
  - Mixin-based: core, assembly, topology, inversion, schwarz, hierarchy, woodbury
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| RESTART_THRESHOLD | 0.3 | Powell's restart threshold |

## Testing

```bash
# MAS package unit tests
cd n_E_demos
python test_mas_pkg_unittest.py -v

# Other tests
cd demo
python test_mas_preconditioner.py
python test_metis_reorder.py
```

## Version Control

**CRITICAL:** After EVERY code modification, immediately commit and push:

```bash
git add -A && git commit -m "[module] description" && git push
```

## Known Issues

- **NumPy:** Use numpy 1.26 or older (meshtaichi compatibility)
- **MeshTaichi:** .face files required even if empty
- **Platform:** Linux recommended
- **pymetis:** Optional; METIS features disabled if not installed
