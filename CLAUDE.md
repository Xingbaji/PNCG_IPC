# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNCG_IPC implements the **MAS-PNCG** framework from the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact". The project features a GPU-accelerated **Multilevel Additive Schwarz (MAS) preconditioner** using Taichi.

### Reference Documents
- `docs/papers/MAS_PNCG_clean.tex` - Main paper with algorithm overview
- `docs/papers/supplementary.tex` - Detailed derivations
- `docs/algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md` - Implementation details
- `docs/algorithm/MAS_PRECONDITIONER_PKG_TESTING.md` - Testing framework

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
├── docs/                   # Documentation
│   ├── algorithm/          # Algorithm implementation docs
│   ├── design/             # Design & planning docs
│   ├── reference/          # Reference implementation docs
│   └── papers/             # Papers (PDF/tex)
├── experiment_reports/     # Experiment reports
├── unittest/               # Unit tests
│   ├── tests/              # Test cases
│   ├── debug/              # Debug utilities
│   └── run_all_tests.py    # Test runner
└── model/                  # 3D mesh models
```

### Class Hierarchy
```
base_deformer → collision_detection_bvh → pncg_base_ipc → Demo classes

MASPreconditioner (algorithm/mas_preconditioner_pkg/)
  - Mixin-based: core, assembly, topology, inversion, schwarz, hierarchy, woodbury, metis_integration
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| RESTART_THRESHOLD | 0.3 | Powell's restart threshold |

## Testing

**⚠️ CRITICAL: 修改代码后必须运行测试！**

```bash
# Run all unit tests
cd unittest
python run_all_tests.py

# Or run specific tests
cd demo
python test_mas_preconditioner.py
python test_metis_reorder.py
```

**测试要求：**
1. **每次修改代码后**，必须运行相关的单元测试
2. **提交代码前**，必须确保所有测试通过
3. **添加新功能时**，必须编写对应的测试用例

## Knowledge Base

**⚠️ CRITICAL: 修改代码后必须更新知识库！**

项目文档按以下结构组织：

```
docs/
├── INDEX.md                 # 文档索引
├── KNOWLEDGE_BASE_GUIDE.md  # 知识库维护指南
├── algorithm/               # 算法实现文档
├── design/                  # 设计与规划文档
├── reference/               # 参考实现文档
└── papers/                  # 论文PDF和tex源文件
```

| 目录 | 用途 | 更新时机 |
|------|------|----------|
| `docs/algorithm/` | 算法实现文档 | 代码重构时 |
| `docs/design/` | 设计与规划文档 | 架构变更时 |
| `docs/reference/` | 参考实现分析 | 研究新方法时 |
| `docs/papers/` | 论文和tex源文件 | 较少更新 |
| `experiment_reports/` | 实验报告 | 每次重要实验后 |

**知识库更新要求：**
1. **修改算法实现** → 更新 `docs/algorithm/` 对应文档
2. **修改架构设计** → 更新 `docs/design/` 对应文档
3. **完成重要实验** → 创建 `experiment_reports/` 实验报告
4. **添加新功能** → 更新相关文档并在 `docs/INDEX.md` 中添加索引

```bash
# 创建实验报告
touch experiment_reports/$(date +%Y-%m-%d)_experiment_name.md
```

详细规范见 `docs/KNOWLEDGE_BASE_GUIDE.md`，文档索引见 `docs/INDEX.md`。

## Development Workflow

**每次修改代码的标准流程：**

```
1. 修改代码
2. 运行测试 → python unittest/run_all_tests.py
3. 更新知识库 → docs/ 或 experiment_reports/
4. 提交代码 → git add -A && git commit -m "[module] desc" && git push
```

## Version Control

**CRITICAL:** After EVERY code modification, immediately commit and push:

```bash
git add -A && git commit -m "[module] description" && git push
```
