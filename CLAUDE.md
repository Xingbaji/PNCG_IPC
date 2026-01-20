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
├── algorithm/                      # Core solvers
│   ├── mas_preconditioner_small/   # MAS preconditioner (默认，简化版)
│   ├── mas_preconditioner_abd/     # MAS preconditioner (ABD系统集成)
│   ├── mas_preconditioner_pkg/     # MAS preconditioner (完整版，已弃用)
│   ├── mas_pncg_solver.py          # MAS-PNCG solver
│   ├── base_deformer.py            # 基础变形器（网格、弹性模型）
│   ├── pncg_base_collision_free.py # PNCG基类（无碰撞）
│   ├── collision_detection_bvh.py  # BVH碰撞检测
│   ├── pncg_base_ipc.py            # PNCG-IPC求解器
│   ├── abd_system.py               # ABD仿射体动力学系统
│   ├── pncg_abd_ipc.py             # 混合ABD-FEM求解器
│   └── gcp_contact_potential.py    # GCP接触势能
├── math_utils/             # Elastic energy, matrix utilities
├── config/                 # Configuration system
├── demo/                   # Main demos
├── n_E_demos/              # N-body demos and benchmarks
├── util/                   # Model loading utilities (含METIS自动重排序)
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

**Deformer 继承链（核心物理模拟）：**
```
base_deformer                      # 基类：网格加载、弹性模型、预计算
    ↓
pncg_base_deformer                 # PNCG求解器基类（无碰撞）
    ↓
collision_detection_bvh_module     # 添加BVH碰撞检测（PT/EE）
    ↓
pncg_ipc_deformer                  # IPC接触处理 + MAS预条件
    ↓
pncg_abd_ipc_deformer              # 混合ABD-FEM求解器（可选）
    ↓
Demo classes (cubic_demos, etc.)   # 具体场景
```

**关键类说明：**
| 类名 | 文件 | 职责 |
|------|------|------|
| `base_deformer` | `algorithm/base_deformer.py` | 网格初始化、弹性能量(ARAP/SNH)、质量矩阵 |
| `pncg_base_deformer` | `algorithm/pncg_base_collision_free.py` | PNCG优化框架、线搜索、收敛判定 |
| `collision_detection_bvh_module` | `algorithm/collision_detection_bvh.py` | LBVH宽相、PT/EE窄相碰撞检测 |
| `pncg_ipc_deformer` | `algorithm/pncg_base_ipc.py` | IPC势能、接触Hessian、MAS预条件器集成 |
| `pncg_abd_ipc_deformer` | `algorithm/pncg_abd_ipc.py` | 混合ABD-FEM，支持仿射体动力学 |
| `ABDSystem` | `algorithm/abd_system.py` | ABD仿射体：12D约化坐标、形状能量 |

**MAS预条件器（三个版本）：**
```
mas_preconditioner_small/  (默认，推荐)
  ├── core.py           # MASPreconditionerSmall 主类
  └── metis_reorder.py  # METIS图划分重排序

mas_preconditioner_abd/   (ABD集成)
  └── core.py           # MASPreconditionerABD (FEM+ABD混合)

mas_preconditioner_pkg/   (完整版，已弃用)
  └── Mixin-based: core, assembly, topology, inversion, schwarz, hierarchy, woodbury
```

**METIS自动重排序：**
- `util/model_loading.py` 在加载网格时自动应用METIS重排序
- 顶点ID直接映射到块：`block_id = vertex_id // 16`, `lane_id = vertex_id % 16`
- 无需运行时映射查找，apply时间 ~0.29ms

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BANKSIZE | 16 | Nodes per subdomain |
| MAX_LEVELS | 6 | Maximum hierarchy depth |
| RESTART_THRESHOLD | 0.3 | Powell's restart threshold |

## Testing

构建测试时可以直接基于已有的 base_deformer
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
