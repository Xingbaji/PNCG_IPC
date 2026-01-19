# 文档索引

> 最后更新: 2026-01-19

## 目录结构

```
docs/
├── INDEX.md                 # 本文档
├── KNOWLEDGE_BASE_GUIDE.md  # 知识库维护指南
├── algorithm/               # 算法实现文档
├── design/                  # 设计与规划文档
├── reference/               # 参考实现文档
└── papers/                  # 论文和源文件
```

---

## 算法实现 (algorithm/)

核心算法的实现细节和测试文档。

| 文档 | 描述 |
|------|------|
| [MAS_PRECONDITIONER_IMPLEMENTATION.md](algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md) | MAS预条件器实现 |
| [MAS_IMPLEMENTATION_DETAILS.md](algorithm/MAS_IMPLEMENTATION_DETAILS.md) | MAS算法详细说明 |
| [GCP_IMPLEMENTATION.md](algorithm/GCP_IMPLEMENTATION.md) | 几何接触势能(GCP)实现 |
| [CUBIC_BARRIER_IMPLEMENTATION.md](algorithm/CUBIC_BARRIER_IMPLEMENTATION.md) | 三次障碍函数实现 |
| [MAS_PRECONDITIONER_PKG_TESTING.md](algorithm/MAS_PRECONDITIONER_PKG_TESTING.md) | MAS包测试框架 |

---

## 设计文档 (design/)

架构设计、集成方案和优化计划。

| 文档 | 描述 |
|------|------|
| [MAS_MESHTAICHI_INTEGRATION_DESIGN.md](design/MAS_MESHTAICHI_INTEGRATION_DESIGN.md) | MeshTaichi集成设计 |
| [MESHTAICHI_ARCHITECTURE.md](design/MESHTAICHI_ARCHITECTURE.md) | MeshTaichi架构说明 |
| [MAS_PRECONDITIONER_OPTIMIZATION_TODO.md](design/MAS_PRECONDITIONER_OPTIMIZATION_TODO.md) | MAS优化待办事项 |

---

## 参考实现 (reference/)

外部参考实现的分析文档。

| 文档 | 描述 |
|------|------|
| [STIFFGIPC_IMPLEMENTATION_DETAILS.md](reference/STIFFGIPC_IMPLEMENTATION_DETAILS.md) | Stiff-GIPC实现分析 |
| [STIFFGIPC_OPTIMIZATION_SUMMARY.md](reference/STIFFGIPC_OPTIMIZATION_SUMMARY.md) | Stiff-GIPC优化总结 |

---

## 论文资料 (papers/)

相关论文PDF和LaTeX源文件。

### 核心论文
| 文件 | 描述 |
|------|------|
| siggraphconferencepapers24-96_camera.pdf | MAS-PNCG主论文 |
| PNCG_supplemental_document.pdf | PNCG补充材料 |
| MAS_PNCG_clean.tex | MAS-PNCG论文源文件 |
| supplementary.tex | 补充材料源文件 |

### 参考论文
| 文件 | 描述 |
|------|------|
| stiffgipc.pdf | Stiff-GIPC论文 |
| Geometric Contact Potential.pdf | GCP论文 |
| cubic_barrier.pdf | 三次障碍函数论文 |
| cipc.pdf | C-IPC论文 |
| yu22meshtaichi.pdf | MeshTaichi论文 |
| Wu-2022-GBM.pdf | GBM论文 |
| C5D.pdf | C5D相关论文 |

---

## 实验报告 (experiment_reports/)

位于项目根目录下，记录实验过程和结果。

| 报告 | 日期 | 主题 | 状态 |
|------|------|------|------|
| [MAS_PKG_SPEED_TEST_REPORT.md](../experiment_reports/MAS_PKG_SPEED_TEST_REPORT.md) | 2026-01-19 | MAS包性能测试 | ✅ 完成 |
| [MAS_SYMMETRY_BUG_ANALYSIS.md](../experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md) | 2026-01-19 | 矩阵对称性问题分析 | 🔄 进行中 |

---

## 单元测试 (unittest/)

位于项目根目录下，包含完整的测试框架。

| 文件 | 描述 |
|------|------|
| [unittest/README.md](../unittest/README.md) | 测试说明 |
| [unittest/TEST_REPORT.md](../unittest/TEST_REPORT.md) | 最新测试报告 (44.4% 通过率) |
| `unittest/tests/` | 20个测试文件 |
| `unittest/debug/` | 调试脚本 |

### 核心测试状态
| 测试 | 状态 | 说明 |
|------|------|------|
| test_mas_ground_truth | ✅ 通过 | 22个测试 |
| test_mas_multilevel | ✅ 通过 | 25个测试 (6跳过) |
| test_mas_simple | ✅ 通过 | 4帧模拟 |
| test_assembly_logic | ✅ 通过 | 装配逻辑验证 |

---

## 快速链接

- **入口文件**: [CLAUDE.md](../CLAUDE.md)
- **项目说明**: [README.md](../README.md)
- **实验报告目录**: [experiment_reports/](../experiment_reports/)
- **单元测试目录**: [unittest/](../unittest/)

---

## 文档状态说明

- ✅ 已完成且最新
- 🔄 需要更新
- 📝 草稿状态
- ❌ 已废弃

```bash
# 查看最近修改的文档
ls -lt docs/**/*.md experiment_reports/*.md | head -10
```
