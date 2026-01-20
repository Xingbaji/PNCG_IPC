# 文档索引

> 最后更新: 2026-01-20 (METIS integration in model_loading)

## 目录结构

```
docs/
├── INDEX.md                 # 本文档
├── KNOWLEDGE_BASE_GUIDE.md  # 知识库维护指南
├── algorithm/               # 算法实现文档
├── design/                  # 设计与规划文档
├── reference/               # 参考实现文档
├── papers/                  # 论文和源文件
└── archive/                 # 归档文档
```

---

## MAS Preconditioner 核心文档

`mas_preconditioner_pkg/` 模块的主要文档：

| 文档 | 描述 | 状态 |
|------|------|------|
| [MAS_PRECONDITIONER_IMPLEMENTATION.md](algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md) | **主文档** - 完整实现指南 | ✅ 活跃 |
| [MAS_PRECONDITIONER_PKG_TESTING.md](algorithm/MAS_PRECONDITIONER_PKG_TESTING.md) | 测试框架和用例 | ✅ 活跃 |
| [MAS_PRECONDITIONER_OPTIMIZATION_TODO.md](design/MAS_PRECONDITIONER_OPTIMIZATION_TODO.md) | 优化待办和计划 | ✅ 活跃 |

---

## 算法实现 (algorithm/)

| 文档 | 描述 |
|------|------|
| [MAS_PNCG_ALGORITHM.md](algorithm/MAS_PNCG_ALGORITHM.md) | **MAS-PNCG 算法详解** (纯算法) |
| [MAS_PRECONDITIONER_IMPLEMENTATION.md](algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md) | MAS预条件器完整实现 (含METIS集成) |
| [MAS_PRECONDITIONER_PKG_TESTING.md](algorithm/MAS_PRECONDITIONER_PKG_TESTING.md) | MAS包测试框架 |
| [GCP_IMPLEMENTATION.md](algorithm/GCP_IMPLEMENTATION.md) | 几何接触势能(GCP)实现 |
| [CUBIC_BARRIER_IMPLEMENTATION.md](algorithm/CUBIC_BARRIER_IMPLEMENTATION.md) | 三次障碍函数实现 |
| [CONTACT_FILTER_IMPLEMENTATION.md](algorithm/CONTACT_FILTER_IMPLEMENTATION.md) | **Contact Filter双半径碰撞检测** ✅ 新增 |

### METIS Pre-Reordering (新增 2026-01-20)

METIS重排序现已集成到 `util/model_loading.py` 中，所有网格加载时自动进行METIS重排序：

```python
# 加载模型时自动METIS重排序
model = model_loading(demo='cube_40')
# 顶点ID直接对应METIS分区: block_id = vid // 16, lane_id = vid % 16

# MAS预条件器无需额外参数
preconditioner = MASPreconditionerSmall(model.mesh)
```

详见: [MAS_PRECONDITIONER_IMPLEMENTATION.md § 13](algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md#13-model-loading-with-metis-pre-reordering)

---

## 设计文档 (design/)

| 文档 | 描述 |
|------|------|
| [MESHTAICHI_ARCHITECTURE.md](design/MESHTAICHI_ARCHITECTURE.md) | MeshTaichi架构说明 |
| [MAS_PRECONDITIONER_OPTIMIZATION_TODO.md](design/MAS_PRECONDITIONER_OPTIMIZATION_TODO.md) | MAS优化待办事项 |

---

## 参考实现 (reference/)

外部参考实现的分析文档。代码路径: `/root/Stiff-GIPC_init/`

### Stiff-GIPC 核心文档

| 文档 | 描述 | 状态 |
|------|------|------|
| [STIFFGIPC_MAS_REFERENCE.md](reference/STIFFGIPC_MAS_REFERENCE.md) | **MAS预条件器CUDA参考实现详解** | ✅ 新增 |
| [STIFFGIPC_PROJECT_STRUCTURE.md](reference/STIFFGIPC_PROJECT_STRUCTURE.md) | 项目结构和模块概览 | ✅ 新增 |
| [STIFFGIPC_IMPLEMENTATION_DETAILS.md](reference/STIFFGIPC_IMPLEMENTATION_DETAILS.md) | 算法实现细节 | ✅ 活跃 |
| [STIFFGIPC_OPTIMIZATION_SUMMARY.md](reference/STIFFGIPC_OPTIMIZATION_SUMMARY.md) | 优化策略总结 | ✅ 活跃 |

### 关键参考文件

| 文件路径 | 行数 | 描述 |
|----------|------|------|
| `StiffGIPC/MASPreconditioner.cu` | 2,361 | MAS 完整 CUDA 实现 |
| `StiffGIPC/MASPreconditioner.cuh` | 103 | MAS 类声明 |
| `StiffGIPC/eigen_data.h` | 206 | 核心数据结构定义 |
| `StiffGIPC/linear_system/` | - | 线性系统框架 |
| `MeshProcess/metis_partition/` | - | METIS 网格分区工具 |

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
| [MAS_DIRECTION_ACCURACY_ANALYSIS.md](../experiment_reports/MAS_DIRECTION_ACCURACY_ANALYSIS.md) | 2026-01-20 | **MAS方向精度分析** | ✅ 完成 |
| [MAS_PKG_SPEED_TEST_REPORT.md](../experiment_reports/MAS_PKG_SPEED_TEST_REPORT.md) | 2026-01-19 | MAS包性能测试 | ✅ 完成 |
| [MAS_SYMMETRY_BUG_ANALYSIS.md](../experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md) | 2026-01-19 | 矩阵对称性问题分析 | ✅ 已修复 |

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

## Demo配置系统 (demo_settings/)

YAML配置系统用于定义模拟demo的参数。位于项目根目录下。

### 目录结构

| 子目录 | 描述 |
|--------|------|
| `basic/` | 基础demo (cube, banana, armadillo等) |
| `contact/` | 接触/碰撞demo (IPC启用) |
| `dirichlet/` | Dirichlet边界条件demo |
| `stiff_gipc/` | 高刚度GIPC测试 |
| `freefall/` | 自由落体demo |
| `mas_test/` | MAS预条件器测试 |
| `unittest/` | 单元测试用demo |

### 使用方法

```python
# 推荐方式：使用demo名称
from demo_settings import load_demo, load_demo_config, list_demos

# 加载配置
config = load_demo_config('cube_40')  # 返回DemoConfig对象
demo_dict = load_demo('cube_40')       # 返回传统字典格式

# 列出所有demo
demos = list_demos()                   # 返回列表
demos = list_demos(by_category=True)   # 按类别分组
```

### YAML配置示例

```yaml
# demo_settings/basic/cube.yaml
material:
  E: 1.0e4        # 杨氏模量 (支持科学计数法)
  nu: 0.4         # 泊松比
  density: 1000.0
  elastic_type: ARAP_SPD

solver:
  epsilon: 1e-5
  iter_max: 50
  dt: 0.04

scene:
  ground_height: 0.1
  gravity: -9.8
  meshes:
    - path: ../model/mesh/cube_1/cube_1.node
      scale: [0.5, 0.5, 0.5]
      translation: [0.0, 0.5, 0.0]

ipc:
  enabled: false
```

---

## 快速链接

- **入口文件**: [CLAUDE.md](../CLAUDE.md)
- **项目说明**: [README.md](../README.md)
- **Demo配置目录**: [demo_settings/](../demo_settings/)
- **实验报告目录**: [experiment_reports/](../experiment_reports/)
- **单元测试目录**: [unittest/](../unittest/)

---

## 归档文档 (archive/)

历史文档，内容已整合到主文档或已过时。

| 文档 | 原因 |
|------|------|
| MAS_IMPLEMENTATION_DETAILS.md | 内容已整合到 MAS_PRECONDITIONER_IMPLEMENTATION.md |
| MAS_PNCG_IMPLEMENTATION_PLAN.md | 算法计划，内容已实现 |
| MAS_MESHTAICHI_INTEGRATION_DESIGN.md | 未来设计，暂时归档 |

---

## 文档状态说明

- ✅ 活跃/最新
- 🔄 需要更新
- 📝 草稿状态
- 📦 已归档

```bash
# 查看最近修改的文档
ls -lt docs/**/*.md experiment_reports/*.md | head -10
```
