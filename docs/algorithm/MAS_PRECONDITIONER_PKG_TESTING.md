# MAS Preconditioner Package - Testing Documentation

本文档描述了重构后的MAS Preconditioner模块化包(`algorithm/mas_preconditioner_pkg/`)的单元测试框架。

> **最后更新**: 2026-01-19

## 概述

MAS Preconditioner Package是将原始的`mas_preconditioner.py`(约5000行)重构为模块化目录结构(6092行)的结果。测试框架用于验证每个模块的正确性和性能。

> **Note:** 原始的单体版本 `mas_preconditioner.py` 已被弃用并移动到 `tmp/` 文件夹。请使用模块化版本 `mas_preconditioner_pkg/`。

### 测试文件位置

测试框架已迁移到统一的 `unittest/` 目录:

```
unittest/
├── README.md              # 测试说明
├── TEST_REPORT.md         # 最新测试报告 (44.4% 通过率)
├── run_all_tests.py       # 主测试运行器
├── tests/                 # 20个测试文件
│   ├── test_mas_ground_truth.py   # 核心Ground Truth测试 (22 tests) ✅
│   ├── test_mas_multilevel.py     # 多层级测试 (25 tests) ✅
│   ├── test_mas_simple.py         # 简单模拟测试 ✅
│   ├── test_mas_pkg_unittest.py   # 模块化包单元测试
│   └── ...
└── debug/                 # 调试脚本
```

## 模块化包结构

```
algorithm/mas_preconditioner_pkg/
├── __init__.py              # 模块导出
├── constants.py             # 核心常量定义
├── core.py                  # 主MASPreconditioner类
├── topology.py              # 网格拓扑和邻居构建
├── assembly.py              # 矩阵组装 (弹性+接触)
├── inversion.py             # 块矩阵求逆算法
├── schwarz.py               # Schwarz局部求解器
├── hierarchy.py             # 多级限制与延拓
├── woodbury.py              # Woodbury低秩更新
├── metis_integration.py     # METIS重排序集成
├── simple_api.py            # 简化API接口
├── spmv.py                  # 稀疏矩阵向量乘
└── warp_utils.py            # Warp工具函数
```

## 测试类和方法

### 1. TestConstants (8个测试)

测试`constants.py`模块中的常量定义。

| 测试方法 | 描述 |
|----------|------|
| `test_banksize_value` | 验证BANKSIZE=16 |
| `test_max_levels_value` | 验证MAX_LEVELS在4-8之间 |
| `test_sym_block_count` | 验证SYM_BLOCK_COUNT=136 (16*17/2) |
| `test_block_dof` | 验证BLOCK_DOF=48 (16*3) |
| `test_max_neighbors_per_vertex` | 验证邻居数在32-128之间 |
| `test_node_bandwidth` | 验证NODE_BANDWIDTH>0 |
| `test_top_k_updates` | 验证TOP_K_UPDATES>0 |
| `test_eps_value` | 验证EPS在(0, 1e-6)范围内 |

### 2. TestWarpUtils (14个测试)

测试`warp_utils.py`模块中的位操作工具函数。

| 测试方法 | 描述 |
|----------|------|
| `test_popcount_zero` | popcount(0) = 0 |
| `test_popcount_one` | popcount(1) = 1 |
| `test_popcount_powers_of_two` | popcount(2^i) = 1 |
| `test_popcount_all_ones` | popcount(0xFFFFFFFF) = 32 |
| `test_popcount_specific` | 特定值测试 |
| `test_clz_zero` | clz(0) = 32 |
| `test_clz_one` | clz(1) = 31 |
| `test_clz_powers_of_two` | clz(2^i) = 31-i |
| `test_ffs_zero` | ffs(0) = 0 |
| `test_ffs_one` | ffs(1) = 1 |
| `test_ffs_powers_of_two` | ffs(2^i) = i+1 |
| `test_ffs_zero_indexed` | ffs_zero_indexed返回0-based索引 |
| `test_lanemask_lt` | lanemask_lt返回正确的位掩码 |
| `test_bit_reverse` | 32位整数位反转 |

### 3. TestSRBKSpMV (5个测试)

测试`spmv.py`模块中的稀疏矩阵向量乘实现。

| 测试方法 | 描述 |
|----------|------|
| `test_initialization` | SpMV对象初始化 |
| `test_clear` | 清除triplets |
| `test_identity_spmv` | 单位矩阵SpMV: y = I*x |
| `test_symmetric_storage` | 对称存储验证 |
| `test_sort_by_row` | 按行排序triplets |

### 4. TestInversion (2个测试)

测试`inversion.py`模块中的对称索引计算。

| 测试方法 | 描述 |
|----------|------|
| `test_sym_index_diagonal` | 对角线元素索引: [0,16,31,45,...,135] |
| `test_sym_index_upper_triangle` | 上三角覆盖所有136个条目 |

### 5. TestHierarchy (3个测试)

测试层次结构构建逻辑。

| 测试方法 | 描述 |
|----------|------|
| `test_level_count_small_mesh` | 小网格(100顶点)应有1-3个level |
| `test_level_count_large_mesh` | 大网格(10000顶点)应有3-6个level |
| `test_hierarchy_size_estimation` | 总大小应小于n_verts*1.2 |

### 6. TestMASPreconditionerIntegration (5个测试)

集成测试完整的MASPreconditioner类。

| 测试方法 | 描述 |
|----------|------|
| `test_initialization` | MAS初始化验证 |
| `test_build_hierarchy` | 层次结构构建 |
| `test_assemble_and_invert` | 矩阵组装和求逆 |
| `test_apply` | 预条件器应用 |
| `test_get_stats` | 统计信息获取 |

## 性能基准测试

`MASBenchmarks`类提供以下基准测试方法:

### benchmark_hierarchy_build(n_runs=5)
测量层次结构构建时间。

### benchmark_matrix_assembly(n_runs=10)
测量矩阵组装时间。

### benchmark_inversion_methods(n_runs=5)
比较不同求逆算法的性能:
- Gauss-Jordan
- One-way Gauss-Jordan (P4优化)
- Cholesky
- Incomplete Cholesky IC(0)
- Diagonal Only

### benchmark_apply(n_runs=20)
测量预条件器应用时间。

### benchmark_full_pipeline(n_runs=5)
测量完整的rebuild+apply管道时间。

## 使用方法

### 运行所有测试

```bash
cd /root/PNCG_IPC/unittest
python run_all_tests.py
```

### 详细输出

```bash
python run_all_tests.py -v
```

### 运行特定测试

```bash
# 运行核心测试套件
python -m pytest tests/test_mas_ground_truth.py -v
python -m pytest tests/test_mas_multilevel.py -v

# 运行单元测试
python -m pytest tests/test_mas_pkg_unittest.py -v
```

### 运行调试脚本

```bash
cd /root/PNCG_IPC/unittest/debug
python debug_sym_expand.py
python debug_assembly_logic.py
```

## 测试框架设计

### Taichi初始化

测试框架在模块导入时自动初始化Taichi:

```python
_taichi_initialized = False

def init_taichi():
    global _taichi_initialized
    if _taichi_initialized:
        return
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    _taichi_initialized = True

# 模块导入时初始化
init_taichi()
```

### Taichi Kernel测试模式

由于Taichi kernel不能直接作为unittest方法，测试使用以下模式:

```python
@ti.data_oriented
class WarpUtilTester:
    def __init__(self):
        self.result_i32 = ti.field(dtype=ti.i32, shape=())
        self.result_u32 = ti.field(dtype=ti.u32, shape=())

    @ti.kernel
    def test_popcount(self, x: ti.u32):
        self.result_i32[None] = warp_utils.popcount_u32(x)

# 在测试中使用
cls.tester = WarpUtilTester()
cls.tester.test_popcount(10)
self.assertEqual(cls.tester.result_i32[None], 2)
```

### 性能计时器

```python
class PerfTimer:
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed_ms = 0.0

    def __enter__(self):
        ti.sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        ti.sync()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
```

## 重构过程中发现并修复的问题

### 1. topology.py - 缺失build_hierarchy方法

**问题**: `build_hierarchy()`方法在原始文件中存在，但未被移植到模块化版本。

**修复**: 在`topology.py`中添加了完整的`build_hierarchy()`方法实现。

### 2. core.py - Taichi函数条件返回

**问题**: `_lanemask_lt`函数在if分支中使用return，Taichi不支持。

**修复**:
```python
# 修复前
@ti.func
def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
    if lane_id <= 0:
        return ti.u32(0)
    return (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)

# 修复后
@ti.func
def _lanemask_lt(self, lane_id: ti.i32) -> ti.u32:
    result = ti.u32(0)
    if lane_id > 0:
        result = (ti.u32(1) << ti.u32(lane_id)) - ti.u32(1)
    return result
```

### 3. assembly.py - Kernel内部import

**问题**: `math_utils`的import语句放在kernel内部，Taichi不支持。

**修复**: 将import移到文件顶部:
```python
# 文件顶部
from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import (
    compute_d2PsidF2_ARAP_filter, compute_d2PsidF2_SNH, compute_d2PsidF2_FCR_filter
)
```

## 测试结果概要

### 当前状态 (2026-01-19)

| 类别 | 通过 | 失败 | 错误 | 总计 |
|------|------|------|------|------|
| Core Test Suites | 2 | 0 | 0 | 2 |
| Functional Validation | 1 | 1 | 1 | 3 |
| Assembly Tests | 1 | 1 | 1 | 3 |
| Symmetry Tests | 1 | 0 | 1 | 2 |
| Hierarchy Tests | 2 | 0 | 1 | 3 |
| Specific Issue Tests | 1 | 0 | 4 | 5 |
| **总计** | **8** | **2** | **8** | **18** |

**成功率**: 44.4% (8/18 通过)

### 核心测试全部通过:
- `test_mas_ground_truth.py`: 22 tests ✅
- `test_mas_multilevel.py`: 25 tests (6 skipped) ✅
- `test_mas_simple.py`: 4 frames ✅
- `test_assembly_logic.py`: 装配逻辑 ✅

### 已知问题
详见: `unittest/TEST_REPORT.md` 和 `experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md`

## 相关文档

- [MAS_PRECONDITIONER_IMPLEMENTATION.md](MAS_PRECONDITIONER_IMPLEMENTATION.md) - MAS实现详细文档
- [unittest/TEST_REPORT.md](../../unittest/TEST_REPORT.md) - 完整测试报告
- [experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md](../../experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md) - 对称性问题分析

## 版本历史

- **2026-01-19**: 测试框架迁移到 `unittest/` 目录，20个测试文件
- **2026-01-19**: 添加 Ground Truth 测试套件
- **2026-01-19**: 创建初始测试框架，修复3个模块化遗留问题
