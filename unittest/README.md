# MAS Preconditioner Unit Tests

本目录包含 MAS（Multilevel Additive Schwarz）预条件器的单元测试和调试脚本。

## 目录结构

```
unittest/
├── README.md           # 本文档
├── debug/              # 调试脚本（用于问题定位）
└── tests/              # 单元测试（用于验证正确性）
```

## 运行测试

```bash
# 进入测试目录
cd /root/PNCG_IPC/unittest/tests

# 设置 PYTHONPATH
export PYTHONPATH=/root/PNCG_IPC

# 运行 ground truth 测试
python test_mas_ground_truth.py -v

# 运行多层级测试
python test_mas_multilevel.py -v

# 运行简单功能测试
python test_mas_simple.py
```

## 最新测试报告

详见 [TEST_REPORT.md](TEST_REPORT.md)

---

## 调试脚本 (`debug/`)

用于问题定位和深入分析的脚本。

| 文件 | 用途 |
|------|------|
| `debug_assembly_logic.py` | 调试对称块矩阵装配逻辑，排查重复计数问题 |
| `debug_block0_detailed.py` | 详细调试 Block 0（首个 warp），分析负特征值来源 |
| `debug_mas_gTz.py` | 调试 gTz < 0 问题（预条件方向与梯度点积为负） |
| `debug_mas_nan.py` | 排查 Incomplete Cholesky 求逆中的 NaN 问题 |
| `debug_sym_expand.py` | 纯 NumPy 测试对称存储展开逻辑 |

---

## 单元测试 (`tests/`)

### 核心测试套件

| 文件 | 用途 | 优先级 |
|------|------|--------|
| `test_mas_pkg_unittest.py` | MAS 包完整单元测试框架，含常量验证和性能基准 | **高** |
| `test_mas_ground_truth.py` | NumPy ground truth 对比测试，验证装配、求逆、SPMv | **高** |
| `test_mas_multilevel.py` | 多层级 ground truth 测试，含限制/延拓操作验证 | **高** |
| `test_inversion_methods.py` | 不同求逆方法的正确性和性能对比测试（6种方法） | **高** |

### 功能验证测试

| 文件 | 用途 |
|------|------|
| `test_mas_simple.py` | 单帧无碰撞场景下的 MAS 预条件器简单测试 |
| `test_mas_freefall.py` | 自由落体场景验证，与牛顿定律 ground truth 对比 |
| `test_mas_matrix_diagnostic.py` | 模块化诊断测试，分别测试惯性、弹性、组合贡献 |

### 装配逻辑测试

| 文件 | 用途 |
|------|------|
| `test_assembly_detail.py` | 单元素贡献的详细装配诊断 |
| `test_assembly_logic.py` | 多种顶点排序下的装配逻辑综合测试 |
| `test_assembly_precise.py` | 基于 Taichi 原子操作的精确装配测试 |

### 对称性验证测试

| 文件 | 用途 |
|------|------|
| `test_He_symmetry.py` | 验证 Taichi 计算中元素 Hessian 的对称性 |
| `test_He_symmetry_simple.py` | 纯 NumPy 证明 H_e 的对称性构造 |

### 层级结构测试

| 文件 | 用途 |
|------|------|
| `test_hierarchy_mapping.py` | 测试层级构建和 going_next 映射 |
| `test_level0_only.py` | 隔离 Level 0（同 warp）装配，排查不对称来源 |
| `test_crosswarp_issue.py` | 测试跨 warp 装配效应 |

### 特定问题测试

| 文件 | 用途 |
|------|------|
| `test_diagonal_contrib.py` | 验证对角块 H_e[i,i] 贡献的正确性 |
| `test_diagonal_simple.py` | 简化的对角贡献测试 |
| `test_debug_simple.py` | 简单调试测试，分析包含特定顶点的元素 |
| `test_nonopt_kernel.py` | 对比优化和非优化内核，排查 CUDA 优化问题 |
| `test_upper_triangle_bug.py` | 分析上三角处理（j >= i）导致的问题 |

---

## 测试分类

### 按测试类型

- **Ground Truth 测试**: `test_mas_ground_truth.py`, `test_mas_multilevel.py`, `test_mas_freefall.py`
- **单元测试**: `test_mas_pkg_unittest.py`, `test_mas_simple.py`
- **诊断测试**: `test_mas_matrix_diagnostic.py`, `test_assembly_*.py`
- **回归测试**: `test_He_symmetry*.py`, `test_hierarchy_mapping.py`

### 按测试目标

- **装配正确性**: `test_assembly_*.py`, `test_diagonal_*.py`
- **对称性验证**: `test_He_symmetry*.py`, `test_upper_triangle_bug.py`
- **层级结构**: `test_hierarchy_mapping.py`, `test_level0_only.py`, `test_crosswarp_issue.py`
- **求解器验证**: `test_mas_freefall.py`, `test_mas_simple.py`

---

## 开发指南

### 添加新测试

1. 根据测试目的选择放入 `debug/` 或 `tests/`
2. 命名规范：
   - 调试脚本：`debug_<issue>.py`
   - 单元测试：`test_<feature>.py`
3. 更新本 README

### 测试最佳实践

```python
# 推荐的测试结构
import unittest
import numpy as np

class TestFeature(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        pass

    def test_basic_functionality(self):
        """基本功能测试"""
        pass

    def test_edge_cases(self):
        """边界情况测试"""
        pass

if __name__ == '__main__':
    unittest.main()
```

---

## 相关文档

- [MAS 预条件器实现](../docs/algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md)
- [MAS 包测试框架](../docs/algorithm/MAS_PRECONDITIONER_PKG_TESTING.md)
- [对称性问题分析](../experiment_reports/MAS_SYMMETRY_BUG_ANALYSIS.md)
- [性能测试报告](../experiment_reports/MAS_PKG_SPEED_TEST_REPORT.md)
- [主论文](../docs/papers/MAS_PNCG_clean.tex)
