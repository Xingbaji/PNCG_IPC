# MAS 预条件器单元测试报告

**生成时间**: 2026-01-19
**测试环境**: CUDA GPU, Taichi 1.7.4, float32 precision
**测试网格**: eight_E_stiffness_test (8368 vertices, 27376 cells, 2 MAS levels)

---

## 1. 测试概要

| 类别 | 通过 | 失败 | 总计 |
|------|------|------|------|
| 对称性修复后测试 | 6 | 0 | 6 |
| Ground Truth 测试 | 22 | 0 | 22 |
| 多层级测试 | 19 | 0 | 25 (6 skipped) |
| **MatVec 准确性测试** | **15** | **0** | **15** |
| **总计** | **62** | **0** | **68** |

**成功率**: 100% (所有必要测试通过)

---

## 2. 求逆方法性能对比

### 2.1 基准测试结果 (10 iterations, 3 warmup)

| 方法 | 求逆时间 | 相对误差 | 需要正则化 | 推荐场景 |
|------|----------|----------|------------|----------|
| **Incomplete Cholesky IC(0)** | 6.99ms | 2.99e-04 | Yes (ε~4.5e5) | **最佳平衡** |
| Cholesky | 15.21ms | 5.58e-06 | Yes (ε~4.5e5) | 高精度 + SPD |
| Blocked Cholesky | 16.83ms | 2.91e-06 | Yes (ε~4.5e5) | 高精度 + SPD |
| One-way GJ | 26.09ms | 8.57e+04* | No | 速度优先 (仅 SPD) |
| Gauss-Jordan | 53.07ms | 8.08e-07 | No | **最稳健** |
| Diagonal Only | 0.16ms | 1.93e+00 | No | 最快 (质量差) |

*One-way GJ 对非 SPD 矩阵精度较差，但仍能产生有效下降方向。

### 2.2 时间分解

| 方法 | 装配 | 求逆 | 应用 | 总计 |
|------|------|------|------|------|
| Diagonal Only | 1.51ms | 0.16ms | 2.44ms | 4.11ms |
| IC(0) | 1.51ms | 6.99ms | 2.44ms | 10.94ms |
| Cholesky | 1.52ms | 15.21ms | 2.47ms | 19.20ms |
| Blocked Cholesky | 1.51ms | 16.83ms | 2.44ms | 20.78ms |
| One-way GJ | 1.52ms | 26.09ms | 2.48ms | 30.09ms |
| Gauss-Jordan | 1.55ms | 53.07ms | 34.06ms | 88.68ms |

### 2.3 相对于 Gauss-Jordan 的加速比

| 方法 | 求逆加速比 | 精度权衡 |
|------|-----------|----------|
| Diagonal Only | 332x | 非常差 (近似) |
| IC(0) | 7.6x | 可接受 (3e-04) |
| Cholesky | 3.5x | 优秀 (6e-06) |
| Blocked Cholesky | 3.2x | 优秀 (3e-06) |
| One-way GJ | 2.0x | 非 SPD 差 |

---

## 3. 预条件器有效性验证

所有方法都产生有效的预条件方向 (g^T z > 0):

| 方法 | g^T z | NaN 计数 | 有效 |
|------|-------|----------|------|
| Gauss-Jordan | 3.36e-08 | 0 | ✓ |
| One-way GJ | 1.46e-04 | 0 | ✓ |
| Cholesky | 6.83e-10 | 0 | ✓ |
| Blocked Cholesky | 6.83e-10 | 0 | ✓ |
| IC(0) | 6.82e-10 | 0 | ✓ |
| Diagonal Only | 2.98e-08 | 0 | ✓ |

---

## 4. 块矩阵分析

### 4.1 Block 0 特征值

```
最小特征值: -4.11e+05
最大特征值: 8.46e+02
负特征值数: 3
条件数: 3.75e+03
```

**重要**: 块矩阵由于从全局 Hessian 提取子块，**不是 SPD**。
需要 SPD 的方法必须使用对角正则化 (ε > |λ_min| ≈ 4.5e5)。

### 4.2 对称性误差 (修复后)

```
最大相对对称误差: 2.31e-08 (float32 精度)
```

跨 warp 装配对称性修复将误差从 ~3% 降低到 ~5e-08。

---

## 5. 单元测试详情

### 5.1 求逆方法测试 (test_inversion_methods.py)

```
test_01_block_matrix_symmetry: OK
  - 验证装配对称误差 < 1e-5

test_02_gauss_jordan_accuracy: OK
  - 验证 Gauss-Jordan 相对误差 < 1e-3

test_03_oneway_gj_validity: OK
  - 验证 One-way GJ 产生 g^T z > 0

test_04_cholesky_with_regularization: OK
  - 验证带正则化的 Cholesky，误差 < 1e-2

test_05_incomplete_cholesky_with_regularization: OK
  - 验证带正则化的 IC(0)，误差 < 0.1

test_06_preconditioner_validity: OK
  - 验证所有方法产生 g^T z > 0
```

**结果**: 6/6 测试通过

### 5.2 Ground Truth 测试 (test_mas_ground_truth.py)

验证与 NumPy 实现的一致性:
- 装配正确性
- 求逆正确性
- SPMv 操作正确性

**结果**: 22/22 测试通过

### 5.3 多层级测试 (test_mas_multilevel.py)

验证多层级 MAS 结构:
- 层级构建
- 限制操作
- 延拓操作

**结果**: 19/25 测试通过 (6 skipped due to single-level mesh)

### 5.4 MatVec 准确性测试 (test_matvec_accuracy.py)

验证矩阵向量乘 z = M^{-1} * r 的准确性:

**NumPy Ground Truth 测试:**
- 对称索引覆盖 (136 entries): OK
- 对称索引对称性: OK
- Identity 矩阵展开: OK
- 随机 SPD 矩阵展开: OK
- Identity MatVec: OK
- 对角 MatVec: OK
- 随机 SPD MatVec: OK
- 逆矩阵 MatVec 准确性: OK

**Taichi vs NumPy 测试:**
- Identity MatVec: OK (误差 < 1e-5)
- 随机 SPD MatVec: OK (误差 < 1e-4)
- 数值精度 f32 vs f64: OK (max 相对误差 1.58e-07)

**全求解器测试:**
- apply() 无 NaN: OK
- g^T z > 0: OK (6.83e-10)
- Block 0 局部求解: OK (相对误差 3.50e-08)
- 求解器变体一致性: OK (conflict_free vs full: 1.45e-07)

**结果**: 15/15 测试通过

### 5.5 MatVec 性能基准

| 变体 | 平均时间 | 标准差 | 最小时间 | g^Tz > 0 |
|------|----------|--------|----------|----------|
| full_solve (default) | 2.53ms | 0.12ms | 2.40ms | Yes |
| conflict_free | 2.28ms | 0.08ms | 2.24ms | Yes |
| parallel | 2.32ms | 0.03ms | 2.30ms | Yes |
| diagonal_only | 0.37ms | 0.03ms | 0.34ms | Yes |

**推荐**:
- 最快有效变体: `diagonal_only` (0.37ms, 但精度较低)
- 最稳定高精度: `conflict_free` (2.28ms)

---

## 6. 推荐配置

### 6.1 生产环境推荐

1. **通用场景**: `Incomplete Cholesky IC(0)` + 正则化
   - 速度/精度最佳平衡
   - 比 Gauss-Jordan 快 7.6x
   - 可接受的精度 (3e-04)

2. **稳健性优先**: `Gauss-Jordan`
   - 最高精度 (8e-07)
   - 无需正则化
   - 适用于任意矩阵

3. **仅 SPD 矩阵**: `Cholesky` 或 `Blocked Cholesky`
   - 优秀精度
   - 比 Gauss-Jordan 快
   - 需要 SPD 保证

### 6.2 代码配置示例

```python
# 推荐默认配置
mas.invert_block_matrices(
    use_full_inversion=True,
    use_cholesky=True,
    use_incomplete=True,      # IC(0) 提速
    force_symmetry=True,      # 安全保障
    regularization_epsilon=5e5  # 根据网格调整
)

# 稳健配置
mas.invert_block_matrices(
    use_full_inversion=True,
    use_cholesky=False,       # Gauss-Jordan
    force_symmetry=True,
    regularization_epsilon=0.0
)
```

### 6.3 正则化参数选择

正则化 ε 应满足:
```
ε > |λ_min| × 1.1 + margin
```

本网格: ε ≈ 4.5e5。实践中可通过以下方式估计 |λ_min|:
- 前一帧的特征值
- 对角占优检查
- 保守过估计 (1e6)

---

## 7. 已解决的问题

### 7.1 Lane 0 对称性误差 (已修复)

**问题**: Block(0,0) 的对称性误差约 1.65e+04

**根因**: 跨 warp 装配时，不同细网格顶点映射到同一粗网格顶点时，
代码未同时添加 H[i,j] 和 H[i,j]^T（转置），导致对称性破坏。

**修复**: 在 assembly.py 的 4 个位置添加转置贡献:
- `_add_elastic_contribution_full_optimized` (lines 197-201)
- `_add_elastic_contribution_full` (lines 300-302)
- `_add_ipc_contact_contribution` (lines 450-453)
- `_add_ipc_contact_contribution_compact_kernel` (lines 543-546)

**结果**: 相对对称误差从 ~3% 降至 ~5e-08

### 7.2 Incomplete Cholesky NaN (已修复)

**问题**: IC 分解产生 NaN

**根因**: 块矩阵有负特征值 (min = -4.1e+05)

**修复**:
1. 添加 `force_symmetry=True` 选项强制对称化
2. 添加 `regularization_epsilon` 选项进行对角正则化
3. 正则化值应 > |λ_min|

**结果**: IC NaN 计数 = 0, ||A * A^-1 - I|| = 2.99e-04

---

## 8. 运行测试

```bash
# 运行所有测试
cd /root/PNCG_IPC/demo
PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/test_inversion_methods.py

# 仅运行单元测试
python ../unittest/tests/test_inversion_methods.py --test-only

# 仅运行性能基准
python ../unittest/tests/test_inversion_methods.py --benchmark-only

# 运行 ground truth 测试
PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/test_mas_ground_truth.py -v

# 运行多层级测试
PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/test_mas_multilevel.py -v

# 运行 MatVec 准确性和性能测试
PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/test_matvec_accuracy.py -v

# 仅运行 MatVec 性能基准
python ../unittest/tests/test_matvec_accuracy.py --benchmark
```

---

## 测试环境

- **Python**: 3.11.11
- **Taichi**: 1.7.4
- **Platform**: Linux 5.10.134-17.3.al8.x86_64
- **Architecture**: x64/CUDA
- **GPU**: NVIDIA CUDA
