# MAS 矩阵对称性问题分析报告

**日期**: 2026-01-19
**状态**: 已修复 (2026-01-19)
**相关文件**: `algorithm/mas_preconditioner_pkg/assembly.py`

---

## 修复摘要

### 根因
在跨 warp 装配逻辑中，当不同细网格顶点映射到同一粗网格顶点时，代码未同时添加 H[i,j] 和 H[i,j]^T（转置），导致对称性破坏。

### 修复位置
- `assembly.py:197-201` (`_add_elastic_contribution_full_optimized`)
- `assembly.py:300-302` (`_add_elastic_contribution_full`)
- `assembly.py:450-453` (`_add_ipc_contact_contribution`)
- `assembly.py:543-546` (`_add_ipc_contact_contribution_compact_kernel`)

### 修复后结果
- **相对对称误差**: ~3% → ~5e-08 (float32 精度)
- **绝对对称误差**: 1.65e+04 → 3.08e-02
- **IC with regularization**: NaN count = 0, ||A * A^-1 - I|| = 2.99e-04

详见 [TEST_REPORT.md](../unittest/TEST_REPORT.md)

---

## 1. 问题描述

### 1.1 现象
MAS 预处理器的 Block(0,0) 对角块出现严重的对称性误差：
- **Lane 0 对称性误差**: 1.65e+04
- **其他 Lane 对称性误差**: ~1e-06（正常浮点精度）

### 1.2 误差模式
```
Block(0,0):
[[-146392.36,   -6965.78,    2635.62]
 [   3204.43, -156053.44,   -6083.76]
 [    703.84,    -713.07, -205453.00]]

Block(0,0) - Block(0,0)^T (反对称差):
[[     0.00,  -10170.20,    1931.77]
 [ 10170.20,       0.00,   -5370.69]
 [ -1931.77,    5370.69,       0.00]]
```

差矩阵是**反对称的**，表明有非对称的贡献被添加到对角块。

### 1.3 影响
- Incomplete Cholesky 分解失败（需要 SPD 矩阵）
- 导致预处理器产生 NaN

---

## 2. 已确认的事实

### 2.1 数学正确性验证
| 验证项 | 结果 | 说明 |
|--------|------|------|
| H_e 对称性 | ✓ 通过 | `H_e = dFdx^T @ d2PsidF2 @ dFdx` 数学上保证对称 |
| NumPy 模拟装配 | ✓ 通过 | 纯 NumPy 实现的装配逻辑产生对称结果 |
| ARAP d2PsidF2 | ✓ 通过 | `2*mu*I_9x9` 是对称正定的 |

### 2.2 两个版本一致性
```
Optimized kernel:   Lane 0 sym_error = 1.649299e+04
Non-optimized kernel: Lane 0 sym_error = 1.649297e+04
```
两个版本误差几乎相同，说明问题在**共同的逻辑**中，而非优化实现的差异。

### 2.3 只有 Lane 0 受影响
```
Lane 0:  sym_error = 1.649e+04  ← 异常
Lane 1:  sym_error = ~1e-06     ← 正常
Lane 2:  sym_error = ~1e-06     ← 正常
...
Lane 15: sym_error = ~1e-06     ← 正常
```

---

## 3. 包含顶点0的元素分析

测试 demo: `eight_E_stiffness_test` (8368 vertices, 27376 cells)

### 3.1 元素详情
| Element | Cell ID | Vertices | V0 Local Index | Warps | 全在 Warp 0? |
|---------|---------|----------|----------------|-------|--------------|
| 0 | 500 | [299, 0, 300, 3] | 1 | [18, 0, 18, 0] | 否 |
| 1 | 501 | [0, 299, 4, 3] | 0 | [0, 18, 0, 0] | 否 |
| 2 | 0 | [0, 1, 2, 3] | 0 | [0, 0, 0, 0] | **是** |
| 3 | 1 | [0, 4, 1, 3] | 0 | [0, 0, 0, 0] | **是** |
| 4 | 62 | [55, 0, 2, 3] | 1 | [3, 0, 0, 0] | 否 |
| 5 | 642 | [300, 0, 55, 3] | 1 | [18, 0, 3, 0] | 否 |

**观察**:
- 只有 Cell 0 和 Cell 1 的所有顶点都在 Warp 0
- 其他元素会触发 cross-warp 传播到粗网格

### 3.2 层级映射
```
going_next mapping:
v0  -> coarse v8368 (warp 523, lane 0)
v1  -> coarse v8368 (warp 523, lane 0)
...
v15 -> coarse v8368 (warp 523, lane 0)
```
前16个顶点都映射到同一个粗网格顶点。

---

## 4. 当前代码逻辑

### 4.1 装配循环 (assembly.py:137-165)
```python
for i in ti.static(range(4)):
    for j in ti.static(range(i, 4)):  # j >= i: 只处理上三角
        warp_i, warp_j = warp_ids[i], warp_ids[j]
        lane_i, lane_j = lane_ids[i], lane_ids[j]

        if warp_i == warp_j:  # 同 warp: Level 0 装配
            sub_block = H_e[i*3:(i+1)*3, j*3:(j+1)*3]

            if lane_i <= lane_j:
                # 直接存储到 sym_idx(lane_i, lane_j)
                atomic_add(block_matrices[warp_i, sym_idx], sub_block)
            else:
                # 转置存储到 sym_idx(lane_j, lane_i)
                atomic_add(block_matrices[warp_i, sym_idx], sub_block^T)
        else:  # 跨 warp: 传播到粗网格
            # ... coarse level propagation
```

### 4.2 逻辑分析

**对角块 (i == j) 情况**:
- `lane_i == lane_j` (同一顶点)
- 存储 `H_e[i,i]` 到 `Block(lane_i, lane_i)`
- `H_e[i,i]` 是对称的 → 对角块应该对称

**离对角块 (i < j) 情况**:
- 如果 `lane_i < lane_j`: 存 `H_e[i,j]` 到 `Block(lane_i, lane_j)`
- 如果 `lane_i > lane_j`: 存 `H_e[i,j]^T` 到 `Block(lane_j, lane_i)`

**关键问题**: 什么情况下 off-diagonal 的 H_e[i,j] 会被添加到对角块 Block(0,0)？

---

## 5. 假设与分析

### 假设 1: 对角块收到了错误的 off-diagonal 贡献
**可能性**: 当 `i < j` 但 `lane_i == lane_j == 0` 时
**结论**: ❌ 不可能。这要求 `v_ids[i] == v_ids[j]`，但元素的顶点必须两两不同。

### 假设 2: Cross-warp 传播影响了 Level 0
**可能性**: Level 1 的贡献错误地写到了 Level 0
**状态**: 待验证。需要检查 `coarse_warp_i` 的计算是否正确。

### 假设 3: 某个元素的装配逻辑有边界条件错误
**可能性**: 当全局顶点0出现在不同元素的不同 local index 时
**状态**: 待验证。例如：
- Cell 0: v0 在 local index 0
- Cell 62: v0 在 local index 1

### 假设 4: Atomic add 的竞争条件
**可能性**: 多线程写入同一位置时的竞争
**状态**: 可能性较低，因为 `ti.atomic_add` 应该是原子操作。

---

## 6. 下一步调试方向

### 6.1 简化测试（优先级高）
只用一个元素（Cell 0，全在 Warp 0）进行装配，检查结果是否对称：
```python
# 只处理 vertices=[0,1,2,3], warps=[0,0,0,0]
# 预期: Block(0,0) = H_e[0,0] 应该是对称的
```

### 6.2 追踪具体贡献
修改代码记录每次对 Block(0,0) 的 atomic_add：
```python
if lane_i == 0 and lane_j == 0:
    print(f"Adding to Block(0,0): cell={c.id}, i={i}, j={j}, value={sub_block}")
```

### 6.3 手动计算验证
取 Cell 0 的参数，手工计算 H_e 和装配结果，与实际输出对比。

### 6.4 检查 Taichi 编译
当前 Taichi 编译出现 assertion failure，可能影响测试：
```
[E] Assertion failure: offl->task_type == OffloadedStmt::TaskType::range_for || ...
```

---

## 7. 相关测试文件

| 文件 | 用途 |
|------|------|
| `n_E_demos/test_mas_matrix_diagnostic.py` | 综合诊断测试 |
| `n_E_demos/test_nonopt_kernel.py` | 对比优化/非优化版本 |
| `n_E_demos/test_crosswarp_issue.py` | 跨 warp 装配测试 |
| `n_E_demos/test_hierarchy_mapping.py` | 层级映射测试 |
| `n_E_demos/test_He_symmetry_simple.py` | H_e 对称性 NumPy 验证 |
| `n_E_demos/test_assembly_logic.py` | 装配逻辑 NumPy 模拟 |
| `n_E_demos/test_debug_simple.py` | 元素结构分析 |

---

## 8. 临时解决方案

当前已禁用的措施：
- ❌ 正则化 (`add_regularization=False`)

可用的替代方案：
- ✓ Gauss-Jordan 求逆（可处理非 SPD 矩阵）
- ? Modified Cholesky（待实现）
- ? 特征值裁剪（待实现）

---

## 9. 结论

问题的根本原因尚未确定，但已经排除了：
1. H_e 本身不对称
2. NumPy 模拟的装配逻辑错误
3. 优化版本与非优化版本的实现差异

最可能的原因是：
- **装配循环中某个边界条件的处理**
- **或 Taichi 编译器的 bug**

需要进一步的简化测试来定位问题。
