# MAS 预条件器单元测试报告

**生成时间**: 2026-01-19 20:18

## 测试概要

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

---

## 详细测试结果

### Core Test Suites

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_mas_ground_truth.py` | ✅ PASS | 22 tests, OK |
| `test_mas_multilevel.py` | ✅ PASS | 25 tests, OK (6 skipped) |

**说明**: 核心测试套件全部通过，包括：
- Ground Truth 测试：验证装配、求逆、SPMv 操作
- 多层级测试：验证层级构建、限制/延拓操作

---

### Functional Validation

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_mas_simple.py` | ✅ PASS | 4 frames, 30 iters/frame, 6.7 FPS |
| `test_mas_freefall.py` | ❌ FAIL | 位置误差超阈值 (5.99e-02 > 0.01) |
| `test_mas_matrix_diagnostic.py` | ⚠️ ERROR | 4/8 tests passed, 4/8 failed |

**说明**:
- `test_mas_simple.py`: 无碰撞场景下 MAS 预条件器正常工作
- `test_mas_freefall.py`: 自由落体验证失败，收敛速度不足
- `test_mas_matrix_diagnostic.py`: 诊断测试发现块矩阵 SPD 问题

**诊断结果**:
- ✅ Inertia Contribution
- ❌ Elastic Contribution (ARAP_filter) - 非 SPD
- ❌ Combined: Inertia + Elastic - 非 SPD
- ❌ Symmetric Expansion - 展开逻辑问题
- ❌ Incomplete Cholesky Inversion - 需要 SPD

---

### Assembly Tests

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_assembly_logic.py` | ✅ PASS | 装配逻辑正确 |
| `test_assembly_detail.py` | ⚠️ ERROR | ImportError: compute_dFdx_taichi |
| `test_assembly_precise.py` | ✅ PASS | 原子操作正确，非对称来自输入 |

**发现**: Lane 0 存在对称性误差 (~1.65e+04)，其他 Lane 误差 < 1e-05

---

### Symmetry Tests

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_He_symmetry.py` | ⚠️ ERROR | Taichi assertion failure |
| `test_He_symmetry_simple.py` | ✅ PASS | H_e 对称性证明正确 |

**说明**: NumPy 验证证明元素 Hessian 理论上对称

---

### Hierarchy Tests

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_hierarchy_mapping.py` | ✅ PASS | 层级映射正确 |
| `test_crosswarp_issue.py` | ✅ PASS | 跨 warp 问题定位 |
| `test_level0_only.py` | ⚠️ ERROR | Taichi assertion failure |

---

### Specific Issue Tests

| 测试文件 | 状态 | 结果 |
|----------|------|------|
| `test_debug_simple.py` | ✅ PASS | Lane 0 对称误差 1.65e+04 |
| `test_diagonal_contrib.py` | ⚠️ ERROR | Taichi assertion failure |
| `test_diagonal_simple.py` | ⚠️ ERROR | Taichi assertion failure |
| `test_nonopt_kernel.py` | ✅ PASS | 优化/非优化内核误差相同 |
| `test_upper_triangle_bug.py` | ✅ PASS | 上三角处理分析完成 |

---

## 调试脚本状态

| 脚本 | 状态 | 说明 |
|------|------|------|
| `debug_sym_expand.py` | ✅ PASS | 对称展开逻辑验证 |
| `debug_assembly_logic.py` | ⚠️ ERROR | AttributeError: 'MeshElementField' |
| `debug_block0_detailed.py` | - | 未测试 |
| `debug_mas_gTz.py` | - | 未测试 |
| `debug_mas_nan.py` | - | 未测试 |

---

## 已知问题

### 1. Lane 0 对称性误差 (Critical)

**现象**: Block(0,0) 的对称性误差约 1.65e+04，而其他 Lane 误差 < 1e-05

**原因分析**:
- Lane 0 对应 vertex 0，是许多元素的首顶点
- 跨 warp 贡献可能导致不对称累加
- 与 METIS 重排序后的顶点映射有关

**影响**: 导致预条件器质量下降，收敛速度变慢

### 2. Taichi Assertion Failure

**现象**: 多个测试触发 `codegen_llvm.cpp:operator()@1087` 断言失败

**原因**: Taichi 编译器对某些 mesh kernel 模式不支持

**影响**: 部分测试无法完成

### 3. 块矩阵非 SPD

**现象**: 弹性贡献组装后的块矩阵非正定

**原因**: 跨块边界的元素只贡献部分 Hessian，缺失的交叉耦合导致非正定

**解决方案**: 使用 Gauss-Jordan 求逆代替 Incomplete Cholesky

---

## 建议

1. **Lane 0 问题**: 需要进一步调查跨 warp 累加逻辑
2. **Taichi 兼容性**: 考虑简化 mesh kernel 以避免编译器限制
3. **求逆方法**: 优先使用 `gauss_jordan` 或 `oneway_gj` 方法
4. **测试覆盖**: 修复导入错误以提高测试覆盖率

---

---

## 求逆方法性能对比

### 单步测试结果

| 方法 | |z| | g^T*z | NaN | 总时间 |
|------|-----|-------|-----|--------|
| Gauss-Jordan | 1.50e-05 | +3.36e-08 | No | 3275ms |
| One-way GJ | 9.04e-01 | +1.46e-04 | No | 158ms |
| Cholesky | nan | nan | Yes | 130ms |
| Incomplete Cholesky | nan | nan | Yes | 126ms |

### 多帧性能基准 (5 frames, grad_tol=1e-6)

| 方法 | 状态 | 迭代次数 | 帧时间 | 求逆时间 | FPS |
|------|------|----------|--------|----------|-----|
| Gauss-Jordan | OK | 30.0 | 1153ms | 99ms | 0.9 |
| One-way GJ | OK | 30.0 | 1006ms | 80ms | 1.0 |
| Cholesky | NaN | 1.0 | 834ms | 71ms | - |
| Incomplete Cholesky | NaN | 1.0 | 830ms | 65ms | - |

### 结论

1. **推荐方法**: `One-way GJ` (One-way Gauss-Jordan)
   - 比完整 Gauss-Jordan 快约 20%
   - 不产生 NaN
   - g^T*z > 0 (有效下降方向)

2. **不推荐**: Cholesky / Incomplete Cholesky
   - 块矩阵非 SPD 导致 NaN
   - 无法正常收敛

3. **性能瓶颈**: 装配时间 (~610ms) 占帧时间的 60%

---

## 测试环境

- **Python**: 3.11.11
- **Taichi**: 1.7.4
- **Platform**: Linux 5.10.134-17.3.al8.x86_64
- **Architecture**: x64/CUDA
