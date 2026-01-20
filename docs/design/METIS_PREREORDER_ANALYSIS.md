# METIS Pre-Reorder 性能分析

## 问题描述

Pre-reordered 模式（在创建 mesh 前进行 METIS 重排序）比 Runtime mapping 模式更慢：

| 模式 | Assemble (ms) | Apply (ms) | Total (ms) |
|------|--------------|------------|------------|
| No METIS | 0.566 | 0.670 | 1.236 |
| Runtime Mapping | 0.942 | **0.341** | 1.283 |
| Pre-Reordered | 0.970 | **0.550** | 1.519 |

**关键发现**: Apply 性能是主要差异（0.341ms vs 0.550ms）

---

## 根本原因分析

### 1. Banded vs Full Solve

#### Runtime Mapping 使用 `_schwarz_local_solve_banded_metis`

```python
NODE_BANDWIDTH = 2  # 只迭代相邻的 2 个节点

for lane_j in range(lane_j_start, lane_j_end):  # 最多 5 次迭代
    # lane_j_start = max(0, lane_i - 2)
    # lane_j_end = min(16, lane_i + 3)
```

每个节点只计算与其相邻 ±2 节点的交互 → **O(5)** 复杂度

#### Pre-Reordered 使用 `_schwarz_local_solve_full`

```python
for lane_j in range(BANKSIZE):  # 16 次迭代
    # 迭代 block 内所有 16 个节点
```

每个节点计算与 block 内所有 16 个节点的交互 → **O(16)** 复杂度

**结论**: Full solve 比 banded solve 慢 ~3x（16/5 ≈ 3.2）

### 2. 为什么 Runtime Mapping 可以用 Banded Solve？

METIS 分区的核心优势是**拓扑局部性**：
- 同一 partition 内的顶点在网格拓扑上相邻
- 相邻顶点的 Hessian 耦合更强
- 大部分有意义的耦合集中在对角线附近

因此 banded approximation 对 METIS 分区是有效的近似：
- 保留主要耦合（对角线 ±2）
- 忽略弱耦合（远离对角线）

### 3. 为什么 No-METIS 和 Pre-Reordered 必须用 Full Solve？

#### No-METIS 模式

顺序分块（v0-v15 = block0, v16-v31 = block1, ...）：
- 相邻顶点可能分布在网格的不同位置
- 无拓扑局部性保证
- Banded approximation 不准确

#### Pre-Reordered 模式（当前实现）

虽然 mesh 数据已按 METIS 排序，但 **Apply 使用了错误的 kernel**：

```python
if self.metis_reordered:
    # 使用 _schwarz_local_solve_full (O(16))
    self._schwarz_local_solve_full()  # ← 错误！应该用 banded
elif self.use_metis:
    # 使用 _schwarz_local_solve_banded_metis (O(5))
    self._schwarz_local_solve_banded_metis()
```

Pre-reordered 模式的 mesh 数据已经是 METIS 顺序，**应该可以使用 banded solve**！

---

## MeshTaichi Patch vs METIS Partition 分析

根据 [MESHTAICHI_ARCHITECTURE.md](./MESHTAICHI_ARCHITECTURE.md)，MeshTaichi 有独立的 Patch 系统：

| 特性 | MeshTaichi Patch | METIS Partition |
|------|-----------------|-----------------|
| 大小 | ~2048 元素 | 16 顶点 (BANKSIZE) |
| 目的 | GPU 共享内存优化 | 块矩阵质量 |
| 排序影响 | 影响属性缓存局部性 | 影响 MAS block 分配 |

### METIS 重排序对 MeshTaichi 的影响

当对 mesh 数据进行 METIS 重排序后创建 MeshTaichi mesh：

1. **Patch 分区会重新计算**
   - MeshTaichi 基于新的拓扑结构重新分区
   - METIS 重排序改变了顶点 ID，但不改变拓扑连接关系
   - Patch 质量（owned/ribbon ratio）应该不变

2. **内存访问模式**
   - 使用 `for vert in self.mesh.verts` 时，按 MeshTaichi Patch 顺序迭代
   - MeshTaichi 会将属性加载到共享内存
   - METIS 重排序不影响 MeshTaichi 的内部优化

3. **MAS Block 与 MeshTaichi Patch 独立**
   - MAS block = 每 16 个顺序顶点
   - MeshTaichi patch = 每 ~256 个拓扑相邻元素
   - 两者优化目标不同，互不干扰

### 关键洞察

METIS Pre-reordered 模式的问题**不是** MeshTaichi Patch 冲突，而是**使用了错误的 solve kernel**。

---

## 解决方案

### 方案 1: 为 Pre-Reordered 模式添加 Banded Solve (推荐)

```python
@ti.kernel
def _schwarz_local_solve_banded_reordered(self):
    """Banded solve for METIS pre-reordered mesh.

    Since mesh is pre-reordered, vertex IDs directly correspond to partitions:
    - block_id = vertex_id // BANKSIZE
    - lane_id = vertex_id % BANKSIZE

    No mapping lookup needed, but can use banded approximation!
    """
    NODE_BANDWIDTH = 2
    n_blocks = (self.n_verts + BANKSIZE - 1) // BANKSIZE

    for block_id, lane_i in ti.ndrange(n_blocks, BANKSIZE):
        idx_i = block_id * BANKSIZE + lane_i
        if idx_i < self.n_verts:
            z0, z1, z2 = ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)

            # Banded iteration (same as _schwarz_local_solve_banded_metis)
            lane_j_start = ti.max(0, lane_i - NODE_BANDWIDTH)
            lane_j_end = ti.min(BANKSIZE, lane_i + NODE_BANDWIDTH + 1)

            for lane_j in range(lane_j_start, lane_j_end):
                idx_j = block_id * BANKSIZE + lane_j
                if idx_j < self.n_verts:
                    r_j = self.multi_level_r[idx_j]
                    # ... (matrix-vector multiplication)

            self.multi_level_z[idx_i] = ti.Vector([z0, z1, z2], dt=ti.f32)
```

### 方案 2: 直接在 Apply 中选择正确的 Kernel

修改 `apply()` 方法：

```python
def apply(self):
    self._clear_multi_level_buffers()

    if self.metis_reordered:
        # Pre-reordered mode: use banded solve (can use METIS locality)
        self._build_multi_level_r()
        self._schwarz_local_solve_banded_reordered()  # ← 新 kernel
    elif self.use_metis:
        self._build_multi_level_r_metis()
        self._schwarz_local_solve_banded_metis()
    else:
        self._build_multi_level_r()
        self._schwarz_local_solve_full()  # No-METIS 必须用 full

    self._collect_final_z(self.level_num)
```

### 预期收益

| 操作 | 当前 Pre-Reordered | 优化后 Pre-Reordered | Runtime Mapping |
|------|-------------------|---------------------|-----------------|
| Apply (Level 0) | O(16) | O(5) | O(5) |
| 映射查找 | 无 | 无 | 有 |
| 预期 Apply 时间 | 0.550ms | ~0.25ms | 0.341ms |

**优化后 Pre-Reordered 应该比 Runtime Mapping 更快**：
- 同样使用 banded solve (O(5))
- 无需映射查找 (`partId_map_real`)
- 内存访问模式相同

---

## 实现计划

1. [x] 添加 `_schwarz_local_solve_banded_reordered()` kernel
2. [x] 修改 `apply()` 使用正确的 kernel
3. [x] 运行 benchmark 验证性能

---

## 修复结果 (2026-01-20)

### 修复后 Benchmark (cube_20)

| Mode | Assemble (ms) | Apply (ms) | Total (ms) |
|------|--------------|------------|------------|
| No METIS | 0.502 | 0.563 | 1.065 |
| Runtime Mapping | 0.947 | 0.326 | 1.273 |
| **Pre-Reordered** | 0.966 | **0.287** | **1.253** |

### 关键改进

| 指标 | 修复前 | 修复后 | 改善 |
|------|-------|-------|------|
| Apply 时间 | 0.550ms | 0.287ms | **1.92x** |
| Total 时间 | 1.519ms | 1.253ms | **1.21x** |
| vs Runtime Mapping | 0.84x | **1.02x** | 从更慢变更快 |

修复验证了预期：
- 预期 Apply: ~0.25ms，实际: 0.287ms ✓
- Pre-reordered 比 Runtime Mapping 快 ✓

---

## 总结

Pre-Reordered 模式更慢的原因**不是** MeshTaichi 架构冲突，而是：

1. **使用了错误的 solve kernel** (`full` instead of `banded`)
2. Full solve O(16) vs Banded solve O(5) 导致 ~3x 性能差距

**修复完成**: 添加了 `_schwarz_local_solve_banded_reordered()` kernel。

Pre-reordered 现在是**最快的 Apply 模式**：
- 无需 mapping 查找（vs Runtime Mapping 的 `partId_map_real`）
- 使用 banded solve（vs No-METIS 的 full solve）
- Apply: 0.287ms < 0.326ms (Runtime) < 0.563ms (No-METIS)
