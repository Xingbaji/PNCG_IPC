# MAS 预条件器与 MeshTaichi 深度集成设计

本文档分析 MeshTaichi 和 MAS 预条件器的核心优化原理，并提出一种**融合两者优点**的新架构。

---

## 1. 核心优化原理对比

### 1.1 MeshTaichi 的核心优化

| 优化策略 | 目标 | 实现方式 |
|---------|------|----------|
| **Patch 分区** | 最小化 ribbon 元素 | 贪心算法选择最连通的邻居 |
| **属性局部化** | 减少全局内存访问 | 属性加载到共享内存 |
| **Prologue-Compute-Epilogue** | 隐藏内存延迟 | 批量加载→计算→批量写回 |
| **内存重排序** | 合并内存访问 | Patch 内元素连续存储 |

**核心洞察**: MeshTaichi 的 Patch 大小 (~2048) 是为了**完全装入共享内存**。

### 1.2 MAS 预条件器的核心优化

| 优化策略 | 目标 | 实现方式 |
|---------|------|----------|
| **BANKSIZE=16 分组** | Warp 内协作 | 48×48 块矩阵求逆 |
| **多层级层次** | 捕获全局耦合 | 限制-求解-延拓循环 |
| **METIS 分区** | 块内拓扑连通 | 图分区使连通顶点相邻 |
| **Woodbury 更新** | 避免重建预条件器 | 低秩更新处理碰撞变化 |

**核心洞察**: MAS 的 BANKSIZE=16 是为了**单个 warp (32线程) 处理两个块**，利用 warp-level 原语加速。

### 1.3 核心冲突

| 特性 | MeshTaichi | MAS |
|------|-----------|-----|
| 分区大小 | ~2048 元素 | 16 顶点 |
| 分区目标 | 最小化边界 | 最大化块内连通性 |
| 层级数 | 单层 | 多层 (最多6层) |
| 计算模式 | 元素并行 | 块并行 |

---

## 2. 新架构：层次化统一分区 (HUP)

### 2.1 核心思想

**将 MeshTaichi 的 Patch 作为 MAS 的 subdomain，在 Patch 内部进行 METIS 细分区。**

```
┌────────────────────────────────────────────────────────────────┐
│                    层次化统一分区 (HUP)                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Level 0: MeshTaichi Patches (~2048 verts each)               │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │    Patch 0       │  │    Patch 1       │  ...              │
│  │  ┌────┬────┬───┐ │  │  ┌────┬────┬───┐ │                   │
│  │  │B0  │B1  │...│ │  │  │B128│B129│...│ │                   │
│  │  ├────┼────┼───┤ │  │  ├────┼────┼───┤ │                   │
│  │  │B16 │B17 │...│ │  │  │B144│B145│...│ │                   │
│  │  └────┴────┴───┘ │  │  └────┴────┴───┘ │                   │
│  │  (128 MAS blocks)│  │  (128 MAS blocks)│                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                │
│  Level 1: Patch Representatives (~128 per patch → 1)          │
│  ┌──────────────────────────────────────────┐                 │
│  │  [P0_rep, P1_rep, P2_rep, ...]           │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                │
│  Level 2+: Standard MAS coarsening                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 数据布局

```python
# 顶点按 Patch → 块 → 顶点 三级组织
# vertex_id = patch_id * PATCH_SIZE + block_local * BANKSIZE + lane_id

PATCH_SIZE = 2048  # MeshTaichi 兼容
BLOCKS_PER_PATCH = PATCH_SIZE // BANKSIZE  # = 128

# 内存布局 (连续存储)
# [Patch 0: Block 0-127][Patch 1: Block 128-255][...]
#   ↓
# [B0: v0-15][B1: v16-31]...[B127: v2032-2047][B128: v2048-2063]...
```

### 2.3 分区算法

```python
def hierarchical_unified_partition(mesh, n_verts, cells):
    """
    Step 1: MeshTaichi 风格的 Patch 分区 (最小化边界)
    """
    n_patches = (n_verts + PATCH_SIZE - 1) // PATCH_SIZE
    patch_assignment = greedy_patch_partition(mesh, patch_size=PATCH_SIZE)

    """
    Step 2: 每个 Patch 内部用 METIS 细分为 BANKSIZE 块
    """
    block_assignment = np.zeros(n_verts, dtype=np.int32)

    for patch_id in range(n_patches):
        # 获取该 Patch 内的顶点
        patch_verts = np.where(patch_assignment == patch_id)[0]

        # 提取子图
        subgraph = extract_subgraph(cells, patch_verts)

        # METIS 分区为 BLOCKS_PER_PATCH 个块
        n_blocks_in_patch = len(patch_verts) // BANKSIZE
        local_partition = metis_partition(subgraph, n_blocks_in_patch)

        # 映射回全局块 ID
        for local_id, global_vert in enumerate(patch_verts):
            local_block = local_partition[local_id]
            global_block = patch_id * BLOCKS_PER_PATCH + local_block
            block_assignment[global_vert] = global_block

    """
    Step 3: 根据块分配重排顶点
    """
    sort_index = np.argsort(block_assignment)  # sort_index[new] = old

    return {
        'patch_assignment': patch_assignment,
        'block_assignment': block_assignment,
        'sort_index': sort_index,
        'n_patches': n_patches
    }
```

---

## 3. 执行流程：Patch-MAS 协同

### 3.1 预条件器构建

```python
class HierarchicalMASPreconditioner:
    def __init__(self, mesh, n_verts, n_cells, cells_np):
        # 1. 执行层次化统一分区
        self.partition_result = hierarchical_unified_partition(
            mesh, n_verts, cells_np
        )

        # 2. 按 Patch 组织的数据结构
        self.n_patches = self.partition_result['n_patches']

        # 每个 Patch 的块矩阵 (Patch 内连续)
        # Shape: (n_patches, BLOCKS_PER_PATCH, SYM_BLOCK_COUNT, 3, 3)
        self.patch_block_matrices = ti.Matrix.field(
            3, 3, dtype=ti.f64,
            shape=(self.n_patches, BLOCKS_PER_PATCH, SYM_BLOCK_COUNT)
        )

        # 每个 Patch 的逆矩阵
        self.patch_inv_blocks = ti.Matrix.field(
            3, 3, dtype=ti.f32,
            shape=(self.n_patches, BLOCKS_PER_PATCH, SYM_BLOCK_COUNT)
        )

        # 重排序后的顶点数据 (Patch 对齐)
        self.reordered_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.reordered_grad = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.reordered_z = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
```

### 3.2 MeshTaichi Patch 执行模型集成

```python
@ti.kernel
def build_and_solve_patch_parallel(self):
    """
    利用 MeshTaichi 的 Prologue-Compute-Epilogue 模型
    每个 GPU Block 处理一个 Patch
    """
    # 共享内存分配 (MeshTaichi 风格)
    # 每个 Patch: 2048 顶点 × 3 × 8 bytes = 49KB (适合共享内存)

    ti.loop_config(block_dim=256)  # 256 threads per GPU block
    for patch_id in range(self.n_patches):
        # ============ Prologue ============
        # 加载 Patch 内所有顶点数据到共享内存
        # (由 Taichi 自动处理 ti.block_local)

        # ============ Compute ============
        # Phase 1: 构建块矩阵 (128 blocks in parallel)
        for local_block in range(BLOCKS_PER_PATCH):
            self._build_block_matrix_in_patch(patch_id, local_block)

        # Phase 2: 块矩阵求逆 (128 blocks in parallel)
        for local_block in range(BLOCKS_PER_PATCH):
            self._invert_block_in_patch(patch_id, local_block)

        # Phase 3: 求解 z = B^{-1} r (128 blocks in parallel)
        for local_block in range(BLOCKS_PER_PATCH):
            self._solve_block_in_patch(patch_id, local_block)

        # ============ Epilogue ============
        # 写回结果 (自动处理)
```

### 3.3 Patch 边界处理

```python
@ti.kernel
def handle_patch_boundaries(self):
    """
    处理 Patch 边界上的 ribbon 元素
    类似 MeshTaichi 的 ribbon 处理策略
    """
    for patch_id in range(self.n_patches):
        # 1. 识别边界块 (与其他 Patch 有连接的块)
        for local_block in range(BLOCKS_PER_PATCH):
            if self.is_boundary_block[patch_id, local_block]:
                # 2. 对边界块执行额外的修正
                # 考虑跨 Patch 的 Hessian 贡献
                self._add_ribbon_contribution(patch_id, local_block)
```

---

## 4. 性能优势分析

### 4.1 内存访问优化

| 访问类型 | 传统 MAS | HUP (新方案) |
|---------|---------|-------------|
| 顶点数据加载 | 随机访问 | Patch 内连续 (合并访问) |
| 块矩阵访问 | 分散存储 | Patch 内连续 |
| 缓存命中率 | 低 | 高 (Patch 适合 L2 缓存) |

### 4.2 计算并行度

| 层级 | 传统 MAS | HUP (新方案) |
|-----|---------|-------------|
| GPU Block | 1 warp = 2 MAS blocks | 1 GPU Block = 1 Patch = 128 MAS blocks |
| Warp 内协作 | ✅ | ✅ |
| 共享内存利用 | 有限 | 充分 (49KB/Patch) |

### 4.3 分区质量

| 指标 | 纯 METIS | 纯 MeshTaichi | HUP |
|-----|---------|--------------|-----|
| 块内连通性 | 最优 | 无保证 | 优 (Patch 内 METIS) |
| Patch 边界开销 | N/A | 最优 | 最优 (继承 MeshTaichi) |
| 多层级支持 | ✅ | ❌ | ✅ |

---

## 5. 实现路径

### Phase 1: Patch 分区集成 (低风险)

```python
# 在现有 MASPreconditioner 中添加 Patch 概念

class MASPreconditioner:
    def __init__(self, ...):
        # 新增: Patch 分区
        self.use_patch_organization = True
        self.patch_size = 2048

        if self.use_patch_organization:
            self._init_patch_structure()

    def _init_patch_structure(self):
        """初始化 Patch 结构"""
        self.n_patches = (self.n_verts + self.patch_size - 1) // self.patch_size

        # 贪心 Patch 分区 (可选择使用 MeshTaichi 的分区)
        self.patch_assignment = self._greedy_patch_partition()

        # Patch 内 METIS 细分区
        self._metis_partition_within_patches()
```

### Phase 2: Patch-Local 块构建 (中风险)

```python
@ti.kernel
def _build_block_matrices_patch_local(self):
    """Patch-local 块矩阵构建"""
    for patch_id in range(self.n_patches):
        # 每个 Patch 独立构建
        patch_offset = patch_id * self.patch_size

        for local_vert in range(self.patch_size):
            global_vert = self.sort_index[patch_offset + local_vert]
            local_block = local_vert // BANKSIZE
            lane_id = local_vert % BANKSIZE

            # 使用 Patch-local 块索引
            self._add_vertex_contribution_to_patch_block(
                patch_id, local_block, lane_id, global_vert
            )
```

### Phase 3: 完整 Prologue-Compute-Epilogue (高收益)

```python
# 利用 Taichi 的 ti.block_local 实现共享内存缓存

@ti.kernel
def apply_patch_parallel(self):
    """Patch 并行的预条件器应用"""
    for patch_id in range(self.n_patches):
        # Taichi 会自动处理共享内存加载
        patch_offset = patch_id * self.patch_size

        # 限制 (restriction)
        for local_vert in range(self.patch_size):
            idx = patch_offset + local_vert
            original_idx = self.sort_index[idx]
            self.reordered_grad[idx] = self.mesh.verts.grad[original_idx]

        # 求解 (solve) - 128 blocks in parallel
        for local_block in range(BLOCKS_PER_PATCH):
            self._solve_block(patch_id, local_block)

        # 延拓 (prolongation)
        for local_vert in range(self.patch_size):
            idx = patch_offset + local_vert
            original_idx = self.sort_index[idx]
            self.mesh.verts.z[original_idx] = self.reordered_z[idx]
```

---

## 6. 与现有代码的兼容性

### 6.1 接口保持不变

```python
# 现有接口
preconditioner = MASPreconditioner(n_verts, n_cells, mesh)
preconditioner.build_hierarchy()
preconditioner.assemble_matrices(solver)
preconditioner.apply()

# 新接口 (完全兼容)
preconditioner = MASPreconditioner(
    n_verts, n_cells, mesh,
    use_patch_organization=True,  # 新参数
    patch_size=2048  # 新参数
)
preconditioner.build_hierarchy()
preconditioner.assemble_matrices(solver)
preconditioner.apply()  # 内部自动选择 Patch 并行路径
```

### 6.2 渐进式迁移

1. **第一步**: 保持现有代码，添加 Patch 分区作为可选功能
2. **第二步**: 测试 Patch 分区对分区质量的影响
3. **第三步**: 实现 Patch-local 块构建和求解
4. **第四步**: 性能调优和完整集成

---

## 7. 预期收益

| 指标 | 预期改进 | 原因 |
|-----|---------|------|
| 内存带宽利用率 | 2-3x | Patch 内合并访问 |
| 缓存命中率 | 3-5x | 共享内存缓存 |
| 块矩阵质量 | 10-30% | Patch 内 METIS 分区 |
| 总体求解时间 | 2-4x | 以上因素叠加 |

---

## 8. 总结

**层次化统一分区 (HUP)** 的核心创新是：

1. **保留 MeshTaichi 的 Patch 优化**：大粒度分区最小化边界开销
2. **引入 METIS 细分区**：在 Patch 内部保证块的拓扑连通性
3. **统一执行模型**：利用 Prologue-Compute-Epilogue 隐藏内存延迟
4. **多层级兼容**：Patch 自然成为 MAS 的中间层级

这种设计避免了"选择 MeshTaichi 还是 MAS"的两难，而是将两者的优点融合到统一的架构中。
