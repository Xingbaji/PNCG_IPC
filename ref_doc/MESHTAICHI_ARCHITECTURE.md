# MeshTaichi 架构与实现原理

本文档总结 MeshTaichi 论文的核心设计，并分析其与 MAS 预条件器集成的可能性。

> 参考: Yu et al., "MeshTaichi: A Compiler for Efficient Mesh-based Operations", ACM SIGGRAPH Asia 2022

---

## 1. 核心问题：网格操作的性能瓶颈

### 1.1 网格 vs 网格（Grid）的本质差异

| 特性 | 结构化网格 (Grid) | 非结构化网格 (Mesh) |
|------|-----------------|-------------------|
| 邻居查询 | 索引偏移计算 | 需查表（关系表） |
| 内存访问 | 连续、可预测 | 离散、不可预测 |
| 缓存命中率 | 高 | 低 |
| GPU 合并访问 | 容易实现 | 难以实现 |

### 1.2 网格操作的四大性能问题

1. **全局内存访问慢**：GPU 全局内存访问比片上内存慢 10~100 倍
2. **非合并访问**：邻居属性访问模式不规则，难以合并
3. **原子操作开销**：Scattering 风格操作需要昂贵的原子操作避免竞争
4. **分支发散**：动态关系（如 VV）导致不同线程访问不同数量的邻居

---

## 2. MeshTaichi 核心设计

### 2.1 编程模型

```python
# 定义网格数据类型
mesh = ti.TetMesh()
mesh.verts.place({'pos': ti.math.vec3, 'vel': ti.math.vec3, 'force': ti.math.vec3})
mesh.cells.place({'B': ti.math.mat3, 'w': ti.f32})

# 实例化网格
bunny = mesh.build('./bunny.mesh')

# 网格计算 (mesh-for 循环)
@ti.kernel
def substep():
    for c in bunny.cells:  # 并行遍历所有单元
        # 通过引用访问邻居属性
        Ds0 = c.verts[0].pos - c.verts[3].pos
        Ds1 = c.verts[1].pos - c.verts[3].pos
        Ds2 = c.verts[2].pos - c.verts[3].pos
        F = ti.Matrix(Ds0, Ds1, Ds2).transpose() @ c.B
        # ... 计算力并散射到顶点
        c.verts[0].force += H[:, 0]  # 自动处理原子操作
```

**关键特性**:
- `mesh-for` 循环隐藏索引系统
- 引用风格的邻居访问（如 `c.verts[i].pos`）
- `+=` 操作自动转换为原子操作（编译器会自动降级不必要的原子操作）

### 2.2 网格元素与关系

**元素类型**:
- 0-d: 顶点 (Vertex, V)
- 1-d: 边 (Edge, E)
- 2-d: 面 (Face, F)
- 3-d: 单元 (Cell, C)

**关系类型**:

| 关系 | 类型 | 说明 |
|------|------|------|
| CV | 静态 | 每个单元固定有 4 个顶点邻居 |
| EV | 静态 | 每条边固定有 2 个顶点邻居 |
| FV | 静态 | 每个面固定有 3 个顶点邻居 |
| VV | 动态 | 每个顶点有不定数量的顶点邻居 |
| VE | 动态 | 每个顶点连接不定数量的边 |
| VF | 动态 | 每个顶点关联不定数量的面 |

**操作风格**:
- **Gathering**: 从邻居读取，写入自身（如 VV 计算顶点力）
- **Scattering**: 从自身读取，写入邻居（如 EV 散射边力到顶点）

---

## 3. 编译器优化策略

### 3.1 网格分区 (Mesh Partitioning)

将网格划分为小的 **Patch**，每个 Patch 可完全装入 GPU 共享内存。

```
┌─────────────────────────────────────────────────────────┐
│                    原始网格                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Patch 1 │  │ Patch 2 │  │ Patch 3 │  ...            │
│  │ (owned) │  │ (owned) │  │ (owned) │                 │
│  │ +ribbon │  │ +ribbon │  │ +ribbon │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
└─────────────────────────────────────────────────────────┘
```

**两类元素**:
- **Owned elements**: 属于该 Patch 的元素
- **Ribbon elements**: 从相邻 Patch 填充的边界元素（类似 ghost cells）

**分区算法** (贪心算法，优于 k-means):

```
while 存在未分区的元素:
    找一个未分区元素 u
    S = {u}
    while |S| < patch_size:
        f(v) = v 与 S 中元素的连接数
        v = argmax(f(v))，其中 v 未被分区且不在 S 中
        S = S ∪ {v}
    将 S 保存为新 Patch
```

**性能指标**: γ = (n_ribbon + n_owned) / n_owned
- γ 越小越好（ribbon 元素越少，重复加载的属性越少）

### 3.2 属性局部化 (Attribute Localization)

**核心决策**: 只缓存属性，不缓存关系

| 缓存策略 | 效果 |
|---------|------|
| 不缓存 | 最慢（硬件调度不足） |
| 只缓存关系 | 中等（RXMesh 策略） |
| **只缓存属性** | **最快**（MeshTaichi 策略） |
| 缓存两者 | 较慢（共享内存不足，占用率下降） |

**原因分析**:
- 关系在编译时预计算并存储在全局内存，访问模式相对规则
- 属性被多个邻居重复访问，缓存收益高
- 缓存属性还能减少 Patch 间的写冲突

**优化提示语法**:
```python
# 手动指定缓存属性
ti.mesh_local(bunny.verts.pos, bunny.verts.force)
for c in bunny.cells:
    ...

# 编译器自动分析（无提示时）
# 优先级: 原子写操作 > 高频访问属性
```

### 3.3 执行流程：Prologue-Compute-Epilogue

```
┌──────────────────────────────────────────────────────────────┐
│ GPU Block (对应一个 Patch)                                    │
│                                                              │
│  Prologue: 从全局内存加载属性到共享内存                         │
│     global_memory[owned + ribbon] → shared_memory            │
│                                                              │
│  __syncthreads()                                             │
│                                                              │
│  Compute: 在共享内存中执行计算                                  │
│     for each element in patch:                               │
│         read neighbors from shared_memory                    │
│         compute                                              │
│         write to shared_memory                               │
│                                                              │
│  __syncthreads()                                             │
│                                                              │
│  Epilogue: 将更新的属性写回全局内存                             │
│     shared_memory[owned] → global_memory                     │
│     (ribbon 元素可能需要原子操作处理冲突)                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 内存排序管理

**问题**: 默认的全局内存顺序与 Patch 的局部顺序不一致，导致非合并访问。

**解决方案**: 可选的内存重排序

```python
# 使用 Patch 对齐的内存顺序
mesh.verts.place({'vel': ti.math.vec3, 'force': ti.math.vec3}, reorder=True)

# 保持自然顺序（需要与外部数据交互时）
mesh.verts.place({'pos': ti.math.vec3}, reorder=False)
```

**重排序后的内存布局**:
```
自然顺序:     [v1, v2, v3, v4, v5, v6, ...]
             (属于不同 Patch，访问分散)

Patch 顺序:  [Patch1 元素...][Patch2 元素...][Patch3 元素...]
             (同一 Patch 的元素连续，合并访问)
```

---

## 4. 中间表示 (IR) 设计

### 4.1 关系访问语句 (RelationAccessStmt)

统一表示静态和动态关系访问：
- 输入: from-end 元素的局部索引
- 输出: to-end 元素的局部索引

### 4.2 索引转换语句 (IndexConversionStmt)

将局部索引转换为全局索引（考虑可能的重排序）：
- 局部索引 → 全局索引（自然顺序）
- 局部索引 → 重排序索引（Patch 顺序）

### 4.3 局部化转换

属性访问从 `GlobalPtrStmt` 转换为 `BlockLocalPtrStmt`（共享内存访问）。

---

## 5. 性能数据

| 应用 | 相比 RXMesh 加速 | 相比原生 Taichi 加速 |
|------|-----------------|-------------------|
| 显式弹簧系统 | 2.30x | - |
| 隐式弹簧系统 | 2.65x | - |
| 顶点法线计算 | 2.23x | - |
| 测地距离 | 3.46x | - |
| 投影动力学 | 1.61x | 1.26x |
| XPBD 布料 | 3.75x (stretch) | 1.42x |

**关键观察**:
- Scattering 风格加速更明显（减少原子操作）
- 大规模网格加速更显著（数据局部性收益更大）

---

## 6. 与 MAS 预条件器集成分析

### 6.1 MeshTaichi 的局限性

1. **拓扑必须静态**: 不支持运行时修改网格拓扑
2. **分区在编译时完成**: Patch 划分基于网格拓扑，非基于物理特性
3. **不支持跨 Patch 的全局操作**: 适合局部操作，不适合全局归约

### 6.2 MAS 预条件器的需求

| MAS 需求 | MeshTaichi 支持情况 |
|---------|-------------------|
| 顶点按拓扑分组 (BANKSIZE=16) | ❌ Patch 大小固定 ~2048 |
| 分组内顶点连续存储 | ❌ MeshTaichi 的 Patch 顺序与 METIS 分区无关 |
| 多层级层次结构 | ❌ MeshTaichi 只有单层 Patch |
| 48x48 块矩阵操作 | ✅ 可在 kernel 中实现 |
| 碰撞导致的动态连接 | ❌ 关系在编译时固定 |

### 6.3 推荐的集成方案

**方案: 在 MAS 层面引入 METIS 重排序的数据副本**

```
┌───────────────────────────────────────────────────────────────┐
│                    数据流架构                                  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  MeshTaichi (原始顺序)              MAS 内部 (METIS 顺序)       │
│  ┌──────────────────┐              ┌────────────────────┐    │
│  │ mesh.verts.x     │──reorder──→ │ reordered_x        │    │
│  │ mesh.verts.grad  │──reorder──→ │ reordered_grad     │    │
│  │ mesh.cells       │              │ reordered_cells    │    │
│  └──────────────────┘              └────────────────────┘    │
│          ↑                                  │                 │
│          │                                  ↓                 │
│  ┌───────┴───────┐              ┌────────────────────────┐   │
│  │ mesh.verts.z  │←─reorder─── │ MAS 求解               │   │
│  └───────────────┘              │ (METIS 分区内连续)     │   │
│                                  └────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**具体步骤**:

1. **在 MAS 初始化时**:
   ```python
   # 执行 METIS 分区
   reorder_result = metis_reorder_mesh(n_verts, cells, block_size=16)

   # 分配重排序缓冲区
   self.reordered_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
   self.reordered_grad = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
   self.reordered_cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

   # 存储映射
   self.sort_index_ti = ti.field(dtype=ti.i32, shape=n_verts)
   self.old_to_new_ti = ti.field(dtype=ti.i32, shape=n_verts)
   ```

2. **在 apply() 前重排**:
   ```python
   @ti.kernel
   def _reorder_to_metis(self):
       for new_idx in range(self.n_verts):
           old_idx = self.sort_index_ti[new_idx]
           self.reordered_x[new_idx] = self.mesh.verts.x[old_idx]
           self.reordered_grad[new_idx] = self.mesh.verts.grad[old_idx]
   ```

3. **MAS 内部使用重排数据**:
   - `warp_id = new_idx // BANKSIZE` 现在对应 METIS 分区
   - 同一 METIS 分区的顶点拓扑相邻
   - 块矩阵质量显著提升

4. **在 apply() 后还原**:
   ```python
   @ti.kernel
   def _reorder_from_metis(self):
       for new_idx in range(self.n_verts):
           old_idx = self.sort_index_ti[new_idx]
           self.mesh.verts.z[old_idx] = self.reordered_z[new_idx]
   ```

### 6.4 预期收益

| 指标 | 当前实现 | 改进后 |
|------|---------|-------|
| BANKSIZE 块内连通性 | 随机 | 拓扑相邻 |
| 块矩阵条件数 | 差 | 好 |
| 收敛速度 | 慢 | 快 |
| 内存访问模式 | 跨步 | 连续 |

---

## 7. 总结

MeshTaichi 的核心优化思想是 **属性局部化**：
1. 将网格划分为 Patch
2. 预计算关系存储在全局内存
3. 运行时将属性加载到共享内存
4. 在共享内存中完成计算
5. 将结果写回全局内存

对于 MAS 预条件器：
- **不能直接使用** MeshTaichi 的 Patch 作为 MAS 的 subdomain
- **应该在 MAS 内部**维护 METIS 重排序的数据副本
- 保持 MeshTaichi 的编程接口不变，仅在预条件器层面优化数据布局
