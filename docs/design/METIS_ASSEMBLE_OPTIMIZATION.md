# METIS Assemble 优化策略

## 问题分析

### 当前性能数据

在 cube_40 (19661 verts) 上的测试结果：
- Sequential assemble: ~X ms
- METIS assemble: ~1.6X ms (约 0.62x 速度)

### 瓶颈分析

#### 1. 间接寻址开销

METIS版本需要额外的映射查询：
```python
# 非METIS: 直接计算
warp_id = v0 // BANKSIZE
lane_id = v0 % BANKSIZE

# METIS: 需要查表
part_info = self.real_map_partId[v0]  # 额外内存访问
block_id = part_info // BANKSIZE
lane_id = part_info % BANKSIZE
```

这增加了每个顶点的内存访问次数。

#### 2. 原子操作冲突增加

METIS分区的目标是让拓扑相邻的顶点分到同一block，但这**反而增加**了原子操作冲突：
- 同一cell的4个顶点更可能在同一block
- 多个相邻cell对同一block的并发写入增多

#### 3. MeshTaichi Patch vs METIS Partition 的不匹配

| 特性 | MeshTaichi Patch | METIS Partition |
|------|------------------|-----------------|
| 大小 | ~2048 元素 | 16 顶点 |
| 目的 | 共享内存缓存 | 块矩阵质量 |
| 迭代顺序 | 按Patch分组 | 按原始顺序 |

当前 `for c in self.mesh.cells` 按MeshTaichi的Patch顺序迭代，但METIS分区与此无关。

---

## 优化方案

### 方案1: 预排序Cell（推荐）

**核心思想**: 按METIS分区对cell进行重排序，使同一partition内的cell连续处理。

```python
class MASPreconditionerSmall:
    def __init__(self, ...):
        if metis_result is not None:
            # 预计算cell到partition的映射
            self._precompute_cell_partition_mapping(cells, metis_result)

    def _precompute_cell_partition_mapping(self, cells, metis_result):
        """
        为每个cell计算其"主分区"（4个顶点所属分区中最常见的）
        然后按主分区排序cell
        """
        n_cells = len(cells)
        cell_main_partition = np.zeros(n_cells, dtype=np.int32)

        for c_idx, cell in enumerate(cells):
            parts = [metis_result.partition[v] for v in cell]
            # 选择出现最多的分区作为主分区
            cell_main_partition[c_idx] = max(set(parts), key=parts.count)

        # 按主分区排序
        self.cell_order = np.argsort(cell_main_partition)
        self.cell_order_ti = ti.field(dtype=ti.i32, shape=n_cells)
        self.cell_order_ti.from_numpy(self.cell_order)
```

**优点**:
- 减少原子操作冲突：同一分区的cell连续处理，冲突更局部化
- 更好的GPU缓存利用：连续处理的cell访问相近的block_matrices
- 一次性预计算开销，可以和METIS一起完成

**实现难点**:
- 需要改变cell迭代方式：不能直接用 `for c in mesh.cells`
- 需要通过cell index访问cell数据

### 方案2: 两阶段组装

**核心思想**: 分离"同分区组装"和"跨分区传播"。

```python
@ti.kernel
def _add_elastic_contribution_phase1(self, ...):
    """Phase 1: 只处理4个顶点在同一分区的cell"""
    for c in self.mesh.cells:
        # 检查是否所有顶点在同一分区
        if all_same_partition(c):
            # 直接组装，无需检查
            assemble_to_block(...)

@ti.kernel
def _add_elastic_contribution_phase2(self, ...):
    """Phase 2: 处理跨分区的cell"""
    for c in self.mesh.cells:
        if not all_same_partition(c):
            # 分别处理同分区和跨分区的顶点对
            ...
```

**优点**:
- 减少分支发散
- Phase 1 完全无原子冲突检查

**缺点**:
- 需要遍历两次
- 实际收益取决于"同分区cell"的比例

### 方案3: 使用局部缓冲区（MeshTaichi风格）

**核心思想**: 借鉴MeshTaichi的"属性局部化"，在组装时使用线程本地缓冲区。

```python
@ti.kernel
def _add_elastic_contribution_with_local_buffer(self, ...):
    # 使用shared memory风格的本地缓冲
    ti.loop_config(block_dim=256)

    for c in self.mesh.cells:
        # 计算element Hessian到本地变量
        H_e = compute_element_hessian(c, ...)

        # 使用ti.simt的原子操作优化
        # Taichi会自动将连续的原子操作合并
        for i, j in ti.static(...):
            if same_block:
                # 使用ti.atomic_add，编译器会优化
                ti.atomic_add(block_matrices[...], H_e[...])
```

**优点**:
- 利用Taichi编译器的原子操作优化
- 代码改动较小

**缺点**:
- 依赖编译器优化，效果不确定

### 方案4: 重排序后的数据副本（文档6.3推荐）

**核心思想**: 维护METIS顺序的数据副本，所有操作在重排序后的数据上进行。

```python
class MASPreconditionerSmall:
    def __init__(self, ...):
        if metis_result:
            # 分配重排序缓冲区
            self.reordered_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
            self.reordered_cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

            # 预计算重排序后的cell connectivity
            self._reorder_cell_connectivity(cells, metis_result)

    def _reorder_cell_connectivity(self, cells, metis_result):
        """将cell的顶点ID替换为METIS重排序后的ID"""
        reordered = np.zeros_like(cells)
        for c_idx, cell in enumerate(cells):
            for i in range(4):
                reordered[c_idx, i] = metis_result.old_to_new[cell[i]]
        self.reordered_cells_np = reordered
```

然后在assemble中：
```python
@ti.kernel
def _add_elastic_contribution_reordered(self, ...):
    for c_idx in range(self.n_cells):
        # 使用重排序后的顶点ID
        v0 = self.reordered_cells[c_idx, 0]
        v1 = self.reordered_cells[c_idx, 1]
        ...

        # 现在warp_id = v / BANKSIZE 直接对应METIS分区
        warp_id = v0 // BANKSIZE
        lane_id = v0 % BANKSIZE
        # 无需额外的映射查询！
```

**优点**:
- 消除了间接寻址开销
- 与MeshTaichi的设计理念一致
- 预计算开销可以和METIS一起完成

**缺点**:
- 需要额外的内存存储重排序后的cells
- 不能直接使用 `for c in mesh.cells` 访问顶点位置

---

## 推荐方案

结合上述分析，**推荐方案4 + 方案1**：

1. **预计算重排序的cell connectivity** (方案4)
   - 消除间接寻址开销
   - 重排序ID后可以直接计算block/lane

2. **按分区排序cell处理顺序** (方案1)
   - 减少原子操作冲突
   - 提高缓存局部性

### 实现步骤

```python
class MASPreconditionerSmall:
    def _init_metis_optimized_assembly(self, cells_np, metis_result):
        """初始化METIS优化的组装数据结构"""
        n_cells = len(cells_np)

        # Step 1: 重排序cell的顶点ID
        reordered_cells = np.zeros((n_cells, 4), dtype=np.int32)
        cell_main_partition = np.zeros(n_cells, dtype=np.int32)

        for c_idx in range(n_cells):
            parts = []
            for i in range(4):
                old_v = cells_np[c_idx, i]
                new_v = metis_result.old_to_new[old_v]
                reordered_cells[c_idx, i] = new_v
                parts.append(new_v // BANKSIZE)  # partition = block

            # 主分区
            cell_main_partition[c_idx] = max(set(parts), key=parts.count)

        # Step 2: 按主分区排序cell
        cell_order = np.argsort(cell_main_partition, kind='stable')

        # Step 3: 重排cells
        sorted_cells = reordered_cells[cell_order]

        # 存储到Taichi field
        self.sorted_cells = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)
        self.sorted_cells.from_numpy(sorted_cells)

        # 存储顶点位置的重排序版本
        self.reordered_x = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)
```

### 性能预期

| 优化 | 预期收益 |
|------|---------|
| 消除间接寻址 | 10-20% |
| cell排序减少冲突 | 5-15% |
| 总计 | 15-30% |

如果优化后METIS assemble速度接近或超过Sequential，那么METIS带来的apply加速(1.8x)将使整体性能显著提升。

---

## 实现计划

1. [x] 在 `metis_reorder.py` 中添加 `compute_optimized_cell_data()` 函数
2. [x] 在 `core.py` 中添加 `init_optimized_assembly()` 方法
3. [x] 实现 `_add_elastic_contribution_arap_optimized_full()` kernel
4. [x] 运行性能测试比较优化前后

---

## 实验结果 (2026-01-20)

### cube_20 (2931 verts, 13200 cells)

| 方法 | Assemble时间 | 加速比 |
|------|-------------|--------|
| Sequential | 0.460ms | 1.00x (baseline) |
| METIS | 0.888ms | 0.52x (slower) |
| **METIS Optimized** | **0.804ms** | **1.10x vs METIS** |

Same partition cells: 951/13200 (7.2%)

### cube_40 (19661 verts, 100800 cells)

| 方法 | Assemble时间 | 加速比 |
|------|-------------|--------|
| Sequential | 0.952ms | 1.00x (baseline) |
| METIS | 1.603ms | 0.59x (slower) |
| **METIS Optimized** | **2.198ms** | **0.73x vs METIS (反而更慢!)** |

Same partition cells: 5576/100800 (5.5%)

### 分析

优化方案在较大mesh上**没有**达到预期效果，原因：

1. **Same-partition比例太低**: BANKSIZE=16的约束使得大部分四面体的4个顶点分布在多个partition中
   - cube_20: 7.2%
   - cube_40: 5.5%

2. **额外的间接访问开销**: 优化版本需要：
   - 访问 `sorted_B[sorted_idx]` (额外field访问)
   - 访问 `sorted_W[sorted_idx]` (额外field访问)
   - 访问 `sorted_orig_cells[sorted_idx]` (额外field访问)
   - 通过原始vertex ID访问位置 `mesh.verts.x[orig_cell[i]]`

   这些额外的间接访问抵消了排序带来的缓存局部性改善。

3. **MeshTaichi的限制**: 不能直接通过索引访问cells，必须预提取数据到额外的field中。

### 结论

对于当前的MAS实现，METIS cell排序优化**不推荐使用**：
- 小mesh: 略有改善 (1.10x)，但开销不值得
- 大mesh: 反而变慢

更好的优化方向：
1. 使用更大的partition size (但会影响块矩阵质量)
2. 仅对apply()使用METIS优化 (已证实有效：1.8-2.0x加速)
3. 接受assemble稍慢的代价，换取apply的加速
