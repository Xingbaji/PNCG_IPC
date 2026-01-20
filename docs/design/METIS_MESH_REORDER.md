# METIS Mesh Reorder 优化方案

## 问题分析

### 当前 Pipeline

```
1. model_loading: 加载原始 mesh 数据 (vertices, cells)
2. Patcher.load_mesh(): 创建 MeshTaichi mesh 对象
3. MASPreconditioner: 初始化时接收 mesh，计算 METIS 分区
4. 每帧:
   - assemble: 需要 real_map_partId 查找 (间接寻址开销)
   - apply: 需要 partId_map_real 查找 (间接寻址开销)
```

### 当前问题

1. **双重映射开销**:
   - assemble 时: 原始顶点ID → METIS分区ID → block/lane
   - apply 时: METIS分区ID → 原始顶点ID

2. **内存访问不连续**:
   - MeshTaichi 按原始顺序存储顶点属性
   - METIS 分区后，同一 block 的顶点在内存中不连续

3. **MeshTaichi Patch vs METIS Partition 不匹配**:
   - MeshTaichi 内部有自己的 Patch 分区（约2048个元素）
   - METIS 分区（16个顶点）与 Patch 无关

---

## 优化方案：Mesh 级别的 METIS 重排序

### 核心思想

在创建 `self.mesh` **之前**，对顶点和 cells 进行 METIS 重排序，使得：
- MeshTaichi mesh 内部的顶点顺序已经是 METIS 优化后的顺序
- 无需运行时的映射查找
- 内存访问自然连续

### 新 Pipeline

```
1. model_loading: 加载原始 mesh 数据 (vertices, cells)
2. **METIS reorder**: 对 vertices 和 cells 进行重排序
3. Patcher.load_mesh(): 用重排序后的数据创建 mesh
4. MASPreconditioner: 初始化时无需映射，顶点ID直接对应分区
5. 每帧:
   - assemble: block_id = vertex_id // BANKSIZE (直接计算)
   - apply: 直接读写，无需映射
```

### 数据流对比

```
当前实现:
┌─────────────────────────────────────────────────────────────────┐
│  原始顺序                    METIS 分区                         │
│  vertex 0 ──mapping──→ partition 5, lane 3                     │
│  vertex 1 ──mapping──→ partition 2, lane 7                     │
│  vertex 2 ──mapping──→ partition 5, lane 1                     │
│  ...                                                           │
│  每次 assemble/apply 都需要查表                                 │
└─────────────────────────────────────────────────────────────────┘

优化后:
┌─────────────────────────────────────────────────────────────────┐
│  METIS 顺序 (mesh 内部已重排)                                   │
│  vertex 0  = partition 0, lane 0  (直接计算)                   │
│  vertex 1  = partition 0, lane 1  (直接计算)                   │
│  ...                                                           │
│  vertex 15 = partition 0, lane 15 (直接计算)                   │
│  vertex 16 = partition 1, lane 0  (直接计算)                   │
│  ...                                                           │
│  无需查表！block_id = v // 16, lane_id = v % 16                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实现方案

### 1. 新增工具函数

```python
# algorithm/mas_preconditioner_small/metis_reorder.py

def reorder_mesh_data_metis(
    vertices: np.ndarray,  # (n_verts, 3)
    cells: np.ndarray,     # (n_cells, 4)
    block_size: int = 16
) -> tuple:
    """
    对 mesh 数据进行 METIS 重排序。

    Returns:
        reordered_vertices: 重排序后的顶点位置
        reordered_cells: 重排序后的 cell 连接（顶点ID已更新）
        metis_result: MetisReorderResult 对象（保存映射关系用于输出）
    """
    # 1. 构建图邻接表
    # 2. 调用 pymetis 分区
    # 3. 按分区排序顶点
    # 4. 更新 cell 连接
    # 5. 返回重排序后的数据
```

### 2. 修改 model_loading.py

```python
def load_demo_n_object_collision_free(self, demo, demo_dict):
    # ... 加载原始数据 ...
    models = [self.add_object(...) for ...]

    # 如果启用 METIS，在创建 mesh 前重排序
    if demo_dict.get('use_metis', False):
        from algorithm.mas_preconditioner_small.metis_reorder import reorder_mesh_data_metis

        # 合并所有 model 的数据
        all_verts, all_cells = merge_models(models)

        # METIS 重排序
        reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(
            all_verts, all_cells
        )

        # 重建 models 列表
        models = split_models(reordered_verts, reordered_cells, ...)

        # 保存 metis_result 供后续使用
        self.metis_result = metis_result

    # 用重排序后的数据创建 mesh
    self.mesh = Patcher.load_mesh(models, relations=["CV"])
```

### 3. 简化 MASPreconditionerSmall

```python
class MASPreconditionerSmall:
    def __init__(self, mesh, metis_reordered=False, ...):
        """
        Args:
            mesh: MeshTaichi mesh (可能已经是 METIS 重排序后的)
            metis_reordered: 如果 True，mesh 数据已经按 METIS 排序，
                            无需映射，可直接使用 v // BANKSIZE 计算 block
        """
        self.mesh = mesh
        self.metis_reordered = metis_reordered

        if metis_reordered:
            # 不需要映射表！
            self.use_metis = True
            # block_id = vertex_id // BANKSIZE
            # lane_id = vertex_id % BANKSIZE
        else:
            # 传统模式：需要运行时映射
            ...
```

---

## 优势分析

### 性能优势

| 操作 | 当前实现 | 优化后 |
|------|---------|-------|
| assemble block/lane 计算 | `real_map_partId[v]` 查表 | `v // 16`, `v % 16` 直接计算 |
| apply 读取残差 | 原始顺序访问 | 按 block 连续访问 |
| apply 写入结果 | 原始顺序访问 | 按 block 连续访问 |
| 内存带宽利用 | 低（跨步访问）| 高（连续访问）|

### 代码简化

1. 删除 `real_map_partId`, `partId_map_real` 字段
2. 删除 METIS 映射相关的 kernel 分支
3. assemble/apply 代码统一，无需 `_metis` 后缀版本

---

## 挑战与解决方案

### 1. 多物体场景

**问题**: 多个物体的顶点需要合并后再 METIS 分区

**解决**:
- 在 `model_loading.py` 中合并所有 models
- 分区后需要记录每个物体的顶点范围
- 输出时需要逆映射回原始顺序

### 2. 输出/可视化

**问题**: 重排序后的顶点顺序与原始 mesh 文件不对应

**解决**:
- 保存 `new_to_old` 映射
- 输出时逆映射回原始顺序
- 或者接受新顺序，更新可视化代码

### 3. 碰撞检测

**问题**: 碰撞检测可能依赖原始顶点顺序

**解决**:
- 碰撞检测使用的是 `mesh.verts.x`，与顶点顺序无关
- boundary 信息需要在重排序后重新计算

### 4. Dirichlet 边界条件

**问题**: `is_dirichlet` 数组基于原始顶点ID

**解决**:
- 重排序后重新映射 Dirichlet 顶点
- `new_is_dirichlet[new_id] = old_is_dirichlet[old_id]`

---

## 实现计划

1. [ ] 在 `metis_reorder.py` 中添加 `reorder_mesh_data_metis()` 函数
2. [ ] 添加 `merge_models()` 和 `split_models()` 工具函数
3. [ ] 修改 `model_loading.py` 支持 mesh 级别重排序
4. [ ] 简化 `MASPreconditionerSmall`，添加 `metis_reordered` 模式
5. [ ] 更新测试用例
6. [ ] 性能对比测试

---

## 预期收益

| 指标 | 预期改善 |
|------|---------|
| assemble 性能 | 消除映射查找，预计 10-20% 提升 |
| apply 性能 | 内存连续访问，预计 20-30% 提升 |
| 代码复杂度 | 删除多个 `_metis` 版本 kernel |
| 内存占用 | 删除映射表（约 2 * n_verts * 4 bytes）|

---

## 注意事项

1. **兼容性**: 需要保持非 METIS 模式的兼容性
2. **单次计算**: METIS 重排序仍然只在模拟开始时计算一次
3. **顺序敏感**: 任何依赖原始顶点顺序的代码都需要检查
