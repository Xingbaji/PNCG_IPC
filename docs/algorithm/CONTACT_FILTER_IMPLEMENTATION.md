# Contact Filter 实现文档

> 最后更新: 2026-01-20

## 概述

Contact Filter 模块实现了**双半径碰撞检测策略**，用于优化 IPC (Incremental Potential Contact) 仿真中的碰撞检测效率。

### 核心思想

- **检测半径 (detection_dHat)**: 使用较大的距离阈值 (默认 5×dHat) 缓存潜在碰撞对
- **激活半径 (active_dHat)**: 使用实际的 dHat 进行 barrier 函数计算

这种策略的优势：
1. 减少 BVH 遍历频率（缓存的碰撞对在更大范围内保持有效）
2. 预过滤不相关的碰撞，减少 barrier 计算量
3. 检测和激活阈值分离，提供更大的灵活性

---

## 模块结构

### 文件位置

```
algorithm/
├── contact_filter.py          # Contact Filter 模块 (新增)
├── collision_detection_bvh.py # BVH碰撞检测 (已修改)
└── pncg_base_ipc.py           # PNCG求解器 (可选集成)

unittest/tests/
└── test_contact_filter.py     # 单元测试
```

### 类图

```
ContactFilter
├── 配置
│   ├── detection_dHat: ti.field    # 检测阈值 (5×dHat)
│   └── active_dHat: ti.field       # 激活阈值 (dHat)
├── 存储
│   ├── filtered_contacts: pair.field  # 过滤后的碰撞对
│   └── n_filtered: ti.field           # 过滤后的数量
└── 方法
    ├── configure()                 # 配置阈值
    ├── filter_contacts()           # 过滤碰撞对
    └── get_stats()                 # 获取统计信息
```

---

## API 参考

### ContactFilter 类

```python
@ti.data_oriented
class ContactFilter:
    """
    双半径碰撞检测的Contact Filter。

    使用 detection_dHat (默认 5×dHat) 缓存碰撞对，
    然后过滤到 active_dHat (dHat) 用于 barrier 计算。
    """
```

#### 构造函数

```python
def __init__(self, max_contacts: int = 2**21):
    """
    初始化 Contact Filter。

    Args:
        max_contacts: 最大碰撞对数量
    """
```

#### 配置方法

```python
def configure(self, active_dHat: float, multiplier: float = 5.0) -> float:
    """
    配置检测和激活阈值。

    Args:
        active_dHat: 实际的 barrier 阈值 (dHat)
        multiplier: 检测半径倍数 (默认: 5.0)

    Returns:
        detection_dHat: 配置的检测阈值，用于 BVH
    """
```

#### 过滤方法

```python
@ti.kernel
def filter_contacts(self, cached_contacts: ti.template(), n_cached: ti.i32):
    """
    过滤缓存的碰撞对，只保留 active_dHat 范围内的。

    Args:
        cached_contacts: 来自 collision_detection_bvh 的碰撞对
        n_cached: 缓存的碰撞对数量
    """
```

#### 统计方法

```python
def get_stats(self) -> dict:
    """
    获取过滤统计信息。

    Returns:
        {'n_cached': int, 'n_filtered': int, 'filter_ratio': float}
    """

def print_stats(self):
    """打印过滤统计信息。"""
```

---

## 碰撞对数据结构

Contact Filter 使用与 `collision_detection_bvh.py` 相同的碰撞对结构：

```python
pair = ti.types.struct(
    a=ti.types.vector(4, ti.u32),   # 顶点索引 [v0, v1, v2, v3]
    b=float,                         # 距离
    c=ti.types.vector(4, float),    # 重心坐标
    d=ti.types.vector(3, float)     # 方向向量 (归一化)
)
```

### Point-Triangle (PT) 碰撞对

```
a = [point_id, tri_v0, tri_v1, tri_v2]
b = 点到三角形的距离
c = [1.0, -cord0, -cord1, -cord2]  # 重心坐标
d = (xp - xt) / dist              # 归一化方向
```

### Edge-Edge (EE) 碰撞对

```
a = [edge1_v0, edge1_v1, edge2_v0, edge2_v1]
b = 边之间的距离
c = [sc - 1.0, -sc, 1.0 - tc, tc]  # 参数坐标
d = t_ee / dist                    # 归一化方向
```

---

## 与 collision_detection_bvh.py 的集成

### 新增方法

```python
def set_detection_dHat(self, detection_dHat: float):
    """
    设置自定义检测阈值用于 contact filtering。

    允许使用更大的检测半径 (如 5×dHat) 来缓存碰撞对，
    而实际的 barrier 仍使用原始 dHat。

    Args:
        detection_dHat: 检测阈值 (应 >= dHat)
    """
    self._detection_dHat = detection_dHat
    self.bvh_gap = ti.sqrt(detection_dHat)

def get_detection_dHat(self) -> float:
    """获取当前检测阈值。"""
    return self._detection_dHat
```

### 修改的函数

`attempt_PT_no_adj` 和 `attempt_EE_no_adj` 现在使用 `self._detection_dHat` 而不是 `self.dHat`：

```python
@ti.func
def attempt_PT_no_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
    # 使用 _detection_dHat 进行过滤 (可以比 dHat 更大用于缓存)
    if p != t0 and p != t1 and p != t2 and \
       point_triangle_ccd_broadphase(xp, x0, x1, x2, self._detection_dHat):
        # ... 计算距离 ...
        if dist < self._detection_dHat and ti.abs(dist) > self.SMALL_NUM:
            # 添加到缓存
            self._add_contact_pair(ids, dist, cord, t_pt)
```

### 向后兼容性

默认情况下，`_detection_dHat = dHat`，确保不调用 `set_detection_dHat` 时行为不变。

---

## 使用示例

### 基本用法

```python
from algorithm.contact_filter import ContactFilter
from algorithm.pncg_base_ipc import pncg_ipc_deformer

# 初始化
deformer = pncg_ipc_deformer(demo='eight_E_drop_demo_contact')
contact_filter = ContactFilter(max_contacts=2**21)

# 配置: detection_dHat = 5 × dHat
detection_dHat = contact_filter.configure(deformer.dHat, multiplier=5.0)
deformer.set_detection_dHat(detection_dHat)

# 仿真循环
for frame in range(100):
    # 碰撞检测 (使用 5×dHat 检测)
    deformer.find_cnts()

    # 过滤到激活范围 (dHat)
    contact_filter.filter_contacts(
        deformer.contact_pairs,
        deformer.n_contacts[None]
    )

    # 使用过滤后的碰撞对计算 barrier
    n_active = contact_filter.n_filtered[None]
    # 遍历 contact_filter.filtered_contacts[:n_active]

    # 打印统计
    contact_filter.print_stats()
```

### 与 Barrier 函数集成

```python
@ti.kernel
def compute_barrier_energy_filtered(
    self,
    filtered_contacts: ti.template(),
    n_filtered: ti.i32
) -> float:
    """使用过滤后的碰撞对计算 barrier 能量。"""
    E = 0.0
    for idx in range(n_filtered):
        pair = filtered_contacts[idx]
        dist = pair.b
        # barrier 使用 active_dHat (self.dHat)
        E += self.get_barrier_E(dist)
    return E
```

---

## 工作流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Contact Filter 工作流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 配置阶段                                                     │
│     ┌──────────────────┐     ┌───────────────────┐              │
│     │ contact_filter.  │────▶│ detection_dHat    │              │
│     │ configure(dHat,5)│     │ = 5 × dHat        │              │
│     └──────────────────┘     └───────────────────┘              │
│                                       │                         │
│                                       ▼                         │
│     ┌──────────────────┐     ┌───────────────────┐              │
│     │ deformer.        │────▶│ BVH gap =         │              │
│     │ set_detection_   │     │ sqrt(5×dHat)      │              │
│     │ dHat()           │     └───────────────────┘              │
│     └──────────────────┘                                        │
│                                                                 │
│  2. 检测阶段                                                     │
│     ┌──────────────────┐     ┌───────────────────┐              │
│     │ deformer.        │────▶│ cached_contacts   │              │
│     │ find_cnts()      │     │ (dist < 5×dHat)   │              │
│     └──────────────────┘     │ n_cached = 1000   │              │
│                              └───────────────────┘              │
│                                       │                         │
│                                       ▼                         │
│  3. 过滤阶段                                                     │
│     ┌──────────────────┐     ┌───────────────────┐              │
│     │ contact_filter.  │────▶│ filtered_contacts │              │
│     │ filter_contacts()│     │ (dist < dHat)     │              │
│     └──────────────────┘     │ n_filtered = 200  │              │
│                              └───────────────────┘              │
│                                       │                         │
│                                       ▼                         │
│  4. Barrier 计算                                                 │
│     ┌──────────────────┐                                        │
│     │ 遍历 filtered_   │                                        │
│     │ contacts 计算     │                                        │
│     │ barrier E/g/H    │                                        │
│     └──────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 性能分析

### 过滤效果

| 场景 | n_cached | n_filtered | 过滤比例 |
|------|----------|------------|----------|
| 初始无接触 | 1000 | 0 | 0% |
| 接近但未接触 | 5000 | 500 | 10% |
| 轻微接触 | 10000 | 2000 | 20% |
| 深度接触 | 50000 | 40000 | 80% |

### 计算开销

- **过滤内核**: O(n_cached)，每个碰撞对一次距离比较
- **内存开销**: 额外的 `filtered_contacts` 数组 (MAX_C × 56 bytes)
- **GPU 效率**: 使用原子计数器实现线程安全插入

### 优化建议

1. **缓存更新策略**: 大位移时才重建 BVH，小位移时只 refit
2. **自适应倍数**: 根据仿真状态调整 multiplier
3. **批量过滤**: 多次迭代共享相同的缓存

---

## 单元测试

测试文件: `unittest/tests/test_contact_filter.py`

| 测试 | 描述 |
|------|------|
| test_configuration | 验证阈值配置 |
| test_filter_basic | 基本过滤功能 |
| test_filter_empty | 无激活碰撞时返回空 |
| test_filter_all_pass | 所有碰撞在阈值内 |
| test_filter_preserves_data | 验证数据完整性 |
| test_stats | 统计信息准确性 |

运行测试:

```bash
cd /root/PNCG_IPC
python unittest/tests/test_contact_filter.py
```

---

## 相关文档

- [collision_detection_bvh.py](../../algorithm/collision_detection_bvh.py) - BVH 碰撞检测实现
- [pncg_base_ipc.py](../../algorithm/pncg_base_ipc.py) - PNCG IPC 求解器
- [GCP_IMPLEMENTATION.md](GCP_IMPLEMENTATION.md) - GCP 几何接触势能实现
- [CUBIC_BARRIER_IMPLEMENTATION.md](CUBIC_BARRIER_IMPLEMENTATION.md) - 三次障碍函数实现

---

## 变更历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-01-20 | 1.0 | 初始实现 |
