# MAS Preconditioner Optimization TODO

本文档记录了MAS预条件器及相关代码的Taichi优化计划。

## 目录
1. [问题概述](#1-问题概述)
2. [稀疏字典遍历问题（全局影响）](#2-稀疏字典遍历问题全局影响)
3. [MAS预条件器特定优化](#3-mas预条件器特定优化)
4. [优化优先级与计划](#4-优化优先级与计划)
5. [稀疏字典问题的IPC模拟影响分析](#5-稀疏字典问题的ipc模拟影响分析)
6. [参考资料](#6-参考资料)
7. [更新日志](#7-更新日志)

---

## 1. 问题概述

当前实现从CUDA参考代码移植而来，保留了许多CUDA特有的编程模式。虽然功能正确，但未充分利用Taichi的优势。

### 主要问题分类

| 问题类型 | 影响范围 | 性能影响 | 优化难度 |
|----------|----------|----------|----------|
| 稀疏字典遍历 | **全局（16+文件，56+处）** | 高 | 中 |
| 过度原子操作 | mas_preconditioner.py | 中-高 | 中 |
| 串行块求逆 | mas_preconditioner.py | 中 | 高 |
| 位操作模拟 | mas_preconditioner.py | 低-中 | 低 |
| 数据布局 | 多个文件 | 低-中 | 低 |

---

## 2. 稀疏字典遍历问题（全局影响）

### 2.1 问题描述

当前使用Taichi的`bitmasked`稀疏字段存储接触对：

```python
# collision_detection_bvh.py:92-99
self.pair = ti.types.struct(
    a=ti.types.vector(4, ti.u32),  # vertex ids
    b=float,                         # distance
    c=ti.types.vector(4, float),    # barycentric coords
    d=ti.types.vector(3, float)     # normal/direction
)
self.cid = self.pair.field()
self.cid_root = ti.root.bitmasked(ti.ij, (2, self.MAX_C)).place(self.cid)
```

遍历模式：
```python
for k, j in self.cid:
    pair = self.cid[k, j]
    # 处理接触对...
```

### 2.2 性能问题

1. **哈希表查找开销**：每次`self.cid[k,j]`都需要哈希计算和查找
2. **内存访问不连续**：bitmasked结构的物理存储是稀疏的
3. **无法向量化**：遍历顺序不确定，编译器无法优化
4. **分支发散**：不同线程处理的接触对位置随机

### 2.2.1 技术深度分析

#### 当前哈希机制

```python
# collision_detection_bvh.py:118-120
@ti.func
def hash_coords_2(self, x, y):
    h = (x * 92837111) ^ (y * 689287499)
    return ti.abs(h) % self.MAX_C  # MAX_C = 2^21
```

**问题1：哈希冲突风险**
- 使用简单乘法哈希，冲突概率随接触数增加
- 冲突时后写入的接触对会覆盖前一个
- 可能导致漏检接触对

**问题2：双重索引开销**
```python
# 写入时
self.cid[0, hash_index] = self.pair(ids, dist, cord, t_pt)

# 读取时（每次循环都执行）
for k, j in self.cid:           # Step 1: 遍历bitmasked查找有效entry
    pair = self.cid[k, j]       # Step 2: 再次哈希查找取值
```

#### Taichi bitmasked实现原理

```
bitmasked(ti.ij, (2, MAX_C)) 底层结构:
┌────────────────────────────────────────┐
│  Bitmask Array (2 * MAX_C / 32 bits)   │  <- 判断slot是否有效
├────────────────────────────────────────┤
│  Data Array (2 * MAX_C * sizeof(pair)) │  <- 实际数据
└────────────────────────────────────────┘

遍历过程:
1. 线性扫描bitmask找到非零位
2. 计算对应的(k,j)索引
3. 访问Data Array[k * MAX_C + j]
```

**性能瓶颈**：
- Bitmask扫描是串行的
- 数据分散在2GB范围内（2 * 2^21 * 32bytes ≈ 256MB）
- Cache利用率极低

#### 性能对比估算

| 操作 | bitmasked方式 | 紧凑数组方式 |
|------|---------------|--------------|
| 遍历N个接触对 | O(MAX_C) bitmask扫描 | O(N) 连续访问 |
| 读取一个接触对 | 哈希计算 + 随机访存 | 直接索引 |
| 内存带宽利用 | ~10-20% | ~80-90% |
| 向量化可能性 | 无 | 有（SIMD） |

**实际影响**：当N=10000接触对时，
- bitmasked需扫描2^22个bitmask位（~500KB）
- 紧凑数组只需读取10000*32=320KB连续数据

### 2.3 影响范围

#### 核心算法文件

| 文件 | 行号 | 用途 |
|------|------|------|
| `algorithm/mas_preconditioner.py` | 1122, 1919, 2038 | IPC Hessian组装、Woodbury更新 |
| `algorithm/mas_pncg_solver.py` | 212, 307, 571, 636, 684 | 能量、梯度、Hv、力计算 |
| `algorithm/pncg_base_ipc.py` | 270, 317, 367, 413, 609 | 能量、梯度、Hv、力计算 |
| `algorithm/gcp_contact_potential.py` | 910, 926, 955, 1005 | GCP能量、梯度、Hv |
| `algorithm/pncg_base_ipc_spatial_hash.py` | 108, 138, 173, 318 | 能量、梯度、Hv、力 |
| `algorithm/pncg_abd_ipc.py` | 431 | ABD-FEM混合求解 |
| `algorithm/collision_detection_bvh.py` | 527 | 调试统计 |
| `algorithm/collision_detection.py` | 726 | 空间哈希遍历 |

#### Demo文件

| 文件 | 行号 | 用途 |
|------|------|------|
| `demo/n_E_demo.py` | 78 | pHp计算 |
| `demo/gcp_demo.py` | 207, 242, 290 | 渲染、力可视化、应力 |
| `demo/compare_collision_detection.py` | 115, 179 | 性能测试 |
| `n_E_demos/cubic_barrier_n_E_demo.py` | 247, 291, 382, 417, 524, 613 | 完整IPC流程 |
| `n_E_demos/spatial_hash_n_E_demo.py` | 138 | 空间哈希 |
| `n_E_demos/initial_n_E_demo.py` | 120 | 能量计算 |
| `n_E_demos/bvh_n_E_demo.py` | 269 | BVH遍历 |
| `n_E_demos/gcp_n_E_demo.py` | 354, 387, 432 | GCP计算 |

### 2.4 优化方案

#### 方案A：紧凑数组存储（推荐）

```python
# 定义紧凑的接触对存储
@ti.dataclass
class ContactPair:
    ids: ti.types.vector(4, ti.u32)
    dist: ti.f32
    cord: ti.types.vector(4, ti.f32)
    normal: ti.types.vector(3, ti.f32)

class CollisionDetection:
    def __init__(self, ...):
        # 紧凑数组存储
        self.max_contacts = 2 ** 21
        self.contact_pairs = ContactPair.field(shape=self.max_contacts)
        self.n_contacts = ti.field(ti.i32, shape=())

        # 可选：用于快速查找的索引（如果需要按顶点ID查找）
        self.vertex_contact_start = ti.field(ti.i32, shape=n_verts + 1)
        self.vertex_contact_count = ti.field(ti.i32, shape=n_verts)

    @ti.kernel
    def detect_contacts(self):
        self.n_contacts[None] = 0
        # 碰撞检测...
        for ...:
            if is_contact:
                idx = ti.atomic_add(self.n_contacts[None], 1)
                if idx < self.max_contacts:
                    self.contact_pairs[idx].ids = ids
                    self.contact_pairs[idx].dist = dist
                    # ...

    @ti.kernel
    def compute_energy(self) -> ti.f64:
        E = 0.0
        # 连续访问，cache友好
        for i in range(self.n_contacts[None]):
            pair = self.contact_pairs[i]
            E += barrier_energy(pair.dist, ...)
        return E
```

**优点**：
- 连续内存访问
- 可向量化
- 无哈希开销
- 支持并行reduction

**缺点**：
- 需要修改碰撞检测输出格式
- 如需按顶点查找需要额外索引

#### 方案B：Morton码排序

```python
@ti.kernel
def sort_contacts_by_morton(self):
    """按Morton码排序以提高空间局部性"""
    for i in range(self.n_contacts[None]):
        ids = self.contact_pairs[i].ids
        # 计算接触对质心
        centroid = compute_centroid(ids)
        # 计算Morton码
        self.morton_codes[i] = compute_morton_code(centroid)
        self.contact_indices[i] = i

    # 排序（使用Taichi的radix sort或外部排序）
    sort_by_key(self.morton_codes, self.contact_indices, self.n_contacts[None])
```

**优点**：
- 空间相近的接触对在内存中也相邻
- 提高cache命中率

**缺点**：
- 排序开销
- 复杂度增加

#### 方案C：分桶存储（按子域）

```python
class BucketedContacts:
    def __init__(self, n_subdomains, max_per_subdomain):
        self.n_subdomains = n_subdomains
        self.max_per_subdomain = max_per_subdomain

        # 每个子域有独立的接触对数组
        self.subdomain_contacts = ContactPair.field(
            shape=(n_subdomains, max_per_subdomain))
        self.subdomain_counts = ti.field(ti.i32, shape=n_subdomains)

    @ti.kernel
    def add_contact_to_subdomain(self, subdomain_id: ti.i32, pair: ti.template()):
        idx = ti.atomic_add(self.subdomain_counts[subdomain_id], 1)
        if idx < self.max_per_subdomain:
            self.subdomain_contacts[subdomain_id, idx] = pair
```

**优点**：
- 天然支持子域并行（MAS预条件器）
- 减少跨子域的数据竞争

**缺点**：
- 接触对可能跨子域（需要复制到两个子域）
- 内存使用可能不均衡

### 2.5 迁移策略

**阶段1：添加紧凑数组接口（兼容层）**
```python
class CollisionDetectionBVH:
    def __init__(self, ...):
        # 保留原有bitmasked结构
        self.cid = self.pair.field()
        self.cid_root = ti.root.bitmasked(...)

        # 新增紧凑数组
        self.contact_pairs = ContactPair.field(shape=self.max_contacts)
        self.n_contacts = ti.field(ti.i32, shape=())

    def sync_to_compact_array(self):
        """将bitmasked cid同步到紧凑数组"""
        self._sync_kernel()

    @ti.kernel
    def _sync_kernel(self):
        self.n_contacts[None] = 0
        for k, j in self.cid:
            idx = ti.atomic_add(self.n_contacts[None], 1)
            pair = self.cid[k, j]
            self.contact_pairs[idx].ids = pair.a
            # ...
```

**阶段2：逐步迁移使用方**
- 优先迁移热点代码（能量、梯度、Hv计算）
- 保留旧接口用于非关键路径

**阶段3：移除bitmasked结构**
- 所有代码迁移完成后移除旧结构

---

## 3. MAS预条件器特定优化

### 3.1 原子操作优化

**问题位置**：`mas_preconditioner.py:991`

```python
# 当前：大量原子操作
ti.atomic_add(self.block_matrices[warp_i, sym_idx][di, dj], val)
```

**优化方案**：使用`ti.block_local`共享内存

```python
@ti.kernel
def _add_elastic_contribution_optimized(self, ...):
    # 每个block分配局部累加器
    ti.block_local(local_accum)  # shape: [BANKSIZE, BANKSIZE, 3, 3]

    # Phase 1: 累积到局部内存（无竞争）
    for c in self.mesh.cells:
        # 计算H_e...
        # 写入local_accum（每个warp内）

    ti.block_sync()  # 同步

    # Phase 2: 合并写入全局内存（每个entry只写一次）
    for lane in ti.ndrange(BANKSIZE * BANKSIZE):
        # 单次原子操作
```

**预期收益**：20-30%（弹性Hessian组装）

---

### 3.2 块矩阵求逆优化

**问题位置**：`mas_preconditioner.py:1446-1492`

```python
# 当前：每个块串行Gauss-Jordan消元
for pivot in range(BLOCK_DOF):  # 48次迭代
    for r in range(BLOCK_DOF):  # 48次
        # 消元
```

**优化方案A**：CPU批量求逆

```python
def invert_blocks_cpu(self):
    """利用NumPy/SciPy的优化BLAS"""
    # 同步GPU数据到CPU
    mats = self.full_block_matrix.to_numpy()  # shape: [n_blocks, 48, 48]

    # 批量求逆（利用多线程BLAS）
    invs = np.linalg.inv(mats)

    # 回传GPU
    self.full_block_inverse.from_numpy(invs.astype(np.float32))
```

**优化方案B**：Cholesky分解（对SPD矩阵）

```python
@ti.kernel
def invert_blocks_cholesky(self):
    for block_id in range(n_blocks):
        # L = cholesky(A)
        L = cholesky_48x48(self.full_block_matrix[block_id])
        # A^{-1} = L^{-T} L^{-1}
        self.full_block_inverse[block_id] = inv_cholesky(L)
```

**预期收益**：30-50%（矩阵求逆阶段）

---

### 3.3 位操作优化

**问题位置**：`mas_preconditioner.py:273-291`

```python
# 当前：循环模拟
@ti.func
def _popcount(self, x: ti.u32) -> ti.i32:
    count = 0
    while x:
        count += ti.i32(x & 1)
        x >>= 1
    return count
```

**优化方案**：使用Taichi内置函数（1.7+）

```python
@ti.func
def _popcount(self, x: ti.u32) -> ti.i32:
    return ti.math.popcnt(x)

@ti.func
def _find_first_set(self, x: ti.u32) -> ti.i32:
    return ti.math.ctz(x) if x != 0 else -1
```

**预期收益**：10-20%（层级构建阶段）

---

### 3.4 数据布局优化

**问题位置**：`mas_preconditioner.py:198-203`

```python
# 当前：分离的r和z数组
self.multi_level_r = ti.Vector.field(3, dtype=ti.f64, shape=...)
self.multi_level_z = ti.Vector.field(3, dtype=ti.f64, shape=...)
```

**优化方案**：合并为Struct

```python
@ti.dataclass
class LevelData:
    r: ti.types.vector(3, ti.f64)
    z: ti.types.vector(3, ti.f64)

self.level_data = LevelData.field(shape=total_nodes)

# 访问时
data = self.level_data[idx]
r = data.r
z = data.z
```

**预期收益**：5-10%（预条件应用阶段）

---

### 3.5 局部求解优化

**问题位置**：`mas_preconditioner.py:1637-1684`

```python
# 当前：嵌套循环访问
for lane_i in range(BANKSIZE):
    for lane_j in range(BANKSIZE):
        sym_idx = ...  # 每次计算
        inv_block = self.inv_block_matrices[block_id, sym_idx]
```

**优化方案**：Tile-based访问

```python
@ti.kernel
def _schwarz_local_solve_tiled(self):
    for block_id in range(n_blocks):
        # 预加载residual
        r_local = ti.Matrix.zero(ti.f64, BANKSIZE, 3)
        for lane in range(BANKSIZE):
            idx = block_id * BANKSIZE + lane
            if idx < self.n_verts:
                r_local[lane, :] = self.multi_level_r[idx]

        # 密集矩阵乘法（更规则的访问模式）
        z_local = block_matvec(block_id, r_local)

        # 写回
        for lane in range(BANKSIZE):
            idx = block_id * BANKSIZE + lane
            if idx < self.n_verts:
                self.multi_level_z[idx] = z_local[lane, :]
```

**预期收益**：15-25%（局部求解阶段）

---

## 4. 优化优先级与计划

### 4.1 优先级排序

| 优先级 | 优化项 | 影响范围 | 预期收益 | 工作量 |
|--------|--------|----------|----------|--------|
| **P0** | 稀疏字典→紧凑数组 | 全局 | 30-50% | 大 |
| **P1** | 位操作内置函数 | MAS | 10-20% | 小 |
| **P2** | 原子操作优化 | MAS | 20-30% | 中 |
| **P3** | 数据布局合并 | MAS | 5-10% | 小 |
| **P4** | 局部求解tile优化 | MAS | 15-25% | 中 |
| **P5** | 块求逆CPU/Cholesky | MAS | 30-50% | 中-大 |

### 4.2 实施计划

#### Phase 1: 快速收益（1-2天）
- [ ] P1: 替换位操作为`ti.math.popcnt/ctz`
- [ ] P3: 合并`multi_level_r/z`为Struct field

#### Phase 2: 稀疏字典迁移（3-5天）
- [ ] P0-1: 在`collision_detection_bvh.py`添加紧凑数组存储
- [ ] P0-2: 添加`sync_to_compact_array()`方法
- [ ] P0-3: 迁移核心算法文件（pncg_base_ipc, mas_pncg_solver）
- [ ] P0-4: 迁移其他文件
- [ ] P0-5: 移除旧bitmasked结构

#### Phase 3: MAS深度优化（3-5天）
- [ ] P2: 实现`ti.block_local`弹性Hessian组装
- [ ] P4: 实现tile-based局部求解
- [ ] P5: 评估CPU批量求逆 vs GPU Cholesky

#### Phase 4: 验证与调优
- [ ] 性能基准测试（before/after对比）
- [ ] 数值精度验证
- [ ] 大规模网格测试

---

## 5. 稀疏字典问题的IPC模拟影响分析

### 5.1 IPC求解器典型调用链

每个时间步的典型流程：

```
时间步循环:
├── 碰撞检测 (find_cnts)
│   └── 写入 cid (bitmasked)
│
├── 牛顿迭代 (iter_max次)
│   ├── compute_E()           ← 遍历 cid 1次
│   ├── compute_grad_and_diagH() ← 遍历 cid 1次
│   ├── CG迭代 (cg_iter_max次)
│   │   └── compute_pHp() / compute_Hv()  ← 每次遍历 cid 1次
│   └── line_search()         ← 可能多次遍历 cid
│
└── 更新速度/位置
```

### 5.2 遍历次数估算

假设：
- 牛顿迭代次数：`iter_max = 10`
- CG迭代次数：`cg_iter_max = 50`
- Line search步数：平均 3 次

**每时间步遍历cid次数**：
```
= 1 (energy)
+ 1 (gradient)
+ 50 (CG Hv)
+ 3 (line search)
= ~55次/牛顿迭代

总计 = 55 * 10 = 550次/时间步
```

### 5.3 性能影响量化

假设场景：
- 顶点数：10,000
- 接触对数：5,000
- 时间步数：1,000

**当前bitmasked方式**：
- 每次遍历：扫描4MB bitmask + 5000次随机访存
- 总计：550 * 1000 = 55万次遍历
- 估计耗时占比：**15-25%** 总模拟时间

**紧凑数组方式**：
- 每次遍历：160KB连续读取（5000 * 32B）
- 可向量化，GPU利用率高
- 估计性能提升：**3-5x**（此部分）

### 5.4 为什么原始实现使用bitmasked

推测原因：
1. **去重需求**：碰撞检测可能多次检测到同一接触对，哈希自动去重
2. **动态大小**：不需要预分配精确大小
3. **移植便利**：从其他框架移植时的常见模式

**更好的替代方案**：
```python
# 使用原子计数器 + 紧凑数组
@ti.kernel
def add_contact(self, ...):
    # 检查是否重复（可选：用separate哈希表）
    idx = ti.atomic_add(self.n_contacts[None], 1)
    if idx < self.max_contacts:
        self.contact_pairs[idx] = pair
    else:
        print("Warning: contact overflow")

# 去重可以在CPU端进行（每帧一次），或使用并行去重算法
```

---

## 6. 参考资料

- Taichi官方文档：[Sparse Data Structures](https://docs.taichi-lang.org/docs/sparse)
- Taichi优化指南：[Performance Tuning](https://docs.taichi-lang.org/docs/performance)
- MAS论文：ref_doc/MAS_PNCG_clean.tex
- CUDA参考实现：/root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu

---

## 7. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-01-19 | 初始版本，完成问题分析和优化方案设计 |
