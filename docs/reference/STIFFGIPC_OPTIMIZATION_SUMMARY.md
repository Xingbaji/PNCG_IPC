# Stiff-GIPC 关键优化实现总结

本文档总结了 `/root/Stiff-GIPC_init/StiffGIPC/` 中三个关键的 GPU 优化技术，用于指导 Taichi 复现。

---

## 1. Warp Reduction 优化

### 1.1 核心思想

CUDA 中的 Warp Reduction 利用 GPU 硬件的 32 线程 warp 并行性，使用 `__shfl_down_sync` 指令在 warp 内部进行无冲突的并行归约。

### 1.2 CUDA 实现关键代码

**文件:** `MASPreconditioner.cu` (Lines 930-948)

```cpp
// 边界检测：lane 0 或 warp 边界
bool bBoundary = (lane_id == 0) || (warpId == 0);

// 创建边界位掩码
unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
mark = __brev(mark);  // 反转位顺序
int clzlen = __clz(mark << (warpId + 1));  // 计算前导零
unsigned int interval = std::min(clzlen, 31 - warpId);  // 归约区间

// Warp 级归约：指数步长
for (int iter = 1; iter < maxSize; iter <<= 1) {
    float tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
    if (interval >= iter) {
        rdata += tmpx;
    }
}
```

### 1.3 关键算法步骤

1. **边界检测**: 使用 `__ballot_sync` 检测分段边界
2. **区间计算**: `__brev` + `__clz` 计算每个线程的归约区间大小
3. **并行归约**: `__shfl_down_sync` 在区间内进行 O(log N) 归约
4. **结果写回**: 只有段首线程写回归约结果

### 1.4 Taichi 复现策略

由于 Taichi 1.7.4 不支持 `ti.simt` warp 原语，采用以下替代方案：

```python
# 方案1: Field-based tree reduction (当前实现)
@ti.kernel
def tree_reduce(warp_sum_buffer: ti.template(), n_warps: ti.i32):
    for warp_id in range(n_warps):
        # Step 0: stride = 8
        for lane in range(8):
            warp_sum_buffer[warp_id, lane] += warp_sum_buffer[warp_id, lane + 8]
        # Step 1: stride = 4
        for lane in range(4):
            warp_sum_buffer[warp_id, lane] += warp_sum_buffer[warp_id, lane + 4]
        # ... 继续到 stride = 1

# 方案2: 共享内存模拟 (如果支持)
# ti.simt.block.SharedArray (Taichi 1.8+)
```

---

## 2. 连通性哈希编码

### 2.1 核心思想

使用 32 位无符号整数作为位掩码，编码 BANKSIZE=16 节点的子域内连通性。每个 bit 位置代表一个节点。

### 2.2 CUDA 实现关键代码

**文件:** `MASPreconditioner.cu` (Lines 29-61)

```cpp
// 构建 Level 0 连通性掩码
__global__ void _buildCML0(...) {
    int warpId = idx / BANKSIZE;
    int laneId = idx % BANKSIZE;

    // 初始化：自连接
    unsigned int connectMsk = (1U << laneId);

    // 遍历邻居
    for (int i = 0; i < numNeighbor; i++) {
        int vIdConnected = _neighborList[startId + i];
        int warpIdxConnected = vIdConnected / BANKSIZE;

        if (warpId == warpIdxConnected) {
            // 同一 warp 内：设置连通位
            unsigned int laneIdxConnected = vIdConnected % BANKSIZE;
            connectMsk |= (1U << laneIdxConnected);
        }
    }
    _fineConnectedMsk[idx] = connectMsk;
}
```

**传递闭包计算 (Lines 128-139):**

```cpp
// 迭代扩展连通性
unsigned int visited = (1U << laneId);
while (connectMsk != -1) {  // -1 = 0xFFFFFFFF (全连通)
    unsigned int todo = visited ^ connectMsk;
    if (!todo) break;

    unsigned int nextVist = __ffs(todo) - 1;  // 找到第一个未访问的连通节点
    visited |= (1U << nextVist);
    connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  // 合并连通性
}
```

### 2.3 关键位操作

| 操作 | CUDA | Taichi | 说明 |
|------|------|--------|------|
| 设置位 | `mask \|= (1U << bit)` | `mask \|= ti.u32(1) << bit` | 标记连通 |
| 统计位数 | `__popc(mask)` | `_popcount(mask)` | 统计连通节点数 |
| 首位索引 | `__ffs(mask) - 1` | `_find_first_set(mask)` | 找到第一个1的位置 |
| 前缀掩码 | `(1U << lane) - 1` | `_lanemask_lt(lane)` | lane 之前的掩码 |
| 位反转 | `__brev(mask)` | 手动实现 | 用于区间计算 |
| 前导零 | `__clz(mask)` | 手动实现 | 用于区间计算 |

### 2.4 对称存储哈希编码

**文件:** `MASPreconditioner.cu` (Lines 1030-1042)

```cpp
// 从线性哈希索引反推 (row, col)
__device__ void get_index(int& row, int& col, const int& hash, const int& size) {
    for (row = 0; row < size; row++) {
        col = hash - size * row + row * (row + 1) / 2;
        if (col >= 0 && col < size) {
            if (size * row - row * (row + 1) / 2 + col == hash)
                return;
        }
    }
}

// 正向公式：(row, col) -> hash (row <= col)
// hash = BANKSIZE * row - row * (row + 1) / 2 + col
```

**存储大小:** 16×16 对称矩阵 → 16×17/2 = 136 个 3×3 子矩阵

---

## 3. SRBK SpMV (Symmetric Reduce-By-Key Sparse Matrix-Vector)

### 3.1 核心思想

对于对称稀疏矩阵 A，计算 y = A * x 时：
1. 只存储上三角元素 (row <= col)
2. 按行索引排序后，使用 warp 级分段归约减少原子操作
3. 同时处理下三角贡献 (对称性)

### 3.2 CUDA 实现关键代码

**文件:** `linear_system/utils/spmv.cu` (Lines 52-131)

```cpp
__device__ void warp_reduce_sym_spmv(...) {
    using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;

    // 获取当前和前一个行索引
    int prev_i = (global_thread_id > 0) ? rows[global_thread_id - 1] : -1;
    int i = rows[global_thread_id];
    int j = cols[global_thread_id];
    auto block_value = Mats3[global_thread_id];

    // 主 SpMV: y[i] += A[i,j] * x[j]
    Vector3 vec = block_value * x.segment<3>(j * 3);

    // 对称处理下三角: y[j] += A[j,i] * x[i] = A[i,j]^T * x[i]
    if (i != j) {
        Vector3 vec_ = a * block_value.transpose() * x.segment<3>(i * 3);
        y.segment<3>(j * 3).atomic_add(vec_);
    }

    // 分段边界检测
    char flags = ((lane_id == 0) || (prev_i != i)) ? 1 : 0;

    // CUB 分段归约 (每个分量独立)
    vec.x() = WarpReduceFloat(temp_storage[warp_id])
        .HeadSegmentedReduce(vec.x(), flags, [](Float a, Float b) { return a + b; });
    vec.y() = WarpReduceFloat(...)...;
    vec.z() = WarpReduceFloat(...)...;

    // 段首写回 (原子，因为可能跨 warp)
    if (flags) {
        y.segment<3>(i * 3).atomic_add((a * vec).eval());
    }
}
```

### 3.3 关键优化点

1. **排序**: 三元组按 (row, col) 排序，使同行元素连续
2. **分段检测**: 使用 `flags` 标记新行开始
3. **Warp 归约**: CUB 的 `HeadSegmentedReduce` 在 warp 内无冲突归约
4. **对称处理**: 上三角同时计算下三角贡献
5. **原子操作最小化**: 只有段首需要原子操作

### 3.4 Taichi 复现策略

```python
@ti.kernel
def srbk_spmv(triplet_values: ti.template(),
              row_ids: ti.template(),
              col_ids: ti.template(),
              triplet_count: ti.i32,
              x: ti.template(),
              y: ti.template()):
    # 清零输出
    for i in range(y.shape[0]):
        y[i] = 0.0

    # 并行处理三元组
    for tid in range(triplet_count):
        i = row_ids[tid]
        j = col_ids[tid]
        mat = triplet_values[tid]

        # 主贡献 y[i] += A[i,j] * x[j]
        result = mat @ x[j]
        for d in ti.static(range(3)):
            ti.atomic_add(y[i][d], result[d])

        # 对称贡献 (下三角)
        if i != j:
            result_t = mat.transpose() @ x[i]
            for d in ti.static(range(3)):
                ti.atomic_add(y[j][d], result_t[d])
```

**优化版本 (模拟分段归约):**

```python
# 1. 预排序三元组
# 2. 计算每行的起始/结束索引
# 3. 每行并行归约后单次写入
```

---

## 4. 数据结构对照

| CUDA 结构 | Taichi 等价 | 说明 |
|-----------|-------------|------|
| `MasMatrixSymT` | `ti.Matrix.field(3,3,shape=(N,136))` | 对称块存储 |
| `itable` | `ti.Vector.field(6,shape=N)` | 层级索引表 |
| `_fineConnectedMsk` | `ti.field(ti.u32,shape=N)` | 连通性掩码 |
| `SharedArray` | `ti.simt.block.SharedArray` (1.8+) | 共享内存 |

---

## 5. 常量定义

```cpp
#define BANKSIZE 16          // 每子域节点数
#define DEFAULT_BLOCKSIZE 256  // CUDA 块大小
#define DEFAULT_WARPNUM 16    // 每块 warp 数
#define warp_size 32          // NVIDIA warp 大小
#define SYM_BLOCK_COUNT 136   // 16*(16+1)/2
```

---

## 6. 实现优先级

1. **高优先级**: 连通性哈希编码 - 基础数据结构
2. **中优先级**: SRBK SpMV - 求解器核心
3. **较低优先级**: Warp Reduction - 性能优化 (Taichi 原子操作已较优)

---

## 参考文件

- `/root/Stiff-GIPC_init/StiffGIPC/MASPreconditioner.cu` (2361 行)
- `/root/Stiff-GIPC_init/StiffGIPC/linear_system/utils/spmv.cu` (135 行)
- `/root/Stiff-GIPC_init/StiffGIPC/eigen_data.h` (206 行)
