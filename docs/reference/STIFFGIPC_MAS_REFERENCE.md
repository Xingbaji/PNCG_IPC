# Stiff-GIPC MAS Preconditioner 参考实现详解

> 本文档详细分析 `/root/Stiff-GIPC_init/` 中的 MAS Preconditioner CUDA 参考实现。

## 目录

1. [概述](#1-概述)
2. [项目结构](#2-项目结构)
3. [MASPreconditioner 类详解](#3-maspreconditioner-类详解)
4. [核心数据结构](#4-核心数据结构)
5. [层次构建算法](#5-层次构建算法)
6. [矩阵装配与求逆](#6-矩阵装配与求逆)
7. [预条件器应用](#7-预条件器应用)
8. [碰撞连接更新](#8-碰撞连接更新)
9. [线性系统框架](#9-线性系统框架)
10. [关键常量与参数](#10-关键常量与参数)
11. [与 Taichi 实现的对比](#11-与-taichi-实现的对比)

---

## 1. 概述

Stiff-GIPC 是一个 GPU 加速的 IPC 仿真框架，其核心创新之一是**连接性增强的多级加性 Schwarz (CEMAS) 预条件器**。

### 参考论文
- **StiffGIPC**: "Advancing GPU IPC for Stiff Affine-Deformable Simulation" (ACM TOG 2025)
- **作者**: Kemeng Huang, Xinyu Lu, Huancheng Lin, Taku Komura, Minchen Li

### 关键文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `MASPreconditioner.cuh` | 103 | 类声明和接口定义 |
| `MASPreconditioner.cu` | 2,361 | 完整 CUDA 实现 |
| `eigen_data.h` | 206 | 核心数据结构定义 |
| `fem_mas_preconditioner.h/cu` | 61 | 线性系统框架封装 |

---

## 2. 项目结构

```
/root/Stiff-GIPC_init/
├── StiffGIPC/                          # 主仿真器
│   ├── MASPreconditioner.cuh           # MAS 类声明
│   ├── MASPreconditioner.cu            # MAS 完整实现 (2361行)
│   ├── eigen_data.h                    # 数据结构定义
│   ├── linear_system/                  # 线性系统框架
│   │   ├── linear_system/              # 全局系统
│   │   │   ├── i_preconditioner.h      # 预条件器接口
│   │   │   ├── global_linear_system.h  # 全局线性系统
│   │   │   └── linear_subsystem.h      # 子系统抽象
│   │   ├── preconditioner/             # 预条件器实现
│   │   │   ├── fem_mas_preconditioner.h
│   │   │   ├── abd_preconditioner.h
│   │   │   └── diag_preconditioner.h
│   │   ├── solver/                     # 求解器
│   │   │   └── pcg_solver.h            # PCG 求解器
│   │   └── utils/                      # 工具函数
│   │       └── spmv.h                  # 稀疏矩阵向量积
│   └── muda/                           # GPU 抽象层
├── MeshProcess/                        # METIS 网格分区工具
│   └── metis_partition/
└── Assets/                             # 测试网格
```

---

## 3. MASPreconditioner 类详解

### 3.1 类声明 (`MASPreconditioner.cuh`)

```cpp
class MASPreconditioner
{
    // 基本配置
    int totalNodes;              // 总节点数
    int totalMapNodes;           // 映射后节点数（含填充）
    int levelnum;                // 层级数量
    int collision_node_Offset;   // 碰撞节点偏移
    int totalNumberClusters;     // 总簇数量
    int2 h_clevelSize;           // 当前层大小 (host)

    // 碰撞对
    int4* _collisonPairs;

    // 层级数据
    int2* d_levelSize;               // 各层大小
    int* d_coarseSpaceTables;        // 粗空间映射表
    int* d_prefixOriginal;           // 前缀和原始值
    int* d_prefixSumOriginal;        // 前缀和结果
    int* d_goingNext;                // 到下一层的映射
    int* d_denseLevel;               // 稠密层标记
    __GEIGEN__::itable* d_coarseTable;  // 粗化路径表

    // 连接掩码
    unsigned int* d_fineConnectMask;    // 细层连接掩码
    unsigned int* d_nextConnectMask;    // 下一层连接掩码
    unsigned int* d_nextPrefix;         // 下一层前缀
    unsigned int* d_nextPrefixSum;      // 下一层前缀和

    // 矩阵存储
    __GEIGEN__::MasMatrixT* d_MatMas;           // 全矩阵存储
    __GEIGEN__::MasMatrixSymT* d_inverseMatMas; // 对称存储 (double)
    __GEIGEN__::MasMatrixSymf* d_precondMatMas; // 求逆后 (float)
    Eigen::Vector3f* d_multiLevelR;             // 多级残差
    Precision_T3* d_multiLevelZ;                // 多级解

public:
    // 邻接信息
    int neighborListSize;
    unsigned int* d_neighborList;
    unsigned int* d_neighborStart;
    unsigned int* d_neighborNum;
    unsigned int* d_neighborListInit;
    unsigned int* d_neighborNumInit;

    // 节点映射
    int* d_partId_map_real;   // 分区ID → 真实ID
    int* d_real_map_partId;   // 真实ID → 分区ID
};
```

### 3.2 主要公共方法

```cpp
// 初始化
void initPreconditioner_Neighbor(int vertNum, int collisionOffset,
                                  int totalNeighborNum, int4* collisionPairs,
                                  int partMapSize);
void initPreconditioner_Matrix();
void computeNumLevels(int vertNum);

// 运行时重排序
int ReorderRealtime(int cpNum);

// 层级构建
void BuildConnectMaskL0();
void PreparePrefixSumL0();
void BuildLevel1();
void BuildConnectMaskLx(int level);
void NextLevelCluster(int level);
void PrefixSumLx(int level);
void ComputeNextLevel(int level);
void AggregationKernel();
void BuildCollisionConnection(...);

// 矩阵操作
void PrepareHessian_bcoo(...);
void setPreconditioner_bcoo(...);

// 预条件器应用
void preconditioning(const double3* R, double3* Z);
void BuildMultiLevelR(const double3* R);
void SchwarzLocalXSym();
void SchwarzLocalXSym_block3();
void SchwarzLocalXSym_sym();
void CollectFinalZ(double3* Z);

// 清理
void FreeMAS();
```

---

## 4. 核心数据结构

### 4.1 常量定义 (`eigen_data.h`)

```cpp
#define BANKSIZE 16              // 每个子域的节点数
#define DEFAULT_BLOCKSIZE 256    // CUDA block 大小
#define DEFAULT_WARPNUM 16       // 每 block 的 warp 数

// 对称存储元素数: 16*(16+1)/2 = 136
#define SYM_BLOCK_COUNT (BANKSIZE * (BANKSIZE + 1) / 2)
```

### 4.2 矩阵存储结构

```cpp
namespace __GEIGEN__ {

// 完整 48×48 矩阵 (BANKSIZE*3 × BANKSIZE*3)
struct MasMatrixT {
    Precision_T m[BANKSIZE * 3][BANKSIZE * 3];  // 2304 个 double
};

// 对称存储 (用于装配) - 136 个 3×3 块
struct MasMatrixSymT {
    Eigen::Matrix3d M[BANKSIZE * (BANKSIZE + 1) / 2];  // 136 × 9 = 1224 个 double
};

// 对称存储 (用于应用) - 单精度
struct MasMatrixSymf {
    Eigen::Matrix3f M[BANKSIZE * (BANKSIZE + 1) / 2];  // 136 × 9 = 1224 个 float
};

// 层级路径表
struct itable {
    int index[6];  // 最多 6 层的路径
};

}
```

### 4.3 对称索引公式

```cpp
// 对于 row <= col (上三角):
int sym_idx(int row, int col) {
    int r = min(row, col);
    int c = max(row, col);
    return BANKSIZE * r - r * (r + 1) / 2 + c;
}

// 示例 (BANKSIZE=16):
// (0,0) → 0
// (0,1) → 1
// (0,15) → 15
// (1,1) → 16
// (15,15) → 135
```

---

## 5. 层次构建算法

### 5.1 整体流程 (`ReorderRealtime`)

```cpp
int MASPreconditioner::ReorderRealtime(int cpNum) {
    // 阶段 1: 构建 Level-0 连接掩码
    BuildConnectMaskL0();

    // 阶段 2: 计算传递闭包和前缀和
    PreparePrefixSumL0();

    // 阶段 3: 如有碰撞，添加碰撞连接
    if (cpNum)
        BuildCollisionConnection(d_fineConnectMask, nullptr, 0, cpNum);

    // 阶段 4: 构建 Level-1
    BuildLevel1();

    // 阶段 5: 逐层构建更高层级
    for (int level = 2; level <= levelnum; level++) {
        BuildConnectMaskLx(level);
        if (cpNum)
            BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum);
        NextLevelCluster(level);
        PrefixSumLx(level);
        ComputeNextLevel(level);
    }

    // 阶段 6: 聚合内核
    AggregationKernel();

    return totalNumberClusters;
}
```

### 5.2 Level-0 连接掩码构建

```cuda
__global__ void _buildCML0_new(
    const unsigned int* _neighborStart,
    unsigned int*       _neighborNum,
    unsigned int*       _neighborList,
    unsigned int*       _fineConnectedMsk,
    int*                _partId_map_real,
    int*                _real_map_partId,
    int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tdx >= number) return;

    int warpId = tdx / BANKSIZE;
    int laneId = tdx % BANKSIZE;
    int idx = _partId_map_real[tdx];

    if (idx >= 0) {
        int numNeighbor = _neighborNum[idx];
        unsigned int connectMsk = (1U << laneId);  // 自连接
        int nk = 0;
        int startId = _neighborStart[idx];

        for (int i = 0; i < numNeighbor; i++) {
            int vIdConnected = _neighborList[startId + i];
            int warpIdxConnected = _real_map_partId[vIdConnected] / BANKSIZE;

            if (warpId == warpIdxConnected) {
                // 同一 bank 内的邻居
                unsigned int laneIdxConnected = _real_map_partId[vIdConnected] % BANKSIZE;
                connectMsk |= (1U << laneIdxConnected);
            } else {
                // 保留跨 bank 的邻居供高层使用
                _neighborList[startId + nk] = vIdConnected;
                nk++;
            }
        }
        _neighborNum[idx] = nk;
        _fineConnectedMsk[idx] = connectMsk;
    }
}
```

### 5.3 传递闭包计算

```cuda
__global__ void _preparePrefixSumL0_new(
    int*          _prefixOriginal,
    unsigned int* _fineConnectedMsk,
    int*          _partId_map_real,
    int           vertNum)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tdx >= vertNum) return;

    int warpId = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId = tdx % BANKSIZE;
    int idx = _partId_map_real[tdx];

    __shared__ unsigned int cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int prefixSum[DEFAULT_WARPNUM];

    if (idx >= 0) {
        unsigned int connectMsk = _fineConnectedMsk[idx];

        if (laneId == 0)
            prefixSum[localWarpId] = 0;

        cacheMask[threadIdx.x] = connectMsk;
        unsigned int visited = (1U << laneId);

        // BFS 式位合并
        while (connectMsk != 0xFFFFFFFF) {
            unsigned int todo = visited ^ connectMsk;
            if (!todo) break;

            unsigned int nextVist = __ffs(todo) - 1;
            visited |= (1U << nextVist);
            connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];
        }

        _fineConnectedMsk[idx] = connectMsk;

        // 选举代表节点
        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
        if (electedPrefix == 0) {
            atomicAdd(prefixSum + localWarpId, 1);
        }

        if (laneId == 0)
            _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}
```

### 5.4 辅助函数

```cuda
__device__ unsigned int _LanemaskLt(int laneIdx) {
    return (1U << laneIdx) - 1;
}
```

---

## 6. 矩阵装配与求逆

### 6.1 Hessian 装配到多级结构

```cuda
// 在 PrepareHessian_bcoo 中
ParallelFor().apply(tripletNum,
    [=] __device__(int I) mutable {
        int index = indices[I];
        auto vertRid_real = row_ids[index] - offset;
        auto vertCid_real = col_ids[index] - offset;
        auto H = triplet_values[index];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid = vertCid / BANKSIZE;

        // 同一 bank 内：直接存储
        if (vertCid / BANKSIZE == vertRid / BANKSIZE) {
            if (vertCid >= vertRid) {
                int bvRid = vertRid % BANKSIZE;
                int bvCid = vertCid % BANKSIZE;
                int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
                _invMatrix[cPid].M[index] = H;
            }
        } else {
            // 跨 bank：累加到粗层
            int level = 0;
            while (level < levelNum - 1) {
                level++;
                vertCid = (level == 1) ? _goingNext[vertCid_real] : _goingNext[vertCid];
                vertRid = (level == 1) ? _goingNext[vertRid_real] : _goingNext[vertRid];

                if (vertCid / BANKSIZE == vertRid / BANKSIZE) {
                    // 在此层相遇
                    int bvRid = min(vertRid, vertCid) % BANKSIZE;
                    int bvCid = max(vertRid, vertCid) % BANKSIZE;
                    int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;

                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            atomicAdd(&(_invMatrix[cPid].M[index](i, j)), H(i, j));
                        }
                    }
                    break;
                }
            }
        }
    });
```

### 6.2 块矩阵求逆 (Gauss-Jordan)

```cuda
__global__ void __inverse6_P96x96(
    __GEIGEN__::MasMatrixSymf* _preMatrix,   // 输出 (float)
    __GEIGEN__::MasMatrixSymT* _invMatrix,   // 输入 (double)
    int numbers)
{
    __shared__ double sPMas[32/BANKSIZE][BANKSIZE*3][BANKSIZE*3];

    // 1. 展开对称存储到完整矩阵 (共享内存)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        if (colId >= rowId) {
            index = SYM_INDEX(rowId, colId);
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](j%3, i%3);
        } else {
            index = SYM_INDEX(colId, rowId);
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](i%3, j%3);  // 转置
        }
    }

    // 2. Gauss-Jordan 消元 (48 个主元)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        __syncthreads();
        double rt = sPMas[block_matId][j][j];
        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        if (i == j) sPMas[block_matId][i][j] = 1;
        else sPMas[block_matId][i][j] = 0;

        __syncthreads();
        sPMas[block_matId][j][i] /= rt;

        // 列消元
        for (int k = 0; k < BANKSIZE * 3; k++) {
            if (k != j) {
                double rate = -colm[block_matId][k];
                __syncthreads();
                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }
    }

    // 3. 存回对称格式 (单精度)
    for (int j = 0; j < BANKSIZE * 3; j++) {
        if (colId >= rowId) {
            _preMatrix[matId].M[SYM_INDEX(rowId, colId)](j%3, i%3) =
                (float)sPMas[block_matId][j][i];
        }
    }
}
```

---

## 7. 预条件器应用

### 7.1 应用流程

```cpp
void MASPreconditioner::preconditioning(const double3* R, double3* Z) {
    // 1. 构建多级残差 (限制)
    BuildMultiLevelR(R);

    // 2. 局部 Schwarz 求解
    SchwarzLocalXSym();  // 或 SchwarzLocalXSym_block3() / SchwarzLocalXSym_sym()

    // 3. 收集最终解 (延拓)
    CollectFinalZ(Z);
}
```

### 7.2 限制操作 (`BuildMultiLevelR`)

```cuda
__global__ void __buildMultiLevelR_optimized_new(
    const double3*     R,
    Eigen::Vector3f*   _multiLR,
    const int*         _goingNext,
    const int*         _prefixOriginal,
    const unsigned int* _fineConnectMsk,
    const int*         _partId_map_real,
    int                levelNum,
    int                number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tdx >= number) return;

    int warpId = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId = tdx % BANKSIZE;
    int idx = _partId_map_real[tdx];

    Eigen::Vector3f r;
    if (idx >= 0) {
        r[0] = R[idx].x;
        r[1] = R[idx].y;
        r[2] = R[idx].z;
    }

    _multiLR[tdx] = r;

    // 检查是否需要 warp shuffle 归约
    int prefixSum = _prefixOriginal[warpId];
    unsigned int connectMsk = (idx >= 0) ? _fineConnectMsk[idx] : 0;

    if (prefixSum == 1) {
        // 整个 bank 是一个簇：warp shuffle 归约
        auto mask_val = __activemask();
        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
        bool bBoundary = (laneId == 0) || (electedPrefix == 0);

        for (int iter = 1; iter < BANKSIZE; iter <<= 1) {
            float tmpx = __shfl_down_sync(mask_val, r[0], iter);
            float tmpy = __shfl_down_sync(mask_val, r[1], iter);
            float tmpz = __shfl_down_sync(mask_val, r[2], iter);
            r[0] += tmpx;
            r[1] += tmpy;
            r[2] += tmpz;
        }

        // 传播到粗层
        if (bBoundary) {
            int level = 0;
            while (level < levelNum - 1) {
                level++;
                idx = _goingNext[idx];
                atomicAdd(&(_multiLR[idx][0]), r[0]);
                atomicAdd(&(_multiLR[idx][1]), r[1]);
                atomicAdd(&(_multiLR[idx][2]), r[2]);
            }
        }
    }
    // ... 其他情况处理
}
```

### 7.3 Schwarz 局部求解

```cuda
__global__ void _schwarzLocalXSym3(
    const __GEIGEN__::MasMatrixSymf* Pred,
    const Eigen::Vector3f*           mR,
    Precision_T3*                    mZ,
    int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    int hessianSize = (BANKSIZE * 3) * BANKSIZE;
    int Hid = idx / hessianSize;
    int MRid = (idx % hessianSize) / BANKSIZE;
    int MCid = (idx % hessianSize) % BANKSIZE;

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;
    int r3id = MRid % 3;

    int lvrid = vrid % BANKSIZE;
    int lvcid = vcid % BANKSIZE;

    // 加载残差到共享内存
    __shared__ Eigen::Vector3f smR[BANKSIZE];
    if (threadIdx.x < BANKSIZE)
        smR[threadIdx.x] = mR[vcid];
    __syncthreads();

    // 矩阵-向量乘法 (处理对称性)
    FloatP rdata = 0;
    if (lvcid >= lvrid) {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata = Pred[Hid].M[index](r3id, 0) * smR[lvcid][0]
              + Pred[Hid].M[index](r3id, 1) * smR[lvcid][1]
              + Pred[Hid].M[index](r3id, 2) * smR[lvcid][2];
    } else {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata = Pred[Hid].M[index](0, r3id) * smR[lvcid][0]
              + Pred[Hid].M[index](1, r3id) * smR[lvcid][1]
              + Pred[Hid].M[index](2, r3id) * smR[lvcid][2];
    }

    // Warp 归约
    int warpId = threadIdx.x & 0x1f;
    int landidx = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
    mark = __brev(mark);
    int clzlen = __clz(mark << (warpId + 1));
    unsigned int interval = min(clzlen, 31 - warpId);

    for (int iter = 1; iter < min(32, BANKSIZE); iter <<= 1) {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if (interval >= iter)
            rdata += tmpx;
    }

    if (bBoundary)
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
}
```

### 7.4 延拓操作 (`CollectFinalZ`)

```cuda
__global__ void __collectFinalZ_new(
    double3*                  _Z,
    const Precision_T3*       d_multiLevelZ,
    const __GEIGEN__::itable* _coarseTable,
    int*                      _real_map_partId,
    int                       levelnum,
    int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    // 从分区空间读取 Level-0 解
    int rdx = _real_map_partId[idx];
    Precision_T3 cz;
    cz.x = d_multiLevelZ[rdx].x;
    cz.y = d_multiLevelZ[rdx].y;
    cz.z = d_multiLevelZ[rdx].z;

    // 累加所有粗层的解
    __GEIGEN__::itable table = _coarseTable[idx];
    for (int i = 1; i < levelnum; i++) {
        int now = table.index[i - 1];
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}
```

---

## 8. 碰撞连接更新

### 8.1 动态添加碰撞连接

```cuda
__global__ void _buildCollisionConnection_new(
    unsigned int*     _pConnect,
    const int*        _pCoarseSpaceTable,
    const int4*       _collisionPair,
    const int*        _real_map_partId,
    int               level,
    int               node_offset,
    int               vertNum,
    int               number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    int4 MMCVIDI = _collisionPair[idx];
    int* cpVidPtr = &(MMCVIDI.x);

    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w < 0)
            MMCVIDI.w = -MMCVIDI.w - 1;

        // 偏移碰撞顶点
        for (int i = 0; i < 4; i++)
            cpVidPtr[i] -= node_offset;

        // 获取分区 ID
        int cpVid[4];
        if (_pCoarseSpaceTable) {
            for (int i = 0; i < 4; i++)
                cpVid[i] = (cpVidPtr[i] >= 0) ?
                    _pCoarseSpaceTable[cpVidPtr[i] + (level-1) * vertNum] : -1;
        } else {
            for (int i = 0; i < 4; i++)
                cpVid[i] = (cpVidPtr[i] >= 0) ?
                    _real_map_partId[cpVidPtr[i]] : -1;
        }

        // 添加同 bank 内的连接
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if (myId == otId || myId < 0 || otId < 0)
                    continue;

                if (myId / BANKSIZE == otId / BANKSIZE) {
                    atomicOr(_pConnect + myId, (1U << (otId % BANKSIZE)));
                    atomicOr(_pConnect + otId, (1U << (myId % BANKSIZE)));
                }
            }
        }
    }
}
```

---

## 9. 线性系统框架

### 9.1 预条件器接口

```cpp
// i_preconditioner.h
class IPreconditioner {
public:
    virtual ~IPreconditioner();
    virtual Json as_json() const;

protected:
    muda::LinearSystemContext& ctx() const;

private:
    virtual void do_apply(muda::CDenseVectorView<Float> r,
                          muda::DenseVectorView<Float>  z) = 0;
    virtual void do_assemble(GIPCTripletMatrix& global_triplets) = 0;
};

class LocalPreconditioner : public IPreconditioner {
protected:
    virtual void assemble() {};
    virtual void apply(muda::CDenseVectorView<Float> r,
                       muda::DenseVectorView<Float> z) = 0;
};
```

### 9.2 MAS 预条件器封装

```cpp
// fem_mas_preconditioner.h
class MAS_Preconditioner : public LocalPreconditioner {
    MASPreconditioner& MAS_Prec;
    double* masses;
    uint32_t* cpNum;

public:
    MAS_Preconditioner(FEMLinearSubsystem& subsystem,
                       MASPreconditioner& mMAS,
                       double* mMasses,
                       uint32_t* mCpNum);

    virtual void assemble() override;
    virtual void apply(muda::CDenseVectorView<Float> r,
                       muda::DenseVectorView<Float> z) override;
};
```

### 9.3 PCG 求解器

```cpp
// pcg_solver.h
class PCGSolverConfig {
public:
    Float max_iter_ratio  = 0.3;    // 最大迭代 = DOF × 0.3
    Float global_tol_rate = 1e-4;   // 相对容差
    bool  use_bsr         = true;   // 使用 BSR 格式
};

class PCGSolver : public IterativeSolver {
    DeviceDenseVector z;    // 预条件后残差
    DeviceDenseVector r;    // 残差
    DeviceDenseVector p;    // 搜索方向
    DeviceDenseVector Ap;   // A*p

protected:
    SizeT solve(muda::DenseVectorView<Float> x,
                muda::CDenseVectorView<Float> b) override;

private:
    SizeT pcg(muda::DenseVectorView<Float> x,
              muda::CDenseVectorView<Float> b,
              SizeT max_iter);
};
```

---

## 10. 关键常量与参数

### 10.1 编译时常量

| 常量 | 值 | 描述 |
|------|-----|------|
| `BANKSIZE` | 16 | 每个子域的节点数 |
| `DEFAULT_BLOCKSIZE` | 256 | CUDA block 大小 |
| `DEFAULT_WARPNUM` | 16 | 每 block 的 warp 数 (256/16) |
| `SYM_BLOCK_COUNT` | 136 | 对称存储元素数 (16×17/2) |
| `MAX_LEVELS` | 6 | 最大层级深度 (itable.index[6]) |

### 10.2 运行时参数

| 参数 | 典型值 | 描述 |
|------|--------|------|
| `max_iter_ratio` | 0.3 | PCG 最大迭代比例 |
| `global_tol_rate` | 1e-4 | 相对收敛容差 |

### 10.3 内存布局

```
每个子域的矩阵存储:
- MasMatrixT: 48 × 48 × 8 bytes = 18,432 bytes (完整)
- MasMatrixSymT: 136 × 9 × 8 bytes = 9,792 bytes (对称 double)
- MasMatrixSymf: 136 × 9 × 4 bytes = 4,896 bytes (对称 float)
```

---

## 11. 与 Taichi 实现的对比

### 11.1 架构对比

| 方面 | CUDA 参考实现 | Taichi 实现 |
|------|--------------|-------------|
| 编程模型 | CUDA C++ | Taichi Python |
| 内存管理 | 显式 (cudaMalloc/cudaMemcpy) | 自动 (ti.field) |
| Warp 原语 | `__shfl_*`, `__ballot_sync` | `ti.simt.warp.*` |
| 原子操作 | `atomicAdd`, `atomicOr` | `ti.atomic_add` |
| 共享内存 | `__shared__` | `ti.simt.block.shared_array` |

### 11.2 关键差异

1. **节点重排序**
   - CUDA: METIS 离线分区 + 运行时碰撞更新
   - Taichi: 可选 METIS 或简单分区

2. **矩阵存储**
   - CUDA: `MasMatrixSymT/MasMatrixSymf` 自定义结构
   - Taichi: `ti.field` 或 `ti.types.matrix`

3. **并行模式**
   - CUDA: 显式 kernel launch + cooperative groups
   - Taichi: `@ti.kernel` 自动并行化

4. **精度控制**
   - CUDA: 装配 double → 应用 float
   - Taichi: 可配置 `ti.f32` 或 `ti.f64`

### 11.3 移植建议

1. **保持 BANKSIZE=16**：这是 GPU warp 优化的关键
2. **使用对称存储**：节省 ~47% 内存
3. **混合精度**：装配用 double，应用用 float
4. **层级路径表**：每节点存储 6 个索引的路径

---

## 参考

1. Stiff-GIPC 论文: ACM TOG 2025
2. CUDA 参考实现: `/root/Stiff-GIPC_init/StiffGIPC/`
3. METIS 库: https://github.com/KemengHuang/METIS
4. muda GPU 库: https://github.com/KemengHuang/muda
