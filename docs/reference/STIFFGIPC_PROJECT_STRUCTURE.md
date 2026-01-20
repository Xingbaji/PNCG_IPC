# Stiff-GIPC 项目结构概览

> 本文档描述 `/root/Stiff-GIPC_init/` 参考实现的完整项目结构。

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [核心模块](#3-核心模块)
4. [依赖关系](#4-依赖关系)
5. [构建系统](#5-构建系统)
6. [外部库](#6-外部库)

---

## 1. 项目概述

**Stiff-GIPC** (Stiff GPU Incremental Potential Contact) 是一个统一的 GPU IPC 仿真框架，发表于 ACM TOG 2025。

### 核心特性

- **CEMAS 预条件器**: 连接性增强的多级加性 Schwarz 预条件器
- **三次应变限制能量**: C²-连续，解析特征系统
- **两级归约 Hessian 装配**: 高效的哈希并行归约
- **对称 RBK SpMV**: 1.85× 加速对比 cuSPARSE BSR
- **ABD 耦合**: 刚体-可变形体混合仿真

### 作者

- Kemeng Huang, Xinyu Lu, Huancheng Lin, Taku Komura, Minchen Li

---

## 2. 目录结构

```
/root/Stiff-GIPC_init/
├── README.md                           # 项目说明
├── CMakeLists.txt                      # 顶层 CMake 配置
│
├── StiffGIPC/                          # 主仿真器 (C++/CUDA)
│   ├── CMakeLists.txt
│   │
│   ├── MASPreconditioner.cuh           # MAS 类声明 (103 行)
│   ├── MASPreconditioner.cu            # MAS 完整实现 (2,361 行)
│   │
│   ├── eigen_data.h                    # 核心数据结构 (206 行)
│   ├── device_fem_data.cuh             # FEM 设备数据
│   ├── type_define.h                   # 类型定义
│   │
│   ├── linear_system/                  # 线性系统框架
│   │   ├── linear_system/              # 全局系统
│   │   │   ├── i_preconditioner.h/cu   # 预条件器接口
│   │   │   ├── i_linear_system_solver.h/cu
│   │   │   ├── global_linear_system.h/cu
│   │   │   ├── global_matrix.h/cu
│   │   │   └── linear_subsystem.h/cu
│   │   │
│   │   ├── preconditioner/             # 预条件器实现
│   │   │   ├── fem_mas_preconditioner.h/cu  # FEM MAS
│   │   │   ├── abd_preconditioner.h/cu      # ABD
│   │   │   └── diag_preconditioner.h/cu     # 对角
│   │   │
│   │   ├── solver/                     # 求解器
│   │   │   └── pcg_solver.h/cu         # PCG 求解器
│   │   │
│   │   ├── subsystem/                  # 子系统
│   │   │   ├── fem_linear_subsystem.h/cu
│   │   │   └── abd_linear_subsystem.h/cu
│   │   │
│   │   └── utils/                      # 工具
│   │       ├── spmv.h/cu               # 稀疏矩阵向量积
│   │       └── converter.h/cu          # 格式转换
│   │
│   ├── abd_system/                     # Affine Body Dynamics
│   │   └── ...
│   │
│   ├── gipc/                           # IPC 接触
│   │   └── ...
│   │
│   ├── cuda_tools/                     # CUDA 工具
│   │   └── cuda_tools.h
│   │
│   └── muda/                           # GPU 抽象层 (fork)
│       └── ...
│
├── MeshProcess/                        # 网格处理工具
│   ├── CMakeLists.txt
│   ├── metis_partition/
│   │   └── src/
│   │       ├── main.cpp                # 分区工具入口
│   │       ├── mesh.h/cpp              # 网格数据结构
│   │       ├── metis_sort.h/cpp        # METIS 排序
│   │       └── node_edge_model.h/cpp   # 节点边模型
│   │
│   └── External/                       # 外部依赖
│       ├── METIS/                      # METIS 库 (fork)
│       └── GKlib/                      # METIS 依赖
│
└── Assets/                             # 资源文件
    ├── tetMesh/                        # 四面体网格
    ├── triMesh/                        # 三角形网格
    ├── sorted_mesh/                    # 预分区网格
    └── scene/                          # 场景配置
```

---

## 3. 核心模块

### 3.1 MAS Preconditioner

**文件**: `MASPreconditioner.cuh/cu`

| 功能 | 方法 |
|------|------|
| 初始化 | `initPreconditioner_Neighbor`, `initPreconditioner_Matrix` |
| 层级构建 | `BuildConnectMaskL0`, `BuildLevel1`, `ComputeNextLevel` |
| 矩阵操作 | `PrepareHessian_bcoo`, `__inverse6_P96x96` |
| 应用 | `preconditioning`, `SchwarzLocalXSym`, `CollectFinalZ` |
| 碰撞更新 | `BuildCollisionConnection` |

### 3.2 Linear System Framework

```
IPreconditioner (接口)
├── LocalPreconditioner (子系统级)
│   ├── MAS_Preconditioner (FEM)
│   ├── ABDPreconditioner (刚体)
│   └── DiagonalPreconditioner (对角)
└── GlobalPreconditioner (全局级)

IterativeSolver (接口)
└── PCGSolver (共轭梯度)

DiagonalSubsystem (子系统)
├── FEMLinearSubsystem (可变形体)
└── ABDLinearSubsystem (刚体)
```

### 3.3 GIPC 模块

- **IPC Barrier**: 接触势能和力
- **Collision Detection**: BVH 碰撞检测
- **Energy**: 弹性能量计算

---

## 4. 依赖关系

### 4.1 模块依赖图

```
MASPreconditioner
    ├── eigen_data.h          # 数据结构
    ├── device_fem_data.cuh   # FEM 数据
    ├── cuda_tools/           # CUDA 工具
    └── muda/                 # GPU 抽象

LinearSystem
    ├── MASPreconditioner
    ├── gipc/                 # IPC 模块
    └── muda/                 # GPU 抽象

MeshProcess
    ├── METIS/                # 图分区
    └── GKlib/                # METIS 依赖
```

### 4.2 头文件依赖

```cpp
// MASPreconditioner.cuh
#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <muda/ext/linear_system/bcoo_matrix_view.h>
#include "linear_system/linear_system/global_matrix.h"

// MASPreconditioner.cu
#include "MASPreconditioner.cuh"
#include "cuda_tools/cuda_tools.h"
#include <muda/launch/launch.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
```

---

## 5. 构建系统

### 5.1 CMake 配置

**顶层 CMakeLists.txt**:
```cmake
cmake_minimum_required(VERSION 3.18)
project(StiffGIPC CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(StiffGIPC)
add_subdirectory(MeshProcess)
```

**StiffGIPC/CMakeLists.txt**:
```cmake
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

add_executable(StiffGIPC
    MASPreconditioner.cu
    # ... 其他源文件
)

target_link_libraries(StiffGIPC
    CUDA::cusparse
    CUDA::cublas
    CUDA::cusolver
    Eigen3::Eigen
    tbb
)

target_compile_options(StiffGIPC PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
```

### 5.2 编译选项

| 选项 | 值 | 描述 |
|------|-----|------|
| `CMAKE_CUDA_ARCHITECTURES` | 75, 80, 86 | GPU 架构 |
| `--use_fast_math` | - | 快速数学 |
| `-lineinfo` | - | 调试信息 |
| Separable Compilation | ON | 分离编译 |

### 5.3 依赖安装

**Linux**:
```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev nlohmann-json3-dev
```

**Windows/Linux (vcpkg)**:
```bash
vcpkg install eigen3 freeglut glew nlohmann-json
```

---

## 6. 外部库

### 6.1 依赖列表

| 库 | 版本 | 用途 | 来源 |
|----|------|------|------|
| CUDA | >=11.0 | GPU 编程 | 系统安装 |
| Eigen3 | 3.4.0 | 矩阵计算 | vcpkg |
| FreeGLUT | 3.4.0 | 可视化 | vcpkg |
| GLEW | 2.2.0 | OpenGL | vcpkg |
| cuSPARSE | - | 稀疏矩阵 | CUDA |
| cuBLAS | - | 稠密线性代数 | CUDA |
| cuSOLVER | - | 分解求解 | CUDA |
| Thrust | - | 并行算法 | CUDA |
| TBB | - | CPU 并行 | 系统安装 |

### 6.2 Fork 库

这些库是作者 fork 的定制版本:

| 库 | 原始仓库 | Fork 仓库 |
|----|---------|-----------|
| muda | https://github.com/MuGdxy/muda | https://github.com/KemengHuang/muda |
| METIS | https://github.com/KarypisLab/METIS | https://github.com/KemengHuang/METIS |
| GKlib | https://github.com/KarypisLab/GKlib | https://github.com/KemengHuang/GKlib |

### 6.3 muda 库

**muda** 是一个现代 CUDA 抽象层，提供:

```cpp
// 并行 for
muda::ParallelFor().apply(N, [=] __device__(int i) { ... });

// 设备缓冲区
muda::DeviceBuffer<T> buffer;

// 线性系统上下文
muda::LinearSystemContext ctx;

// BCOO 矩阵视图
muda::CBCOOMatrixView<Float, 3> view;
```

---

## 相关文档

- [STIFFGIPC_MAS_REFERENCE.md](STIFFGIPC_MAS_REFERENCE.md) - MAS 预条件器详解
- [STIFFGIPC_IMPLEMENTATION_DETAILS.md](STIFFGIPC_IMPLEMENTATION_DETAILS.md) - 算法实现细节
- [STIFFGIPC_OPTIMIZATION_SUMMARY.md](STIFFGIPC_OPTIMIZATION_SUMMARY.md) - 优化总结
