# MAS-PNCG 算法详解

本文档描述 MAS-PNCG（Multilevel Additive Schwarz Preconditioned Nonlinear Conjugate Gradient）框架的核心算法。

> **参考论文**: "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact"

---

## 1. 算法概述

MAS-PNCG 是一种用于求解 IPC（Incremental Potential Contact）优化问题的高效框架，结合了：

- **MAS 预条件器**：多级 Additive Schwarz 预条件，处理刚度矩阵的病态问题
- **2D 子空间最小化**：利用 Hessian 信息计算最优搜索方向
- **Sparse-Input Woodbury 更新**：高效更新 Level-0 子域逆矩阵
- **Powell 重启准则**：监控共轭性损失，决定何时完全重建预条件器
- **保守 CCD**：快速保守碰撞检测，确保无穿透轨迹

---

## 2. IPC 能量函数

IPC 将接触动力学转化为无约束优化问题：

$$
\mathbf{x}^{t+1} = \arg\min_{\mathbf{x}} E(\mathbf{x})
$$

能量函数包含四部分：

$$
E(\mathbf{x}) = \underbrace{\frac{1}{2}(\mathbf{x}-\tilde{\mathbf{x}})^\top \mathbf{M}(\mathbf{x}-\tilde{\mathbf{x}})}_{\text{惯性项}} + \underbrace{h^2 \Psi(\mathbf{x})}_{\text{弹性能}} + \underbrace{B(\mathbf{x})}_{\text{接触势垒}} + \underbrace{D(\mathbf{x})}_{\text{摩擦}}
$$

其中：
- $\tilde{\mathbf{x}} = \mathbf{x}^t + h\mathbf{v}^t + h^2\mathbf{M}^{-1}\mathbf{f}_{\text{ext}}$（预测位置）
- $B(\mathbf{x}) = \kappa \sum_{c \in \mathcal{C}} b(d_c(\mathbf{x}))$（接触势垒）

---

## 3. MAS 预条件器

### 3.1 Additive Schwarz (Level 0)

将计算域 $\Omega$ 分解为 $D$ 个子域 $\Omega_d$，Level-0 预条件器为：

$$
\mathbf{M}_{(0)}^{-1} = \sum_{d=1}^{D} \mathbf{S}_d^\top (\mathbf{M}_{(0)}^d)^{-1} \mathbf{S}_d
$$

其中：
- $\mathbf{S}_d$：选择矩阵，提取子域 $d$ 的自由度
- $\mathbf{M}_{(0)}^d = \mathbf{S}_d \mathbf{A} \mathbf{S}_d^\top$：子域局部 Hessian

### 3.2 多级扩展

MAS 预条件器扩展到多个层级：

$$
\mathbf{P} = \mathbf{M}_{(0)}^{-1} + \sum_{l=1}^{L} \mathbf{C}_{(l)}^\top \mathbf{M}_{(l)}^{-1} \mathbf{C}_{(l)}
$$

其中：
- $\mathbf{C}_{(l)}$：第 $l$ 层的粗化/限制矩阵（二值聚合）
- $\mathbf{M}_{(l)}^{-1}$：粗化系统 $\mathbf{A}_{(l)} = \mathbf{C}_{(l)} \mathbf{A} \mathbf{C}_{(l)}^\top$ 的 AS 预条件器
- $L$：层级总数（通常 ≤ 6）

---

## 4. Sparse-Input Woodbury 更新

### 4.1 动机

- 粗层级组件捕获低频误差模式，变化缓慢，可以冻结
- 严重病态主要来自接触势垒项，空间上稀疏
- 仅更新 Level-0 子域即可捕获局部变化

### 4.2 接触 Hessian 的 Gauss-Newton 近似

为保持 SPD 特性，使用 Gauss-Newton 近似：

$$
\nabla^2 b(d_c(\mathbf{x})) \approx k \mathbf{n} \mathbf{n}^\top
$$

其中：
- $k = b''(d_c)$：刚度标量
- $\mathbf{n} = \nabla d_c$：距离梯度方向

### 4.3 更新矩阵构建

Hessian 近似为：

$$
\mathbf{H}_{\text{new}} = \mathbf{H}_{\text{base}} + \mathbf{U} \mathbf{U}^\top
$$

其中 $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_m]$ 收集 rank-1 贡献。

**接触分类策略**：

| 情况 | 条件 | 更新向量 | 显著性 |
|------|------|----------|--------|
| **新接触** | $c \notin \mathcal{C}_{\text{base}}$ | $\mathbf{u} = \sqrt{k_{\text{curr}}} \cdot \mathbf{n}_{\text{curr}}$ | $\Delta S = k_{\text{curr}}$ |
| **法向旋转** | $\mathbf{n}_{\text{curr}} \cdot \mathbf{n}_{\text{base}} < \epsilon_{\text{rot}}$ | 视为新接触 | $\Delta S = k_{\text{curr}}$ |
| **刚度增加** | $k_{\text{curr}} > k_{\text{base}}$ | $\mathbf{u} = \sqrt{k_{\text{curr}} - k_{\text{base}}} \cdot \mathbf{n}_{\text{curr}}$ | $\Delta S = k_{\text{curr}} - k_{\text{base}}$ |
| **刚度减少** | $k_{\text{curr}} \le k_{\text{base}}$ | 忽略（保留更刚的基准近似） | $\Delta S = 0$ |

**Top-K 截断**：每个子域只保留 $K$ 个最显著的更新向量（如 $K=8$）。

### 4.4 Woodbury 公式

更新后的子域逆矩阵：

$$
\hat{\mathbf{B}}_d = \mathbf{B}_d - \mathbf{B}_d \mathbf{U}_d \left( \mathbf{I}_K + \mathbf{U}_d^\top \mathbf{B}_d \mathbf{U}_d \right)^{-1} \mathbf{U}_d^\top \mathbf{B}_d
$$

**求解步骤**：
1. **基准求解**：$\mathbf{z}_{\text{base}} = \mathbf{B}_d \mathbf{g}_d$
2. **更新投影**：$\mathbf{W}_d = \mathbf{B}_d \mathbf{U}_d$，$\mathbf{r} = \mathbf{U}_d^\top \mathbf{z}_{\text{base}}$
3. **电容矩阵求解**：$(\mathbf{I}_K + \mathbf{U}_d^\top \mathbf{W}_d) \boldsymbol{\lambda} = \mathbf{r}$
4. **修正**：$\mathbf{z}_d = \mathbf{z}_{\text{base}} - \mathbf{W}_d \boldsymbol{\lambda}$

---

## 5. 2D 子空间最小化

### 5.1 搜索方向参数化

在 2D 子空间中搜索最优方向：

$$
\mathbf{p}_{k+1}(\mu, \nu) = -\mu \mathbf{z}_{k+1} + \nu \mathbf{p}_k
$$

其中：
- $\mathbf{z}_{k+1} = \mathbf{P}_{k+1} \mathbf{g}_{k+1}$：预条件梯度
- $\mathbf{p}_k$：上一步搜索方向

### 5.2 最优系数求解

最小化局部二次模型 $Q(\mathbf{p}_{k+1}) = \mathbf{g}_{k+1}^\top \mathbf{p}_{k+1} + \frac{1}{2} \mathbf{p}_{k+1}^\top \tilde{\mathbf{H}} \mathbf{p}_{k+1}$，得到 2×2 系统：

$$
\begin{bmatrix}
\mathbf{z}_{k+1}^\top \tilde{\mathbf{H}} \mathbf{z}_{k+1} & -\mathbf{z}_{k+1}^\top \tilde{\mathbf{H}} \mathbf{p}_k \\
-\mathbf{p}_k^\top \tilde{\mathbf{H}} \mathbf{z}_{k+1} & \mathbf{p}_k^\top \tilde{\mathbf{H}} \mathbf{p}_k
\end{bmatrix}
\begin{bmatrix} \mu \\ \nu \end{bmatrix}
=
\begin{bmatrix} \mathbf{z}_{k+1}^\top \mathbf{g}_{k+1} \\ -\mathbf{p}_k^\top \mathbf{g}_{k+1} \end{bmatrix}
$$

### 5.3 高效计算

维护 $\mathbf{w}_k = \tilde{\mathbf{H}} \mathbf{p}_k$，则只需计算：
- $\mathbf{v} = \tilde{\mathbf{H}} \mathbf{z}_{k+1}$（一次 Hessian-vector 乘积）
- 系统矩阵：$A_{11} = \mathbf{z}_{k+1}^\top \mathbf{v}$，$A_{12} = -\mathbf{p}_k^\top \mathbf{v}$，$A_{22} = \mathbf{p}_k^\top \mathbf{w}_k$

更新：$\mathbf{w}_{k+1} = -\mu \mathbf{v} + \nu \mathbf{w}_k$

### 5.4 优势

- 无需启发式 $\beta$ 公式（如 Polak-Ribière、Fletcher-Reeves）
- $\mu$ 作为"自然步长"，初始试探步长可设为 $\alpha = 1.0$
- 只需通过 CCD 夹紧以防止穿透

---

## 6. Powell 重启准则

### 6.1 正交性监控

监控当前梯度与上一步预条件梯度的正交性损失：

$$
r_k = \frac{|\mathbf{g}_{k+1}^\top \mathbf{z}_k|}{\mathbf{g}_{k+1}^\top \mathbf{z}_{k+1}}
$$

### 6.2 重启条件

当 $r_k > \delta$（如 $\delta = 0.3$）时触发重启：
- 完全重建 MAS 预条件器
- 重置搜索方向为 $\mathbf{p}_{k+1} = -\mathbf{z}_{k+1}$

### 6.3 物理意义

判断梯度方向在预条件器消除病态后是否仍与前一步线性相关。这比在原始空间判断更准确。

---

## 7. 保守 CCD (Conservative CCD)

### 7.1 核心思想

- 不追求精确 TOI（碰撞时间），而是快速保守夹紧到安全区域
- 新检测到的接触将在下一迭代通过 Woodbury 更新注入
- **按子域独立计算**，避免"数值锁定"

### 7.2 算法

对于子域 $d$ 中的每个接触 $c$：

```
α_d ← 1.0
for each contact c in C_d:
    α_mon ← GetFirstCriticalPoint(f'_c(α))  # 单调区域边界
    α ← min(1.0, α_mon, α_l)
    while sign(f_c(0)) ≠ sign(f_c(α)) and α > α_l:
        α ← α / 2  # 二分夹紧
    α_d ← min(α_d, α)
return α_d
```

### 7.3 位置更新

按子域应用不同步长：

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \sum_{d=1}^{D} \mathbf{S}_d^\top \alpha_d \mathbf{S}_d \mathbf{p}_{k+1}
$$

---

## 8. 改进的步长下界

### 8.1 标准 ACCD 下界

$$
l_{\text{original}} = \max\|\mathbf{p}_P\| + \max\|\mathbf{p}_{T_j}\|
$$

### 8.2 改进的帧不变下界

**点-三角形 (PT)**：
$$
l_{\text{tight}}^{PT} = \max_{j \in \{0,1,2\}} \|\mathbf{p}_P - \mathbf{p}_{T_j}\|
$$

**边-边 (EE)**：
$$
l_{\text{tight}}^{EE} = \max_{i,j \in \{a,b\}} \|\mathbf{p}_{E1i} - \mathbf{p}_{E2j}\|
$$

**优势**：帧不变（不随全局平移改变），当图元运动相似时更紧。

---

## 9. 完整算法流程

```
输入: x^t, v^t, M, d̂, ε, δ
输出: x^{t+1}

x_0 ← x^t
x̃ ← x^t + h·v^t + h²·M⁻¹·f_ext
Restart ← True

for k = 0 to IterMax:
    C ← ComputeConstraintSet(x_k, d̂)

    if Restart:
        P_base, H̃_base ← RebuildMASPreconditioner(x_k, C)
        P_{k+1} ← P_base, H̃ ← H̃_base
    else:
        P_{k+1} ← SparseInputWoodburyUpdate(P_base, C)

    g_{k+1} ← ∇E(x_k)
    z_{k+1} ← P_{k+1} · g_{k+1}
    v ← H̃ · z_{k+1}

    if Restart:
        μ ← (z_{k+1}ᵀ·g_{k+1}) / (z_{k+1}ᵀ·v)
        ν ← 0
    else:
        求解 2×2 系统得 (μ, ν)

    p_{k+1} ← -μ·z_{k+1} + ν·p_k
    w_{k+1} ← -μ·v + ν·w_k

    {α_d} ← ConservativeCCD(x_k, p_{k+1})
    x_{k+1} ← x_k + Σ_d S_dᵀ·α_d·S_d·p_{k+1}

    if ‖α·p_{k+1}‖ ≤ ε:
        break
    else:
        r_k ← |g_{k+1}ᵀ·z_k| / (g_{k+1}ᵀ·z_{k+1})
        Restart ← (r_k > δ)

    z_k ← z_{k+1}

return x_{k+1}
```

---

## 10. 关键参数

| 参数 | 典型值 | 描述 |
|------|--------|------|
| BANKSIZE | 16 | 每个子域的节点数 |
| MAX_LEVELS | 6 | 最大层级深度 |
| $\delta$ | 0.3 | Powell 重启阈值 |
| $K$ | 8 | Top-K Woodbury 更新数 |
| $\epsilon_{\text{rot}}$ | cos(25°) | 法向旋转阈值 |
| $\alpha_l$ | 1e-6 | CCD 最小步长 |

---

## 参考文献

1. MAS-PNCG 论文: `docs/papers/MAS_PNCG_clean.tex`
2. 补充材料: `docs/papers/supplementary.tex`
3. MAS 预条件器实现: `docs/algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md`
