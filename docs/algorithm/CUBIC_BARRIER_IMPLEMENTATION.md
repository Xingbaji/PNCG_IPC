# Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness

## Reference

**Paper**: "A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness" (Ryoichi Ando, SIGGRAPH 2024)

**Reference Implementation**: `/cpfs/user/shenxing/ppf-contact-solver` (Rust + CUDA)

---

## 1. Core Concepts

### 1.1 The Problem with Logarithmic Barriers

The standard IPC uses a **logarithmic barrier**:

$$\psi_{\text{ln}}(g) = -\kappa (g - \hat{g})^2 \ln(g/\hat{g}), \quad \text{if } g \leq \hat{g}$$

**Problem**: When minimized with Newton's solver, the search direction obtained by dividing force by curvature yields an **extremely small magnitude near gap zero** (search direction locking).

### 1.2 The Cubic Barrier Solution

The cubic barrier replaces the logarithmic barrier with:

$$\psi_{\text{weak}}(g_i, \hat{g}_i, \kappa_i) = \begin{cases} -\frac{2\kappa_i}{3\hat{g}_i}(g_i - \hat{g}_i)^3, & \text{if } g \leq \hat{g} \\ 0, & \text{otherwise} \end{cases}$$

**Properties**:
- **C² continuous** at $g_i = \hat{g}_i$
- Search directions are **not too small or too large** near the critical point
- Does NOT have the property of infinite energy at gap zero (addressed by dynamic stiffness)

### 1.3 Derivatives

**Gradient** (first derivative):
$$\frac{\partial \psi}{\partial g} = -\frac{2\kappa}{\hat{g}}(g - \hat{g})^2$$

**Curvature** (second derivative, Hessian):
$$\frac{\partial^2 \psi}{\partial g^2} = 4\kappa \left(1 - \frac{g}{\hat{g}}\right)$$

Note: The barrier functions in the paper do NOT include $\kappa$ - the stiffness is applied as a multiplier when computing forces/Hessians.

---

## 2. Elasticity-Inclusive Dynamic Stiffness

### 2.1 Design Principle

The key innovation is treating $\kappa_i(\mathbf{x})$ as a function that approaches infinity as $g_i \to 0$, but evaluating it **semi-implicitly** (treating as constant when computing derivatives).

**Stiffness formula** (Equation 4 in paper):

$$\bar{\kappa} = \frac{m}{g^2} + \mathbf{n} \cdot (H\mathbf{n})$$

Where:
- $m$: average vertex mass of the contact
- $g$: gap distance
- $\mathbf{n}$: unit contact normal direction
- $H$: elasticity Hessian (enforced SPD)

### 2.2 Physical Interpretation

1. **Inertial term** $m/g^2$: Ensures barrier becomes infinitely stiff as gap approaches zero
2. **Elasticity term** $\mathbf{n} \cdot (H\mathbf{n})$: Includes elasticity to resist incoming forces

The idea is that by assigning stiffness in this form, the barrier would be stiff enough to resist incoming forces. If the stiffness turns out to be weak, the $m/g^2$ term ensures the barrier eventually becomes strong enough as the gap shrinks.

---

## 3. Reference Implementation Details

### 3.1 Barrier Functions (cubic.hpp)

```cpp
// Energy: ψ = -2κ/(3ĝ) * (g - ĝ)³
__device__ static float energy(float g, float ghat, float offset) {
    g -= offset;
    float y = g - ghat;
    if (y < 0.0f) {
        return -2.0f * (y * y * y) / (3.0f * ghat);
    } else {
        return 0.0f;
    }
}

// Gradient: ∂ψ/∂g = -2κ/ĝ * (g - ĝ)²
__device__ static float gradient(float g, float ghat, float offset) {
    g -= offset;
    float y = g - ghat;
    if (y < 0.0f) {
        return -2.0f * y * y / ghat;
    } else {
        return 0.0f;
    }
}

// Curvature: ∂²ψ/∂g² = 4κ * (1 - g/ĝ)
__device__ static float curvature(float g, float ghat, float offset) {
    g -= offset;
    if (g - ghat < 0.0f) {
        return 4.0f * (1.0f - g / ghat);
    } else {
        return 0.0f;
    }
}
```

**IMPORTANT**: Note that `κ` is NOT included in these functions! The stiffness is computed separately and multiplied when computing forces/Hessians.

### 3.2 Stiffness Computation (barrier.cu)

```cpp
template <unsigned N>
__device__ float compute_stiffness(
    const Proximity<N> &prox,
    const SVecf<N> &mass,
    const FixedCSRMat &hess,   // Full elasticity Hessian matrix!
    const Vec3f &e,
    float ghat,
    float offset,
    const ParamSet &param)
{
    SMatf<N * 3, N * 3> local_hess = SMatf<N * 3, N * 3>::Zero();
    float norm = e.norm();
    float g = norm - offset;
    float sqr_x = g * g;

    // Build local Hessian block for the contact vertices
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            Mat3x3f val = Mat3x3f::Zero();
            val += hess(prox.index[ii], prox.index[jj]);  // Elasticity Hessian block
            if (ii == jj) {
                val += (mass[ii] / sqr_x) * Mat3x3f::Identity();  // m/g² term
            }
            local_hess.template block<3, 3>(3 * ii, 3 * jj) = val;
        }
    }

    // Extended contact direction: w_i = coord_i * e
    SVecf<N * 3> w = SVecf<N * 3>::Zero();
    for (unsigned ii = 0; ii < N; ++ii) {
        float val = prox.value[ii];  // Barycentric coordinate
        Map<Vec3f>(w.data() + 3 * ii) = val * e;
    }
    w.normalize();

    // κ̄ = w · (H_local · w)
    return (local_hess * w).dot(w);
}
```

### 3.3 Force and Hessian Application (contact.cu)

```cpp
// Stiffness is computed from full Hessian
float stiff_k = barrier::compute_stiffness<N>(prox, mass, fixed_in, ex, ghat, offset, param);

// Force = κ̄ * barrier_gradient
Vec3f f = stiff_k * barrier::compute_edge_gradient(ex, ghat, offset, barrier);

// Hessian = κ̄ * barrier_hessian
Mat3x3f H = stiff_k * barrier::compute_edge_hessian(ex, ghat, offset, barrier);
```

---

## 4. Current Implementation Analysis

### 4.1 Current Code (pncg_base_ipc.py)

```python
def compute_adaptive_kappa(self, ids, cord, t, dist):
    # Compute average mass
    m_avg = 0.0
    for i in ti.static(range(4)):
        m_avg += self.mesh.verts.m[ids[i]] * ti.abs(cord[i])

    g_clamped = ti.max(dist, 1e-8)

    # First term: m/g²
    kappa_inertia = m_avg / (g_clamped * g_clamped)

    # Second term: n·(H·n) using diagonal approximation
    kappa_elastic = 0.0
    w_norm_sq = 0.0
    for i in ti.static(range(4)):
        w_i = cord[i] * t
        diagH_i = self.mesh.verts.diagH[ids[i]]
        kappa_elastic += w_i[0] * diagH_i[0] * w_i[0] + \
                        w_i[1] * diagH_i[1] * w_i[1] + \
                        w_i[2] * diagH_i[2] * w_i[2]
        w_norm_sq += w_i.norm_sqr()

    if w_norm_sq > 1e-10:
        kappa_elastic = kappa_elastic / w_norm_sq

    return kappa_inertia + ti.max(kappa_elastic, 0.0)
```

### 4.2 Problem: Why is kappa_max Not Large Enough?

**Issue 1: Current implementation uses fixed `self.kappa` instead of dynamic stiffness**

In `cubic_barrier_n_E_demo.py`, the barrier functions use `self.kappa` (a fixed constant):

```python
@ti.func
def barrier_H(self, d):
    H = 0.0
    if d < self.dHat:
        H = 4.0 * self.kappa * (1.0 - d / self.dHat)  # Fixed kappa!
    return H
```

The demo configuration sets `kappa = 0.001`, so:
- When `d → 0`: `max_kappa = 4 × 0.001 = 0.004`

**This is the root cause!** The implementation uses a **fixed kappa** rather than the **elasticity-inclusive dynamic stiffness**.

**Issue 2: The `compute_adaptive_kappa` function exists but is only used when `adaptive_kappa=True`**

The dynamic stiffness computation is implemented but gated behind `adaptive_kappa` flag, which is `False` by default.

---

## 5. Correct Implementation

### 5.1 How Barrier Should Work

According to the paper and reference implementation:

1. **Barrier functions should NOT include κ** - they should return values without stiffness multiplied
2. **Stiffness κ̄ should be computed dynamically** using Equation 4: $\bar{\kappa} = m/g^2 + \mathbf{n} \cdot (H\mathbf{n})$
3. **Forces and Hessians should be**: `f = κ̄ × barrier_gradient`, `H = κ̄ × barrier_hessian`

### 5.2 Expected Effective Kappa Values

With proper dynamic stiffness:

For a vertex with mass `m ≈ 0.05 kg` (typical for cloth) and gap `g = 0.001 m`:
$$\kappa_{\text{inertia}} = \frac{m}{g^2} = \frac{0.05}{0.000001} = 50000$$

Plus the elasticity term, the effective stiffness can be **orders of magnitude larger** than the fixed `κ = 0.001`.

### 5.3 Reference: Strain Limiting Stiffness (Equation 9)

$$\bar{\kappa}_{\text{SL}} = \frac{m_{\text{face}}}{(1 + \tau + \hat{\varepsilon} - \max(\sigma_1, \sigma_2))^2} + \mathbf{w}_r \cdot (H_{9\times9} \mathbf{w}_r)$$

---

## 6. Summary of Key Differences

| Aspect | Paper/Reference | Current Implementation |
|--------|-----------------|------------------------|
| Barrier κ | Dynamic: $m/g^2 + \mathbf{n}·(H\mathbf{n})$ | Fixed constant (0.001) |
| Max effective kappa | Can be $10^4$ - $10^6$ | 0.004 |
| Elasticity Hessian | Full matrix blocks | Diagonal approximation |
| κ in barrier function | Not included | Included directly |

---

## 7. Recommendations

1. **Enable `adaptive_kappa=True`** in demo configurations to use dynamic stiffness
2. **Modify barrier functions** to not include κ, and multiply by dynamic κ when computing forces/Hessians
3. **Consider using full Hessian blocks** instead of diagonal approximation for more accurate stiffness estimation
4. **Verify stiffness magnitude**: Dynamic κ should be on the order of $10^3$ - $10^6$ for tight contacts, not $10^{-3}$
