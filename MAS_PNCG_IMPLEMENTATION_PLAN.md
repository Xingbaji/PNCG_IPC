# MAS-PNCG Implementation Plan

Based on the paper "An Efficient Multilevel Preconditioned Nonlinear Conjugate Gradient Framework for Incremental Potential Contact" and its supplementary document.

## Algorithm Overview (Algorithm 1 from Paper)

```
Input: x^t, v^t, M, d_hat, ε, δ (restart threshold)
Output: x^{t+1}

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
    v ← H̃ · z_{k+1}                    # Hessian-vector product

    if Restart:
        μ ← (z_{k+1}ᵀ·g_{k+1}) / (z_{k+1}ᵀ·v)
        ν ← 0
    else:
        Solve 2×2 system for (μ, ν)    # Eq. 6 in paper

    p_{k+1} ← -μ·z_{k+1} + ν·p_k
    w_{k+1} ← -μ·v + ν·w_k             # Maintains w = H̃·p

    {α_d} ← ConservativeCCD(x_k, p_{k+1})
    x_{k+1} ← x_k + Σ_d S_dᵀ·α_d·S_d·p_{k+1}

    if ‖α·p_{k+1}‖ ≤ ε:
        break
    else:
        r_k ← |g_{k+1}ᵀ·z_k| / (g_{k+1}ᵀ·z_{k+1})
        Restart ← (r_k > δ)            # Powell's criterion

    z_k ← z_{k+1}                       # Cache for restart judgment

return x_{k+1}
```

## Key Components to Implement

### 1. Optimal 2D Subspace Minimization (Section 3.2)

**Current status**: Not implemented. Currently using Dai-Kai CG formula.

**What it does**: Instead of using heuristic β formulas (Fletcher-Reeves, Polak-Ribière), directly optimize the search direction in the 2D subspace spanned by:
- z_{k+1} = P·g_{k+1} (preconditioned gradient)
- p_k (previous search direction)

**Formula**: p_{k+1} = -μ·z_{k+1} + ν·p_k

**2×2 System** (Eq. 6):
```
[z_{k+1}ᵀ·H̃·z_{k+1}    -z_{k+1}ᵀ·H̃·p_k  ] [μ]   [z_{k+1}ᵀ·g_{k+1} ]
[-p_kᵀ·H̃·z_{k+1}       p_kᵀ·H̃·p_k       ] [ν] = [-p_kᵀ·g_{k+1}    ]
```

**Key insight**: μ acts as a "natural step size" - start with α=1.0, only clamp for CCD.

**Implementation**:
1. Compute v = H̃·z_{k+1} (Hessian-vector product)
2. Cache w_k = H̃·p_k from previous iteration
3. Build 2×2 matrix:
   - A11 = z_{k+1}ᵀ·v
   - A12 = -p_kᵀ·v
   - A22 = p_kᵀ·w_k
4. Build RHS:
   - b1 = z_{k+1}ᵀ·g_{k+1}
   - b2 = -p_kᵀ·g_{k+1}
5. Solve 2×2 system for (μ, ν)
6. Update: p_{k+1} = -μ·z_{k+1} + ν·p_k
7. Update: w_{k+1} = -μ·v + ν·w_k

**Fallback**: If matrix is singular (first iteration or linear dependence), use μ=z·g/(z·v), ν=0.

---

### 2. Powell's Restart Criterion (Section 3.3)

**Current status**: Not implemented. Currently rebuilds preconditioner every iteration.

**What it does**: Monitor orthogonality loss between gradient and previous preconditioned gradient. Restart when conjugacy is lost.

**Formula**:
```
r_k = |g_{k+1}ᵀ·z_k| / (g_{k+1}ᵀ·z_{k+1})

if r_k > δ (e.g., δ = 0.3):
    Restart ← True  # Full preconditioner rebuild
```

**Why it works**: Measures whether gradient direction remains linearly independent after factoring out ill-conditioning.

**Implementation**:
1. Cache z_k from previous iteration
2. Compute r_k = |g_{k+1}·z_k| / (g_{k+1}·z_{k+1})
3. If r_k > 0.3: trigger full rebuild

---

### 3. Sparse-Input Woodbury Level-0 Update (Section 3.1)

**Current status**: Not implemented. Currently does full rebuild each time.

**What it does**: Efficiently update Level-0 subdomain inverses when contact state changes, without full rebuild.

**Core idea**: Contact barrier Hessian changes can be modeled as low-rank updates:
```
H_new = H_base + U·Uᵀ
```
where U = [u_1, ..., u_m] collects rank-1 contributions u_i = √k_i·n_i from contacts.

**Woodbury formula**:
```
B̂_d = B_d - B_d·U_d·(I_K + U_dᵀ·B_d·U_d)⁻¹·U_dᵀ·B_d
```

**Contact classification** (for each contact c):
1. **New contact** (c ∉ C_base): u = √k_curr·n_curr, ΔS = k_curr
2. **Existing contact** (c ∈ C_base):
   - **Rotated normal** (n_curr·n_base < ε_rot): Treat as new contact
   - **Stable normal**:
     - **Stiffening** (k_curr > k_base): u = √(k_curr - k_base)·n_curr, ΔS = k_curr - k_base
     - **Softening**: Ignore (ΔS = 0)

**Top-K truncation**: For each subdomain, keep only K=8 most significant updates.

**Implementation**:
1. Track base contact set C_base from last restart
2. For each current contact, compute update vector and significance
3. For each subdomain, select Top-K updates
4. Apply Woodbury formula to update B_d

---

### 4. Conservative CCD (Section 3.4)

**Current status**: Not implemented. Currently using global step size clamping.

**What it does**: Per-subdomain conservative step clamping to ensure penetration-free motion.

**Philosophy**: Instead of exact TOI, quickly clamp to safe region. New contacts will be injected in next iteration via Woodbury update.

**Per-subdomain**: Prevents "numerical locking" where one tight contact slows entire system.

**Algorithm** (Algorithm 2):
```
for each subdomain d:
    α_d ← 1.0
    for each contact c in C_d:
        α_mon ← GetFirstCriticalPoint(f'_c(α))  # Monotonic region
        α ← min(1.0, α_mon, α_l)
        while sign(f_c(0)) ≠ sign(f_c(α)) and α > α_l:
            α ← α/2  # Bisection
        α_d ← min(α_d, α)
    return α_d
```

**Position update**:
```
x_{k+1} = x_k + Σ_d S_dᵀ·α_d·S_d·p_{k+1}
```

---

### 5. Improved Lower Bound for CCD (Section 3.5)

**Current status**: Not implemented.

**What it does**: Tighter, frame-invariant step size lower bound.

**Original ACCD bound**: l_original = max‖p_P‖ + max‖p_{T_j}‖

**Improved bound** (Point-Triangle):
```
l_tight^PT = max_{j∈{0,1,2}} ‖p_P - p_{T_j}‖
```

**Improved bound** (Edge-Edge):
```
l_tight^EE = max_{i,j∈{a,b}} ‖p_{E1i} - p_{E2j}‖
```

**Advantage**: Frame-invariant (doesn't change with global translation), tighter when primitives move similarly.

---

## Implementation Order

**Phase 1: Core Algorithm Changes** (Priority)
1. 2D Subspace Minimization - replaces Dai-Kai CG
2. Powell's Restart Criterion - controls when to rebuild
3. Add w field to track H̃·p

**Phase 2: Efficiency Improvements**
4. Sparse-Input Woodbury Update - faster between restarts
5. Conservative CCD - per-subdomain step sizes

**Phase 3: Accuracy Improvements**
6. Improved Lower Bound for CCD

---

## Data Structures Needed

```python
# New vertex fields
mesh.verts.place({
    'z': ti.types.vector(3, float),      # Preconditioned gradient (existing)
    'w': ti.types.vector(3, float),      # H̃·p (Hessian-vector product with search direction)
    'z_prev': ti.types.vector(3, float), # Previous z for Powell's criterion
})

# Solver state
self.restart = True                       # Restart flag
self.restart_threshold = 0.3              # Powell's δ

# For Woodbury updates
self.base_contact_set = {}                # Contact state at last restart
self.base_contact_stiffness = {}          # k values at last restart
self.base_contact_normal = {}             # n values at last restart
```

## Key Equations Reference

**MAS Preconditioner** (Eq. 2):
```
P = M_{(0)}^{-1} + Σ_{l=1}^L C_{(l)}ᵀ·M_{(l)}^{-1}·C_{(l)}
```

**Gauss-Newton Contact Hessian** (Eq. 3):
```
∇²b(d_c) ≈ k·n·nᵀ    where k = b''(d_c), n = ∇d_c
```

**Woodbury Update** (Eq. 5):
```
B̂_d = B_d - B_d·U_d·(I_K + U_dᵀ·B_d·U_d)⁻¹·U_dᵀ·B_d
```

**2D Subspace System** (Eq. 6):
```
[z·H̃·z  -p·H̃·z] [μ]   [z·g ]
[-p·H̃·z  p·H̃·p] [ν] = [-p·g]
```

**Powell's Restart** (Eq. 7):
```
r_k = |g_{k+1}ᵀ·z_k| / (g_{k+1}ᵀ·z_{k+1})
Restart if r_k > δ
```
