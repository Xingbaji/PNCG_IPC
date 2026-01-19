# Geometric Contact Potential (GCP) Implementation Guide

## Overview

This document describes the implementation of **Geometric Contact Potential (GCP)** from the SIGGRAPH 2025 paper "Geometric Contact Potential" into the PNCG_IPC codebase. GCP enables significantly larger `dHat` values (10x or more) without requiring adjacency matrix filtering.

### Key References
- Paper: "Geometric Contact Potential" (SIGGRAPH 2025) - Huang et al.
- Source: `/cpfs/user/shenxing/geometric-contact-potential`
- Paper PDF: `ref_doc/Geometric Contact Potential.pdf`

---

## Problem with Standard IPC

Standard IPC (Incremental Potential Contact) requires:
1. **Small `dHat`**: Detection distance must be smaller than minimum edge length
2. **Adjacency Filtering**: At rest configuration, adjacent boundary elements within `dHat` must be explicitly excluded via `define_adj_matrix()`

**Why this is limiting:**
- Small `dHat` requires more iterations to detect approaching contacts
- Adjacency matrix consumes O(n²) memory in worst case
- Mesh-dependent: changing mesh requires recomputing adjacency

---

## GCP Solution: Directional Factors

GCP introduces **directional factors** `γ(x,y)` that naturally filter out adjacent element pairs without precomputation.

### Key Insight

For two surfaces approaching contact:
- True contact: surfaces approach **perpendicular** to each other
- Adjacent elements: surfaces are **parallel** (same surface)

GCP uses geometric constraints to distinguish these cases.

---

## Mathematical Formulation

### 1. Interaction Set Definition

For smooth surfaces, the interaction set C(x,f) is defined by two constraints:

**Local Minimum Constraint Φᵐ(x,y):**
```
Φᵐ(x,y) := ||(f(y) - f(x))₊ × n(y)|| = 0
```
- Identifies points that are local minima of the distance function
- Cross product with normal detects tangential deviation

**Exterior Direction Constraint Φᵉ(x,y):**
```
Φᵉ(x,y) := -n(y) · (f(y) - f(x))₊
```
- Ensures the direction vector points toward interior at one end
- Prevents spurious forces between compressed nearby surfaces

### 2. Directional Factor γ(x,y)

The directional factor combines both constraints using smooth step functions:

```
γˢ(x,y) = δ_α(Φᵐ(x,y)) · H_α(Φᵉ(x,y)) · δ_α(Φᵐ(y,x)) · H_α(Φᵉ(y,x))
```

Where:
- `δ_α(z)` is a smooth bump function centered at 0
- `H_α(z)` is a smoothed Heaviside step function

For our implementation (piecewise smooth meshes):
```
gamma = smooth_step(||d_tangent||, alpha) * smooth_step(d · n, alpha)
```

### 3. Mollified Barrier Function

The GCP barrier potential is:
```
p_ε(d) = h_ε(d) · barrier(d)
```

Where:
- `h_ε(d)` is a C² mollifier that transitions from 1 to 0 over [0, ε]
- `barrier(d)` is the standard barrier (e.g., `-log(d/ε)`)
- ε is the detection distance (can be much larger than IPC's dHat)

**C² Cubic Mollifier:**
```
h_ε(d) = 1                           if d ≤ 0
       = (1-t)² · (1+2t)             if 0 < d < ε, where t = d/ε
       = 0                           if d ≥ ε
```

### 4. Adaptive Epsilon Per Primitive

To avoid spurious forces at rest configuration:
```
ε(x) = min(d_rest(x) / 2, ε_target)
```

Where `d_rest(x)` is the minimum distance from point x to non-adjacent elements at rest configuration.

---

## Implementation Details

### Directional Factor for Point-Triangle (PT)

```python
@ti.func
def compute_gamma_PT(xp, x0, x1, x2, alpha):
    """
    Compute directional factor for Point-Triangle contact.

    Args:
        xp: Point position
        x0, x1, x2: Triangle vertices
        alpha: Smooth step parameter

    Returns:
        gamma: Directional factor in [0, 1]
    """
    # Compute closest point on triangle
    cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
    xt = cord0 * x0 + cord1 * x1 + cord2 * x2

    # Direction vector from triangle to point
    d_vec = xp - xt
    dist = d_vec.norm()

    if dist < 1e-10:
        return 0.0

    # Compute triangle normal
    e1 = x1 - x0
    e2 = x2 - x0
    n = e1.cross(e2).normalized()

    # Tangential component (local minimum constraint)
    d_normal = d_vec.dot(n)
    d_tangent = d_vec - d_normal * n
    phi_m = d_tangent.norm()

    # Normal component (exterior direction constraint)
    phi_e = d_normal  # Positive if point is on normal side

    # Smooth step mollification
    gamma_m = smooth_step(phi_m, 0, alpha * dist)
    gamma_e = smooth_step(-phi_e, -alpha * dist, 0)  # Active when phi_e > 0

    return gamma_m * gamma_e
```

### Directional Factor for Edge-Edge (EE)

```python
@ti.func
def compute_gamma_EE(ea0, ea1, eb0, eb1, alpha):
    """
    Compute directional factor for Edge-Edge contact.
    """
    # Compute closest points
    d_vec, sc, tc = dist3D_Segment_to_Segment(ea0, ea1, eb0, eb1)
    dist = d_vec.norm()

    if dist < 1e-10:
        return 0.0

    # Edge tangents
    ta = (ea1 - ea0).normalized()
    tb = (eb1 - eb0).normalized()

    # Cross product gives perpendicular direction
    n = ta.cross(tb)
    n_norm = n.norm()

    if n_norm < 1e-6:
        # Near-parallel edges - reduced weight
        return 0.5

    n = n / n_norm
    if n.dot(d_vec) < 0:
        n = -n

    # Tangential component
    d_normal = d_vec.dot(n)
    d_tangent = d_vec - d_normal * n
    phi_m = d_tangent.norm()

    # Normal component
    phi_e = d_normal

    # Smooth step
    gamma_m = smooth_step(phi_m, 0, alpha * dist)
    gamma_e = smooth_step(-phi_e, -alpha * dist, 0)

    return gamma_m * gamma_e
```

### Mollified Barrier Functions

```python
@ti.func
def smooth_step_cubic(z, a, b):
    """
    C² smooth step function: 1 at a, 0 at b.
    """
    if z <= a:
        return 1.0
    elif z >= b:
        return 0.0
    else:
        t = (z - a) / (b - a)
        return (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t)

@ti.func
def smooth_step_cubic_derivative(z, a, b):
    """
    Derivative of C² smooth step.
    """
    if z <= a or z >= b:
        return 0.0
    else:
        t = (z - a) / (b - a)
        dt_dz = 1.0 / (b - a)
        # d/dt[(1-t)²(1+2t)] = -2(1-t)(1+2t) + 2(1-t)² = -6t(1-t)
        return -6.0 * t * (1.0 - t) * dt_dz

@ti.func
def gcp_barrier_E(d, epsilon, gamma, kappa):
    """GCP mollified barrier energy."""
    if d >= epsilon or gamma < 1e-8:
        return 0.0

    h = smooth_step_cubic(d, 0.0, epsilon)
    barrier = -ti.log(d / epsilon)

    return kappa * gamma * h * barrier

@ti.func
def gcp_barrier_g(d, epsilon, gamma, kappa):
    """GCP barrier gradient w.r.t. distance."""
    if d >= epsilon or gamma < 1e-8:
        return 0.0

    h = smooth_step_cubic(d, 0.0, epsilon)
    dh = smooth_step_cubic_derivative(d, 0.0, epsilon)
    barrier = -ti.log(d / epsilon)
    dbarrier = -1.0 / d

    return kappa * gamma * (dh * barrier + h * dbarrier)

@ti.func
def gcp_barrier_H(d, epsilon, gamma, kappa):
    """GCP barrier Hessian w.r.t. distance."""
    if d >= epsilon or gamma < 1e-8:
        return 0.0

    h = smooth_step_cubic(d, 0.0, epsilon)
    dh = smooth_step_cubic_derivative(d, 0.0, epsilon)
    d2h = smooth_step_cubic_second_derivative(d, 0.0, epsilon)
    barrier = -ti.log(d / epsilon)
    dbarrier = -1.0 / d
    d2barrier = 1.0 / (d * d)

    return kappa * gamma * (d2h * barrier + 2.0 * dh * dbarrier + h * d2barrier)
```

---

## Integration with PNCG_IPC

### Extended Constraint Structure

```python
self.gcp_pair = ti.types.struct(
    a=ti.types.vector(4, ti.u32),     # Vertex IDs
    b=float,                           # Distance
    c=ti.types.vector(4, float),       # Barycentric coords
    d=ti.types.vector(3, float),       # Direction vector
    gamma=float,                        # Directional factor
    epsilon=float,                      # Per-primitive epsilon
)
```

### Usage

```python
# Standard IPC (requires adjacency)
solver = pncg_ipc_deformer(demo='cube')
solver.contact_type = 'ipc'
solver.dHat = 0.01
solver.adj = 1  # Enable adjacency filtering

# GCP (no adjacency needed)
solver = pncg_ipc_deformer(demo='cube')
solver.contact_type = 'gcp'
solver.dHat = 0.1  # 10x larger!
# adj = 0 by default, gamma filtering handles adjacent elements
```

---

## Advantages Over IPC

| Aspect | IPC | GCP |
|--------|-----|-----|
| dHat limit | < min edge length | >> min edge length |
| Adjacency matrix | Required for large dHat | Not needed |
| Rest forces | Possible if adj matrix incomplete | Zero by construction |
| Mesh dependence | Strong | Weak |
| Convergence | Standard | Often faster (fewer constraints) |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon_target` | 0.1 | Maximum detection distance |
| `alpha` | 0.1 | Smooth step transition width |
| `kappa` | 1.0 | Barrier stiffness |
| `adaptive_epsilon` | True | Use per-primitive epsilon |

---

## File Structure

```
algorithm/
    gcp_contact_potential.py   # Main GCP implementation
    collision_detection_bvh.py # Existing BVH (unchanged)
    pncg_base_ipc.py           # Existing IPC (unchanged)

ref_doc/
    GCP_IMPLEMENTATION.md      # This document

demo/
    gcp_demo.py                # GCP demo and comparison
```

---

## Testing

1. **Unit Test**: Verify gamma ≈ 0 for adjacent elements
2. **Integration Test**: Compare GCP vs IPC on contact scenarios
3. **Large dHat Test**: Verify stable simulation with dHat = 0.1

```bash
cd demo && python gcp_demo.py --headless --frames 100
```
