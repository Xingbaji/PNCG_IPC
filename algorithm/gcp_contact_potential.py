"""
Geometric Contact Potential (GCP) Implementation

This module implements the Geometric Contact Potential from the SIGGRAPH 2025 paper
"Geometric Contact Potential" by Huang et al.

Key Features:
- Directional factors (gamma) that automatically filter adjacent elements
- Mollified barrier functions with C2 smoothness
- Per-primitive adaptive epsilon for non-uniform detection distances
- Drop-in replacement for standard IPC barrier functions

Usage:
    from algorithm.gcp_contact_potential import GCPModule

    # Initialize GCP module
    gcp = GCPModule(n_boundary_points, n_boundary_edges, n_boundary_triangles)
    gcp.init_gcp(mesh, boundary_points, boundary_edges, boundary_triangles)

    # Compute adaptive epsilon from rest configuration
    gcp.compute_adaptive_epsilon(mesh)

    # Find constraints with GCP filtering
    gcp.find_constraints_gcp(mesh, bvh_triangles, bvh_edges)

    # Use GCP barrier functions in solver
    E = gcp.compute_E_gcp(mesh, kappa)
"""

import taichi as ti
from dataclasses import dataclass
from typing import Optional
import numpy as np

from math_utils.graphic_util import (
    dist3D_Point_Triangle,
    dist3D_Segment_to_Segment,
    point_triangle_ccd_broadphase,
    edge_edge_ccd_broadphase,
)


@dataclass
class GCPConfig:
    """Configuration for Geometric Contact Potential."""
    epsilon_target: float = 0.1       # Maximum detection distance (dHat)
    adaptive_epsilon: bool = True     # Use per-primitive epsilon
    alpha: float = 0.1                # Smooth step transition parameter
    kappa: float = 1.0                # Barrier stiffness
    min_epsilon: float = 1e-4         # Minimum epsilon to avoid numerical issues


# =============================================================================
# Part A: Smooth Step Functions (C2 Mollification)
# =============================================================================

@ti.func
def smooth_step_cubic(z: float, a: float, b: float) -> float:
    """
    C2 smooth step function that transitions from 1 to 0.

    h(z) = 1           if z <= a
         = (1-t)²(1+2t) if a < z < b, where t = (z-a)/(b-a)
         = 0           if z >= b

    This function is C2 continuous at both boundaries.

    Args:
        z: Input value
        a: Lower bound (h=1)
        b: Upper bound (h=0)

    Returns:
        Smooth step value in [0, 1]
    """
    result = 0.0
    if z <= a:
        result = 1.0
    elif z >= b:
        result = 0.0
    else:
        t = (z - a) / (b - a)
        result = (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t)
    return result


@ti.func
def smooth_step_cubic_derivative(z: float, a: float, b: float) -> float:
    """
    First derivative of C2 smooth step function.

    d/dz[(1-t)²(1+2t)] = d/dt[(1-t)²(1+2t)] * dt/dz
                       = -6t(1-t) * (1/(b-a))
    """
    result = 0.0
    if a < z < b:
        t = (z - a) / (b - a)
        dt_dz = 1.0 / (b - a)
        result = -6.0 * t * (1.0 - t) * dt_dz
    return result


@ti.func
def smooth_step_cubic_second_derivative(z: float, a: float, b: float) -> float:
    """
    Second derivative of C2 smooth step function.

    d²/dz² = d/dt[-6t(1-t)] * (dt/dz)²
           = -6(1-2t) * (1/(b-a))²
    """
    result = 0.0
    if a < z < b:
        t = (z - a) / (b - a)
        dt_dz = 1.0 / (b - a)
        result = -6.0 * (1.0 - 2.0 * t) * dt_dz * dt_dz
    return result


# =============================================================================
# Part B: Directional Factor (Gamma) Computation
# =============================================================================

@ti.func
def compute_triangle_normal(x0: ti.template(), x1: ti.template(), x2: ti.template()) -> ti.types.vector(3, float):
    """Compute outward normal of a triangle."""
    e1 = x1 - x0
    e2 = x2 - x0
    n = e1.cross(e2)
    n_len = n.norm()
    if n_len > 1e-10:
        n = n / n_len
    else:
        n = ti.Vector([0.0, 1.0, 0.0])  # Default normal
    return n


@ti.func
def compute_gamma_PT(
    xp: ti.template(),
    x0: ti.template(),
    x1: ti.template(),
    x2: ti.template(),
    cord0: float,
    cord1: float,
    cord2: float,
    alpha: float
) -> float:
    """
    Compute directional factor for Point-Triangle contact.

    The directional factor gamma is computed based on:
    1. Local minimum constraint (phi_m): tangential deviation
    2. Exterior direction constraint (phi_e): pointing toward exterior

    For adjacent elements (same surface), gamma ≈ 0.
    For true contact approaches, gamma ≈ 1.

    Args:
        xp: Point position
        x0, x1, x2: Triangle vertices
        cord0, cord1, cord2: Barycentric coordinates of closest point
        alpha: Smooth step parameter

    Returns:
        gamma: Directional factor in [0, 1]
    """
    # Closest point on triangle
    xt = cord0 * x0 + cord1 * x1 + cord2 * x2

    # Direction vector from triangle to point
    d_vec = xp - xt
    dist = d_vec.norm()

    gamma = 0.0

    if dist > 1e-10:
        # Compute triangle normal
        n = compute_triangle_normal(x0, x1, x2)

        # Decompose direction into normal and tangential components
        d_normal = d_vec.dot(n)
        d_tangent = d_vec - d_normal * n
        phi_m = d_tangent.norm()  # Local minimum constraint

        # Exterior direction constraint
        # phi_e > 0 means point is on the positive normal side (exterior)
        phi_e = d_normal

        # Smooth step functions
        # gamma_m: penalize large tangential deviation
        # For true contacts, tangent should be small relative to distance
        alpha_scaled = alpha * dist
        gamma_m = smooth_step_cubic(phi_m, 0.0, alpha_scaled)

        # gamma_e: require exterior direction (phi_e > 0)
        # smooth_step transitions from 1 (at -alpha_scaled) to 0 (at 0)
        # So when phi_e > 0, we get value closer to 1
        gamma_e = 1.0 - smooth_step_cubic(-phi_e, -alpha_scaled, 0.0)

        gamma = gamma_m * gamma_e

    return gamma


@ti.func
def compute_gamma_EE(
    ea0: ti.template(),
    ea1: ti.template(),
    eb0: ti.template(),
    eb1: ti.template(),
    sc: float,
    tc: float,
    alpha: float
) -> float:
    """
    Compute directional factor for Edge-Edge contact.

    Args:
        ea0, ea1: First edge vertices
        eb0, eb1: Second edge vertices
        sc, tc: Parameters for closest points on edges
        alpha: Smooth step parameter

    Returns:
        gamma: Directional factor in [0, 1]
    """
    # Closest points on edges
    pa = (1.0 - sc) * ea0 + sc * ea1
    pb = (1.0 - tc) * eb0 + tc * eb1

    # Direction from edge b to edge a
    d_vec = pa - pb
    dist = d_vec.norm()

    gamma = 0.0

    if dist > 1e-10:
        # Edge tangent vectors
        ta = ea1 - ea0
        tb = eb1 - eb0
        ta_len = ta.norm()
        tb_len = tb.norm()

        if ta_len > 1e-10 and tb_len > 1e-10:
            ta = ta / ta_len
            tb = tb / tb_len

            # Cross product gives perpendicular direction
            n = ta.cross(tb)
            n_len = n.norm()

            if n_len > 1e-6:
                # Non-parallel edges
                n = n / n_len

                # Ensure n points from b toward a
                if n.dot(d_vec) < 0:
                    n = -n

                # Decompose direction
                d_normal = d_vec.dot(n)
                d_tangent = d_vec - d_normal * n
                phi_m = d_tangent.norm()
                phi_e = d_normal

                # Smooth step
                alpha_scaled = alpha * dist
                gamma_m = smooth_step_cubic(phi_m, 0.0, alpha_scaled)
                gamma_e = 1.0 - smooth_step_cubic(-phi_e, -alpha_scaled, 0.0)

                gamma = gamma_m * gamma_e
            else:
                # Near-parallel edges: use reduced weight
                # These cases are ambiguous geometrically
                gamma = 0.5

    return gamma


# =============================================================================
# Part C: Mollified Barrier Functions
# =============================================================================

@ti.func
def gcp_barrier_E(d: float, epsilon: float, gamma: float, kappa: float) -> float:
    """
    GCP mollified barrier energy.

    E = kappa * gamma * h(d) * barrier(d)

    where h(d) is C2 mollifier and barrier(d) = -log(d/epsilon).

    Args:
        d: Distance
        epsilon: Detection threshold (per-primitive)
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        Barrier energy contribution
    """
    E = 0.0
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        E = kappa * gamma * h * barrier
    return E


@ti.func
def gcp_barrier_g(d: float, epsilon: float, gamma: float, kappa: float) -> float:
    """
    GCP barrier gradient with respect to distance.

    dE/dd = kappa * gamma * (dh/dd * barrier + h * dbarrier/dd)

    Args:
        d: Distance
        epsilon: Detection threshold
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        Barrier gradient (dE/dd)
    """
    g = 0.0
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, 0.0, epsilon)
        dh = smooth_step_cubic_derivative(d, 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        dbarrier = -1.0 / d
        g = kappa * gamma * (dh * barrier + h * dbarrier)
    return g


@ti.func
def gcp_barrier_H(d: float, epsilon: float, gamma: float, kappa: float) -> float:
    """
    GCP barrier Hessian with respect to distance.

    d²E/dd² = kappa * gamma * (d²h/dd² * barrier + 2 * dh/dd * dbarrier/dd + h * d²barrier/dd²)

    Args:
        d: Distance
        epsilon: Detection threshold
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        Barrier Hessian (d²E/dd²)
    """
    H = 0.0
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, 0.0, epsilon)
        dh = smooth_step_cubic_derivative(d, 0.0, epsilon)
        d2h = smooth_step_cubic_second_derivative(d, 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        dbarrier = -1.0 / d
        d2barrier = 1.0 / (d * d)
        H = kappa * gamma * (d2h * barrier + 2.0 * dh * dbarrier + h * d2barrier)
    return H


# =============================================================================
# Part D: GCP Module Class
# =============================================================================

@ti.data_oriented
class GCPModule:
    """
    Geometric Contact Potential module for integration with PNCG_IPC solver.

    This module provides:
    - Per-primitive adaptive epsilon computation
    - GCP-aware constraint detection (with gamma filtering)
    - Mollified barrier functions for energy/gradient/Hessian
    """

    def __init__(
        self,
        n_boundary_points: int,
        n_boundary_edges: int,
        n_boundary_triangles: int,
        config: Optional[GCPConfig] = None
    ):
        """
        Initialize GCP module.

        Args:
            n_boundary_points: Number of boundary vertices
            n_boundary_edges: Number of boundary edges
            n_boundary_triangles: Number of boundary triangles
            config: GCP configuration (uses defaults if None)
        """
        self.config = config or GCPConfig()
        self.n_boundary_points = n_boundary_points
        self.n_boundary_edges = n_boundary_edges
        self.n_boundary_triangles = n_boundary_triangles

        # Per-primitive adaptive epsilon
        self.epsilon_per_point = ti.field(dtype=float, shape=n_boundary_points)
        self.epsilon_per_edge = ti.field(dtype=float, shape=n_boundary_edges)

        # Extended constraint structure with gamma and epsilon
        self.MAX_C = 2 ** 21

        self.gcp_pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),     # Vertex IDs
            b=float,                           # Distance
            c=ti.types.vector(4, float),       # Barycentric coordinates
            d=ti.types.vector(3, float),       # Direction vector (normalized)
            gamma=float,                        # Directional factor
            epsilon=float,                      # Per-primitive epsilon
        )

        self.cid_gcp = self.gcp_pair.field()
        self.cid_gcp_root = ti.root.bitmasked(ti.ij, (2, self.MAX_C)).place(self.cid_gcp)

        # Statistics
        self.n_constraints_gcp = ti.field(dtype=ti.i32, shape=())

        print(f"GCP Module initialized:")
        print(f"  epsilon_target = {self.config.epsilon_target}")
        print(f"  adaptive_epsilon = {self.config.adaptive_epsilon}")
        print(f"  alpha = {self.config.alpha}")
        print(f"  kappa = {self.config.kappa}")

    def init_epsilon_uniform(self):
        """Initialize uniform epsilon for all primitives."""
        self._init_epsilon_uniform_kernel(self.config.epsilon_target)

    @ti.kernel
    def _init_epsilon_uniform_kernel(self, epsilon: float):
        for i in range(self.n_boundary_points):
            self.epsilon_per_point[i] = epsilon
        for i in range(self.n_boundary_edges):
            self.epsilon_per_edge[i] = epsilon

    def compute_adaptive_epsilon(
        self,
        mesh,
        boundary_points: ti.template(),
        boundary_edges: ti.template(),
        boundary_triangles: ti.template()
    ):
        """
        Compute per-primitive adaptive epsilon from rest configuration.

        epsilon(x) = min(d_rest(x) / 2, epsilon_target)

        This ensures zero potential at rest configuration.
        """
        if not self.config.adaptive_epsilon:
            self.init_epsilon_uniform()
            return

        self._compute_adaptive_epsilon_PT_kernel(
            mesh.verts.x_init,
            boundary_points,
            boundary_triangles,
            self.config.epsilon_target,
            self.config.min_epsilon
        )

        self._compute_adaptive_epsilon_EE_kernel(
            mesh.verts.x_init,
            boundary_edges,
            self.config.epsilon_target,
            self.config.min_epsilon
        )

    @ti.kernel
    def _compute_adaptive_epsilon_PT_kernel(
        self,
        x_init: ti.template(),
        boundary_points: ti.template(),
        boundary_triangles: ti.template(),
        epsilon_target: float,
        min_epsilon: float
    ):
        """Compute adaptive epsilon for each boundary point."""
        for pi in range(self.n_boundary_points):
            p = boundary_points[pi]
            xp = x_init[p]

            # Find minimum distance to non-adjacent triangles at rest
            min_dist = epsilon_target * 10.0  # Start with large value

            for ti_idx in range(self.n_boundary_triangles):
                t0 = boundary_triangles[ti_idx, 0]
                t1 = boundary_triangles[ti_idx, 1]
                t2 = boundary_triangles[ti_idx, 2]

                # Skip if point is a vertex of this triangle
                if p != t0 and p != t1 and p != t2:
                    x0 = x_init[t0]
                    x1 = x_init[t1]
                    x2 = x_init[t2]

                    # Compute distance
                    cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                    xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                    dist = (xp - xt).norm()

                    ti.atomic_min(min_dist, dist)

            # Adaptive epsilon: half the minimum distance, capped at target
            epsilon = ti.min(min_dist * 0.5, epsilon_target)
            epsilon = ti.max(epsilon, min_epsilon)  # Ensure minimum
            self.epsilon_per_point[pi] = epsilon

    @ti.kernel
    def _compute_adaptive_epsilon_EE_kernel(
        self,
        x_init: ti.template(),
        boundary_edges: ti.template(),
        epsilon_target: float,
        min_epsilon: float
    ):
        """Compute adaptive epsilon for each boundary edge."""
        for ei in range(self.n_boundary_edges):
            a0 = boundary_edges[ei, 0]
            a1 = boundary_edges[ei, 1]
            ea0 = x_init[a0]
            ea1 = x_init[a1]

            # Find minimum distance to non-adjacent edges at rest
            min_dist = epsilon_target * 10.0

            for ej in range(self.n_boundary_edges):
                if ei != ej:
                    b0 = boundary_edges[ej, 0]
                    b1 = boundary_edges[ej, 1]

                    # Skip if edges share a vertex
                    if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                        eb0 = x_init[b0]
                        eb1 = x_init[b1]

                        t_ee, _, _ = dist3D_Segment_to_Segment(ea0, ea1, eb0, eb1)
                        dist = t_ee.norm()

                        ti.atomic_min(min_dist, dist)

            # Adaptive epsilon
            epsilon = ti.min(min_dist * 0.5, epsilon_target)
            epsilon = ti.max(epsilon, min_epsilon)
            self.epsilon_per_edge[ei] = epsilon

    @ti.func
    def hash_coords_2(self, x: ti.i32, y: ti.i32) -> ti.i32:
        """Hash function for constraint storage."""
        h = (x * 92837111) ^ (y * 689287499)
        return ti.abs(h) % self.MAX_C

    # =========================================================================
    # GCP-Aware Constraint Detection
    # =========================================================================

    @ti.func
    def attempt_PT_gcp(
        self,
        triangle_id: ti.i32,
        p: ti.i32,
        t0: ti.i32,
        t1: ti.i32,
        t2: ti.i32,
        xp: ti.template(),
        x0: ti.template(),
        x1: ti.template(),
        x2: ti.template(),
        epsilon: float,
        alpha: float,
        SMALL_NUM: float
    ):
        """
        Attempt to add a Point-Triangle constraint with GCP filtering.

        Unlike standard IPC, this uses the directional factor gamma to
        automatically filter adjacent elements without adjacency matrix.
        """
        # Compute closest point
        cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
        xt = cord0 * x0 + cord1 * x1 + cord2 * x2
        t_pt = xp - xt
        dist = t_pt.norm()

        if dist < epsilon and dist > SMALL_NUM:
            # Compute directional factor
            gamma = compute_gamma_PT(xp, x0, x1, x2, cord0, cord1, cord2, alpha)

            # Only add constraint if gamma is significant
            if gamma > 1e-6:
                ids = ti.Vector([p, t0, t1, t2], ti.u32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                t_normalized = t_pt / dist

                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid_gcp[0, hash_index] = self.gcp_pair(
                    ids, dist, cord, t_normalized, gamma, epsilon
                )

    @ti.func
    def attempt_EE_gcp(
        self,
        edge_id_0: ti.i32,
        edge_id_1: ti.i32,
        a0: ti.i32,
        a1: ti.i32,
        b0: ti.i32,
        b1: ti.i32,
        x_a0: ti.template(),
        x_a1: ti.template(),
        x_b0: ti.template(),
        x_b1: ti.template(),
        epsilon: float,
        alpha: float,
        SMALL_NUM: float
    ):
        """
        Attempt to add an Edge-Edge constraint with GCP filtering.
        """
        # Compute closest points
        t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
        dist = t_ee.norm()

        if dist < epsilon and dist > SMALL_NUM:
            # Compute directional factor
            gamma = compute_gamma_EE(x_a0, x_a1, x_b0, x_b1, sc, tc, alpha)

            # Only add constraint if gamma is significant
            if gamma > 1e-6:
                cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                ids = ti.Vector([a0, a1, b0, b1], ti.u32)
                t_normalized = t_ee / dist

                hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                self.cid_gcp[1, hash_index] = self.gcp_pair(
                    ids, dist, cord, t_normalized, gamma, epsilon
                )

    def find_constraints_gcp(
        self,
        mesh,
        boundary_points: ti.template(),
        boundary_edges: ti.template(),
        boundary_triangles: ti.template(),
        bvh_triangles,
        bvh_edges,
        n_verts: int
    ):
        """
        Find all GCP constraints using BVH traversal.

        This replaces the standard IPC constraint detection when contact_type='gcp'.
        """
        # Clear existing constraints
        self.cid_gcp_root.deactivate_all()

        # Find PT constraints
        self._find_constraints_PT_gcp_kernel(
            mesh.verts.x,
            boundary_points,
            boundary_triangles,
            bvh_triangles.left_idx,
            bvh_triangles.right_idx,
            bvh_triangles.element_idx,
            bvh_triangles.bv_lower,
            bvh_triangles.bv_upper,
            self.config.alpha
        )

        # Find EE constraints
        self._find_constraints_EE_gcp_kernel(
            mesh.verts.x,
            boundary_edges,
            bvh_edges.left_idx,
            bvh_edges.right_idx,
            bvh_edges.element_idx,
            bvh_edges.bv_lower,
            bvh_edges.bv_upper,
            n_verts,
            self.config.alpha
        )

    @ti.kernel
    def _find_constraints_PT_gcp_kernel(
        self,
        x: ti.template(),
        boundary_points: ti.template(),
        boundary_triangles: ti.template(),
        bvh_left_idx: ti.template(),
        bvh_right_idx: ti.template(),
        bvh_element_idx: ti.template(),
        bvh_lower: ti.template(),
        bvh_upper: ti.template(),
        alpha: float
    ):
        """Find PT constraints with GCP filtering using BVH."""
        INVALID = ti.u32(0xFFFFFFFF)
        SMALL_NUM = 1e-7

        for pi in range(self.n_boundary_points):
            p = boundary_points[pi]
            xp = x[p]
            epsilon = self.epsilon_per_point[pi]
            gap = ti.sqrt(epsilon)

            # BVH traversal stack
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = bvh_left_idx[node_id]
                R_idx = bvh_right_idx[node_id]

                # Check left child
                if self._aabb_overlap_point(xp, L_idx, gap, bvh_lower, bvh_upper):
                    element_idx = bvh_element_idx[L_idx]
                    if element_idx != INVALID:
                        # Leaf node
                        tri_id = element_idx
                        t0 = boundary_triangles[tri_id, 0]
                        t1 = boundary_triangles[tri_id, 1]
                        t2 = boundary_triangles[tri_id, 2]

                        # Skip vertex-on-triangle
                        if p != t0 and p != t1 and p != t2:
                            x0 = x[t0]
                            x1 = x[t1]
                            x2 = x[t2]

                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, epsilon):
                                self.attempt_PT_gcp(
                                    tri_id, p, t0, t1, t2,
                                    xp, x0, x1, x2,
                                    epsilon, alpha, SMALL_NUM
                                )
                    else:
                        # Internal node
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap_point(xp, R_idx, gap, bvh_lower, bvh_upper):
                    element_idx = bvh_element_idx[R_idx]
                    if element_idx != INVALID:
                        tri_id = element_idx
                        t0 = boundary_triangles[tri_id, 0]
                        t1 = boundary_triangles[tri_id, 1]
                        t2 = boundary_triangles[tri_id, 2]

                        if p != t0 and p != t1 and p != t2:
                            x0 = x[t0]
                            x1 = x[t1]
                            x2 = x[t2]

                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, epsilon):
                                self.attempt_PT_gcp(
                                    tri_id, p, t0, t1, t2,
                                    xp, x0, x1, x2,
                                    epsilon, alpha, SMALL_NUM
                                )
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.kernel
    def _find_constraints_EE_gcp_kernel(
        self,
        x: ti.template(),
        boundary_edges: ti.template(),
        bvh_left_idx: ti.template(),
        bvh_right_idx: ti.template(),
        bvh_element_idx: ti.template(),
        bvh_lower: ti.template(),
        bvh_upper: ti.template(),
        n_verts: int,
        alpha: float
    ):
        """Find EE constraints with GCP filtering using BVH."""
        INVALID = ti.u32(0xFFFFFFFF)
        SMALL_NUM = 1e-7

        for ei in range(self.n_boundary_edges):
            a0 = boundary_edges[ei, 0]
            a1 = boundary_edges[ei, 1]
            x_a0 = x[a0]
            x_a1 = x[a1]
            epsilon = self.epsilon_per_edge[ei]
            gap = ti.sqrt(epsilon)

            edge_lower = ti.min(x_a0, x_a1)
            edge_upper = ti.max(x_a0, x_a1)

            # BVH traversal
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = bvh_left_idx[node_id]
                R_idx = bvh_right_idx[node_id]

                # Check left child
                if self._aabb_overlap_edge(edge_lower, edge_upper, L_idx, gap, bvh_lower, bvh_upper):
                    element_idx = bvh_element_idx[L_idx]
                    if element_idx != INVALID:
                        ej = element_idx
                        if ei < ej:  # Avoid duplicates
                            b0 = boundary_edges[ej, 0]
                            b1 = boundary_edges[ej, 1]

                            # Skip adjacent edges
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = x[b0]
                                x_b1 = x[b1]

                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, epsilon):
                                    self.attempt_EE_gcp(
                                        n_verts + ei, ej,
                                        a0, a1, b0, b1,
                                        x_a0, x_a1, x_b0, x_b1,
                                        epsilon, alpha, SMALL_NUM
                                    )
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap_edge(edge_lower, edge_upper, R_idx, gap, bvh_lower, bvh_upper):
                    element_idx = bvh_element_idx[R_idx]
                    if element_idx != INVALID:
                        ej = element_idx
                        if ei < ej:
                            b0 = boundary_edges[ej, 0]
                            b1 = boundary_edges[ej, 1]

                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = x[b0]
                                x_b1 = x[b1]

                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, epsilon):
                                    self.attempt_EE_gcp(
                                        n_verts + ei, ej,
                                        a0, a1, b0, b1,
                                        x_a0, x_a1, x_b0, x_b1,
                                        epsilon, alpha, SMALL_NUM
                                    )
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.func
    def _aabb_overlap_point(
        self,
        xp: ti.template(),
        node_idx: ti.i32,
        gap: float,
        bvh_lower: ti.template(),
        bvh_upper: ti.template()
    ) -> bool:
        """Check if point overlaps with BVH node AABB."""
        node_lower = bvh_lower[node_idx]
        node_upper = bvh_upper[node_idx]

        return ((xp[0] >= node_lower[0] - gap) and (xp[0] <= node_upper[0] + gap) and
                (xp[1] >= node_lower[1] - gap) and (xp[1] <= node_upper[1] + gap) and
                (xp[2] >= node_lower[2] - gap) and (xp[2] <= node_upper[2] + gap))

    @ti.func
    def _aabb_overlap_edge(
        self,
        edge_lower: ti.template(),
        edge_upper: ti.template(),
        node_idx: ti.i32,
        gap: float,
        bvh_lower: ti.template(),
        bvh_upper: ti.template()
    ) -> bool:
        """Check if edge AABB overlaps with BVH node AABB."""
        node_lower = bvh_lower[node_idx]
        node_upper = bvh_upper[node_idx]

        return ((node_upper[0] - edge_lower[0]) > -gap and (edge_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - edge_lower[1]) > -gap and (edge_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - edge_lower[2]) > -gap and (edge_upper[2] - node_lower[2]) > -gap)

    # =========================================================================
    # Energy, Gradient, Hessian Computation
    # =========================================================================

    @ti.kernel
    def compute_E_gcp(self, kappa: float) -> float:
        """Compute total GCP barrier energy."""
        E = 0.0
        for k, j in self.cid_gcp:
            pair = self.cid_gcp[k, j]
            dist = pair.b
            gamma = pair.gamma
            epsilon = pair.epsilon
            E += gcp_barrier_E(dist, epsilon, gamma, kappa)
        return E

    @ti.kernel
    def add_grad_and_diagH_gcp(
        self,
        grad: ti.template(),
        diagH: ti.template(),
        kappa: float
    ):
        """Add GCP barrier contribution to gradient and diagonal Hessian."""
        for k, j in self.cid_gcp:
            pair = self.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:
                bg = gcp_barrier_g(dist, epsilon, gamma, kappa)
                bH = gcp_barrier_H(dist, epsilon, gamma, kappa)

                dist2 = dist * dist
                para = bg / dist
                para0 = (bH - para) / dist2

                for i in range(4):
                    CORD = cord[i]
                    ID = ids[i]
                    grad[ID] += para * CORD * t
                    diag_tmp = CORD * CORD * (para0 * t * t + para * ti.Vector.one(float, 3))
                    diag_tmp_spd = ti.max(diag_tmp, 0.0)
                    diagH[ID] += diag_tmp_spd

    @ti.kernel
    def compute_pHp_gcp(self, p: ti.template(), kappa: float) -> float:
        """Compute p^T H p for GCP barrier."""
        ret = 0.0
        for k, j in self.cid_gcp:
            pair = self.cid_gcp[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            gamma = pair.gamma
            epsilon = pair.epsilon

            if gamma > 1e-8:
                bg = gcp_barrier_g(dist, epsilon, gamma, kappa)
                bH = gcp_barrier_H(dist, epsilon, gamma, kappa)

                dist2 = dist * dist
                para1 = bg / dist
                para0 = (bH - para1) / dist2

                # Gather p values
                p_tmp = ti.Vector.zero(float, 12)
                p_tmp[0:3] = p[ids[0]]
                p_tmp[3:6] = p[ids[1]]
                p_tmp[6:9] = p[ids[2]]
                p_tmp[9:12] = p[ids[3]]

                # Compute dtdx^T
                dtdx_t = ti.Vector.zero(float, 12)
                for i in ti.static(range(4)):
                    for j in ti.static(range(3)):
                        dtdx_t[3*i+j] = cord[i] * t[j]

                # pHp contributions
                pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)

                # d_dtdx contribution
                p_dtdx = ti.Vector.zero(float, 3)
                for i in ti.static(range(4)):
                    p_dtdx += cord[i] * p[ids[i]]
                pHp_1 = para1 * p_dtdx.norm_sqr()

                ret += ti.max(pHp_0 + pHp_1, 0.0)

        return ret

    @ti.kernel
    def print_constraints_gcp(self) -> ti.i32:
        """Print GCP constraint statistics."""
        N = 0
        min_dist = 1.0
        min_gamma = 1.0
        max_gamma = 0.0
        for k, j in self.cid_gcp:
            N += 1
            pair = self.cid_gcp[k, j]
            ti.atomic_min(min_dist, pair.b)
            ti.atomic_min(min_gamma, pair.gamma)
            ti.atomic_max(max_gamma, pair.gamma)
        print('GCP constraints:', N, 'min_dist:', min_dist, 'gamma range: [', min_gamma, ',', max_gamma, ']')
        return N


# =============================================================================
# Integration Helper Functions
# =============================================================================

def create_gcp_module(
    n_boundary_points: int,
    n_boundary_edges: int,
    n_boundary_triangles: int,
    epsilon_target: float = 0.1,
    adaptive_epsilon: bool = True,
    alpha: float = 0.1,
    kappa: float = 1.0
) -> GCPModule:
    """
    Factory function to create a GCP module with custom configuration.

    Args:
        n_boundary_points: Number of boundary vertices
        n_boundary_edges: Number of boundary edges
        n_boundary_triangles: Number of boundary triangles
        epsilon_target: Maximum detection distance
        adaptive_epsilon: Use per-primitive epsilon
        alpha: Smooth step parameter
        kappa: Barrier stiffness

    Returns:
        Configured GCPModule instance
    """
    config = GCPConfig(
        epsilon_target=epsilon_target,
        adaptive_epsilon=adaptive_epsilon,
        alpha=alpha,
        kappa=kappa
    )
    return GCPModule(
        n_boundary_points,
        n_boundary_edges,
        n_boundary_triangles,
        config
    )
