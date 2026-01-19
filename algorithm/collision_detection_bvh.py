"""
Collision detection module using Linear BVH (LBVH) for broad-phase.
Replaces spatial hashing with BVH-based collision detection.
"""

import time
from algorithm.pncg_base_collision_free import *
from algorithm.lbvh import LBVH_Triangles, LBVH_Edges
from math_utils.graphic_util import *
from util.model_loading import *


@ti.data_oriented
class collision_detection_bvh_module(pncg_base_deformer):
    """
    Collision detection using LBVH for broad-phase.
    Supports Point-Triangle (PT) and Edge-Edge (EE) collision detection.
    """

    def __init__(self, demo='cube_0'):
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.mu = ti.field(dtype=ti.f32, shape=())
        self.la = ti.field(dtype=ti.f32, shape=())
        self.mu[None], self.la[None] = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.frame = 0
        self.SMALL_NUM = 1e-6

        # Initialize model
        self.mesh.verts.place({'x': ti.types.vector(3, float),
                               'v': ti.types.vector(3, float),
                               'm': float,
                               'x_n': ti.types.vector(3, float),
                               'x_hat': ti.types.vector(3, float),
                               'x_prev': ti.types.vector(3, float),
                               'x_init': ti.types.vector(3, float),
                               'grad': ti.types.vector(3, float),
                               'grad_prev': ti.types.vector(3, float),
                               'diagH': ti.types.vector(3, float),
                               'p': ti.types.vector(3, float),
                               })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print('n_verts,n_cells', self.n_verts, self.n_cells)

        # Precompute
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size ', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.dHat = model.dHat

    def init_bvh(self):
        """Initialize BVH data structures."""
        print('Initializing BVH structures...')

        # Create BVH for triangles (for PT queries)
        self.bvh_triangles = LBVH_Triangles(self.n_boundary_triangles)

        # Create BVH for edges (for EE queries)
        self.bvh_edges = LBVH_Edges(self.n_boundary_edges)

        # Max number of constraints
        self.MAX_C = 2 ** 21

        # Contact pair struct
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),  # ids (vertex indices)
            b=float,  # dist (distance)
            c=ti.types.vector(4, float),  # cord (barycentric coordinates)
            d=ti.types.vector(3, float)  # t (direction vector)
        )

        # Compact array storage for contacts - O(N) iteration
        self.contact_pairs = self.pair.field(shape=self.MAX_C)
        self.n_contacts = ti.field(dtype=ti.i32, shape=())

        # Pre-computed BVH AABB gap for broad-phase filtering
        self.bvh_gap = ti.sqrt(self.dHat)

        self.attempt_PT = self.attempt_PT_no_adj
        self.attempt_EE = self.attempt_EE_no_adj

        print('BVH initialization complete.')

    def build_bvh(self):
        """Build BVH trees for triangles and edges."""
        # Build triangle BVH
        self.bvh_triangles.build(
            self.mesh.verts.x,
            self.boundary_triangles,
            self.n_boundary_triangles
        )

        # Build edge BVH
        self.bvh_edges.build(
            self.mesh.verts.x,
            self.boundary_edges,
            self.n_boundary_edges
        )

    def refit_bvh(self):
        """Refit BVH trees by updating AABBs (preserves tree structure)."""
        # Refit triangle BVH
        self.bvh_triangles.refit(
            self.mesh.verts.x,
            self.boundary_triangles,
            self.n_boundary_triangles
        )

        # Refit edge BVH
        self.bvh_edges.refit(
            self.mesh.verts.x,
            self.boundary_edges,
            self.n_boundary_edges
        )

    @ti.kernel
    def find_constraints_PT_bvh(self):
        """Find Point-Triangle constraints using BVH traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self.bvh_gap

        # For each boundary point, traverse the triangle BVH
        for pi in range(self.n_boundary_points):
            p = self.boundary_points[pi]
            xp = self.mesh.verts.x[p]

            # BVH traversal stack (local array)
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Start from root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = self.bvh_triangles.left_idx[node_id]
                R_idx = self.bvh_triangles.right_idx[node_id]

                # Check left child
                if self.bvh_triangles.aabb_overlap_point(xp, L_idx, gap):
                    element_idx = self.bvh_triangles.element_idx[L_idx]
                    if element_idx != INVALID:
                        # Leaf node - check triangle
                        tri_id = element_idx
                        t0 = self.boundary_triangles[tri_id, 0]
                        t1 = self.boundary_triangles[tri_id, 1]
                        t2 = self.boundary_triangles[tri_id, 2]
                        x0 = self.mesh.verts.x[t0]
                        x1 = self.mesh.verts.x[t1]
                        x2 = self.mesh.verts.x[t2]
                        self.attempt_PT(tri_id, p, t0, t1, t2, xp, x0, x1, x2)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self.bvh_triangles.aabb_overlap_point(xp, R_idx, gap):
                    element_idx = self.bvh_triangles.element_idx[R_idx]
                    if element_idx != INVALID:
                        # Leaf node - check triangle
                        tri_id = element_idx
                        t0 = self.boundary_triangles[tri_id, 0]
                        t1 = self.boundary_triangles[tri_id, 1]
                        t2 = self.boundary_triangles[tri_id, 2]
                        x0 = self.mesh.verts.x[t0]
                        x1 = self.mesh.verts.x[t1]
                        x2 = self.mesh.verts.x[t2]
                        self.attempt_PT(tri_id, p, t0, t1, t2, xp, x0, x1, x2)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.kernel
    def find_constraints_EE_bvh(self):
        """Find Edge-Edge constraints using BVH traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self.bvh_gap
        n_edges = self.n_boundary_edges

        # For each edge, traverse the edge BVH
        for ei in range(n_edges):
            a0 = self.boundary_edges[ei, 0]
            a1 = self.boundary_edges[ei, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]

            # Current edge's AABB
            edge_lower = ti.min(x_a0, x_a1)
            edge_upper = ti.max(x_a0, x_a1)

            # BVH traversal stack (local array)
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Start from root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = self.bvh_edges.left_idx[node_id]
                R_idx = self.bvh_edges.right_idx[node_id]

                # Check left child
                if self._aabb_overlap_with_edge(edge_lower, edge_upper, L_idx, gap):
                    element_idx = self.bvh_edges.element_idx[L_idx]
                    if element_idx != INVALID:
                        # Leaf node - check edge pair
                        ej = element_idx
                        if ei < ej:  # Avoid duplicate pairs
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            x_b0 = self.mesh.verts.x[b0]
                            x_b1 = self.mesh.verts.x[b1]
                            self.attempt_EE(ei, ej, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap_with_edge(edge_lower, edge_upper, R_idx, gap):
                    element_idx = self.bvh_edges.element_idx[R_idx]
                    if element_idx != INVALID:
                        # Leaf node - check edge pair
                        ej = element_idx
                        if ei < ej:  # Avoid duplicate pairs
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            x_b0 = self.mesh.verts.x[b0]
                            x_b1 = self.mesh.verts.x[b1]
                            self.attempt_EE(ei, ej, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.func
    def _aabb_overlap_with_edge(self, edge_lower: ti.template(), edge_upper: ti.template(),
                                 node_idx: ti.i32, gap: ti.f32) -> bool:
        """Check if edge AABB overlaps with BVH node AABB.

        Uses direct boolean logic for better branch prediction and fewer operations.
        """
        node_lower = self.bvh_edges.bv_lower[node_idx]
        node_upper = self.bvh_edges.bv_upper[node_idx]

        # Direct boolean expression - allows compiler to optimize better
        return ((node_upper[0] - edge_lower[0]) > -gap and (edge_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - edge_lower[1]) > -gap and (edge_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - edge_lower[2]) > -gap and (edge_upper[2] - node_lower[2]) > -gap)

    @ti.func
    def _add_contact_pair(self, ids: ti.template(), dist: float, cord: ti.template(), t_vec: ti.template()):
        """
        Add a contact pair to compact storage.
        Uses atomic counter for thread-safe insertion.
        """
        idx = ti.atomic_add(self.n_contacts[None], 1)
        if idx < self.MAX_C:
            self.contact_pairs[idx] = self.pair(ids, dist, cord, t_vec)

    @ti.func
    def attempt_PT_no_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        # Note: triangle_id is unused here but kept for interface compatibility with attempt_PT_adj
        if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                self._add_contact_pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_EE_no_adj(self, edge_id_0, edge_id_1, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1):
        if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
            t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
            dist = t_ee.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                self._add_contact_pair(ids, dist, cord, t_ee)

    def find_cnts(self, PRINT=False, TIME_LOG=False, use_refit=False):
        """Find all collision constraints using BVH.

        Args:
            PRINT: Print constraint count
            TIME_LOG: Print timing information
            use_refit: Use BVH refit instead of full rebuild (faster for small deformations)
        """
        if TIME_LOG:
            ti.sync()
            t_start = time.perf_counter()

        # Reset contact counter
        self.n_contacts[None] = 0

        # Build or refit BVH trees
        if TIME_LOG:
            ti.sync()
            t_bvh_start = time.perf_counter()

        if use_refit:
            self.refit_bvh()
        else:
            self.build_bvh()

        if TIME_LOG:
            ti.sync()
            t_bvh_end = time.perf_counter()

        # Find PT constraints
        if TIME_LOG:
            ti.sync()
            t_pt_start = time.perf_counter()

        self.find_constraints_PT_bvh()

        if TIME_LOG:
            ti.sync()
            t_pt_end = time.perf_counter()

        # Find EE constraints
        if TIME_LOG:
            ti.sync()
            t_ee_start = time.perf_counter()

        self.find_constraints_EE_bvh()

        if TIME_LOG:
            ti.sync()
            t_ee_end = time.perf_counter()
            t_total = t_ee_end - t_start
            bvh_op = "refit" if use_refit else "build"
            print(f"[BVH Time] {bvh_op}: {(t_bvh_end - t_bvh_start)*1000:.2f}ms, "
                  f"PT: {(t_pt_end - t_pt_start)*1000:.2f}ms, "
                  f"EE: {(t_ee_end - t_ee_start)*1000:.2f}ms, "
                  f"total: {t_total*1000:.2f}ms")

        if PRINT:
            N = self.print_cnts()
            return N

    def find_cnts_iter(self, iter, rate=10, PRINT=False):
        """Find constraints with smart BVH update strategy.

        Uses full build on first call, then refit on subsequent calls.
        Optionally rebuilds every 'rate' iterations for better quality.
        """
        self.n_contacts[None] = 0

        if iter == 0:
            # Full build on first call
            self.build_bvh()
        else:
            # Refit on other iterations (fast AABB update)
            self.refit_bvh()

        self.find_constraints_PT_bvh()
        self.find_constraints_EE_bvh()

        if PRINT:
            self.print_cnts()

    @ti.func
    def _aabb_overlap_with_tri(self, tri_lower: ti.template(), tri_upper: ti.template(),
                                node_idx: ti.i32, gap: ti.f32) -> bool:
        """Check if triangle AABB overlaps with BVH node AABB.

        Uses direct boolean logic for better branch prediction and fewer operations.
        """
        node_lower = self.bvh_edges.bv_lower[node_idx]
        node_upper = self.bvh_edges.bv_upper[node_idx]

        # Direct boolean expression - allows compiler to optimize better
        return ((node_upper[0] - tri_lower[0]) > -gap and (tri_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - tri_lower[1]) > -gap and (tri_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - tri_lower[2]) > -gap and (tri_upper[2] - node_lower[2]) > -gap)

    def check_dcd(self):
        """Check discrete collision detection using BVH."""
        self.build_bvh()
        ret = self.check_edge_triangle_intersection_bvh()
        return ret

    # ===================== Penetration Detection API =====================
    # Based on GIPC.cu implementation for intersection checking

    @ti.func
    def _check_edge_tri_leaf(self, edge_idx: ti.i32, t0: ti.i32, t1: ti.i32, t2: ti.i32,
                              x0: ti.template(), x1: ti.template(), x2: ti.template()) -> ti.i32:
        """Check if edge intersects triangle (leaf node processing)."""
        result = 0
        a0 = self.boundary_edges[edge_idx, 0]
        a1 = self.boundary_edges[edge_idx, 1]
        # Skip if edge shares vertex with triangle (adjacent)
        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            if segment_triangle_intersect_cramer(x_a0, x_a1, x0, x1, x2):
                result = 1
        return result

    @ti.kernel
    def _edge_triangle_intersection_impl(self, count_mode: ti.template()) -> ti.i32:
        """
        Unified edge-triangle intersection detection using BVH traversal.
        Based on GIPC.cu _edgeTriIntersectionQuery implementation.

        Args:
            count_mode: If True, count all intersections; if False, early exit on first

        Returns:
            count_mode=False: 1 if any intersection found, 0 otherwise
            count_mode=True: total number of intersections
        """
        INVALID = ti.u32(0xFFFFFFFF)
        result = 0
        gap = 0.0  # No gap for exact intersection test

        # For each triangle, traverse the edge BVH to find potential intersections
        for ti_idx in range(self.n_boundary_triangles):
            # Early exit check (only for check mode)
            should_process = True
            if ti.static(not count_mode):
                should_process = (result == 0)

            if should_process:
                t0 = self.boundary_triangles[ti_idx, 0]
                t1 = self.boundary_triangles[ti_idx, 1]
                t2 = self.boundary_triangles[ti_idx, 2]
                x0 = self.mesh.verts.x[t0]
                x1 = self.mesh.verts.x[t1]
                x2 = self.mesh.verts.x[t2]

                # Triangle AABB
                tri_lower = ti.min(ti.min(x0, x1), x2)
                tri_upper = ti.max(ti.max(x0, x1), x2)

                # BVH traversal using stack
                stack = ti.Vector.zero(ti.u32, 64)
                stack_ptr = 0
                stack[stack_ptr] = 0  # Start from root
                stack_ptr += 1

                while stack_ptr > 0:
                    # Early exit in check mode
                    if ti.static(not count_mode):
                        if result != 0:
                            break

                    stack_ptr -= 1
                    node_id = stack[stack_ptr]

                    L_idx = self.bvh_edges.left_idx[node_id]
                    R_idx = self.bvh_edges.right_idx[node_id]

                    # Check left child
                    if self._aabb_overlap_with_tri(tri_lower, tri_upper, L_idx, gap):
                        element_idx = self.bvh_edges.element_idx[L_idx]
                        if element_idx != INVALID:
                            # Leaf node - check edge-triangle intersection
                            if self._check_edge_tri_leaf(element_idx, t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            # Internal node - push to stack
                            if stack_ptr < 63:
                                stack[stack_ptr] = L_idx
                                stack_ptr += 1

                    # Check right child
                    if self._aabb_overlap_with_tri(tri_lower, tri_upper, R_idx, gap):
                        element_idx = self.bvh_edges.element_idx[R_idx]
                        if element_idx != INVALID:
                            # Leaf node - check edge-triangle intersection
                            if self._check_edge_tri_leaf(element_idx, t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            # Internal node - push to stack
                            if stack_ptr < 63:
                                stack[stack_ptr] = R_idx
                                stack_ptr += 1

        return result

    def check_edge_triangle_intersection_bvh(self) -> int:
        """
        Check if any edge intersects any triangle using BVH traversal.
        Returns 1 if any intersection found, 0 otherwise.
        """
        return self._edge_triangle_intersection_impl(count_mode=False)

    @ti.kernel
    def check_ground_intersection(self, ground_normal: ti.template(), ground_offset: ti.f32) -> ti.i32:
        """
        Check if any vertex has penetrated the ground plane.
        Based on GIPC.cu checkGroundIntersection implementation.

        Args:
            ground_normal: Ground plane normal (vec3), pointing outward (typically (0, 1, 0))
            ground_offset: Ground plane offset (scalar), plane equation: n · x = offset

        Returns:
            1 if any vertex penetrated, 0 otherwise
        """
        result = 0
        for i in range(self.n_verts):
            if result == 0:
                vertex_pos = self.mesh.verts.x[i]
                dist = vertex_pos.dot(ground_normal) - ground_offset
                if dist < 0.0:
                    result = 1
        return result

    def is_intersected(self, check_ground=True):
        """
        High-level API to check if mesh has any penetration/intersection.
        Based on GIPC.cu isIntersected implementation.

        This function checks:
        1. Ground plane intersection (if enabled)
        2. Edge-triangle intersection (self-intersection)

        Args:
            check_ground: Whether to check ground plane penetration

        Returns:
            True if any intersection found, False otherwise
        """
        # Ensure BVH is built
        self.build_bvh()

        # Check ground intersection
        if check_ground and hasattr(self, 'ground') and self.ground is not None:
            ground_y = self.ground
            ground_normal = ti.Vector([0.0, 1.0, 0.0])
            if self.check_ground_intersection(ground_normal, ground_y):
                print("[Penetration] Ground intersection detected!")
                return True

        # Check edge-triangle intersection (self-intersection)
        if self.check_edge_triangle_intersection_bvh():
            print("[Penetration] Edge-triangle intersection detected!")
            return True

        return False

    def check_penetration(self, verbose=True):
        """
        Alias for is_intersected() with verbose output.
        """
        result = self.is_intersected()
        if verbose:
            if result:
                print("[Penetration Check] FAILED - Mesh has penetration!")
            else:
                print("[Penetration Check] PASSED - No penetration detected.")
        return result

    def count_edge_triangle_intersections(self) -> int:
        """
        Count the total number of edge-triangle intersections.
        Useful for debugging and analysis.

        Returns:
            Number of edge-triangle intersections
        """
        return self._edge_triangle_intersection_impl(count_mode=True)

    @ti.kernel
    def count_ground_penetrations(self, ground_normal: ti.template(), ground_offset: ti.f32) -> ti.i32:
        """
        Count the number of vertices that have penetrated the ground plane.

        Args:
            ground_normal: Ground plane normal (vec3)
            ground_offset: Ground plane offset (scalar)

        Returns:
            Number of vertices below ground
        """
        count = 0
        for i in range(self.n_verts):
            vertex_pos = self.mesh.verts.x[i]
            dist = vertex_pos.dot(ground_normal) - ground_offset
            if dist < 0.0:
                ti.atomic_add(count, 1)
        return count

    @ti.kernel
    def print_cnts(self) -> ti.i32:
        N = self.n_contacts[None]
        min_dist = 1.0
        for idx in range(N):
            pair = self.contact_pairs[idx]
            dist = pair.b
            ti.atomic_min(min_dist, dist)
        print('number of cnts', N, 'min dist', min_dist)
        return N

    @ti.kernel
    def compute_mean_of_boundary_edges(self) -> float:
        total = 0.0
        for i in range(self.n_boundary_edges):
            total += (self.mesh.verts.x[self.boundary_edges[i, 0]] - self.mesh.verts.x[
                self.boundary_edges[i, 1]]).norm()
        result = total / ti.cast(self.n_boundary_edges, float)
        print("Mean of boundary edges:", result)
        return result

    @ti.kernel
    def compute_min_of_boundary_edges(self) -> float:
        ret = 10.0
        for i in range(self.n_boundary_edges):
            dist = (self.mesh.verts.x[self.boundary_edges[i, 0]] - self.mesh.verts.x[
                self.boundary_edges[i, 1]]).norm()
            ti.atomic_min(ret, dist)
        print("Min of boundary edges:", ret)
        return ret

