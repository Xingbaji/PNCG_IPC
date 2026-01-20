"""
Final optimized collision detection module using LBVH Final.

Key optimizations:
1. Uses LBVH_Final with packed AABB (vec6) for better cache efficiency
2. Prefetch both children AABBs before deciding traversal order
3. Inline AABB overlap checks to avoid function call overhead
4. Fused narrow-phase operations with early exits

Based on collision_detection_bvh.py with lbvh_final integration.
"""

import time
from algorithm.pncg_base_collision_free import *
from algorithm.lbvh_final import LBVH_Triangles_Final, LBVH_Edges_Final
from math_utils.graphic_util import *
from util.model_loading import *


@ti.data_oriented
class collision_detection_bvh_final_module(pncg_base_deformer):
    """
    Final optimized collision detection using LBVH Final.

    Optimizations over collision_detection_bvh_optimized:
    - LBVH_Final with packed AABB and node topology
    - Prefetch both children before traversal decision
    - Inline AABB overlap for reduced function call overhead
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
        """Initialize BVH data structures using LBVH Final."""
        print('Initializing BVH Final structures...')

        # Create LBVH Final for triangles (for PT queries)
        self.bvh_triangles = LBVH_Triangles_Final(self.n_boundary_triangles)

        # Create LBVH Final for edges (for EE queries)
        self.bvh_edges = LBVH_Edges_Final(self.n_boundary_edges)

        # Max number of constraints
        self.MAX_C = 2 ** 21

        # Contact pair struct
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),  # ids (vertex indices)
            b=float,  # dist (distance)
            c=ti.types.vector(4, float),  # cord (barycentric coordinates)
            d=ti.types.vector(3, float)  # t (direction vector)
        )

        # Compact array storage for contacts
        self.contact_pairs = self.pair.field(shape=self.MAX_C)
        self.n_contacts = ti.field(dtype=ti.i32, shape=())

        # Detection threshold
        self._detection_dHat = self.dHat
        self.bvh_gap = ti.sqrt(self._detection_dHat)

        print('BVH Final initialization complete.')

    def set_detection_dHat(self, detection_dHat: float):
        """Set custom detection threshold for contact filtering."""
        self._detection_dHat = detection_dHat
        self.bvh_gap = ti.sqrt(detection_dHat)

    def get_detection_dHat(self) -> float:
        """Get the current detection threshold."""
        return self._detection_dHat

    def build_bvh(self):
        """Build BVH trees for triangles and edges."""
        self.bvh_triangles.build(
            self.mesh.verts.x,
            self.boundary_triangles,
            self.n_boundary_triangles
        )
        self.bvh_edges.build(
            self.mesh.verts.x,
            self.boundary_edges,
            self.n_boundary_edges
        )

    def refit_bvh(self):
        """Refit BVH trees (fast AABB update, preserves structure)."""
        self.bvh_triangles.refit(
            self.mesh.verts.x,
            self.boundary_triangles,
            self.n_boundary_triangles
        )
        self.bvh_edges.refit(
            self.mesh.verts.x,
            self.boundary_edges,
            self.n_boundary_edges
        )

    @ti.func
    def _point_aabb_overlap(self, point: ti.math.vec3, aabb: ti.template(), gap: ti.f32) -> bool:
        """Check point-AABB overlap. aabb format: [lx, ly, lz, ux, uy, uz]"""
        return (point[0] >= aabb[0] - gap and point[0] <= aabb[3] + gap and
                point[1] >= aabb[1] - gap and point[1] <= aabb[4] + gap and
                point[2] >= aabb[2] - gap and point[2] <= aabb[5] + gap)

    @ti.func
    def _aabb_overlap(self, aabb1: ti.template(), aabb2: ti.template(), gap: ti.f32) -> bool:
        """Check AABB-AABB overlap. Both format: [lx, ly, lz, ux, uy, uz]"""
        return (aabb1[3] + gap >= aabb2[0] and aabb2[3] + gap >= aabb1[0] and
                aabb1[4] + gap >= aabb2[1] and aabb2[4] + gap >= aabb1[1] and
                aabb1[5] + gap >= aabb2[2] and aabb2[5] + gap >= aabb1[2])

    @ti.func
    def _add_contact_pair(self, ids: ti.template(), dist: float,
                          cord: ti.template(), t_vec: ti.template()):
        """Add contact pair with atomic counter."""
        idx = ti.atomic_add(self.n_contacts[None], 1)
        if idx < self.MAX_C:
            self.contact_pairs[idx] = self.pair(ids, dist, cord, t_vec)

    @ti.kernel
    def find_constraints_PT_bvh(self):
        """Find Point-Triangle constraints using LBVH Final traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self.bvh_gap
        n_tris = self.bvh_triangles.num_primitives[None]

        for pi in range(self.n_boundary_points):
            p = self.boundary_points[pi]
            xp = self.mesh.verts.x[p]

            # BVH traversal stack
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Start from root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = ti.cast(stack[stack_ptr], ti.i32)

                # Get node data (packed: [parent, left, right, element])
                node_data = self.bvh_triangles.node_data[node_id]
                L_idx = ti.cast(node_data[1], ti.i32)
                R_idx = ti.cast(node_data[2], ti.i32)

                # Prefetch both children AABBs
                L_aabb = self.bvh_triangles.aabb[L_idx]
                R_aabb = self.bvh_triangles.aabb[R_idx]

                # Check left child
                L_overlap = self._point_aabb_overlap(xp, L_aabb, gap)
                if L_overlap:
                    L_element = self.bvh_triangles.node_data[L_idx][3]
                    if L_element != INVALID:
                        # Leaf node - narrow phase
                        tri_id = ti.cast(L_element, ti.i32)
                        t0 = self.boundary_triangles[tri_id, 0]
                        t1 = self.boundary_triangles[tri_id, 1]
                        t2 = self.boundary_triangles[tri_id, 2]
                        if p != t0 and p != t1 and p != t2:
                            x0 = self.mesh.verts.x[t0]
                            x1 = self.mesh.verts.x[t1]
                            x2 = self.mesh.verts.x[t2]
                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, self._detection_dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()
                                if dist < self._detection_dHat and dist > self.SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.i32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                                    self._add_contact_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(L_idx)
                            stack_ptr += 1

                # Check right child
                R_overlap = self._point_aabb_overlap(xp, R_aabb, gap)
                if R_overlap:
                    R_element = self.bvh_triangles.node_data[R_idx][3]
                    if R_element != INVALID:
                        # Leaf node - narrow phase
                        tri_id = ti.cast(R_element, ti.i32)
                        t0 = self.boundary_triangles[tri_id, 0]
                        t1 = self.boundary_triangles[tri_id, 1]
                        t2 = self.boundary_triangles[tri_id, 2]
                        if p != t0 and p != t1 and p != t2:
                            x0 = self.mesh.verts.x[t0]
                            x1 = self.mesh.verts.x[t1]
                            x2 = self.mesh.verts.x[t2]
                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, self._detection_dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()
                                if dist < self._detection_dHat and dist > self.SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.i32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                                    self._add_contact_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(R_idx)
                            stack_ptr += 1

    @ti.kernel
    def find_constraints_EE_bvh(self):
        """Find Edge-Edge constraints using LBVH Final traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self.bvh_gap
        n_edges = self.n_boundary_edges

        for ei in range(n_edges):
            a0 = self.boundary_edges[ei, 0]
            a1 = self.boundary_edges[ei, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]

            # Precompute edge AABB (packed format)
            edge_lower = ti.min(x_a0, x_a1)
            edge_upper = ti.max(x_a0, x_a1)
            edge_aabb = ti.Vector([edge_lower[0], edge_lower[1], edge_lower[2],
                                   edge_upper[0], edge_upper[1], edge_upper[2]])

            # BVH traversal stack
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = ti.cast(stack[stack_ptr], ti.i32)

                # Get node data
                node_data = self.bvh_edges.node_data[node_id]
                L_idx = ti.cast(node_data[1], ti.i32)
                R_idx = ti.cast(node_data[2], ti.i32)

                # Prefetch both children AABBs
                L_aabb = self.bvh_edges.aabb[L_idx]
                R_aabb = self.bvh_edges.aabb[R_idx]

                # Check left child
                if self._aabb_overlap(edge_aabb, L_aabb, gap):
                    L_element = self.bvh_edges.node_data[L_idx][3]
                    if L_element != INVALID:
                        # Leaf node - narrow phase
                        ej = ti.cast(L_element, ti.i32)
                        if ei < ej:  # Avoid duplicates
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self.mesh.verts.x[b0]
                                x_b1 = self.mesh.verts.x[b1]
                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self._detection_dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()
                                    if dist < self._detection_dHat and dist > self.SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                                        self._add_contact_pair(ids, dist, cord, t_ee)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(L_idx)
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap(edge_aabb, R_aabb, gap):
                    R_element = self.bvh_edges.node_data[R_idx][3]
                    if R_element != INVALID:
                        # Leaf node - narrow phase
                        ej = ti.cast(R_element, ti.i32)
                        if ei < ej:
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self.mesh.verts.x[b0]
                                x_b1 = self.mesh.verts.x[b1]
                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self._detection_dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()
                                    if dist < self._detection_dHat and dist > self.SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                                        self._add_contact_pair(ids, dist, cord, t_ee)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(R_idx)
                            stack_ptr += 1

    def find_cnts(self, PRINT=False, TIME_LOG=False, use_refit=False):
        """Find all collision constraints using BVH."""
        if TIME_LOG:
            ti.sync()
            t_start = time.perf_counter()

        self.n_contacts[None] = 0

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
            t_pt_start = time.perf_counter()

        self.find_constraints_PT_bvh()

        if TIME_LOG:
            ti.sync()
            t_pt_end = time.perf_counter()
            t_ee_start = time.perf_counter()

        self.find_constraints_EE_bvh()

        if TIME_LOG:
            ti.sync()
            t_ee_end = time.perf_counter()
            t_total = t_ee_end - t_start
            bvh_op = "refit" if use_refit else "build"
            print(f"[BVH-Final Time] {bvh_op}: {(t_bvh_end - t_bvh_start)*1000:.2f}ms, "
                  f"PT: {(t_pt_end - t_pt_start)*1000:.2f}ms, "
                  f"EE: {(t_ee_end - t_ee_start)*1000:.2f}ms, "
                  f"total: {t_total*1000:.2f}ms")

        if PRINT:
            N = self.print_cnts()
            return N

    def find_cnts_iter(self, iter, rate=10, PRINT=False):
        """Find constraints with smart BVH update strategy."""
        self.n_contacts[None] = 0

        if iter == 0:
            self.build_bvh()
        else:
            self.refit_bvh()

        self.find_constraints_PT_bvh()
        self.find_constraints_EE_bvh()

        if PRINT:
            self.print_cnts()

    # ===================== Penetration Detection API =====================

    @ti.func
    def _check_edge_tri_leaf(self, edge_idx: ti.i32, t0: ti.i32, t1: ti.i32, t2: ti.i32,
                              x0: ti.template(), x1: ti.template(), x2: ti.template()) -> ti.i32:
        """Check if edge intersects triangle."""
        result = 0
        a0 = self.boundary_edges[edge_idx, 0]
        a1 = self.boundary_edges[edge_idx, 1]
        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            if segment_triangle_intersect_cramer(x_a0, x_a1, x0, x1, x2):
                result = 1
        return result

    @ti.kernel
    def _edge_triangle_intersection_impl(self, count_mode: ti.template()) -> ti.i32:
        """Edge-triangle intersection detection using LBVH Final."""
        INVALID = ti.u32(0xFFFFFFFF)
        result = 0
        gap = 0.0

        for ti_idx in range(self.n_boundary_triangles):
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

                # Triangle AABB (packed)
                tri_lower = ti.min(ti.min(x0, x1), x2)
                tri_upper = ti.max(ti.max(x0, x1), x2)
                tri_aabb = ti.Vector([tri_lower[0], tri_lower[1], tri_lower[2],
                                      tri_upper[0], tri_upper[1], tri_upper[2]])

                stack = ti.Vector.zero(ti.u32, 64)
                stack_ptr = 0
                stack[stack_ptr] = 0
                stack_ptr += 1

                while stack_ptr > 0:
                    if ti.static(not count_mode):
                        if result != 0:
                            break

                    stack_ptr -= 1
                    node_id = ti.cast(stack[stack_ptr], ti.i32)

                    node_data = self.bvh_edges.node_data[node_id]
                    L_idx = ti.cast(node_data[1], ti.i32)
                    R_idx = ti.cast(node_data[2], ti.i32)

                    L_aabb = self.bvh_edges.aabb[L_idx]
                    R_aabb = self.bvh_edges.aabb[R_idx]

                    if self._aabb_overlap(tri_aabb, L_aabb, gap):
                        L_element = self.bvh_edges.node_data[L_idx][3]
                        if L_element != INVALID:
                            if self._check_edge_tri_leaf(ti.cast(L_element, ti.i32), t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            if stack_ptr < 63:
                                stack[stack_ptr] = ti.u32(L_idx)
                                stack_ptr += 1

                    if self._aabb_overlap(tri_aabb, R_aabb, gap):
                        R_element = self.bvh_edges.node_data[R_idx][3]
                        if R_element != INVALID:
                            if self._check_edge_tri_leaf(ti.cast(R_element, ti.i32), t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            if stack_ptr < 63:
                                stack[stack_ptr] = ti.u32(R_idx)
                                stack_ptr += 1

        return result

    def check_edge_triangle_intersection_bvh(self) -> int:
        """Check if any edge intersects any triangle."""
        return self._edge_triangle_intersection_impl(count_mode=False)

    @ti.kernel
    def check_ground_intersection(self, ground_normal: ti.template(),
                                   ground_offset: ti.f32) -> ti.i32:
        """Check if any vertex penetrated ground plane."""
        result = 0
        for i in range(self.n_verts):
            if result == 0:
                vertex_pos = self.mesh.verts.x[i]
                dist = vertex_pos.dot(ground_normal) - ground_offset
                if dist < 0.0:
                    result = 1
        return result

    def check_dcd(self):
        """Check discrete collision detection."""
        self.build_bvh()
        return self.check_edge_triangle_intersection_bvh()

    def is_intersected(self, check_ground=True):
        """Check if mesh has any penetration/intersection."""
        self.build_bvh()

        if check_ground and hasattr(self, 'ground') and self.ground is not None:
            ground_y = self.ground
            ground_normal = ti.Vector([0.0, 1.0, 0.0])
            if self.check_ground_intersection(ground_normal, ground_y):
                print("[Penetration] Ground intersection detected!")
                return True

        if self.check_edge_triangle_intersection_bvh():
            print("[Penetration] Edge-triangle intersection detected!")
            return True

        return False

    def check_penetration(self, verbose=True):
        """Alias for is_intersected()."""
        result = self.is_intersected()
        if verbose:
            if result:
                print("[Penetration Check] FAILED - Mesh has penetration!")
            else:
                print("[Penetration Check] PASSED - No penetration detected.")
        return result

    def count_edge_triangle_intersections(self) -> int:
        """Count total edge-triangle intersections."""
        return self._edge_triangle_intersection_impl(count_mode=True)

    @ti.kernel
    def count_ground_penetrations(self, ground_normal: ti.template(),
                                   ground_offset: ti.f32) -> ti.i32:
        """Count vertices penetrated ground plane."""
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
            total += (self.mesh.verts.x[self.boundary_edges[i, 0]] -
                      self.mesh.verts.x[self.boundary_edges[i, 1]]).norm()
        result = total / ti.cast(self.n_boundary_edges, float)
        print("Mean of boundary edges:", result)
        return result

    @ti.kernel
    def compute_min_of_boundary_edges(self) -> float:
        ret = 10.0
        for i in range(self.n_boundary_edges):
            dist = (self.mesh.verts.x[self.boundary_edges[i, 0]] -
                    self.mesh.verts.x[self.boundary_edges[i, 1]]).norm()
            ti.atomic_min(ret, dist)
        print("Min of boundary edges:", ret)
        return ret
