"""
Collision detection module using Linear BVH (LBVH) for broad-phase.
Replaces spatial hashing with BVH-based collision detection.
"""

import time
from algorithm.pncg_base_collision_free import *
from algorithm.lbvh import LBVH_Triangles, LBVH_Edges
from math_utils.graphic_util import *
from math_utils.matrix_util import compute_dtdx_t, compute_d_dtdx
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
        self.adj = model.adj
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

        # Constraint storage using bitmasked sparse field (legacy, kept for compatibility)
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),  # ids
            b=float,  # dist
            c=ti.types.vector(4, float),  # cord
            d=ti.types.vector(3, float)  # t (direction vector)
        )
        self.cid = self.pair.field()
        self.cid_root = ti.root.bitmasked(ti.ij, (2, self.MAX_C)).place(self.cid)

        # ============================================================
        # Compact array storage for contacts (P0 optimization)
        # This provides O(N) iteration instead of O(MAX_C) bitmask scan
        # ============================================================
        self.contact_pairs = self.pair.field(shape=self.MAX_C)
        self.n_contacts = ti.field(dtype=ti.i32, shape=())
        # Flag to control which storage to use
        self.use_compact_storage = True

        # Stack for BVH traversal (per-thread)
        self.BVH_STACK_SIZE = 64

        # Collision pair counter
        self.cp_count = ti.field(dtype=ti.i32, shape=())

        if self.adj == 1:
            self.define_adj_matrix()
            self.attempt_PT = self.attempt_PT_adj
            self.attempt_EE = self.attempt_EE_adj
        else:
            self.attempt_PT = self.attempt_PT_no_adj
            self.attempt_EE = self.attempt_EE_no_adj

        print('BVH initialization complete.')

    @ti.func
    def hash_coords_2(self, x, y):
        h = (x * 92837111) ^ (y * 689287499)
        return ti.abs(h) % self.MAX_C

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
        gap = ti.sqrt(self.dHat)
        n_triangles = self.n_boundary_triangles

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
        gap = ti.sqrt(self.dHat)
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

            # Leaf index of current edge in BVH
            self_leaf_idx = ei + n_edges - 1

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
        if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                # Write to compact array (primary storage)
                self._add_contact_pair(ids, dist, cord, t_pt)
                # Also write to bitmasked for backward compatibility
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[0, hash_index] = self.pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_PT_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        hash_adj = self.hash_coords_2(p, triangle_id)
        if p != t0 and p != t1 and p != t2 and self.adj_matrix[hash_adj] == 0 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                # Write to compact array (primary storage)
                self._add_contact_pair(ids, dist, cord, t_pt)
                # Also write to bitmasked for backward compatibility
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[0, hash_index] = self.pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_EE_no_adj(self, edge_id_0, edge_id_1, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1):
        if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
            t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
            dist = t_ee.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                # Write to compact array (primary storage)
                self._add_contact_pair(ids, dist, cord, t_ee)
                # Also write to bitmasked for backward compatibility
                hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                self.cid[1, hash_index] = self.pair(ids, dist, cord, t_ee)

    @ti.func
    def attempt_EE_adj(self, edge_id_0, edge_id_1, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1):
        hash_adj = self.hash_coords_2(self.n_verts + edge_id_0, edge_id_1)
        if self.adj_matrix[hash_adj] == 0:
            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
                t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                dist = t_ee.norm()
                if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                    cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                    ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                    # Write to compact array (primary storage)
                    self._add_contact_pair(ids, dist, cord, t_ee)
                    # Also write to bitmasked for backward compatibility
                    hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                    self.cid[1, hash_index] = self.pair(ids, dist, cord, t_ee)

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

        # Reset both storage systems
        self.cid_root.deactivate_all()
        self.n_contacts[None] = 0  # Reset compact array counter

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
        self.cid_root.deactivate_all()
        self.n_contacts[None] = 0  # Reset compact array counter

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

    @ti.kernel
    def check_collision_3d_bvh(self) -> ti.i32:
        """Check if there's any segment-triangle intersection using BVH."""
        INVALID = ti.u32(0xFFFFFFFF)
        result = 0
        gap = ti.sqrt(self.dHat)

        for ti_idx in range(self.n_boundary_triangles):
            t0 = self.boundary_triangles[ti_idx, 0]
            t1 = self.boundary_triangles[ti_idx, 1]
            t2 = self.boundary_triangles[ti_idx, 2]
            x0 = self.mesh.verts.x[t0]
            x1 = self.mesh.verts.x[t1]
            x2 = self.mesh.verts.x[t2]

            # Triangle AABB
            tri_lower = ti.min(ti.min(x0, x1), x2)
            tri_upper = ti.max(ti.max(x0, x1), x2)

            # BVH traversal
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = self.bvh_edges.left_idx[node_id]
                R_idx = self.bvh_edges.right_idx[node_id]

                # Check left child
                if self._aabb_overlap_with_tri(tri_lower, tri_upper, L_idx, gap):
                    element_idx = self.bvh_edges.element_idx[L_idx]
                    if element_idx != INVALID:
                        j = element_idx
                        a0 = self.boundary_edges[j, 0]
                        a1 = self.boundary_edges[j, 1]
                        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
                            x_a0 = self.mesh.verts.x[a0]
                            x_a1 = self.mesh.verts.x[a1]
                            if segment_intersect_triangle_new(x_a0, x_a1, x0, x1, x2):
                                result = 1
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap_with_tri(tri_lower, tri_upper, R_idx, gap):
                    element_idx = self.bvh_edges.element_idx[R_idx]
                    if element_idx != INVALID:
                        j = element_idx
                        a0 = self.boundary_edges[j, 0]
                        a1 = self.boundary_edges[j, 1]
                        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
                            x_a0 = self.mesh.verts.x[a0]
                            x_a1 = self.mesh.verts.x[a1]
                            if segment_intersect_triangle_new(x_a0, x_a1, x0, x1, x2):
                                result = 1
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

        return result

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
        ret = self.check_collision_3d_bvh()
        return ret

    @ti.kernel
    def print_cnts(self) -> ti.i32:
        N = 0
        min_dist = 1.0
        for k, j in self.cid:
            N += 1
            pair = self.cid[k, j]
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

    def define_adj_matrix(self):
        """Define adjacency matrix to filter constraints at rest pose."""
        self.adj_matrix = ti.field(bool)
        ti.root.bitmasked(ti.i, 2 ** 24).place(self.adj_matrix)

        # Build BVH and find initial constraints
        self.build_bvh()
        self.assign_adj_matrix_EE_bvh()
        self.assign_adj_matrix_PT_bvh()
        print('adj matrix defined')

    @ti.kernel
    def assign_adj_matrix_PT_bvh(self):
        """Assign adjacency matrix for PT pairs using BVH."""
        INVALID = ti.u32(0xFFFFFFFF)
        n_PT = 0
        gap = ti.sqrt(self.dHat)

        for pi in range(self.n_boundary_points):
            p = self.boundary_points[pi]
            xp = self.mesh.verts.x[p]

            # BVH traversal
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = self.bvh_triangles.left_idx[node_id]
                R_idx = self.bvh_triangles.right_idx[node_id]

                if self.bvh_triangles.aabb_overlap_point(xp, L_idx, gap):
                    element_idx = self.bvh_triangles.element_idx[L_idx]
                    if element_idx != INVALID:
                        i = element_idx
                        t0 = self.boundary_triangles[i, 0]
                        t1 = self.boundary_triangles[i, 1]
                        t2 = self.boundary_triangles[i, 2]
                        x0 = self.mesh.verts.x[t0]
                        x1 = self.mesh.verts.x[t1]
                        x2 = self.mesh.verts.x[t2]
                        if point_triangle_ccd_broadphase(xp, x0, x1, x2, 1.0 * self.dHat) and p != t0 and p != t1 and p != t2:
                            hash_index = self.hash_coords_2(p, i)
                            self.adj_matrix[hash_index] = 1
                            ti.atomic_add(n_PT, 1)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                if self.bvh_triangles.aabb_overlap_point(xp, R_idx, gap):
                    element_idx = self.bvh_triangles.element_idx[R_idx]
                    if element_idx != INVALID:
                        i = element_idx
                        t0 = self.boundary_triangles[i, 0]
                        t1 = self.boundary_triangles[i, 1]
                        t2 = self.boundary_triangles[i, 2]
                        x0 = self.mesh.verts.x[t0]
                        x1 = self.mesh.verts.x[t1]
                        x2 = self.mesh.verts.x[t2]
                        if point_triangle_ccd_broadphase(xp, x0, x1, x2, 1.0 * self.dHat) and p != t0 and p != t1 and p != t2:
                            hash_index = self.hash_coords_2(p, i)
                            self.adj_matrix[hash_index] = 1
                            ti.atomic_add(n_PT, 1)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

        print('n_PT', n_PT)

    @ti.kernel
    def assign_adj_matrix_EE_bvh(self):
        """Assign adjacency matrix for EE pairs using BVH."""
        INVALID = ti.u32(0xFFFFFFFF)
        n_EE = 0
        gap = ti.sqrt(self.dHat)
        n_edges = self.n_boundary_edges

        for ei in range(n_edges):
            a0 = self.boundary_edges[ei, 0]
            a1 = self.boundary_edges[ei, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]

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

                L_idx = self.bvh_edges.left_idx[node_id]
                R_idx = self.bvh_edges.right_idx[node_id]

                if self._aabb_overlap_with_edge(edge_lower, edge_upper, L_idx, gap):
                    element_idx = self.bvh_edges.element_idx[L_idx]
                    if element_idx != INVALID:
                        ej = element_idx
                        if ei < ej:
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            x_b0 = self.mesh.verts.x[b0]
                            x_b1 = self.mesh.verts.x[b1]
                            if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, 1.0 * self.dHat) and a0 != b0 and a1 != b1 and a0 != b1 and a1 != b0:
                                hash_index = self.hash_coords_2(self.n_verts + ei, ej)
                                self.adj_matrix[hash_index] = 1
                                ti.atomic_add(n_EE, 1)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                if self._aabb_overlap_with_edge(edge_lower, edge_upper, R_idx, gap):
                    element_idx = self.bvh_edges.element_idx[R_idx]
                    if element_idx != INVALID:
                        ej = element_idx
                        if ei < ej:
                            b0 = self.boundary_edges[ej, 0]
                            b1 = self.boundary_edges[ej, 1]
                            x_b0 = self.mesh.verts.x[b0]
                            x_b1 = self.mesh.verts.x[b1]
                            if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, 1.0 * self.dHat) and a0 != b0 and a1 != b1 and a0 != b1 and a1 != b0:
                                hash_index = self.hash_coords_2(self.n_verts + ei, ej)
                                self.adj_matrix[hash_index] = 1
                                ti.atomic_add(n_EE, 1)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

        print('n_EE', n_EE)
