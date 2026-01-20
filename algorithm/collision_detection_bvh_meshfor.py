"""
Collision detection module using Linear BVH (LBVH) for broad-phase.
Uses meshtaichi_custom.BoundaryMesh for optimized mesh-for loops on boundary elements.

This is an optimized version of collision_detection_bvh.py that leverages
MeshTaichi's mesh-for loops and ti.mesh_local for better GPU performance.
"""

import time
import sys
import os

# Add meshtaichi_custom parent directory to path (so it can be imported as package)
_meshtaichi_custom_parent = '/root'
if _meshtaichi_custom_parent not in sys.path:
    sys.path.insert(0, _meshtaichi_custom_parent)

from algorithm.pncg_base_collision_free import *
from algorithm.lbvh import LBVH_Triangles, LBVH_Edges
from math_utils.graphic_util import *
from util.model_loading import *

# Import BoundaryMesh from meshtaichi_custom
from meshtaichi_custom import BoundaryMesh


@ti.data_oriented
class collision_detection_bvh_meshfor_module(pncg_base_deformer):
    """
    Collision detection using LBVH for broad-phase with mesh-for optimization.
    Uses BoundaryMesh for optimized edge/triangle iteration.
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

        # Original boundary data (kept for compatibility)
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size ', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.dHat = model.dHat

        # Create BoundaryMesh for optimized mesh-for loops
        print('[MeshFor] Creating BoundaryMesh...')
        boundary_points_np = self.boundary_points.to_numpy()
        boundary_edges_np = self.boundary_edges.to_numpy()
        boundary_triangles_np = self.boundary_triangles.to_numpy()

        self.boundary_mesh = BoundaryMesh(
            parent_mesh=self.mesh,
            boundary_data={
                'boundary_points': boundary_points_np,
                'boundary_edges': boundary_edges_np,
                'boundary_triangles': boundary_triangles_np
            },
            patch_size=256
        )
        print('[MeshFor] BoundaryMesh created successfully')

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

        # Detection threshold
        self._detection_dHat = self.dHat

        # Pre-computed BVH AABB gap for broad-phase filtering
        self.bvh_gap = ti.sqrt(self._detection_dHat)

        self.attempt_PT = self.attempt_PT_no_adj
        self.attempt_EE = self.attempt_EE_no_adj

        print('BVH initialization complete.')

    def set_detection_dHat(self, detection_dHat: float):
        """Set custom detection threshold for contact filtering."""
        self._detection_dHat = detection_dHat
        self.bvh_gap = ti.sqrt(detection_dHat)

    def get_detection_dHat(self) -> float:
        """Get the current detection threshold."""
        return self._detection_dHat

    def build_bvh(self):
        """Build BVH trees for triangles and edges."""
        # Sync boundary mesh positions first
        self.boundary_mesh.sync_positions()

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
        # Sync boundary mesh positions first
        self.boundary_mesh.sync_positions()

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
    def find_constraints_EE_bvh_meshfor(self):
        """
        Find Edge-Edge constraints using BVH traversal with mesh-for loop.
        Uses BoundaryMesh for optimized edge iteration.
        """
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self.bvh_gap

        # Use mesh-for loop with ti.mesh_local optimization
        ti.mesh_local(self.boundary_mesh.verts.x)

        for e in self.boundary_mesh.edges:
            # Get edge vertices using mesh-for syntax
            v0 = e.verts[0]
            v1 = e.verts[1]
            x_a0 = v0.x
            x_a1 = v1.x

            # Get global vertex indices for contact pair storage
            ei = e.id
            global_verts = self.boundary_mesh.edge_global_verts[ei]
            a0 = global_verts[0]
            a1 = global_verts[1]

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

    @ti.kernel
    def find_constraints_EE_bvh_range(self):
        """
        Find Edge-Edge constraints using BVH traversal with range loop.
        Original implementation for comparison.
        """
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
        """Check if edge AABB overlaps with BVH node AABB."""
        node_lower = self.bvh_edges.bv_lower[node_idx]
        node_upper = self.bvh_edges.bv_upper[node_idx]

        return ((node_upper[0] - edge_lower[0]) > -gap and (edge_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - edge_lower[1]) > -gap and (edge_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - edge_lower[2]) > -gap and (edge_upper[2] - node_lower[2]) > -gap)

    @ti.func
    def _add_contact_pair(self, ids: ti.template(), dist: float, cord: ti.template(), t_vec: ti.template()):
        """Add a contact pair to compact storage."""
        idx = ti.atomic_add(self.n_contacts[None], 1)
        if idx < self.MAX_C:
            self.contact_pairs[idx] = self.pair(ids, dist, cord, t_vec)

    @ti.func
    def attempt_PT_no_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self._detection_dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self._detection_dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                self._add_contact_pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_EE_no_adj(self, edge_id_0, edge_id_1, a0, a1, b0, b1, x_a0, x_a1, x_b0, x_b1):
        if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self._detection_dHat):
            t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
            dist = t_ee.norm()
            if dist < self._detection_dHat and ti.abs(dist) > self.SMALL_NUM:
                cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                self._add_contact_pair(ids, dist, cord, t_ee)

    def find_cnts(self, PRINT=False, TIME_LOG=False, use_refit=False, use_meshfor=True):
        """Find all collision constraints using BVH.

        Args:
            PRINT: Print constraint count
            TIME_LOG: Print timing information
            use_refit: Use BVH refit instead of full rebuild
            use_meshfor: Use mesh-for loop for EE detection (default True)
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

        if use_meshfor:
            self.find_constraints_EE_bvh_meshfor()
        else:
            self.find_constraints_EE_bvh_range()

        if TIME_LOG:
            ti.sync()
            t_ee_end = time.perf_counter()
            t_total = t_ee_end - t_start
            bvh_op = "refit" if use_refit else "build"
            ee_method = "meshfor" if use_meshfor else "range"
            print(f"[BVH Time] {bvh_op}: {(t_bvh_end - t_bvh_start)*1000:.2f}ms, "
                  f"PT: {(t_pt_end - t_pt_start)*1000:.2f}ms, "
                  f"EE({ee_method}): {(t_ee_end - t_ee_start)*1000:.2f}ms, "
                  f"total: {t_total*1000:.2f}ms")

        if PRINT:
            N = self.print_cnts()
            return N

    def find_cnts_iter(self, iter, rate=10, PRINT=False, use_meshfor=True):
        """Find constraints with smart BVH update strategy."""
        self.n_contacts[None] = 0

        if iter == 0:
            self.build_bvh()
        else:
            self.refit_bvh()

        self.find_constraints_PT_bvh()

        if use_meshfor:
            self.find_constraints_EE_bvh_meshfor()
        else:
            self.find_constraints_EE_bvh_range()

        if PRINT:
            self.print_cnts()

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

    def benchmark_ee_methods(self, n_iterations=10, warmup=3):
        """
        Benchmark mesh-for vs range loop for EE detection.

        Args:
            n_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations

        Returns:
            Dict with timing results
        """
        print(f"\n{'='*60}")
        print("Benchmarking EE Detection Methods")
        print(f"{'='*60}")
        print(f"Boundary edges: {self.n_boundary_edges}")
        print(f"Iterations: {n_iterations} (warmup: {warmup})")

        # Ensure BVH is built
        self.build_bvh()

        # Warmup
        print("\nWarming up...")
        for _ in range(warmup):
            self.n_contacts[None] = 0
            self.find_constraints_EE_bvh_range()
            ti.sync()
            self.n_contacts[None] = 0
            self.find_constraints_EE_bvh_meshfor()
            ti.sync()

        # Benchmark range loop
        print("\nBenchmarking range loop...")
        range_times = []
        for i in range(n_iterations):
            self.n_contacts[None] = 0
            ti.sync()
            t_start = time.perf_counter()
            self.find_constraints_EE_bvh_range()
            ti.sync()
            t_end = time.perf_counter()
            range_times.append((t_end - t_start) * 1000)

        range_contacts = self.n_contacts[None]

        # Benchmark mesh-for loop
        print("Benchmarking mesh-for loop...")
        meshfor_times = []
        for i in range(n_iterations):
            self.n_contacts[None] = 0
            ti.sync()
            t_start = time.perf_counter()
            self.find_constraints_EE_bvh_meshfor()
            ti.sync()
            t_end = time.perf_counter()
            meshfor_times.append((t_end - t_start) * 1000)

        meshfor_contacts = self.n_contacts[None]

        # Calculate statistics
        import numpy as np
        range_mean = np.mean(range_times)
        range_std = np.std(range_times)
        meshfor_mean = np.mean(meshfor_times)
        meshfor_std = np.std(meshfor_times)

        speedup = range_mean / meshfor_mean if meshfor_mean > 0 else 0

        # Print results
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(f"Range loop:    {range_mean:.3f} ± {range_std:.3f} ms  (contacts: {range_contacts})")
        print(f"Mesh-for loop: {meshfor_mean:.3f} ± {meshfor_std:.3f} ms  (contacts: {meshfor_contacts})")
        print(f"Speedup:       {speedup:.2f}x")
        print(f"{'='*60}\n")

        return {
            'range_mean': range_mean,
            'range_std': range_std,
            'meshfor_mean': meshfor_mean,
            'meshfor_std': meshfor_std,
            'speedup': speedup,
            'range_contacts': range_contacts,
            'meshfor_contacts': meshfor_contacts,
        }

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
