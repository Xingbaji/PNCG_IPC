"""
Final optimized BVH-based collision detector using LBVH Final.

Key optimizations over bvh_detector.py:
1. Uses LBVH_Final with packed AABB (vec6) for better cache efficiency
2. Prefetch both children AABBs before deciding traversal order
3. Inline AABB overlap checks to avoid function call overhead
4. Fused narrow-phase operations with early exits

Based on algorithm/collision_detection_bvh_final.py.
"""

import taichi as ti
from typing import Optional, Any
from dataclasses import dataclass

from .registry import CollisionDetectorRegistry
from .contact_pair import ContactPairStorage
from .lbvh import LBVH_Triangles_Final, LBVH_Edges_Final
from .queries.distance import dist3D_Point_Triangle, dist3D_Segment_to_Segment
from .queries.ccd import point_triangle_ccd_broadphase, edge_edge_ccd_broadphase
from .queries.intersection import segment_triangle_intersect_cramer
from ..core.precision import PrecisionType, PrecisionMixin


@dataclass
class SurfaceData:
    """Container for surface mesh data used by collision detection."""
    boundary_points: Any
    boundary_edges: Any
    boundary_triangles: Any
    n_boundary_points: int
    n_boundary_edges: int
    n_boundary_triangles: int


@CollisionDetectorRegistry.register('bvh_final')
@ti.data_oriented
class BVHCollisionDetectorFinal(PrecisionMixin):
    """
    Final optimized BVH collision detector using LBVH Final.

    Optimizations:
    - Packed AABB (vec6) storage for cache efficiency
    - Packed node topology (vec4<u32>)
    - Inline AABB overlap checks
    - Prefetch both children before traversal decision
    - Supports refit for incremental updates

    Usage:
        detector = BVHCollisionDetectorFinal(precision='f32')
        detector.init(mesh, surface_data)
        n_contacts = detector.find_contacts(mesh, dHat=0.01)
    """

    def __init__(
        self,
        precision: PrecisionType = 'f32',
        max_contacts: int = 2**21,
    ):
        self.init_precision(precision)
        self._max_contacts = max_contacts
        self.SMALL_NUM = 1e-6

        self._storage: Optional[ContactPairStorage] = None
        self._bvh_triangles: Optional[LBVH_Triangles_Final] = None
        self._bvh_edges: Optional[LBVH_Edges_Final] = None

        self._mesh = None
        self._boundary_points = None
        self._boundary_edges = None
        self._boundary_triangles = None
        self._n_boundary_points = 0
        self._n_boundary_edges = 0
        self._n_boundary_triangles = 0

        self._detection_dHat = 0.0
        self._bvh_gap = 0.0

        self._initialized = False

    def init(self, mesh: Any, surface_data: SurfaceData) -> None:
        """Initialize collision detection structures."""
        self._mesh = mesh
        self._boundary_points = surface_data.boundary_points
        self._boundary_edges = surface_data.boundary_edges
        self._boundary_triangles = surface_data.boundary_triangles
        self._n_boundary_points = surface_data.n_boundary_points
        self._n_boundary_edges = surface_data.n_boundary_edges
        self._n_boundary_triangles = surface_data.n_boundary_triangles

        # Create LBVH Final structures
        self._bvh_triangles = LBVH_Triangles_Final(
            self._n_boundary_triangles,
            precision=self.precision
        )
        self._bvh_edges = LBVH_Edges_Final(
            self._n_boundary_edges,
            precision=self.precision
        )

        # Create contact storage
        self._storage = ContactPairStorage(
            max_contacts=self._max_contacts,
            precision=self.precision
        )

        self._initialized = True

    def find_contacts(self, mesh: Any, dHat: float, use_refit: bool = False) -> int:
        """
        Find contact pairs within distance threshold.

        Args:
            mesh: MeshTaichi mesh object
            dHat: Distance threshold for contact detection
            use_refit: If True, use refit instead of rebuild (faster for small motions)

        Returns:
            Number of contact pairs found
        """
        if not self._initialized:
            raise RuntimeError("BVHCollisionDetectorFinal not initialized.")

        self._set_detection_threshold(dHat)
        self._storage.reset()

        # Build or refit BVH
        if use_refit:
            self._refit_bvh()
        else:
            self._build_bvh()

        # Find PT and EE contacts
        self._find_constraints_PT_bvh()
        self._find_constraints_EE_bvh()

        return self._storage.count

    @property
    def contact_pairs(self) -> Any:
        if self._storage is None:
            raise RuntimeError("Not initialized.")
        return self._storage.contact_pairs

    @property
    def n_contacts(self) -> int:
        if self._storage is None:
            return 0
        return self._storage.count

    def _set_detection_threshold(self, detection_dHat: float):
        self._detection_dHat = detection_dHat
        self._bvh_gap = ti.sqrt(detection_dHat)

    def _build_bvh(self):
        """Build BVH trees from scratch."""
        self._bvh_triangles.build(
            self._mesh.verts.x,
            self._boundary_triangles,
            self._n_boundary_triangles
        )
        self._bvh_edges.build(
            self._mesh.verts.x,
            self._boundary_edges,
            self._n_boundary_edges
        )

    def _refit_bvh(self):
        """Refit BVH trees (fast AABB update, preserves structure)."""
        self._bvh_triangles.refit(
            self._mesh.verts.x,
            self._boundary_triangles,
            self._n_boundary_triangles
        )
        self._bvh_edges.refit(
            self._mesh.verts.x,
            self._boundary_edges,
            self._n_boundary_edges
        )

    # ==================== Inline AABB Overlap Checks ====================

    @ti.func
    def _point_aabb_overlap(self, point: ti.template(), aabb: ti.template(), gap: ti.template()) -> bool:
        """Check point-AABB overlap. aabb format: [lx, ly, lz, ux, uy, uz]"""
        return (point[0] >= aabb[0] - gap and point[0] <= aabb[3] + gap and
                point[1] >= aabb[1] - gap and point[1] <= aabb[4] + gap and
                point[2] >= aabb[2] - gap and point[2] <= aabb[5] + gap)

    @ti.func
    def _aabb_overlap(self, aabb1: ti.template(), aabb2: ti.template(), gap: ti.template()) -> bool:
        """Check AABB-AABB overlap. Both format: [lx, ly, lz, ux, uy, uz]"""
        return (aabb1[3] + gap >= aabb2[0] and aabb2[3] + gap >= aabb1[0] and
                aabb1[4] + gap >= aabb2[1] and aabb2[4] + gap >= aabb1[1] and
                aabb1[5] + gap >= aabb2[2] and aabb2[5] + gap >= aabb1[2])

    @ti.func
    def _add_contact_pair(self, ids: ti.template(), dist: ti.template(),
                          cord: ti.template(), t_vec: ti.template()):
        """Add contact pair with atomic counter."""
        self._storage.add_pair(ids, dist, cord, t_vec)

    # ==================== PT Detection with LBVH Final ====================

    @ti.kernel
    def _find_constraints_PT_bvh(self):
        """Find Point-Triangle constraints using LBVH Final traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self._bvh_gap
        dHat = self._detection_dHat
        SMALL_NUM = self.SMALL_NUM

        for pi in range(self._n_boundary_points):
            p = self._boundary_points[pi]
            xp = self._mesh.verts.x[p]

            # BVH traversal stack
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Start from root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = ti.cast(stack[stack_ptr], ti.i32)

                # Get node data (packed: [parent, left, right, element])
                node_data = self._bvh_triangles.node_data[node_id]
                L_idx = ti.cast(node_data[1], ti.i32)
                R_idx = ti.cast(node_data[2], ti.i32)

                # Prefetch both children AABBs
                L_aabb = self._bvh_triangles.aabb[L_idx]
                R_aabb = self._bvh_triangles.aabb[R_idx]

                # Check left child
                L_overlap = self._point_aabb_overlap(xp, L_aabb, gap)
                if L_overlap:
                    L_element = self._bvh_triangles.node_data[L_idx][3]
                    if L_element != INVALID:
                        # Leaf node - narrow phase
                        tri_id = ti.cast(L_element, ti.i32)
                        t0 = self._boundary_triangles[tri_id, 0]
                        t1 = self._boundary_triangles[tri_id, 1]
                        t2 = self._boundary_triangles[tri_id, 2]
                        if p != t0 and p != t1 and p != t2:
                            x0 = self._mesh.verts.x[t0]
                            x1 = self._mesh.verts.x[t1]
                            x2 = self._mesh.verts.x[t2]
                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()
                                if dist < dHat and dist > SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.u32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2])
                                    self._add_contact_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(L_idx)
                            stack_ptr += 1

                # Check right child
                R_overlap = self._point_aabb_overlap(xp, R_aabb, gap)
                if R_overlap:
                    R_element = self._bvh_triangles.node_data[R_idx][3]
                    if R_element != INVALID:
                        # Leaf node - narrow phase
                        tri_id = ti.cast(R_element, ti.i32)
                        t0 = self._boundary_triangles[tri_id, 0]
                        t1 = self._boundary_triangles[tri_id, 1]
                        t2 = self._boundary_triangles[tri_id, 2]
                        if p != t0 and p != t1 and p != t2:
                            x0 = self._mesh.verts.x[t0]
                            x1 = self._mesh.verts.x[t1]
                            x2 = self._mesh.verts.x[t2]
                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()
                                if dist < dHat and dist > SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.u32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2])
                                    self._add_contact_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(R_idx)
                            stack_ptr += 1

    # ==================== EE Detection with LBVH Final ====================

    @ti.kernel
    def _find_constraints_EE_bvh(self):
        """Find Edge-Edge constraints using LBVH Final traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self._bvh_gap
        dHat = self._detection_dHat
        SMALL_NUM = self.SMALL_NUM
        n_edges = self._n_boundary_edges

        for ei in range(n_edges):
            a0 = self._boundary_edges[ei, 0]
            a1 = self._boundary_edges[ei, 1]
            x_a0 = self._mesh.verts.x[a0]
            x_a1 = self._mesh.verts.x[a1]

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
                node_data = self._bvh_edges.node_data[node_id]
                L_idx = ti.cast(node_data[1], ti.i32)
                R_idx = ti.cast(node_data[2], ti.i32)

                # Prefetch both children AABBs
                L_aabb = self._bvh_edges.aabb[L_idx]
                R_aabb = self._bvh_edges.aabb[R_idx]

                # Check left child
                if self._aabb_overlap(edge_aabb, L_aabb, gap):
                    L_element = self._bvh_edges.node_data[L_idx][3]
                    if L_element != INVALID:
                        # Leaf node - narrow phase
                        ej = ti.cast(L_element, ti.i32)
                        if ei < ej:  # Avoid duplicates
                            b0 = self._boundary_edges[ej, 0]
                            b1 = self._boundary_edges[ej, 1]
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self._mesh.verts.x[b0]
                                x_b1 = self._mesh.verts.x[b1]
                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()
                                    if dist < dHat and dist > SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc])
                                        ids = ti.Vector([a0, a1, b0, b1], ti.u32)
                                        self._add_contact_pair(ids, dist, cord, t_ee)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(L_idx)
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap(edge_aabb, R_aabb, gap):
                    R_element = self._bvh_edges.node_data[R_idx][3]
                    if R_element != INVALID:
                        # Leaf node - narrow phase
                        ej = ti.cast(R_element, ti.i32)
                        if ei < ej:
                            b0 = self._boundary_edges[ej, 0]
                            b1 = self._boundary_edges[ej, 1]
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self._mesh.verts.x[b0]
                                x_b1 = self._mesh.verts.x[b1]
                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()
                                    if dist < dHat and dist > SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc])
                                        ids = ti.Vector([a0, a1, b0, b1], ti.u32)
                                        self._add_contact_pair(ids, dist, cord, t_ee)
                    else:
                        if stack_ptr < 63:
                            stack[stack_ptr] = ti.u32(R_idx)
                            stack_ptr += 1

    # ==================== Penetration Detection ====================

    @ti.func
    def _check_edge_tri_leaf(self, edge_idx: ti.i32, t0: ti.i32, t1: ti.i32, t2: ti.i32,
                              x0: ti.template(), x1: ti.template(), x2: ti.template()) -> ti.i32:
        """Check if edge intersects triangle."""
        result = 0
        a0 = self._boundary_edges[edge_idx, 0]
        a1 = self._boundary_edges[edge_idx, 1]
        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
            x_a0 = self._mesh.verts.x[a0]
            x_a1 = self._mesh.verts.x[a1]
            if segment_triangle_intersect_cramer(x_a0, x_a1, x0, x1, x2):
                result = 1
        return result

    @ti.kernel
    def _edge_triangle_intersection_impl(self, count_mode: ti.template()) -> ti.i32:
        """Edge-triangle intersection detection using LBVH Final."""
        INVALID = ti.u32(0xFFFFFFFF)
        result = 0
        gap = 0.0

        for tri_idx in range(self._n_boundary_triangles):
            should_process = True
            if ti.static(not count_mode):
                should_process = (result == 0)

            if should_process:
                t0 = self._boundary_triangles[tri_idx, 0]
                t1 = self._boundary_triangles[tri_idx, 1]
                t2 = self._boundary_triangles[tri_idx, 2]
                x0 = self._mesh.verts.x[t0]
                x1 = self._mesh.verts.x[t1]
                x2 = self._mesh.verts.x[t2]

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

                    node_data = self._bvh_edges.node_data[node_id]
                    L_idx = ti.cast(node_data[1], ti.i32)
                    R_idx = ti.cast(node_data[2], ti.i32)

                    L_aabb = self._bvh_edges.aabb[L_idx]
                    R_aabb = self._bvh_edges.aabb[R_idx]

                    if self._aabb_overlap(tri_aabb, L_aabb, gap):
                        L_element = self._bvh_edges.node_data[L_idx][3]
                        if L_element != INVALID:
                            if self._check_edge_tri_leaf(ti.cast(L_element, ti.i32), t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            if stack_ptr < 63:
                                stack[stack_ptr] = ti.u32(L_idx)
                                stack_ptr += 1

                    if self._aabb_overlap(tri_aabb, R_aabb, gap):
                        R_element = self._bvh_edges.node_data[R_idx][3]
                        if R_element != INVALID:
                            if self._check_edge_tri_leaf(ti.cast(R_element, ti.i32), t0, t1, t2, x0, x1, x2):
                                ti.atomic_add(result, 1)
                        else:
                            if stack_ptr < 63:
                                stack[stack_ptr] = ti.u32(R_idx)
                                stack_ptr += 1

        return result

    def check_penetration(self, check_ground: bool = True, ground_y: float = 0.0) -> bool:
        """Check if mesh has any penetration/intersection."""
        if not self._initialized:
            raise RuntimeError("Not initialized.")

        self._build_bvh()

        if check_ground:
            if self._check_ground_intersection(ground_y):
                return True

        if self._edge_triangle_intersection_impl(count_mode=False):
            return True

        return False

    @ti.kernel
    def _check_ground_intersection(self, ground_y: ti.template()) -> ti.i32:
        """Check if any vertex is below ground."""
        result = 0
        for i in self._mesh.verts:
            if result == 0:
                if self._mesh.verts.x[i][1] < ground_y:
                    result = 1
        return result

    def is_intersected(self, check_ground: bool = True, ground_y: float = 0.0) -> bool:
        """Alias for check_penetration."""
        return self.check_penetration(check_ground, ground_y)

    @ti.kernel
    def print_contacts_info(self) -> ti.i32:
        """Print contact information and return count."""
        N = self._storage._n_contacts[None]
        min_dist = 1.0
        for idx in range(N):
            pair = self._storage.contact_pairs[idx]
            dist = pair.b
            ti.atomic_min(min_dist, dist)
        print('number of contacts:', N, 'min dist:', min_dist)
        return N
