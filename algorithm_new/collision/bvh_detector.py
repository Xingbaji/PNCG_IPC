"""
BVH-based collision detector implementing the CollisionDetector protocol.

Provides Point-Triangle (PT) and Edge-Edge (EE) collision detection
using Linear BVH (LBVH) for broad-phase acceleration.
"""

import taichi as ti
from typing import Optional, Any
from dataclasses import dataclass

from .registry import CollisionDetectorRegistry
from .contact_pair import ContactPairStorage
from .lbvh import LBVH_Triangles, LBVH_Edges
from .queries.distance import dist3D_Point_Triangle, dist3D_Segment_to_Segment
from .queries.ccd import point_triangle_ccd_broadphase, edge_edge_ccd_broadphase
from .queries.intersection import segment_triangle_intersect_cramer
from ..core.precision import PrecisionType, PrecisionMixin


@dataclass
class SurfaceData:
    """Container for surface mesh data used by collision detection."""
    boundary_points: Any  # Indices of boundary vertices
    boundary_edges: Any   # Edge vertex pairs (Nx2)
    boundary_triangles: Any  # Triangle vertex triples (Nx3)
    n_boundary_points: int
    n_boundary_edges: int
    n_boundary_triangles: int


@CollisionDetectorRegistry.register('bvh')
@ti.data_oriented
class BVHCollisionDetector(PrecisionMixin):
    """
    BVH-based collision detector implementing CollisionDetector protocol.

    Uses LBVH (Linear Bounding Volume Hierarchy) for efficient broad-phase
    collision detection, followed by narrow-phase distance computation.

    Features:
    - Point-Triangle (PT) contact detection
    - Edge-Edge (EE) contact detection
    - Penetration detection (edge-triangle intersection)
    - Ground plane collision support
    - Precision support (f32/f64)

    Usage:
        detector = BVHCollisionDetector(precision='f32', max_contacts=2**20)
        detector.init(mesh, surface_data)
        n_contacts = detector.find_contacts(mesh, dHat=0.01)
        pairs = detector.contact_pairs
    """

    def __init__(
        self,
        precision: PrecisionType = 'f32',
        max_contacts: int = 2**21,
    ):
        """
        Initialize BVH collision detector.

        Args:
            precision: Float precision ('f32' or 'f64')
            max_contacts: Maximum number of contact pairs to store
        """
        self.init_precision(precision)
        self._max_contacts = max_contacts
        self.SMALL_NUM = 1e-6

        # Will be initialized in init()
        self._storage: Optional[ContactPairStorage] = None
        self._bvh_triangles: Optional[LBVH_Triangles] = None
        self._bvh_edges: Optional[LBVH_Edges] = None

        # Surface data (set during init)
        self._mesh = None
        self._boundary_points = None
        self._boundary_edges = None
        self._boundary_triangles = None
        self._n_boundary_points = 0
        self._n_boundary_edges = 0
        self._n_boundary_triangles = 0

        # Detection parameters
        self._detection_dHat = 0.0
        self._bvh_gap = 0.0

        self._initialized = False

    # === CollisionDetector Protocol Implementation ===

    def init(self, mesh: Any, surface_data: SurfaceData) -> None:
        """
        Initialize collision detection structures.

        Args:
            mesh: MeshTaichi mesh object with verts.x field
            surface_data: SurfaceData containing boundary elements
        """
        self._mesh = mesh
        self._boundary_points = surface_data.boundary_points
        self._boundary_edges = surface_data.boundary_edges
        self._boundary_triangles = surface_data.boundary_triangles
        self._n_boundary_points = surface_data.n_boundary_points
        self._n_boundary_edges = surface_data.n_boundary_edges
        self._n_boundary_triangles = surface_data.n_boundary_triangles

        # Create BVH structures
        self._bvh_triangles = LBVH_Triangles(
            self._n_boundary_triangles,
            precision=self.precision
        )
        self._bvh_edges = LBVH_Edges(
            self._n_boundary_edges,
            precision=self.precision
        )

        # Create contact storage
        self._storage = ContactPairStorage(
            max_contacts=self._max_contacts,
            precision=self.precision
        )

        self._initialized = True

    def find_contacts(self, mesh: Any, dHat: float) -> int:
        """
        Find contact pairs within distance threshold.

        Args:
            mesh: MeshTaichi mesh object (ignored, uses internal reference)
            dHat: Distance threshold for contact detection

        Returns:
            Number of contact pairs found
        """
        if not self._initialized:
            raise RuntimeError("BVHCollisionDetector not initialized. Call init() first.")

        self._set_detection_threshold(dHat)
        self._storage.reset()

        # Build/refit BVH (currently always rebuilds)
        self._build_bvh()

        # Find PT and EE contacts
        self._find_PT_contacts()
        self._find_EE_contacts()

        return self._storage.count

    @property
    def contact_pairs(self) -> Any:
        """Access to contact pair data array."""
        if self._storage is None:
            raise RuntimeError("BVHCollisionDetector not initialized.")
        return self._storage.contact_pairs

    @property
    def n_contacts(self) -> int:
        """Number of active contacts."""
        if self._storage is None:
            return 0
        return self._storage.count

    # === Additional Public Methods ===

    def set_detection_threshold(self, detection_dHat: float):
        """
        Set custom detection threshold for contact filtering.

        This allows using a larger detection radius (e.g., 5*dHat) to cache
        collision pairs while the actual barrier uses the original dHat.

        Args:
            detection_dHat: Detection threshold (should be >= dHat)
        """
        self._set_detection_threshold(detection_dHat)

    def _set_detection_threshold(self, detection_dHat: float):
        """Internal method to set detection threshold."""
        self._detection_dHat = detection_dHat
        self._bvh_gap = ti.sqrt(detection_dHat)

    def build_bvh(self):
        """Build BVH trees from scratch."""
        self._build_bvh()

    def refit_bvh(self):
        """Refit BVH trees (preserves tree structure, updates AABBs)."""
        self._refit_bvh()

    def _build_bvh(self):
        """Build BVH trees for triangles and edges."""
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
        """Refit BVH trees by updating AABBs."""
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

    def _find_PT_contacts(self):
        """Find Point-Triangle contacts using BVH traversal."""
        self._find_constraints_PT_bvh_kernel()

    def _find_EE_contacts(self):
        """Find Edge-Edge contacts using BVH traversal."""
        self._find_constraints_EE_bvh_kernel()

    @ti.kernel
    def _find_constraints_PT_bvh_kernel(self):
        """Kernel to find Point-Triangle constraints using BVH traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self._bvh_gap
        dHat = self._detection_dHat
        SMALL_NUM = self.SMALL_NUM

        # For each boundary point, traverse the triangle BVH
        for pi in range(self._n_boundary_points):
            p = self._boundary_points[pi]
            xp = self._mesh.verts.x[p]

            # BVH traversal stack (local array)
            stack = ti.Vector.zero(ti.u32, 64)
            stack_ptr = 0
            stack[stack_ptr] = 0  # Start from root
            stack_ptr += 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node_id = stack[stack_ptr]

                L_idx = self._bvh_triangles.left_idx[node_id]
                R_idx = self._bvh_triangles.right_idx[node_id]

                # Check left child
                if self._bvh_triangles.aabb_overlap_point(xp, L_idx, gap):
                    element_idx = self._bvh_triangles.element_idx[L_idx]
                    if element_idx != INVALID:
                        # Leaf node - check triangle
                        tri_id = element_idx
                        t0 = self._boundary_triangles[tri_id, 0]
                        t1 = self._boundary_triangles[tri_id, 1]
                        t2 = self._boundary_triangles[tri_id, 2]

                        # Skip if point is a vertex of the triangle
                        if p != t0 and p != t1 and p != t2:
                            x0 = self._mesh.verts.x[t0]
                            x1 = self._mesh.verts.x[t1]
                            x2 = self._mesh.verts.x[t2]

                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()

                                if dist < dHat and ti.abs(dist) > SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.u32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2])
                                    self._storage.add_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._bvh_triangles.aabb_overlap_point(xp, R_idx, gap):
                    element_idx = self._bvh_triangles.element_idx[R_idx]
                    if element_idx != INVALID:
                        # Leaf node - check triangle
                        tri_id = element_idx
                        t0 = self._boundary_triangles[tri_id, 0]
                        t1 = self._boundary_triangles[tri_id, 1]
                        t2 = self._boundary_triangles[tri_id, 2]

                        # Skip if point is a vertex of the triangle
                        if p != t0 and p != t1 and p != t2:
                            x0 = self._mesh.verts.x[t0]
                            x1 = self._mesh.verts.x[t1]
                            x2 = self._mesh.verts.x[t2]

                            if point_triangle_ccd_broadphase(xp, x0, x1, x2, dHat):
                                cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                                xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                                t_pt = xp - xt
                                dist = t_pt.norm()

                                if dist < dHat and ti.abs(dist) > SMALL_NUM:
                                    ids = ti.Vector([p, t0, t1, t2], ti.u32)
                                    cord = ti.Vector([1.0, -cord0, -cord1, -cord2])
                                    self._storage.add_pair(ids, dist, cord, t_pt)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.kernel
    def _find_constraints_EE_bvh_kernel(self):
        """Kernel to find Edge-Edge constraints using BVH traversal."""
        INVALID = ti.u32(0xFFFFFFFF)
        gap = self._bvh_gap
        dHat = self._detection_dHat
        SMALL_NUM = self.SMALL_NUM
        n_edges = self._n_boundary_edges

        # For each edge, traverse the edge BVH
        for ei in range(n_edges):
            a0 = self._boundary_edges[ei, 0]
            a1 = self._boundary_edges[ei, 1]
            x_a0 = self._mesh.verts.x[a0]
            x_a1 = self._mesh.verts.x[a1]

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

                L_idx = self._bvh_edges.left_idx[node_id]
                R_idx = self._bvh_edges.right_idx[node_id]

                # Check left child
                if self._aabb_overlap_with_edge(edge_lower, edge_upper, L_idx, gap):
                    element_idx = self._bvh_edges.element_idx[L_idx]
                    if element_idx != INVALID:
                        # Leaf node - check edge pair
                        ej = element_idx
                        if ei < ej:  # Avoid duplicate pairs
                            b0 = self._boundary_edges[ej, 0]
                            b1 = self._boundary_edges[ej, 1]

                            # Skip if edges share a vertex
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self._mesh.verts.x[b0]
                                x_b1 = self._mesh.verts.x[b1]

                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()

                                    if dist < dHat and ti.abs(dist) > SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc])
                                        ids = ti.Vector([a0, a1, b0, b1], ti.u32)
                                        self._storage.add_pair(ids, dist, cord, t_ee)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                # Check right child
                if self._aabb_overlap_with_edge(edge_lower, edge_upper, R_idx, gap):
                    element_idx = self._bvh_edges.element_idx[R_idx]
                    if element_idx != INVALID:
                        # Leaf node - check edge pair
                        ej = element_idx
                        if ei < ej:  # Avoid duplicate pairs
                            b0 = self._boundary_edges[ej, 0]
                            b1 = self._boundary_edges[ej, 1]

                            # Skip if edges share a vertex
                            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                                x_b0 = self._mesh.verts.x[b0]
                                x_b1 = self._mesh.verts.x[b1]

                                if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, dHat):
                                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                                    dist = t_ee.norm()

                                    if dist < dHat and ti.abs(dist) > SMALL_NUM:
                                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc])
                                        ids = ti.Vector([a0, a1, b0, b1], ti.u32)
                                        self._storage.add_pair(ids, dist, cord, t_ee)
                    else:
                        # Internal node - push to stack
                        if stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

    @ti.func
    def _aabb_overlap_with_edge(
        self,
        edge_lower: ti.template(),
        edge_upper: ti.template(),
        node_idx: ti.i32,
        gap: ti.template()
    ) -> bool:
        """Check if edge AABB overlaps with BVH node AABB."""
        node_lower = self._bvh_edges.bv_lower[node_idx]
        node_upper = self._bvh_edges.bv_upper[node_idx]

        return ((node_upper[0] - edge_lower[0]) > -gap and (edge_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - edge_lower[1]) > -gap and (edge_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - edge_lower[2]) > -gap and (edge_upper[2] - node_lower[2]) > -gap)

    # === Penetration Detection ===

    def check_penetration(self, check_ground: bool = True, ground_y: float = 0.0) -> bool:
        """
        Check if mesh has any penetration/intersection.

        Args:
            check_ground: Whether to check ground plane penetration
            ground_y: Ground plane y-coordinate

        Returns:
            True if any intersection found, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("BVHCollisionDetector not initialized.")

        # Build BVH if needed
        self._build_bvh()

        # Check ground intersection
        if check_ground:
            if self._check_ground_intersection(ground_y):
                return True

        # Check edge-triangle intersection
        if self._check_edge_triangle_intersection():
            return True

        return False

    def is_intersected(self, check_ground: bool = True, ground_y: float = 0.0) -> bool:
        """Alias for check_penetration."""
        return self.check_penetration(check_ground, ground_y)

    @ti.kernel
    def _check_ground_intersection(self, ground_y: ti.template()) -> ti.i32:
        """Check if any vertex is below ground."""
        result = 0
        for i in self._mesh.verts:
            if result == 0:
                if self._mesh.verts.x[i][1] < ground_y:
                    result = 1
        return result

    @ti.kernel
    def _check_edge_triangle_intersection(self) -> ti.i32:
        """Check if any edge intersects any triangle."""
        INVALID = ti.u32(0xFFFFFFFF)
        result = 0
        gap = 0.0  # No gap for exact intersection test

        # For each triangle, traverse the edge BVH
        for ti_idx in range(self._n_boundary_triangles):
            if result == 0:
                t0 = self._boundary_triangles[ti_idx, 0]
                t1 = self._boundary_triangles[ti_idx, 1]
                t2 = self._boundary_triangles[ti_idx, 2]
                x0 = self._mesh.verts.x[t0]
                x1 = self._mesh.verts.x[t1]
                x2 = self._mesh.verts.x[t2]

                # Triangle AABB
                tri_lower = ti.min(ti.min(x0, x1), x2)
                tri_upper = ti.max(ti.max(x0, x1), x2)

                # BVH traversal
                stack = ti.Vector.zero(ti.u32, 64)
                stack_ptr = 0
                stack[stack_ptr] = 0
                stack_ptr += 1

                while stack_ptr > 0 and result == 0:
                    stack_ptr -= 1
                    node_id = stack[stack_ptr]

                    L_idx = self._bvh_edges.left_idx[node_id]
                    R_idx = self._bvh_edges.right_idx[node_id]

                    # Check left child
                    if self._aabb_overlap_with_tri(tri_lower, tri_upper, L_idx, gap):
                        element_idx = self._bvh_edges.element_idx[L_idx]
                        if element_idx != INVALID:
                            if self._check_edge_tri_leaf(element_idx, t0, t1, t2, x0, x1, x2):
                                result = 1
                        elif stack_ptr < 63:
                            stack[stack_ptr] = L_idx
                            stack_ptr += 1

                    # Check right child
                    if result == 0 and self._aabb_overlap_with_tri(tri_lower, tri_upper, R_idx, gap):
                        element_idx = self._bvh_edges.element_idx[R_idx]
                        if element_idx != INVALID:
                            if self._check_edge_tri_leaf(element_idx, t0, t1, t2, x0, x1, x2):
                                result = 1
                        elif stack_ptr < 63:
                            stack[stack_ptr] = R_idx
                            stack_ptr += 1

        return result

    @ti.func
    def _aabb_overlap_with_tri(
        self,
        tri_lower: ti.template(),
        tri_upper: ti.template(),
        node_idx: ti.i32,
        gap: ti.template()
    ) -> bool:
        """Check if triangle AABB overlaps with BVH node AABB."""
        node_lower = self._bvh_edges.bv_lower[node_idx]
        node_upper = self._bvh_edges.bv_upper[node_idx]

        return ((node_upper[0] - tri_lower[0]) > -gap and (tri_upper[0] - node_lower[0]) > -gap and
                (node_upper[1] - tri_lower[1]) > -gap and (tri_upper[1] - node_lower[1]) > -gap and
                (node_upper[2] - tri_lower[2]) > -gap and (tri_upper[2] - node_lower[2]) > -gap)

    @ti.func
    def _check_edge_tri_leaf(
        self,
        edge_idx: ti.i32,
        t0: ti.i32, t1: ti.i32, t2: ti.i32,
        x0: ti.template(), x1: ti.template(), x2: ti.template()
    ) -> ti.i32:
        """Check if edge intersects triangle (leaf node processing)."""
        result = 0
        a0 = self._boundary_edges[edge_idx, 0]
        a1 = self._boundary_edges[edge_idx, 1]
        # Skip if edge shares vertex with triangle
        if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
            x_a0 = self._mesh.verts.x[a0]
            x_a1 = self._mesh.verts.x[a1]
            if segment_triangle_intersect_cramer(x_a0, x_a1, x0, x1, x2):
                result = 1
        return result

    # === Utility Methods ===

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
