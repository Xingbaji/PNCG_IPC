"""
Unit tests for collision detection module.

Tests LBVH construction and BVHCollisionDetector functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import taichi as ti
import numpy as np

# Initialize Taichi before importing collision modules
ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache_collision_test")

from algorithm_new.collision import (
    CollisionDetectorRegistry,
    BVHCollisionDetector,
    ContactPairStorage,
    SurfaceData,
)
from algorithm_new.collision.lbvh import LBVH_Triangles, LBVH_Edges


def test_collision_detector_registry():
    """Test that BVH detector is registered."""
    print("Testing CollisionDetectorRegistry...")

    assert CollisionDetectorRegistry.is_registered('bvh'), "BVH detector should be registered"
    available = CollisionDetectorRegistry.list_available()
    assert 'bvh' in available, f"'bvh' should be in available list: {available}"

    # Create detector via registry
    detector = CollisionDetectorRegistry.create('bvh', precision='f32')
    assert detector is not None, "Should create detector via registry"
    print("  Registry test passed!")


def test_contact_pair_storage():
    """Test ContactPairStorage basic functionality."""
    print("Testing ContactPairStorage...")

    storage = ContactPairStorage(max_contacts=1000, precision='f32')
    storage.reset()

    assert storage.count == 0, "Initial count should be 0"
    print("  ContactPairStorage test passed!")


def test_lbvh_construction():
    """Test LBVH construction with simple geometry."""
    print("Testing LBVH construction...")

    # Create simple test data: 2 triangles
    n_triangles = 2
    n_vertices = 4

    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
    triangles = ti.field(dtype=ti.i32, shape=(n_triangles, 3))

    # Set up vertices: two triangles sharing an edge
    vertices[0] = [0.0, 0.0, 0.0]
    vertices[1] = [1.0, 0.0, 0.0]
    vertices[2] = [0.5, 1.0, 0.0]
    vertices[3] = [0.5, 0.0, 1.0]

    triangles[0, 0] = 0
    triangles[0, 1] = 1
    triangles[0, 2] = 2
    triangles[1, 0] = 0
    triangles[1, 1] = 1
    triangles[1, 2] = 3

    # Create and build BVH
    bvh = LBVH_Triangles(n_triangles, precision='f32')
    bvh.build(vertices, triangles, n_triangles)

    assert bvh.tree_built, "Tree should be marked as built"
    print("  LBVH construction test passed!")


def test_lbvh_edges():
    """Test LBVH_Edges construction."""
    print("Testing LBVH_Edges construction...")

    n_edges = 3
    n_vertices = 4

    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
    edges = ti.field(dtype=ti.i32, shape=(n_edges, 2))

    vertices[0] = [0.0, 0.0, 0.0]
    vertices[1] = [1.0, 0.0, 0.0]
    vertices[2] = [0.5, 1.0, 0.0]
    vertices[3] = [0.5, 0.5, 0.5]

    edges[0, 0] = 0
    edges[0, 1] = 1
    edges[1, 0] = 1
    edges[1, 1] = 2
    edges[2, 0] = 2
    edges[2, 1] = 3

    bvh = LBVH_Edges(n_edges, precision='f32')
    bvh.build(vertices, edges, n_edges)

    assert bvh.tree_built, "Edge BVH tree should be marked as built"
    print("  LBVH_Edges construction test passed!")


def test_precision_support():
    """Test f32 and f64 precision support."""
    print("Testing precision support...")

    for precision in ['f32', 'f64']:
        storage = ContactPairStorage(max_contacts=100, precision=precision)
        assert storage.precision == precision, f"Precision should be {precision}"

        bvh_tri = LBVH_Triangles(10, precision=precision)
        assert bvh_tri.precision == precision, f"Triangle BVH precision should be {precision}"

        bvh_edge = LBVH_Edges(10, precision=precision)
        assert bvh_edge.precision == precision, f"Edge BVH precision should be {precision}"

    print("  Precision support test passed!")


class SimpleMesh:
    """Simple mock mesh for testing."""
    def __init__(self, n_verts):
        self.verts = SimpleVerts(n_verts)


class SimpleVerts:
    """Simple mock verts for testing."""
    def __init__(self, n_verts):
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.n = n_verts

    def __iter__(self):
        return iter(range(self.n))


def test_bvh_collision_detector_basic():
    """Test BVHCollisionDetector basic functionality."""
    print("Testing BVHCollisionDetector basic...")

    # Create simple test geometry: a cube
    n_verts = 8
    n_edges = 12
    n_triangles = 12  # 6 faces * 2 triangles per face

    # Create mock mesh
    mesh = SimpleMesh(n_verts)

    # Cube vertices
    mesh.verts.x[0] = [0.0, 0.0, 0.0]
    mesh.verts.x[1] = [1.0, 0.0, 0.0]
    mesh.verts.x[2] = [1.0, 1.0, 0.0]
    mesh.verts.x[3] = [0.0, 1.0, 0.0]
    mesh.verts.x[4] = [0.0, 0.0, 1.0]
    mesh.verts.x[5] = [1.0, 0.0, 1.0]
    mesh.verts.x[6] = [1.0, 1.0, 1.0]
    mesh.verts.x[7] = [0.0, 1.0, 1.0]

    # Boundary data
    boundary_points = ti.field(dtype=ti.i32, shape=n_verts)
    for i in range(n_verts):
        boundary_points[i] = i

    boundary_edges = ti.field(dtype=ti.i32, shape=(n_edges, 2))
    edge_list = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    for i, (a, b) in enumerate(edge_list):
        boundary_edges[i, 0] = a
        boundary_edges[i, 1] = b

    boundary_triangles = ti.field(dtype=ti.i32, shape=(n_triangles, 3))
    tri_list = [
        # bottom
        (0, 1, 2), (0, 2, 3),
        # top
        (4, 6, 5), (4, 7, 6),
        # front
        (0, 5, 1), (0, 4, 5),
        # back
        (2, 7, 3), (2, 6, 7),
        # left
        (0, 3, 7), (0, 7, 4),
        # right
        (1, 5, 6), (1, 6, 2),
    ]
    for i, (a, b, c) in enumerate(tri_list):
        boundary_triangles[i, 0] = a
        boundary_triangles[i, 1] = b
        boundary_triangles[i, 2] = c

    surface_data = SurfaceData(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        boundary_triangles=boundary_triangles,
        n_boundary_points=n_verts,
        n_boundary_edges=n_edges,
        n_boundary_triangles=n_triangles,
    )

    # Create and initialize detector
    detector = BVHCollisionDetector(precision='f32')
    detector.init(mesh, surface_data)

    # Find contacts with a small dHat (cube edges are length 1, so use 0.1)
    dHat = 0.1
    n_contacts = detector.find_contacts(mesh, dHat)

    print(f"  Found {n_contacts} contacts with dHat={dHat}")

    # For a single cube, we shouldn't have any self-intersections
    # but we might have some close contacts at corners
    assert n_contacts >= 0, "Contact count should be non-negative"

    # Test penetration check
    has_penetration = detector.check_penetration(check_ground=False)
    assert not has_penetration, "Single cube should not self-intersect"

    print("  BVHCollisionDetector basic test passed!")


def run_all_tests():
    """Run all collision detection tests."""
    print("=" * 60)
    print("Running Collision Detection Module Tests")
    print("=" * 60)

    test_collision_detector_registry()
    test_contact_pair_storage()
    test_lbvh_construction()
    test_lbvh_edges()
    test_precision_support()
    test_bvh_collision_detector_basic()

    print("=" * 60)
    print("All collision detection tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
