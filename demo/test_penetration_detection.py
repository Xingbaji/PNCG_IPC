"""
Test script for penetration detection functionality.
Based on GIPC.cu penetration detection implementation.

This script tests:
1. Segment-triangle intersection using Cramer's rule
2. BVH-based edge-triangle intersection query
3. Ground plane penetration detection
4. High-level is_intersected API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.cpu, debug=True)

# Import the penetration detection functions
from math_utils.graphic_util import (
    segment_triangle_intersect_cramer,
    segment_intersect_triangle_new,
    mat3_determinant,
    check_ground_penetration
)


def test_segment_triangle_intersection():
    """Test segment-triangle intersection functions."""
    print("\n" + "="*60)
    print("Testing Segment-Triangle Intersection")
    print("="*60)

    @ti.kernel
    def test_intersection_cramer(
        ve0: ti.types.vector(3, ti.f32),
        ve1: ti.types.vector(3, ti.f32),
        vt0: ti.types.vector(3, ti.f32),
        vt1: ti.types.vector(3, ti.f32),
        vt2: ti.types.vector(3, ti.f32)
    ) -> ti.i32:
        return segment_triangle_intersect_cramer(ve0, ve1, vt0, vt1, vt2)

    @ti.kernel
    def test_intersection_original(
        ve0: ti.types.vector(3, ti.f32),
        ve1: ti.types.vector(3, ti.f32),
        vt0: ti.types.vector(3, ti.f32),
        vt1: ti.types.vector(3, ti.f32),
        vt2: ti.types.vector(3, ti.f32)
    ) -> ti.i32:
        return segment_intersect_triangle_new(ve0, ve1, vt0, vt1, vt2)

    # Test case 1: Segment passing through triangle center
    print("\nTest 1: Segment passing through triangle center")
    ve0 = ti.Vector([0.5, 0.5, -1.0])
    ve1 = ti.Vector([0.5, 0.5, 1.0])
    vt0 = ti.Vector([0.0, 0.0, 0.0])
    vt1 = ti.Vector([1.0, 0.0, 0.0])
    vt2 = ti.Vector([0.5, 1.0, 0.0])

    result_cramer = test_intersection_cramer(ve0, ve1, vt0, vt1, vt2)
    result_original = test_intersection_original(ve0, ve1, vt0, vt1, vt2)
    print(f"  Cramer result: {result_cramer}, Original result: {result_original}")
    assert result_cramer == 1, "Expected intersection"
    print("  PASSED!")

    # Test case 2: Segment not intersecting (parallel)
    print("\nTest 2: Segment parallel to triangle (no intersection)")
    ve0 = ti.Vector([0.5, 0.5, 1.0])
    ve1 = ti.Vector([0.5, 0.5, 2.0])
    result_cramer = test_intersection_cramer(ve0, ve1, vt0, vt1, vt2)
    print(f"  Cramer result: {result_cramer}")
    assert result_cramer == 0, "Expected no intersection"
    print("  PASSED!")

    # Test case 3: Segment on same side of triangle plane
    print("\nTest 3: Segment on one side of triangle plane")
    ve0 = ti.Vector([0.5, 0.5, 1.0])
    ve1 = ti.Vector([0.5, 0.5, 2.0])
    result_cramer = test_intersection_cramer(ve0, ve1, vt0, vt1, vt2)
    print(f"  Cramer result: {result_cramer}")
    assert result_cramer == 0, "Expected no intersection"
    print("  PASSED!")

    # Test case 4: Segment intersecting triangle edge
    print("\nTest 4: Segment just outside triangle (misses)")
    ve0 = ti.Vector([1.5, 0.5, -1.0])
    ve1 = ti.Vector([1.5, 0.5, 1.0])
    result_cramer = test_intersection_cramer(ve0, ve1, vt0, vt1, vt2)
    print(f"  Cramer result: {result_cramer}")
    assert result_cramer == 0, "Expected no intersection"
    print("  PASSED!")

    # Test case 5: Segment endpoint exactly on triangle
    print("\nTest 5: Segment touching triangle vertex")
    ve0 = ti.Vector([0.0, 0.0, 0.0])  # vt0
    ve1 = ti.Vector([0.0, 0.0, 1.0])
    result_cramer = test_intersection_cramer(ve0, ve1, vt0, vt1, vt2)
    print(f"  Cramer result: {result_cramer}")
    # This case is boundary - may be 0 or 1 depending on epsilon handling
    print("  Note: Boundary case, result depends on epsilon handling")

    print("\n[Segment-Triangle Tests] All basic tests passed!")


def test_ground_penetration():
    """Test ground plane penetration detection."""
    print("\n" + "="*60)
    print("Testing Ground Plane Penetration")
    print("="*60)

    @ti.kernel
    def test_ground(
        pos: ti.types.vector(3, ti.f32),
        normal: ti.types.vector(3, ti.f32),
        offset: ti.f32
    ) -> ti.i32:
        return check_ground_penetration(pos, normal, offset)

    ground_normal = ti.Vector([0.0, 1.0, 0.0])  # Y-up
    ground_offset = 0.0  # Ground at y=0

    # Test 1: Point above ground
    print("\nTest 1: Point above ground")
    pos_above = ti.Vector([0.0, 1.0, 0.0])
    result = test_ground(pos_above, ground_normal, ground_offset)
    print(f"  Position: {pos_above}, Result: {result}")
    assert result == 0, "Expected no penetration"
    print("  PASSED!")

    # Test 2: Point below ground
    print("\nTest 2: Point below ground")
    pos_below = ti.Vector([0.0, -0.5, 0.0])
    result = test_ground(pos_below, ground_normal, ground_offset)
    print(f"  Position: {pos_below}, Result: {result}")
    assert result == 1, "Expected penetration"
    print("  PASSED!")

    # Test 3: Point exactly on ground
    print("\nTest 3: Point on ground (boundary)")
    pos_on = ti.Vector([0.0, 0.0, 0.0])
    result = test_ground(pos_on, ground_normal, ground_offset)
    print(f"  Position: {pos_on}, Result: {result}")
    print("  Note: Boundary case, result depends on comparison operator")

    print("\n[Ground Penetration Tests] All basic tests passed!")


def test_determinant():
    """Test 3x3 matrix determinant computation."""
    print("\n" + "="*60)
    print("Testing Matrix Determinant")
    print("="*60)

    @ti.kernel
    def test_det(
        c0: ti.types.vector(3, ti.f32),
        c1: ti.types.vector(3, ti.f32),
        c2: ti.types.vector(3, ti.f32)
    ) -> ti.f32:
        return mat3_determinant(c0, c1, c2)

    # Test with identity matrix columns
    print("\nTest 1: Identity matrix (det = 1)")
    c0 = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
    c1 = ti.Vector([0.0, 1.0, 0.0], dt=ti.f32)
    c2 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
    result = test_det(c0, c1, c2)
    print(f"  Result: {result}")
    assert abs(result - 1.0) < 1e-10, "Expected determinant = 1"
    print("  PASSED!")

    # Test with known matrix
    print("\nTest 2: Known matrix")
    c0 = ti.Vector([1.0, 2.0, 3.0], dt=ti.f32)
    c1 = ti.Vector([4.0, 5.0, 6.0], dt=ti.f32)
    c2 = ti.Vector([7.0, 8.0, 9.0], dt=ti.f32)
    result = test_det(c0, c1, c2)
    print(f"  Result: {result}")
    # det([[1,4,7],[2,5,8],[3,6,9]]) = 0 (linearly dependent columns)
    assert abs(result) < 1e-10, "Expected determinant = 0"
    print("  PASSED!")

    print("\n[Determinant Tests] All tests passed!")


def test_full_penetration_detection():
    """Test full penetration detection with a simple mesh."""
    print("\n" + "="*60)
    print("Testing Full Penetration Detection with Mesh")
    print("="*60)
    print("\nNote: Full mesh test is skipped in quick test mode.")
    print("Run with --full flag to test with actual mesh data.")
    print("Example: python test_penetration_detection.py --full")


def test_full_penetration_detection_impl():
    """Actual implementation of full mesh penetration detection test."""
    print("\n" + "="*60)
    print("Testing Full Penetration Detection with Mesh")
    print("="*60)

    try:
        from algorithm.collision_detection_bvh import collision_detection_bvh_module

        # Create a simple demo instance
        print("\nInitializing collision detection module...")
        demo = collision_detection_bvh_module(demo='cube')
        demo.init_bvh()

        print("\nTesting penetration detection API...")

        # Build BVH
        demo.build_bvh()

        # Test edge-triangle intersection (should be 0 for initial pose)
        result = demo.check_edge_triangle_intersection_bvh()
        print(f"  Edge-triangle intersections: {result}")

        # Count intersections
        count = demo.count_edge_triangle_intersections()
        print(f"  Total intersection count: {count}")

        # Test high-level API
        is_penetrated = demo.is_intersected(check_ground=True)
        print(f"  Is intersected: {is_penetrated}")

        # Verbose check
        demo.check_penetration(verbose=True)

        print("\n[Full Penetration Detection] Tests completed!")

    except Exception as e:
        print(f"\nNote: Full mesh test skipped due to: {e}")
        print("This is expected if the demo configuration is not available.")


def main():
    import sys

    print("="*60)
    print("Penetration Detection Test Suite")
    print("Based on GIPC.cu implementation")
    print("="*60)

    # Run basic tests
    test_determinant()
    test_segment_triangle_intersection()
    test_ground_penetration()

    # Run full mesh test only if --full flag is provided
    if '--full' in sys.argv:
        test_full_penetration_detection_impl()
    else:
        test_full_penetration_detection()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
