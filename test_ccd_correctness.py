"""
Comprehensive test suite for CCD (Continuous Collision Detection) lower bound functions.

Tests the non-penetration property as described in the MAS-PNCG paper:
- The CCD lower bound must be conservative (no false negatives)
- At the returned TOC, the distance should still be >= 0 (no penetration)
- The algorithm should handle edge cases (parallel edges, degenerate triangles, etc.)

The new CubicNoRootRegionPrecise-based implementation doesn't use eta or thickness parameters.
"""

import taichi as ti
import numpy as np
import random

# Initialize Taichi with float64 for higher precision
ti.init(arch=ti.cpu, default_fp=ti.f64)

from math_utils.graphic_util import (
    point_triangle_ccd_lower_bound,
    edge_edge_ccd_lower_bound,
    point_triangle_ccd,
    edge_edge_ccd,
    point_triangle_distance_unclassified,
    edge_edge_distance_unclassified,
)


# ============================================================================
# Helper functions for testing
# ============================================================================

@ti.kernel
def compute_pt_distance_at_toc(
    p: ti.types.vector(3, ti.f64),
    t0: ti.types.vector(3, ti.f64),
    t1: ti.types.vector(3, ti.f64),
    t2: ti.types.vector(3, ti.f64),
    dp: ti.types.vector(3, ti.f64),
    dt0: ti.types.vector(3, ti.f64),
    dt1: ti.types.vector(3, ti.f64),
    dt2: ti.types.vector(3, ti.f64),
    toc: ti.f64
) -> ti.f64:
    """Compute the point-triangle distance at a given time of contact."""
    p_new = p + dp * toc
    t0_new = t0 + dt0 * toc
    t1_new = t1 + dt1 * toc
    t2_new = t2 + dt2 * toc
    dist2 = point_triangle_distance_unclassified(p_new, t0_new, t1_new, t2_new)
    return ti.sqrt(dist2)


@ti.kernel
def compute_ee_distance_at_toc(
    ea0: ti.types.vector(3, ti.f64),
    ea1: ti.types.vector(3, ti.f64),
    eb0: ti.types.vector(3, ti.f64),
    eb1: ti.types.vector(3, ti.f64),
    dea0: ti.types.vector(3, ti.f64),
    dea1: ti.types.vector(3, ti.f64),
    deb0: ti.types.vector(3, ti.f64),
    deb1: ti.types.vector(3, ti.f64),
    toc: ti.f64
) -> ti.f64:
    """Compute the edge-edge distance at a given time of contact."""
    ea0_new = ea0 + dea0 * toc
    ea1_new = ea1 + dea1 * toc
    eb0_new = eb0 + deb0 * toc
    eb1_new = eb1 + deb1 * toc
    dist2 = edge_edge_distance_unclassified(ea0_new, ea1_new, eb0_new, eb1_new)
    return ti.sqrt(dist2)


@ti.kernel
def test_pt_ccd_lower_bound(
    p: ti.types.vector(3, ti.f64),
    t0: ti.types.vector(3, ti.f64),
    t1: ti.types.vector(3, ti.f64),
    t2: ti.types.vector(3, ti.f64),
    dp: ti.types.vector(3, ti.f64),
    dt0: ti.types.vector(3, ti.f64),
    dt1: ti.types.vector(3, ti.f64),
    dt2: ti.types.vector(3, ti.f64)
) -> ti.f64:
    """Test PT CCD lower bound and return TOC."""
    return point_triangle_ccd_lower_bound(p, t0, t1, t2, dp, dt0, dt1, dt2)


@ti.kernel
def test_ee_ccd_lower_bound(
    ea0: ti.types.vector(3, ti.f64),
    ea1: ti.types.vector(3, ti.f64),
    eb0: ti.types.vector(3, ti.f64),
    eb1: ti.types.vector(3, ti.f64),
    dea0: ti.types.vector(3, ti.f64),
    dea1: ti.types.vector(3, ti.f64),
    deb0: ti.types.vector(3, ti.f64),
    deb1: ti.types.vector(3, ti.f64)
) -> ti.f64:
    """Test EE CCD lower bound and return TOC."""
    return edge_edge_ccd_lower_bound(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1)


@ti.kernel
def test_pt_ccd_iterative(
    p: ti.types.vector(3, ti.f64),
    t0: ti.types.vector(3, ti.f64),
    t1: ti.types.vector(3, ti.f64),
    t2: ti.types.vector(3, ti.f64),
    dp: ti.types.vector(3, ti.f64),
    dt0: ti.types.vector(3, ti.f64),
    dt1: ti.types.vector(3, ti.f64),
    dt2: ti.types.vector(3, ti.f64),
    eta: ti.f64,
    thickness: ti.f64
) -> ti.f64:
    """Test PT CCD (iterative version) and return TOC."""
    return point_triangle_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness)


@ti.kernel
def test_ee_ccd_iterative(
    ea0: ti.types.vector(3, ti.f64),
    ea1: ti.types.vector(3, ti.f64),
    eb0: ti.types.vector(3, ti.f64),
    eb1: ti.types.vector(3, ti.f64),
    dea0: ti.types.vector(3, ti.f64),
    dea1: ti.types.vector(3, ti.f64),
    deb0: ti.types.vector(3, ti.f64),
    deb1: ti.types.vector(3, ti.f64),
    eta: ti.f64,
    thickness: ti.f64
) -> ti.f64:
    """Test EE CCD (iterative version) and return TOC."""
    return edge_edge_ccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness)


# ============================================================================
# Test Cases
# ============================================================================

class CCDTestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.failures = []

    def check(self, condition, test_name, details=""):
        self.total += 1
        if condition:
            self.passed += 1
            return True
        else:
            self.failed += 1
            self.failures.append((test_name, details))
            return False

    def report(self):
        print("\n" + "=" * 70)
        print(f"TEST RESULTS: {self.passed}/{self.total} passed, {self.failed} failed")
        print("=" * 70)
        if self.failures:
            print("\nFailed tests:")
            for name, details in self.failures[:20]:  # Show first 20 failures
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
            if len(self.failures) > 20:
                print(f"  ... and {len(self.failures) - 20} more failures")
        return self.failed == 0


def run_pt_ccd_test(suite, name, p, t0, t1, t2, dp, dt0, dt1, dt2, expect_collision=True):
    """
    Run a single point-triangle CCD test using CubicNoRootRegionPrecise.

    Verifies:
    1. TOC is in [0, 1]
    2. At TOC, distance >= 0 (non-penetration)
    3. If expect_collision, TOC < 1.0; otherwise TOC should be 1.0
    """
    # Convert to Taichi vectors
    p_ti = ti.Vector([p[0], p[1], p[2]], dt=ti.f64)
    t0_ti = ti.Vector([t0[0], t0[1], t0[2]], dt=ti.f64)
    t1_ti = ti.Vector([t1[0], t1[1], t1[2]], dt=ti.f64)
    t2_ti = ti.Vector([t2[0], t2[1], t2[2]], dt=ti.f64)
    dp_ti = ti.Vector([dp[0], dp[1], dp[2]], dt=ti.f64)
    dt0_ti = ti.Vector([dt0[0], dt0[1], dt0[2]], dt=ti.f64)
    dt1_ti = ti.Vector([dt1[0], dt1[1], dt1[2]], dt=ti.f64)
    dt2_ti = ti.Vector([dt2[0], dt2[1], dt2[2]], dt=ti.f64)

    # Get TOC
    toc = test_pt_ccd_lower_bound(p_ti, t0_ti, t1_ti, t2_ti, dp_ti, dt0_ti, dt1_ti, dt2_ti)

    # Compute distance at TOC
    dist_at_toc = compute_pt_distance_at_toc(p_ti, t0_ti, t1_ti, t2_ti, dp_ti, dt0_ti, dt1_ti, dt2_ti, toc)

    # Test 1: TOC in valid range
    suite.check(
        0.0 <= toc <= 1.0,
        f"{name}: TOC in [0,1]",
        f"TOC = {toc}"
    )

    # Test 2: Non-penetration (distance >= 0 at TOC)
    tolerance = 1e-10
    suite.check(
        dist_at_toc >= -tolerance,
        f"{name}: Non-penetration",
        f"dist_at_toc = {dist_at_toc}, toc = {toc}"
    )

    # Test 3: Collision expectation
    if expect_collision:
        suite.check(
            toc < 1.0 - 1e-10,
            f"{name}: Expected collision detected",
            f"TOC = {toc}"
        )
    else:
        suite.check(
            toc > 0.99,
            f"{name}: No collision as expected",
            f"TOC = {toc}"
        )

    return toc, dist_at_toc


def run_ee_ccd_test(suite, name, ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, expect_collision=True):
    """
    Run a single edge-edge CCD test using CubicNoRootRegionPrecise.
    """
    # Convert to Taichi vectors
    ea0_ti = ti.Vector([ea0[0], ea0[1], ea0[2]], dt=ti.f64)
    ea1_ti = ti.Vector([ea1[0], ea1[1], ea1[2]], dt=ti.f64)
    eb0_ti = ti.Vector([eb0[0], eb0[1], eb0[2]], dt=ti.f64)
    eb1_ti = ti.Vector([eb1[0], eb1[1], eb1[2]], dt=ti.f64)
    dea0_ti = ti.Vector([dea0[0], dea0[1], dea0[2]], dt=ti.f64)
    dea1_ti = ti.Vector([dea1[0], dea1[1], dea1[2]], dt=ti.f64)
    deb0_ti = ti.Vector([deb0[0], deb0[1], deb0[2]], dt=ti.f64)
    deb1_ti = ti.Vector([deb1[0], deb1[1], deb1[2]], dt=ti.f64)

    # Get TOC
    toc = test_ee_ccd_lower_bound(ea0_ti, ea1_ti, eb0_ti, eb1_ti, dea0_ti, dea1_ti, deb0_ti, deb1_ti)

    # Compute distance at TOC
    dist_at_toc = compute_ee_distance_at_toc(ea0_ti, ea1_ti, eb0_ti, eb1_ti, dea0_ti, dea1_ti, deb0_ti, deb1_ti, toc)

    # Test 1: TOC in valid range
    suite.check(
        0.0 <= toc <= 1.0,
        f"{name}: TOC in [0,1]",
        f"TOC = {toc}"
    )

    # Test 2: Non-penetration
    tolerance = 1e-10
    suite.check(
        dist_at_toc >= -tolerance,
        f"{name}: Non-penetration",
        f"dist_at_toc = {dist_at_toc}, toc = {toc}"
    )

    # Test 3: Collision expectation
    if expect_collision:
        suite.check(
            toc < 1.0 - 1e-10,
            f"{name}: Expected collision detected",
            f"TOC = {toc}"
        )
    else:
        suite.check(
            toc > 0.99,
            f"{name}: No collision as expected",
            f"TOC = {toc}"
        )

    return toc, dist_at_toc


def run_pt_ccd_non_penetration_only(suite, name, p, t0, t1, t2, dp, dt0, dt1, dt2):
    """
    Run a PT CCD test focusing ONLY on the non-penetration property.
    Does NOT check whether collision was expected - just verifies safety.
    """
    p_ti = ti.Vector([p[0], p[1], p[2]], dt=ti.f64)
    t0_ti = ti.Vector([t0[0], t0[1], t0[2]], dt=ti.f64)
    t1_ti = ti.Vector([t1[0], t1[1], t1[2]], dt=ti.f64)
    t2_ti = ti.Vector([t2[0], t2[1], t2[2]], dt=ti.f64)
    dp_ti = ti.Vector([dp[0], dp[1], dp[2]], dt=ti.f64)
    dt0_ti = ti.Vector([dt0[0], dt0[1], dt0[2]], dt=ti.f64)
    dt1_ti = ti.Vector([dt1[0], dt1[1], dt1[2]], dt=ti.f64)
    dt2_ti = ti.Vector([dt2[0], dt2[1], dt2[2]], dt=ti.f64)

    toc = test_pt_ccd_lower_bound(p_ti, t0_ti, t1_ti, t2_ti, dp_ti, dt0_ti, dt1_ti, dt2_ti)
    dist_at_toc = compute_pt_distance_at_toc(p_ti, t0_ti, t1_ti, t2_ti, dp_ti, dt0_ti, dt1_ti, dt2_ti, toc)

    suite.check(0.0 <= toc <= 1.0, f"{name}: TOC in [0,1]", f"TOC = {toc}")
    suite.check(dist_at_toc >= -1e-10, f"{name}: Non-penetration", f"dist = {dist_at_toc}, toc = {toc}")

    return toc, dist_at_toc


def run_ee_ccd_non_penetration_only(suite, name, ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1):
    """
    Run an EE CCD test focusing ONLY on the non-penetration property.
    """
    ea0_ti = ti.Vector([ea0[0], ea0[1], ea0[2]], dt=ti.f64)
    ea1_ti = ti.Vector([ea1[0], ea1[1], ea1[2]], dt=ti.f64)
    eb0_ti = ti.Vector([eb0[0], eb0[1], eb0[2]], dt=ti.f64)
    eb1_ti = ti.Vector([eb1[0], eb1[1], eb1[2]], dt=ti.f64)
    dea0_ti = ti.Vector([dea0[0], dea0[1], dea0[2]], dt=ti.f64)
    dea1_ti = ti.Vector([dea1[0], dea1[1], dea1[2]], dt=ti.f64)
    deb0_ti = ti.Vector([deb0[0], deb0[1], deb0[2]], dt=ti.f64)
    deb1_ti = ti.Vector([deb1[0], deb1[1], deb1[2]], dt=ti.f64)

    toc = test_ee_ccd_lower_bound(ea0_ti, ea1_ti, eb0_ti, eb1_ti, dea0_ti, dea1_ti, deb0_ti, deb1_ti)
    dist_at_toc = compute_ee_distance_at_toc(ea0_ti, ea1_ti, eb0_ti, eb1_ti, dea0_ti, dea1_ti, deb0_ti, deb1_ti, toc)

    suite.check(0.0 <= toc <= 1.0, f"{name}: TOC in [0,1]", f"TOC = {toc}")
    suite.check(dist_at_toc >= -1e-10, f"{name}: Non-penetration", f"dist = {dist_at_toc}, toc = {toc}")

    return toc, dist_at_toc


# ============================================================================
# Test Cases Implementation
# ============================================================================

def test_basic_pt_collision(suite):
    """Test basic point-triangle collision scenarios."""
    print("\n--- Basic Point-Triangle Collision Tests ---")

    # Test 1: Point directly above triangle center, moving down
    run_pt_ccd_test(suite, "PT_basic_down",
        p=[0.5, 0.2, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.3, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )

    # Test 2: Point moving parallel to triangle (no collision)
    run_pt_ccd_test(suite, "PT_parallel_no_collision",
        p=[0.5, 1.0, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[1.0, 0.0, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=False
    )

    # Test 3: Point and triangle moving towards each other
    # Note: with dp=-0.2 and dt=+0.1, relative motion is -0.3, which passes through y=0
    # However the triangle is also moving up, so they may not actually collide during [0,1]
    # Let's use a larger downward motion to guarantee collision
    run_pt_ccd_test(suite, "PT_mutual_approach",
        p=[0.5, 0.3, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.35, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )

    # Test 4: Point moving away from triangle
    run_pt_ccd_test(suite, "PT_moving_away",
        p=[0.5, 0.1, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, 0.5, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=False
    )


def test_basic_ee_collision(suite):
    """Test basic edge-edge collision scenarios."""
    print("\n--- Basic Edge-Edge Collision Tests ---")

    # Test 1: Perpendicular edges, one moving towards other
    run_ee_ccd_test(suite, "EE_perpendicular",
        ea0=[0.0, 0.0, 0.0], ea1=[1.0, 0.0, 0.0],
        eb0=[0.5, 0.2, -0.5], eb1=[0.5, 0.2, 0.5],
        dea0=[0.0, 0.0, 0.0], dea1=[0.0, 0.0, 0.0],
        deb0=[0.0, -0.3, 0.0], deb1=[0.0, -0.3, 0.0],
        expect_collision=True
    )

    # Test 2: Parallel edges moving apart
    run_ee_ccd_test(suite, "EE_parallel_apart",
        ea0=[0.0, 0.0, 0.0], ea1=[1.0, 0.0, 0.0],
        eb0=[0.0, 0.5, 0.0], eb1=[1.0, 0.5, 0.0],
        dea0=[0.0, -0.1, 0.0], dea1=[0.0, -0.1, 0.0],
        deb0=[0.0, 0.1, 0.0], deb1=[0.0, 0.1, 0.0],
        expect_collision=False
    )

    # Test 3: Skew edges approaching
    run_ee_ccd_test(suite, "EE_skew_approach",
        ea0=[0.0, 0.0, 0.0], ea1=[1.0, 0.0, 0.0],
        eb0=[0.3, 0.15, -0.5], eb1=[0.7, 0.15, 0.5],
        dea0=[0.0, 0.0, 0.0], dea1=[0.0, 0.0, 0.0],
        deb0=[0.0, -0.2, 0.0], deb1=[0.0, -0.2, 0.0],
        expect_collision=True
    )


def test_close_proximity(suite):
    """Test cases where primitives are very close initially."""
    print("\n--- Close Proximity Tests ---")

    # Test 1: Point very close to triangle
    run_pt_ccd_test(suite, "PT_very_close",
        p=[0.5, 0.005, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.01, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )

    # Test 2: Edges very close
    run_ee_ccd_test(suite, "EE_very_close",
        ea0=[0.0, 0.0, 0.0], ea1=[1.0, 0.0, 0.0],
        eb0=[0.5, 0.005, -0.5], eb1=[0.5, 0.005, 0.5],
        dea0=[0.0, 0.0, 0.0], dea1=[0.0, 0.0, 0.0],
        deb0=[0.0, -0.01, 0.0], deb1=[0.0, -0.01, 0.0],
        expect_collision=True
    )


def test_grazing_collision(suite):
    """Test grazing/tangential collisions."""
    print("\n--- Grazing Collision Tests ---")

    # Test 1: Point grazing triangle edge - move more directly towards the triangle
    # The point starts at y=0.05 and we move down by more than that
    run_pt_ccd_test(suite, "PT_graze_edge",
        p=[0.5, 0.05, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.1, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )

    # Test 2: Point grazing triangle vertex
    run_pt_ccd_test(suite, "PT_graze_vertex",
        p=[0.0, 0.05, 0.05],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.1, -0.1],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )


def test_high_velocity(suite):
    """Test high velocity scenarios (large displacements)."""
    print("\n--- High Velocity Tests ---")

    # Test 1: Fast moving point
    run_pt_ccd_test(suite, "PT_high_velocity",
        p=[0.5, 10.0, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -15.0, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True
    )

    # Test 2: Fast moving edges
    run_ee_ccd_test(suite, "EE_high_velocity",
        ea0=[0.0, 0.0, 0.0], ea1=[1.0, 0.0, 0.0],
        eb0=[0.5, 5.0, -0.5], eb1=[0.5, 5.0, 0.5],
        dea0=[0.0, 0.0, 0.0], dea1=[0.0, 0.0, 0.0],
        deb0=[0.0, -8.0, 0.0], deb1=[0.0, -8.0, 0.0],
        expect_collision=True
    )


def test_zero_motion(suite):
    """Test with zero relative motion."""
    print("\n--- Zero Motion Tests ---")

    # Test 1: No motion at all
    run_pt_ccd_test(suite, "PT_zero_motion",
        p=[0.5, 0.5, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, 0.0, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=False
    )

    # Test 2: Same motion (no relative motion)
    run_pt_ccd_test(suite, "PT_same_motion",
        p=[0.5, 0.5, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[1.0, 2.0, 3.0],
        dt0=[1.0, 2.0, 3.0], dt1=[1.0, 2.0, 3.0], dt2=[1.0, 2.0, 3.0],
        expect_collision=False
    )


def test_random_scenarios(suite, n_tests=50):
    """
    Generate random test scenarios to stress test the CCD non-penetration property.
    Only verifies safety, doesn't assume collisions will occur.
    """
    print(f"\n--- Random Scenario Tests ({n_tests} tests each for PT and EE) ---")

    random.seed(42)

    for i in range(n_tests):
        p = [random.uniform(-1, 1), random.uniform(0.1, 1), random.uniform(-1, 1)]
        t0 = [random.uniform(-1, 1), 0.0, random.uniform(-1, 1)]
        t1 = [random.uniform(-1, 1), 0.0, random.uniform(-1, 1)]
        t2 = [random.uniform(-1, 1), 0.0, random.uniform(-1, 1)]
        dp = [random.uniform(-0.5, 0.5) for _ in range(3)]
        dt0 = [random.uniform(-0.2, 0.2) for _ in range(3)]
        dt1 = [random.uniform(-0.2, 0.2) for _ in range(3)]
        dt2 = [random.uniform(-0.2, 0.2) for _ in range(3)]

        run_pt_ccd_non_penetration_only(suite, f"PT_random_{i}",
            p=p, t0=t0, t1=t1, t2=t2, dp=dp, dt0=dt0, dt1=dt1, dt2=dt2)

    for i in range(n_tests):
        ea0 = [random.uniform(-1, 1), 0.0, random.uniform(-1, 1)]
        ea1 = [random.uniform(-1, 1), 0.0, random.uniform(-1, 1)]
        eb0 = [random.uniform(-1, 1), random.uniform(0.1, 0.5), random.uniform(-1, 1)]
        eb1 = [random.uniform(-1, 1), random.uniform(0.1, 0.5), random.uniform(-1, 1)]
        dea0 = [random.uniform(-0.3, 0.3) for _ in range(3)]
        dea1 = [random.uniform(-0.3, 0.3) for _ in range(3)]
        deb0 = [random.uniform(-0.3, 0.3) for _ in range(3)]
        deb1 = [random.uniform(-0.3, 0.3) for _ in range(3)]

        run_ee_ccd_non_penetration_only(suite, f"EE_random_{i}",
            ea0=ea0, ea1=ea1, eb0=eb0, eb1=eb1,
            dea0=dea0, dea1=dea1, deb0=deb0, deb1=deb1)


def test_guaranteed_collision_scenarios(suite, n_tests=20):
    """
    Generate scenarios that GUARANTEE a collision will occur.
    """
    print(f"\n--- Guaranteed Collision Tests ({n_tests} tests each for PT and EE) ---")

    random.seed(123)

    for i in range(n_tests):
        # Large triangle in XZ plane at y=0
        t0 = [-1.0, 0.0, -1.0]
        t1 = [1.0, 0.0, -1.0]
        t2 = [0.0, 0.0, 1.0]

        center_x = (t0[0] + t1[0] + t2[0]) / 3
        center_z = (t0[2] + t1[2] + t2[2]) / 3
        start_height = random.uniform(0.05, 0.3)
        p = [center_x + random.uniform(-0.05, 0.05), start_height, center_z + random.uniform(-0.05, 0.05)]

        # Move point straight down
        dp = [0.0, -(start_height + 0.05), 0.0]
        dt0 = [0.0, 0.0, 0.0]
        dt1 = [0.0, 0.0, 0.0]
        dt2 = [0.0, 0.0, 0.0]

        run_pt_ccd_test(suite, f"PT_guaranteed_{i}",
            p=p, t0=t0, t1=t1, t2=t2, dp=dp, dt0=dt0, dt1=dt1, dt2=dt2,
            expect_collision=True)

    for i in range(n_tests):
        # Perpendicular edges that will cross
        ea0 = [-0.5, 0.0, 0.0]
        ea1 = [0.5, 0.0, 0.0]

        start_height = random.uniform(0.05, 0.2)
        eb0 = [0.0, start_height, -0.5]
        eb1 = [0.0, start_height, 0.5]

        # Move edge B straight down
        dea0 = [0.0, 0.0, 0.0]
        dea1 = [0.0, 0.0, 0.0]
        deb0 = [0.0, -(start_height + 0.05), 0.0]
        deb1 = [0.0, -(start_height + 0.05), 0.0]

        run_ee_ccd_test(suite, f"EE_guaranteed_{i}",
            ea0=ea0, ea1=ea1, eb0=eb0, eb1=eb1,
            dea0=dea0, dea1=dea1, deb0=deb0, deb1=deb1,
            expect_collision=True)


def test_numerical_precision(suite):
    """Test numerical precision with various values."""
    print("\n--- Numerical Precision Tests ---")

    # Test 1: Very small distances - non-penetration only
    run_pt_ccd_non_penetration_only(suite, "PT_tiny_distance",
        p=[0.5, 1e-6, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -1e-7, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0])

    # Test 2: Small but reasonable distances
    run_pt_ccd_test(suite, "PT_small_distance",
        p=[0.5, 0.001, 0.5],
        t0=[0.0, 0.0, 0.0], t1=[1.0, 0.0, 0.0], t2=[0.5, 0.0, 1.0],
        dp=[0.0, -0.002, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True)

    # Test 3: Large coordinates
    run_pt_ccd_test(suite, "PT_large_coords",
        p=[1000.5, 1000.2, 1000.5],
        t0=[1000.0, 1000.0, 1000.0], t1=[1001.0, 1000.0, 1000.0], t2=[1000.5, 1000.0, 1001.0],
        dp=[0.0, -0.3, 0.0],
        dt0=[0.0, 0.0, 0.0], dt1=[0.0, 0.0, 0.0], dt2=[0.0, 0.0, 0.0],
        expect_collision=True)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("CCD Lower Bound Correctness Test Suite")
    print("Testing CubicNoRootRegionPrecise-based CCD implementation")
    print("=" * 70)

    suite = CCDTestSuite()

    # Run all test categories
    test_basic_pt_collision(suite)
    test_basic_ee_collision(suite)
    test_close_proximity(suite)
    test_grazing_collision(suite)
    test_high_velocity(suite)
    test_zero_motion(suite)
    test_random_scenarios(suite, n_tests=50)
    test_guaranteed_collision_scenarios(suite, n_tests=20)
    test_numerical_precision(suite)

    # Report results
    success = suite.report()

    if success:
        print("\n✓ All tests passed! CCD lower bound maintains non-penetration property.")
    else:
        print("\n✗ Some tests failed. Check the implementation.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
