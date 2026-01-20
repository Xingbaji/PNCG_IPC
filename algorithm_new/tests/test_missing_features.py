"""
Test suite for critical missing features added to algorithm_new.

Tests:
1. SubdomainCCD - Per-subdomain CCD (Algorithm 2 from paper)
2. MASPreconditioner with contacts - Woodbury update support
3. CCD lower bound functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import taichi as ti
import numpy as np


# Initialize Taichi
ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path='.taichi_cache_test_features')


def test_subdomain_ccd():
    """Test SubdomainCCD module for per-subdomain step sizes."""
    print("\n" + "="*60)
    print("Test 1: SubdomainCCD - Per-subdomain CCD")
    print("="*60)

    from algorithm_new.optimizer.subdomain_ccd import SubdomainCCD

    # Setup: Create simple vertex positions and search direction
    n_verts = 64
    banksize = 16
    n_subdomains = (n_verts + banksize - 1) // banksize

    # Create test fields
    x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    p = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    # Initialize vertices on a grid
    for i in range(n_verts):
        x[i] = [i % 4, i // 4, 0]
        p[i] = [0, -1, 0]  # All moving down

    # Create SubdomainCCD
    subdomain_ccd = SubdomainCCD(
        n_verts=n_verts,
        banksize=banksize,
        safety_factor=0.9,
    )

    # Test with ground plane
    subdomain_ccd.set_ground(ground_y=-0.5, enabled=True)
    subdomain_ccd.set_dHat(0.1)

    # Compute subdomain alphas
    min_alpha = subdomain_ccd.compute_subdomain_alphas(x, p)

    # Check results
    stats = subdomain_ccd.get_alpha_statistics()
    print(f"  n_subdomains: {stats['n_subdomains']}")
    print(f"  min_alpha: {stats['min']:.4f}")
    print(f"  max_alpha: {stats['max']:.4f}")
    print(f"  mean_alpha: {stats['mean']:.4f}")
    print(f"  n_constrained: {stats['n_constrained']}")

    # All alphas should be <= 1.0
    assert stats['max'] <= 1.0, "Alpha should not exceed 1.0"

    # Some subdomains should be constrained due to ground
    # (vertices near y=0 moving down toward ground at y=-0.5)
    print("  ✓ SubdomainCCD basic test passed")

    # Test update_x_subdomain
    x_copy = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    for i in range(n_verts):
        x_copy[i] = x[i]

    subdomain_ccd.update_x_subdomain(x_copy, p)

    # Verify positions were updated
    any_changed = False
    for i in range(n_verts):
        if (x_copy[i] - x[i]).norm() > 1e-6:
            any_changed = True
            break
    assert any_changed, "Positions should have been updated"
    print("  ✓ update_x_subdomain test passed")

    return True


def test_ccd_lower_bound():
    """Test CCD lower bound functions for conservative collision detection."""
    print("\n" + "="*60)
    print("Test 2: CCD Lower Bound Functions")
    print("="*60)

    from algorithm_new.collision.queries.ccd import (
        point_triangle_ccd_lower_bound,
        edge_edge_ccd_lower_bound,
    )

    @ti.kernel
    def test_pt_ccd() -> ti.f32:
        # Point above triangle, moving down
        p = ti.Vector([0.5, 1.0, 0.5])
        t0 = ti.Vector([0.0, 0.0, 0.0])
        t1 = ti.Vector([1.0, 0.0, 0.0])
        t2 = ti.Vector([0.5, 0.0, 1.0])

        # Displacement: point moves down by 2 units
        dp = ti.Vector([0.0, -2.0, 0.0])
        dt0 = ti.Vector([0.0, 0.0, 0.0])
        dt1 = ti.Vector([0.0, 0.0, 0.0])
        dt2 = ti.Vector([0.0, 0.0, 0.0])

        toi = point_triangle_ccd_lower_bound(p, t0, t1, t2, dp, dt0, dt1, dt2)
        return toi

    @ti.kernel
    def test_ee_ccd() -> ti.f32:
        # Two parallel edges moving towards each other
        ea0 = ti.Vector([0.0, 0.0, 0.0])
        ea1 = ti.Vector([1.0, 0.0, 0.0])
        eb0 = ti.Vector([0.0, 2.0, 0.0])
        eb1 = ti.Vector([1.0, 2.0, 0.0])

        # Edge a moves up, edge b moves down
        dea0 = ti.Vector([0.0, 1.5, 0.0])
        dea1 = ti.Vector([0.0, 1.5, 0.0])
        deb0 = ti.Vector([0.0, -1.5, 0.0])
        deb1 = ti.Vector([0.0, -1.5, 0.0])

        toi = edge_edge_ccd_lower_bound(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1)
        return toi

    # Test PT CCD
    pt_toi = test_pt_ccd()
    print(f"  Point-Triangle CCD TOI: {pt_toi:.4f}")
    # Point starts at y=1, moves down 2 units, should hit triangle at y=0 around t=0.5
    assert pt_toi < 1.0, "PT CCD should detect collision"
    assert pt_toi > 0.0, "PT CCD should not be immediate collision"
    print("  ✓ Point-Triangle CCD test passed")

    # Test EE CCD
    ee_toi = test_ee_ccd()
    print(f"  Edge-Edge CCD TOI: {ee_toi:.4f}")
    # Edges start 2 units apart, moving 3 units towards each other
    assert ee_toi < 1.0, "EE CCD should detect collision"
    assert ee_toi > 0.0, "EE CCD should not be immediate collision"
    print("  ✓ Edge-Edge CCD test passed")

    return True


def test_mas_preconditioner_wrapper():
    """Test MASPreconditioner wrapper with contacts flag."""
    print("\n" + "="*60)
    print("Test 3: MASPreconditioner Wrapper")
    print("="*60)

    from algorithm_new.preconditioner import (
        MASPreconditioner,
        create_mas_preconditioner,
    )

    # Test basic import and registration
    print("  ✓ MASPreconditioner imported successfully")

    # Test factory function exists
    assert callable(create_mas_preconditioner)
    print("  ✓ create_mas_preconditioner factory function available")

    # Check that with_contacts parameter exists
    import inspect
    sig = inspect.signature(MASPreconditioner.__init__)
    params = list(sig.parameters.keys())
    assert 'with_contacts' in params, "with_contacts parameter should exist"
    print("  ✓ with_contacts parameter available")

    # Check Woodbury methods exist
    methods = ['save_base_state', 'woodbury_update', 'apply_with_woodbury', 'should_use_woodbury']
    for method in methods:
        assert hasattr(MASPreconditioner, method), f"{method} should exist"
    print("  ✓ Woodbury methods available")

    # Check contact methods exist
    contact_methods = ['rebuild_with_contacts', 'rebuild_with_contacts_spd', 'get_contact_stats']
    for method in contact_methods:
        assert hasattr(MASPreconditioner, method), f"{method} should exist"
    print("  ✓ Contact methods available")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Critical Missing Features Tests")
    print("="*60)

    tests = [
        ("SubdomainCCD", test_subdomain_ccd),
        ("CCD Lower Bounds", test_ccd_lower_bound),
        ("MAS Preconditioner Wrapper", test_mas_preconditioner_wrapper),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            import traceback
            results.append((name, False, traceback.format_exc()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = 0
    failed = 0
    for name, success, error in results:
        if success:
            print(f"  ✓ {name}: PASSED")
            passed += 1
        else:
            print(f"  ✗ {name}: FAILED")
            if error:
                print(f"    Error: {error[:200]}...")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
