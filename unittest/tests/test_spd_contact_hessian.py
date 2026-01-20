"""
Test SPD Contact Hessian Implementation.

This test verifies that the SPD barrier and friction Hessian implementations
from PPF-Contact-Solver style are working correctly.

Key tests:
1. Barrier curvature is non-negative
2. SPD contact Hessian is PSD (eigenvalues >= 0)
3. Friction projection matrix is PSD
4. Combined contact + friction Hessian is PSD
"""

import taichi as ti
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Initialize Taichi
ti.init(arch=ti.cpu, debug=True)


def check_psd(matrix, tol=1e-4):
    """Check if a matrix is positive semi-definite.

    Note: We use a relatively large tolerance (1e-4) because floating-point
    computations can introduce small numerical errors. For practical purposes,
    eigenvalues close to zero are acceptable.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -tol), eigenvalues


def test_barrier_curvature():
    """Test that barrier curvature is non-negative."""
    print("\n=== Test Barrier Curvature ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        barrier_curvature_cubic,
        barrier_curvature_log,
    )

    @ti.kernel
    def compute_curvatures(d: ti.f32, dHat: ti.f32, kappa: ti.f32) -> ti.math.vec2:
        c_cubic = barrier_curvature_cubic(d, dHat, kappa)
        c_log = barrier_curvature_log(d, dHat, kappa)
        return ti.Vector([c_cubic, c_log])

    dHat = 0.01
    kappa = 1e5

    # Test various distances
    test_distances = [0.001, 0.002, 0.005, 0.008, 0.009, 0.0095]

    all_passed = True
    for d in test_distances:
        curvatures = compute_curvatures(d, dHat, kappa)
        c_cubic, c_log = curvatures[0], curvatures[1]

        passed_cubic = c_cubic >= 0
        passed_log = c_log >= 0

        status = "✓" if (passed_cubic and passed_log) else "✗"
        print(f"  d={d:.4f}: cubic={c_cubic:.2e} ({'≥0' if passed_cubic else '<0'}), "
              f"log={c_log:.2e} ({'≥0' if passed_log else '<0'}) {status}")

        if not (passed_cubic and passed_log):
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_spd_contact_hessian():
    """Test that SPD contact Hessian is PSD."""
    print("\n=== Test SPD Contact Hessian ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        compute_spd_edge_hessian,
    )

    # Use a field to store the result
    H_result = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_hessian_cubic(ex: ti.f32, ey: ti.f32, ez: ti.f32, dHat: ti.f32, kappa: ti.f32):
        e = ti.Vector([ex, ey, ez])
        H_result[None] = compute_spd_edge_hessian(e, dHat, kappa, True)

    @ti.kernel
    def compute_hessian_log(ex: ti.f32, ey: ti.f32, ez: ti.f32, dHat: ti.f32, kappa: ti.f32):
        e = ti.Vector([ex, ey, ez])
        H_result[None] = compute_spd_edge_hessian(e, dHat, kappa, False)

    dHat = 0.01
    kappa = 1e5

    # Test various contact edge vectors
    test_edges = [
        (0.005, 0.0, 0.0),
        (0.003, 0.004, 0.0),
        (0.002, 0.002, 0.002),
        (0.001, 0.005, 0.003),
    ]

    all_passed = True
    for (ex, ey, ez) in test_edges:
        # Test cubic barrier
        compute_hessian_cubic(ex, ey, ez, dHat, kappa)
        H_cubic = H_result[None].to_numpy()
        is_psd_cubic, eigs_cubic = check_psd(H_cubic)

        # Test log barrier
        compute_hessian_log(ex, ey, ez, dHat, kappa)
        H_log = H_result[None].to_numpy()
        is_psd_log, eigs_log = check_psd(H_log)

        status = "✓" if (is_psd_cubic and is_psd_log) else "✗"
        print(f"  e=({ex:.3f},{ey:.3f},{ez:.3f}): "
              f"cubic {'PSD' if is_psd_cubic else 'NOT PSD'} (min_eig={eigs_cubic.min():.2e}), "
              f"log {'PSD' if is_psd_log else 'NOT PSD'} (min_eig={eigs_log.min():.2e}) {status}")

        if not (is_psd_cubic and is_psd_log):
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_friction_projection_matrix():
    """Test that friction projection matrix is PSD."""
    print("\n=== Test Friction Projection Matrix ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        compute_friction_projection_matrix,
    )

    P_result = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_projection(nx: ti.f32, ny: ti.f32, nz: ti.f32):
        n = ti.Vector([nx, ny, nz])
        P_result[None] = compute_friction_projection_matrix(n)

    # Test various unit normals
    test_normals = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.577, 0.577, 0.577),  # approx (1,1,1)/sqrt(3)
        (0.707, 0.707, 0.0),    # approx (1,1,0)/sqrt(2)
    ]

    all_passed = True
    for (nx, ny, nz) in test_normals:
        # Normalize
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx/norm, ny/norm, nz/norm

        compute_projection(nx, ny, nz)
        P = P_result[None].to_numpy()
        is_psd, eigenvalues = check_psd(P)

        # Check eigenvalues are [1, 1, 0] (approximately)
        sorted_eigs = np.sort(eigenvalues)
        expected_eigs = np.array([0.0, 1.0, 1.0])
        eigs_correct = np.allclose(sorted_eigs, expected_eigs, atol=1e-5)

        status = "✓" if (is_psd and eigs_correct) else "✗"
        print(f"  n=({nx:.3f},{ny:.3f},{nz:.3f}): "
              f"{'PSD' if is_psd else 'NOT PSD'}, eigs={sorted_eigs} {status}")

        if not (is_psd and eigs_correct):
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_friction_hessian():
    """Test that friction Hessian is PSD."""
    print("\n=== Test Friction Hessian ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        compute_friction_hessian,
    )

    H_result = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_fric_hess(
        fx: ti.f32, fy: ti.f32, fz: ti.f32,  # contact force
        dxx: ti.f32, dxy: ti.f32, dxz: ti.f32,  # displacement
        nx: ti.f32, ny: ti.f32, nz: ti.f32,  # normal
        mu: ti.f32, min_dx: ti.f32,
    ):
        f = ti.Vector([fx, fy, fz])
        dx = ti.Vector([dxx, dxy, dxz])
        n = ti.Vector([nx, ny, nz])
        H_result[None] = compute_friction_hessian(f, dx, n, mu, min_dx)

    # Test cases: contact force pointing opposite to normal (into surface)
    test_cases = [
        # (force, displacement, normal, mu)
        ((-10.0, 0.0, 0.0), (0.001, 0.002, 0.0), (1.0, 0.0, 0.0), 0.5),
        ((-5.0, -5.0, 0.0), (0.001, 0.0, 0.001), (0.707, 0.707, 0.0), 0.3),
        ((0.0, -10.0, 0.0), (0.002, 0.0, 0.002), (0.0, 1.0, 0.0), 0.8),
    ]

    min_dx = 1e-4

    all_passed = True
    for (force, disp, normal, mu) in test_cases:
        fx, fy, fz = force
        dxx, dxy, dxz = disp
        nx, ny, nz = normal

        # Normalize normal
        nn = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx/nn, ny/nn, nz/nn

        compute_fric_hess(fx, fy, fz, dxx, dxy, dxz, nx, ny, nz, mu, min_dx)
        H = H_result[None].to_numpy()
        is_psd, eigenvalues = check_psd(H)

        status = "✓" if is_psd else "✗"
        print(f"  μ={mu}, f={force}: "
              f"{'PSD' if is_psd else 'NOT PSD'}, min_eig={eigenvalues.min():.2e} {status}")

        if not is_psd:
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_combined_contact_friction_hessian():
    """Test that combined contact + friction Hessian is PSD."""
    print("\n=== Test Combined Contact + Friction Hessian ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        compute_spd_contact_friction_hessian,
    )

    H_result = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_combined_cubic(
        ex: ti.f32, ey: ti.f32, ez: ti.f32,
        dxx: ti.f32, dxy: ti.f32, dxz: ti.f32,
        dHat: ti.f32, kappa: ti.f32, mu: ti.f32, min_dx: ti.f32,
    ):
        e = ti.Vector([ex, ey, ez])
        dx = ti.Vector([dxx, dxy, dxz])
        H_result[None] = compute_spd_contact_friction_hessian(e, dx, dHat, kappa, mu, min_dx, True)

    @ti.kernel
    def compute_combined_log(
        ex: ti.f32, ey: ti.f32, ez: ti.f32,
        dxx: ti.f32, dxy: ti.f32, dxz: ti.f32,
        dHat: ti.f32, kappa: ti.f32, mu: ti.f32, min_dx: ti.f32,
    ):
        e = ti.Vector([ex, ey, ez])
        dx = ti.Vector([dxx, dxy, dxz])
        H_result[None] = compute_spd_contact_friction_hessian(e, dx, dHat, kappa, mu, min_dx, False)

    dHat = 0.01
    kappa = 1e5
    min_dx = 1e-4

    # Test cases: (edge, displacement, mu)
    test_cases = [
        ((0.005, 0.0, 0.0), (0.001, 0.002, 0.0), 0.0),   # No friction
        ((0.005, 0.0, 0.0), (0.001, 0.002, 0.0), 0.3),   # With friction
        ((0.003, 0.004, 0.0), (0.0, 0.001, 0.001), 0.5),
        ((0.002, 0.002, 0.002), (0.001, 0.001, 0.0), 0.8),
    ]

    all_passed = True
    for (edge, disp, mu) in test_cases:
        ex, ey, ez = edge
        dxx, dxy, dxz = disp

        # Test cubic barrier
        compute_combined_cubic(ex, ey, ez, dxx, dxy, dxz, dHat, kappa, mu, min_dx)
        H_cubic = H_result[None].to_numpy()
        is_psd_cubic, eigs_cubic = check_psd(H_cubic)

        # Test log barrier
        compute_combined_log(ex, ey, ez, dxx, dxy, dxz, dHat, kappa, mu, min_dx)
        H_log = H_result[None].to_numpy()
        is_psd_log, eigs_log = check_psd(H_log)

        status = "✓" if (is_psd_cubic and is_psd_log) else "✗"
        mu_str = f"μ={mu}" if mu > 0 else "no friction"
        print(f"  {mu_str}: "
              f"cubic {'PSD' if is_psd_cubic else 'NOT PSD'} (min={eigs_cubic.min():.2e}), "
              f"log {'PSD' if is_psd_log else 'NOT PSD'} (min={eigs_log.min():.2e}) {status}")

        if not (is_psd_cubic and is_psd_log):
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_extended_hessian_12x12():
    """Test that extended 12x12 Hessian preserves PSD property."""
    print("\n=== Test Extended 12x12 Hessian ===")

    from algorithm.mas_preconditioner_contact.contact_assembly import (
        compute_spd_edge_hessian,
        extend_spd_contact_hessian_12x12,
    )

    H_result = ti.Matrix.field(12, 12, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_extended(
        ex: ti.f32, ey: ti.f32, ez: ti.f32,
        c0: ti.f32, c1: ti.f32, c2: ti.f32, c3: ti.f32,
        dHat: ti.f32, kappa: ti.f32,
    ):
        e = ti.Vector([ex, ey, ez])
        cord = ti.Vector([c0, c1, c2, c3])
        H_3x3 = compute_spd_edge_hessian(e, dHat, kappa, True)  # cubic
        H_result[None] = extend_spd_contact_hessian_12x12(H_3x3, cord)

    dHat = 0.01
    kappa = 1e5

    # Test with different barycentric coordinates
    test_cases = [
        ((0.005, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0)),    # Point-Point
        ((0.003, 0.004, 0.0), (1.0, -0.5, -0.5, 0.0)), # Point-Edge
        ((0.002, 0.002, 0.002), (1.0, -0.33, -0.33, -0.34)),  # Point-Face
    ]

    all_passed = True
    for (edge, cord) in test_cases:
        ex, ey, ez = edge
        c0, c1, c2, c3 = cord

        compute_extended(ex, ey, ez, c0, c1, c2, c3, dHat, kappa)
        H_12x12 = H_result[None].to_numpy()
        is_psd, eigenvalues = check_psd(H_12x12)

        # Count non-zero eigenvalues (should have at most rank 1 for contact Hessian)
        nonzero_eigs = np.sum(np.abs(eigenvalues) > 1e-8)

        status = "✓" if is_psd else "✗"
        print(f"  cord={cord}: "
              f"{'PSD' if is_psd else 'NOT PSD'}, "
              f"rank≈{nonzero_eigs}, min_eig={eigenvalues.min():.2e} {status}")

        if not is_psd:
            all_passed = False

    print(f"  Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("=" * 60)
    print("SPD Contact Hessian Tests (PPF-Contact-Solver style)")
    print("=" * 60)

    results = []

    results.append(("Barrier Curvature", test_barrier_curvature()))
    results.append(("SPD Contact Hessian", test_spd_contact_hessian()))
    results.append(("Friction Projection Matrix", test_friction_projection_matrix()))
    results.append(("Friction Hessian", test_friction_hessian()))
    results.append(("Combined Contact+Friction Hessian", test_combined_contact_friction_hessian()))
    results.append(("Extended 12x12 Hessian", test_extended_hessian_12x12()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
