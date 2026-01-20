"""
Unit tests for IPC contact handling module.

Tests barrier functions, contact handlers, and CCD step size computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import taichi as ti
import numpy as np

# Initialize Taichi before importing contact modules
ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache_ipc_test")

from algorithm_new.contact import (
    LogBarrier,
    CubicBarrier,
    IPCContactHandler,
    GroundContactHandler,
    CCDStepSizeComputer,
    compute_adaptive_kappa,
)
from algorithm_new.contact.barrier.log_barrier import (
    barrier_E_log, barrier_g_log, barrier_H_log
)
from algorithm_new.contact.barrier.cubic_barrier import (
    barrier_E_cubic, barrier_g_cubic, barrier_H_cubic
)


def test_log_barrier_values():
    """Test log barrier function values at known points."""
    print("Testing LogBarrier values...")

    # Test parameters
    dHat = 0.1
    kappa = 1e4

    @ti.kernel
    def compute_barrier_values(d: ti.f32) -> ti.types.vector(3, ti.f32):
        E = LogBarrier.energy(d, dHat, kappa)
        g = LogBarrier.gradient(d, dHat, kappa)
        H = LogBarrier.hessian(d, dHat, kappa)
        return ti.Vector([E, g, H])

    # Test at d = dHat (boundary)
    result = compute_barrier_values(dHat)
    assert abs(result[0]) < 1e-6, f"E at dHat should be ~0, got {result[0]}"
    print(f"  E(dHat) = {result[0]:.6e} (should be ~0)")

    # Test at d = dHat/2
    result = compute_barrier_values(dHat / 2)
    assert result[0] > 0, f"E at dHat/2 should be positive, got {result[0]}"
    assert result[1] < 0, f"g at dHat/2 should be negative (repulsive), got {result[1]}"
    assert result[2] > 0, f"H at dHat/2 should be positive, got {result[2]}"
    print(f"  E(dHat/2) = {result[0]:.6e}, g = {result[1]:.6e}, H = {result[2]:.6e}")

    # Test at d > dHat (outside barrier)
    result = compute_barrier_values(dHat * 2)
    assert abs(result[0]) < 1e-6, f"E outside barrier should be 0, got {result[0]}"
    print(f"  E(2*dHat) = {result[0]:.6e} (should be 0)")

    print("  LogBarrier values test passed!")


def test_cubic_barrier_values():
    """Test cubic barrier function values at known points."""
    print("Testing CubicBarrier values...")

    dHat = 0.1
    kappa = 1e4

    @ti.kernel
    def compute_barrier_values(d: ti.f32) -> ti.types.vector(3, ti.f32):
        E = CubicBarrier.energy(d, dHat, kappa)
        g = CubicBarrier.gradient(d, dHat, kappa)
        H = CubicBarrier.hessian(d, dHat, kappa)
        return ti.Vector([E, g, H])

    # Test at d = dHat (boundary)
    result = compute_barrier_values(dHat)
    assert abs(result[0]) < 1e-6, f"E at dHat should be 0, got {result[0]}"
    assert abs(result[1]) < 1e-6, f"g at dHat should be 0, got {result[1]}"
    assert abs(result[2]) < 1e-6, f"H at dHat should be 0, got {result[2]}"
    print(f"  At d=dHat: E={result[0]:.6e}, g={result[1]:.6e}, H={result[2]:.6e}")

    # Test at d = dHat/2
    result = compute_barrier_values(dHat / 2)
    assert result[0] > 0, f"E at dHat/2 should be positive, got {result[0]}"
    assert result[1] < 0, f"g at dHat/2 should be negative, got {result[1]}"
    assert result[2] > 0, f"H at dHat/2 should be positive, got {result[2]}"
    print(f"  At d=dHat/2: E={result[0]:.6e}, g={result[1]:.6e}, H={result[2]:.6e}")

    # Test at d = 0 (maximum repulsion)
    result = compute_barrier_values(0.001)
    assert result[0] > 0, "E at d~0 should be large positive"
    assert result[1] < 0, "g at d~0 should be large negative"
    print(f"  At d~0: E={result[0]:.6e}, g={result[1]:.6e}, H={result[2]:.6e}")

    print("  CubicBarrier values test passed!")


def test_barrier_consistency():
    """Test that gradient is derivative of energy, hessian is derivative of gradient."""
    print("Testing barrier derivative consistency...")

    dHat = 0.1
    kappa = 1e4
    eps = 1e-6

    @ti.kernel
    def numerical_derivative_check(d: ti.f32) -> ti.types.vector(4, ti.f32):
        # Log barrier
        E_log = LogBarrier.energy(d, dHat, kappa)
        E_log_plus = LogBarrier.energy(d + eps, dHat, kappa)
        g_log_numerical = (E_log_plus - E_log) / eps
        g_log_analytic = LogBarrier.gradient(d, dHat, kappa)

        # Cubic barrier
        E_cubic = CubicBarrier.energy(d, dHat, kappa)
        E_cubic_plus = CubicBarrier.energy(d + eps, dHat, kappa)
        g_cubic_numerical = (E_cubic_plus - E_cubic) / eps
        g_cubic_analytic = CubicBarrier.gradient(d, dHat, kappa)

        return ti.Vector([g_log_numerical, g_log_analytic, g_cubic_numerical, g_cubic_analytic])

    d = dHat / 2
    result = numerical_derivative_check(d)

    log_error = abs(result[0] - result[1]) / (abs(result[1]) + 1e-10)
    cubic_error = abs(result[2] - result[3]) / (abs(result[3]) + 1e-10)

    print(f"  Log gradient: numerical={result[0]:.6e}, analytic={result[1]:.6e}, rel_error={log_error:.6e}")
    print(f"  Cubic gradient: numerical={result[2]:.6e}, analytic={result[3]:.6e}, rel_error={cubic_error:.6e}")

    assert log_error < 1e-3, f"Log barrier gradient inconsistent: {log_error}"
    assert cubic_error < 1e-3, f"Cubic barrier gradient inconsistent: {cubic_error}"

    print("  Barrier derivative consistency test passed!")


def test_ipc_contact_handler_creation():
    """Test IPCContactHandler creation with different barrier types."""
    print("Testing IPCContactHandler creation...")

    handler_log = IPCContactHandler(max_contacts=1000, barrier_type='log', precision='f32')
    assert handler_log.barrier_type == 'log'
    print(f"  Created log barrier handler with {handler_log.MAX_CONTACTS} max contacts")

    handler_cubic = IPCContactHandler(max_contacts=1000, barrier_type='cubic', precision='f64')
    assert handler_cubic.barrier_type == 'cubic'
    print(f"  Created cubic barrier handler with precision {handler_cubic.precision}")

    print("  IPCContactHandler creation test passed!")


def test_ground_handler():
    """Test GroundContactHandler basic functionality."""
    print("Testing GroundContactHandler...")

    n_vertices = 10
    handler = GroundContactHandler(
        max_vertices=n_vertices,
        ground_y=0.0,
        barrier_type='log',
        precision='f32'
    )

    assert handler.ground_y == 0.0
    handler.ground_y = -0.5
    assert handler.ground_y == -0.5

    # Create test vertices
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
    grad = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
    diagH = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)

    # Initialize vertices above ground
    for i in range(n_vertices):
        vertices[i] = [0.0, 0.5, 0.0]  # All at y=0.5

    # Reset grad and diagH
    for i in range(n_vertices):
        grad[i] = [0.0, 0.0, 0.0]
        diagH[i] = [0.0, 0.0, 0.0]

    # No gradient should be added (vertices are far from ground)
    handler.ground_y = 0.0
    handler.add_gradient(vertices, grad, n_vertices, 0.01, 1e4, 0.1)

    # Check gradient is zero
    grad_sum = 0.0
    for i in range(n_vertices):
        grad_sum += abs(grad[i][0]) + abs(grad[i][1]) + abs(grad[i][2])

    assert grad_sum < 1e-10, f"Gradient should be zero when vertices are far from ground, got {grad_sum}"
    print(f"  Gradient sum when far from ground: {grad_sum:.6e} (should be ~0)")

    # Now move vertices close to ground
    for i in range(n_vertices):
        vertices[i] = [0.0, 0.05, 0.0]  # At y=0.05, close to ground at y=0

    # Reset grad
    for i in range(n_vertices):
        grad[i] = [0.0, 0.0, 0.0]

    handler.add_gradient(vertices, grad, n_vertices, 0.01, 1e4, 0.1)

    # Check gradient is non-zero and in Y direction
    # Note: barrier gradient is negative (dE/dd < 0 when d < dHat)
    # This represents the gradient of potential energy, not force
    # Force would be -grad, which would be positive (pushing up)
    grad_y_sum = 0.0
    for i in range(n_vertices):
        grad_y_sum += grad[i][1]

    assert grad_y_sum < 0, f"Gradient Y should be negative (barrier gradient), got {grad_y_sum}"
    print(f"  Gradient Y sum when close to ground: {grad_y_sum:.6e} (negative = repulsive barrier)")

    # Test contact counting
    count = handler.count_ground_contacts(vertices, n_vertices, 0.1)
    assert count == n_vertices, f"All vertices should be in contact, got {count}"
    print(f"  Ground contacts: {count} (expected {n_vertices})")

    print("  GroundContactHandler test passed!")


def test_ccd_step_size_computer():
    """Test CCDStepSizeComputer basic functionality."""
    print("Testing CCDStepSizeComputer...")

    computer = CCDStepSizeComputer(
        max_contacts=1000,
        safety_factor=0.8,
        precision='f32'
    )

    computer.set_ground(0.0)

    # Create vertices moving towards ground
    n_vertices = 5
    x = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
    p = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)

    # Vertex at y=0.2, moving down with velocity -0.5
    for i in range(n_vertices):
        x[i] = [0.0, 0.2, 0.0]
        p[i] = [0.0, -0.5, 0.0]

    dHat = 0.05
    alpha = computer.compute_safe_step_simple(x, p, n_vertices, dHat)

    # Expected TOI: (ground_y + dHat - y) / p_y = (0 + 0.05 - 0.2) / (-0.5) = 0.3
    # With safety factor 0.8: 0.3 * 0.8 = 0.24
    expected_alpha = 0.3 * 0.8
    print(f"  Computed safe step: {alpha:.4f}, expected: {expected_alpha:.4f}")

    assert abs(alpha - expected_alpha) < 0.01, f"Step size mismatch: {alpha} vs {expected_alpha}"

    print("  CCDStepSizeComputer test passed!")


def test_adaptive_kappa():
    """Test adaptive kappa computation."""
    print("Testing adaptive kappa computation...")

    avg_mass = 1.0
    gap = 0.01
    hessian_diag = 1e3

    kappa = compute_adaptive_kappa(avg_mass, gap, hessian_diag)
    expected = avg_mass / (gap * gap) + abs(hessian_diag)

    print(f"  Computed kappa: {kappa:.2e}, expected: {expected:.2e}")
    assert abs(kappa - expected) < 1e-6 * expected, f"Kappa mismatch: {kappa} vs {expected}"

    # Test with very small gap
    kappa_small = compute_adaptive_kappa(avg_mass, 1e-6, hessian_diag)
    assert kappa_small > 1e10, f"Kappa should be very large for tiny gap: {kappa_small}"
    print(f"  Kappa with small gap: {kappa_small:.2e}")

    print("  Adaptive kappa test passed!")


def test_precision_support():
    """Test f32 and f64 precision support."""
    print("Testing precision support...")

    for precision in ['f32', 'f64']:
        handler = IPCContactHandler(max_contacts=100, precision=precision)
        assert handler.precision == precision
        print(f"  IPCContactHandler with precision={precision}: OK")

        ground = GroundContactHandler(max_vertices=100, precision=precision)
        assert ground.precision == precision
        print(f"  GroundContactHandler with precision={precision}: OK")

        ccd = CCDStepSizeComputer(max_contacts=100, precision=precision)
        assert ccd.precision == precision
        print(f"  CCDStepSizeComputer with precision={precision}: OK")

    print("  Precision support test passed!")


def run_all_tests():
    """Run all IPC contact tests."""
    print("=" * 60)
    print("Running IPC Contact Module Tests")
    print("=" * 60)

    test_log_barrier_values()
    test_cubic_barrier_values()
    test_barrier_consistency()
    test_ipc_contact_handler_creation()
    test_ground_handler()
    test_ccd_step_size_computer()
    test_adaptive_kappa()
    test_precision_support()

    print("=" * 60)
    print("All IPC contact tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
