"""
GCP barrier functions with C2 mollification.

Provides barrier energy, gradient, and hessian with smooth step mollification
and directional factor weighting.
"""

import taichi as ti
from .smooth_step import (
    smooth_step_cubic,
    smooth_step_cubic_derivative,
    smooth_step_cubic_second_derivative,
)


@ti.func
def gcp_barrier_energy(d, epsilon, gamma, kappa):
    """
    GCP mollified barrier energy.

    E = kappa * gamma * h(d) * barrier(d)

    where:
    - h(d) is C2 mollifier (smooth step from 1 to 0)
    - barrier(d) = -log(d/epsilon) (standard IPC log barrier)
    - gamma is directional factor (filters adjacent elements)
    - kappa is barrier stiffness

    Args:
        d: Distance (scalar)
        epsilon: Detection threshold (epsilon_target or per-primitive)
        gamma: Directional factor from compute_gamma_PT/EE
        kappa: Barrier stiffness

    Returns:
        GCP barrier energy
    """
    E = d * 0.0  # Preserve type
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, d * 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        E = kappa * gamma * h * barrier
    return E


@ti.func
def gcp_barrier_gradient(d, epsilon, gamma, kappa):
    """
    GCP barrier gradient (dE/dd).

    Using product rule:
    dE/dd = kappa * gamma * (dh/dd * barrier + h * dbarrier/dd)

    where:
    - dh/dd = smooth_step_cubic_derivative
    - barrier = -log(d/epsilon)
    - dbarrier/dd = -1/d

    Args:
        d: Distance
        epsilon: Detection threshold
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        GCP barrier gradient
    """
    g = d * 0.0  # Preserve type
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, d * 0.0, epsilon)
        dh = smooth_step_cubic_derivative(d, d * 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        dbarrier = -1.0 / d

        g = kappa * gamma * (dh * barrier + h * dbarrier)
    return g


@ti.func
def gcp_barrier_hessian(d, epsilon, gamma, kappa):
    """
    GCP barrier hessian (d²E/dd²).

    Using product rule twice:
    d²E/dd² = kappa * gamma * (d²h/dd² * barrier + 2 * dh/dd * dbarrier/dd + h * d²barrier/dd²)

    where:
    - d²h/dd² = smooth_step_cubic_second_derivative
    - barrier = -log(d/epsilon)
    - dbarrier/dd = -1/d
    - d²barrier/dd² = 1/d²

    Args:
        d: Distance
        epsilon: Detection threshold
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        GCP barrier hessian
    """
    H = d * 0.0  # Preserve type
    if d < epsilon and gamma > 1e-8 and d > 1e-10:
        h = smooth_step_cubic(d, d * 0.0, epsilon)
        dh = smooth_step_cubic_derivative(d, d * 0.0, epsilon)
        d2h = smooth_step_cubic_second_derivative(d, d * 0.0, epsilon)
        barrier = -ti.log(d / epsilon)
        dbarrier = -1.0 / d
        d2barrier = 1.0 / (d * d)

        H = kappa * gamma * (d2h * barrier + 2.0 * dh * dbarrier + h * d2barrier)
    return H


@ti.func
def gcp_barrier_curvature(d, epsilon, gamma, kappa):
    """
    GCP barrier curvature for SPD Hessian construction.

    Same as hessian but clamped to non-negative for SPD guarantee.

    Args:
        d: Distance
        epsilon: Detection threshold
        gamma: Directional factor
        kappa: Barrier stiffness

    Returns:
        Non-negative curvature
    """
    H = gcp_barrier_hessian(d, epsilon, gamma, kappa)
    return ti.max(H, d * 0.0)


class GCPBarrier:
    """
    GCP Barrier class with static methods.

    Provides a class-based interface similar to LogBarrier and CubicBarrier
    but with additional gamma and epsilon parameters.
    """

    @staticmethod
    @ti.func
    def energy(d, epsilon, gamma, kappa):
        """Compute GCP barrier energy."""
        return gcp_barrier_energy(d, epsilon, gamma, kappa)

    @staticmethod
    @ti.func
    def gradient(d, epsilon, gamma, kappa):
        """Compute GCP barrier gradient."""
        return gcp_barrier_gradient(d, epsilon, gamma, kappa)

    @staticmethod
    @ti.func
    def hessian(d, epsilon, gamma, kappa):
        """Compute GCP barrier hessian."""
        return gcp_barrier_hessian(d, epsilon, gamma, kappa)

    @staticmethod
    @ti.func
    def curvature(d, epsilon, gamma, kappa):
        """Compute GCP barrier curvature (clamped hessian)."""
        return gcp_barrier_curvature(d, epsilon, gamma, kappa)
