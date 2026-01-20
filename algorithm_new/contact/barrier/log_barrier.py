"""
Log barrier function for IPC contact handling.

Implements the log barrier: E(d) = -kappa * (d - dHat)^2 * log(d / dHat)

The log barrier is the standard IPC barrier function, providing C2 continuity
and infinite stiffness as d -> 0.
"""

import taichi as ti


class LogBarrier:
    """
    Log barrier functions for IPC contact.

    All methods are static @ti.func for use in GPU kernels.

    Barrier formula: E(d) = -kappa * (d - dHat)^2 * log(d / dHat)

    Properties:
    - Active when d < dHat
    - E(dHat) = 0 (barrier starts at threshold)
    - E(d) -> infinity as d -> 0
    - C2 continuous
    """

    @staticmethod
    @ti.func
    def energy(d, dHat, kappa):
        """
        Compute log barrier energy.

        E(d) = -kappa * (d - dHat)^2 * log(d / dHat)

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Barrier energy (scalar)
        """
        E = d * 0.0  # Preserve type
        if d < dHat and d > 1e-10:
            t2 = d - dHat
            E = -kappa * t2 * t2 * ti.log(d / dHat)
        return E

    @staticmethod
    @ti.func
    def gradient(d, dHat, kappa):
        """
        Compute log barrier gradient (dE/dd).

        g(d) = kappa * [(d-dHat) * (-2*log(d/dHat)) - (d-dHat)^2 / d]

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Gradient (scalar)
        """
        g = d * 0.0  # Preserve type
        if d < dHat and d > 1e-10:
            t2 = d - dHat
            g = kappa * (t2 * ti.log(d / dHat) * (-2.0) - (t2 * t2) / d)
        return g

    @staticmethod
    @ti.func
    def hessian(d, dHat, kappa):
        """
        Compute log barrier hessian (d2E/dd2).

        H(d) = kappa * [(-2) * log(d/dHat) - 4 + 4*dHat/d + (d-dHat)^2 / d^2]

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Hessian (scalar)
        """
        H = d * 0.0  # Preserve type
        if d < dHat and d > 1e-10:
            H = kappa * ((-2.0) * ti.log(d / dHat) - 4.0 + 4.0 * dHat / d +
                        (d - dHat) ** 2 / (d * d))
        return H

    @staticmethod
    @ti.func
    def curvature(d, dHat, kappa):
        """
        Compute barrier curvature for SPD Hessian construction.

        Same as hessian but clamped to be non-negative for SPD guarantee.

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Curvature (non-negative scalar)
        """
        c = d * 0.0  # Preserve type
        if d < dHat and d > 1e-10:
            c = kappa * ((-2.0) * ti.log(d / dHat) - 4.0 + 4.0 * dHat / d +
                        (d - dHat) ** 2 / (d * d))
            c = ti.max(c, d * 0.0)  # Clamp to non-negative
        return c


# Standalone functions for backward compatibility
@ti.func
def barrier_E_log(d, dHat, kappa):
    """Standalone log barrier energy function."""
    return LogBarrier.energy(d, dHat, kappa)


@ti.func
def barrier_g_log(d, dHat, kappa):
    """Standalone log barrier gradient function."""
    return LogBarrier.gradient(d, dHat, kappa)


@ti.func
def barrier_H_log(d, dHat, kappa):
    """Standalone log barrier hessian function."""
    return LogBarrier.hessian(d, dHat, kappa)


@ti.func
def barrier_curvature_log(d, dHat, kappa):
    """Standalone log barrier curvature function."""
    return LogBarrier.curvature(d, dHat, kappa)
