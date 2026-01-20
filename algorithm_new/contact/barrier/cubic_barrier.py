"""
Cubic barrier function for IPC contact handling.

Implements the cubic barrier: E(d) = -2*kappa/(3*dHat) * (d - dHat)^3

The cubic barrier provides a simpler alternative to the log barrier with:
- Faster computation
- Guaranteed non-negative Hessian
- Linear gradient at d=0
"""

import taichi as ti


class CubicBarrier:
    """
    Cubic barrier functions for IPC contact.

    All methods are static @ti.func for use in GPU kernels.

    Barrier formula: E(d) = -2*kappa/(3*dHat) * (d - dHat)^3

    Properties:
    - Active when d < dHat
    - E(dHat) = 0 (barrier starts at threshold)
    - C2 continuous (cubic polynomial)
    - Guaranteed non-negative Hessian (always SPD)
    """

    @staticmethod
    @ti.func
    def energy(d, dHat, kappa):
        """
        Compute cubic barrier energy.

        E(d) = -2*kappa/(3*dHat) * (d - dHat)^3

        Note: When d < dHat, (d - dHat) is negative, so (d - dHat)^3 is negative,
        and with the negative sign in front, E is positive.

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Barrier energy (scalar)
        """
        E = d * 0.0  # Preserve type
        if d < dHat:
            y = d - dHat  # y is negative when d < dHat
            E = -2.0 * kappa * (y * y * y) / (3.0 * dHat)
        return E

    @staticmethod
    @ti.func
    def gradient(d, dHat, kappa):
        """
        Compute cubic barrier gradient (dE/dd).

        g(d) = -2*kappa/dHat * (d - dHat)^2

        Note: (d - dHat)^2 is always non-negative, so gradient is always <= 0
        (repulsive force).

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Gradient (scalar)
        """
        g = d * 0.0  # Preserve type
        if d < dHat:
            y = d - dHat
            g = -2.0 * kappa * (y * y) / dHat
        return g

    @staticmethod
    @ti.func
    def hessian(d, dHat, kappa):
        """
        Compute cubic barrier hessian (d2E/dd2).

        H(d) = 4*kappa * (1 - d/dHat)

        Note: When d < dHat, (1 - d/dHat) > 0, so H is always non-negative.
        This guarantees SPD without additional projection.

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Hessian (scalar)
        """
        H = d * 0.0  # Preserve type
        if d < dHat:
            H = 4.0 * kappa * (1.0 - d / dHat)
        return H

    @staticmethod
    @ti.func
    def curvature(d, dHat, kappa):
        """
        Compute barrier curvature for SPD Hessian construction.

        For cubic barrier, curvature equals hessian (always non-negative).

        Args:
            d: Distance (scalar)
            dHat: Barrier threshold distance
            kappa: Barrier stiffness

        Returns:
            Curvature (non-negative scalar)
        """
        return CubicBarrier.hessian(d, dHat, kappa)


# Standalone functions for backward compatibility
@ti.func
def barrier_E_cubic(d, dHat, kappa):
    """Standalone cubic barrier energy function."""
    return CubicBarrier.energy(d, dHat, kappa)


@ti.func
def barrier_g_cubic(d, dHat, kappa):
    """Standalone cubic barrier gradient function."""
    return CubicBarrier.gradient(d, dHat, kappa)


@ti.func
def barrier_H_cubic(d, dHat, kappa):
    """Standalone cubic barrier hessian function."""
    return CubicBarrier.hessian(d, dHat, kappa)


@ti.func
def barrier_curvature_cubic(d, dHat, kappa):
    """Standalone cubic barrier curvature function."""
    return CubicBarrier.curvature(d, dHat, kappa)
