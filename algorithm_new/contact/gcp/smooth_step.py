"""
C2 smooth step functions for GCP mollification.

Provides smooth transition functions that ensure C2 continuity of the
barrier function, crucial for Newton-type optimization.
"""

import taichi as ti


@ti.func
def smooth_step_cubic(z, a, b):
    """
    C2 smooth step function that transitions from 1 to 0.

    h(z) = 1           if z <= a
         = (1-t)²(1+2t) if a < z < b, where t = (z-a)/(b-a)
         = 0           if z >= b

    This is a cubic Hermite interpolation polynomial that:
    - Has value 1 at z=a and value 0 at z=b
    - Has zero first derivative at both endpoints
    - Is C2 continuous (smooth second derivative)

    Args:
        z: Input value
        a: Start of transition (h(a) = 1)
        b: End of transition (h(b) = 0)

    Returns:
        Smooth step value in [0, 1]
    """
    result = z * 0.0  # Preserve type
    if z <= a:
        result = z * 0.0 + 1.0
    elif z >= b:
        result = z * 0.0
    else:
        t = (z - a) / (b - a)
        result = (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t)
    return result


@ti.func
def smooth_step_cubic_derivative(z, a, b):
    """
    First derivative of C2 smooth step function.

    d/dz[(1-t)²(1+2t)] = d/dt[(1-t)²(1+2t)] * dt/dz
                       = [(1-t)²(2) + (1+2t)(-2)(1-t)] * (1/(b-a))
                       = -6t(1-t) * (1/(b-a))

    Args:
        z: Input value
        a: Start of transition
        b: End of transition

    Returns:
        First derivative of smooth step
    """
    result = z * 0.0  # Preserve type
    if z > a and z < b:
        t = (z - a) / (b - a)
        dt_dz = 1.0 / (b - a)
        result = -6.0 * t * (1.0 - t) * dt_dz
    return result


@ti.func
def smooth_step_cubic_second_derivative(z, a, b):
    """
    Second derivative of C2 smooth step function.

    d²/dz²[h(z)] = d/dz[-6t(1-t) * dt_dz]
                 = -6(1-2t) * (dt_dz)²

    Args:
        z: Input value
        a: Start of transition
        b: End of transition

    Returns:
        Second derivative of smooth step
    """
    result = z * 0.0  # Preserve type
    if z > a and z < b:
        t = (z - a) / (b - a)
        dt_dz = 1.0 / (b - a)
        result = -6.0 * (1.0 - 2.0 * t) * dt_dz * dt_dz
    return result


@ti.func
def smooth_step_linear(z, a, b):
    """
    Linear interpolation (C0 smooth step).

    For comparison and debugging, not recommended for optimization.

    Args:
        z: Input value
        a: Start of transition (h(a) = 1)
        b: End of transition (h(b) = 0)

    Returns:
        Linear interpolation value
    """
    result = z * 0.0
    if z <= a:
        result = z * 0.0 + 1.0
    elif z >= b:
        result = z * 0.0
    else:
        result = 1.0 - (z - a) / (b - a)
    return result


@ti.func
def smooth_step_quintic(z, a, b):
    """
    C2 quintic smooth step (alternative to cubic).

    h(z) = 1 - 6t^5 + 15t^4 - 10t^3  where t = (z-a)/(b-a)

    This has zero first AND second derivatives at endpoints,
    making it C2 continuous.

    Args:
        z: Input value
        a: Start of transition
        b: End of transition

    Returns:
        Quintic smooth step value
    """
    result = z * 0.0
    if z <= a:
        result = z * 0.0 + 1.0
    elif z >= b:
        result = z * 0.0
    else:
        t = (z - a) / (b - a)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        result = 1.0 - 6.0 * t5 + 15.0 * t4 - 10.0 * t3
    return result
