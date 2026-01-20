"""
Barrier functions for IPC contact handling.

Provides log barrier and cubic barrier implementations with energy, gradient, and hessian.
"""

from .log_barrier import LogBarrier
from .cubic_barrier import CubicBarrier

__all__ = ['LogBarrier', 'CubicBarrier']
