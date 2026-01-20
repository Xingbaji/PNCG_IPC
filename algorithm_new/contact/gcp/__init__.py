"""
GCP (Geometric Contact Potential) module.

Provides GCP-based contact handling with automatic adjacent element filtering
via directional factors and C2 smooth mollification.
"""

from .config import GCPConfig
from .smooth_step import (
    smooth_step_cubic,
    smooth_step_cubic_derivative,
    smooth_step_cubic_second_derivative,
)
from .directional_factor import (
    compute_gamma_PT,
    compute_gamma_EE,
    compute_triangle_normal,
)
from .gcp_barrier import GCPBarrier, gcp_barrier_energy, gcp_barrier_gradient, gcp_barrier_hessian
from .gcp_handler import GCPContactHandler

__all__ = [
    # Configuration
    'GCPConfig',
    # Smooth step functions
    'smooth_step_cubic',
    'smooth_step_cubic_derivative',
    'smooth_step_cubic_second_derivative',
    # Directional factors
    'compute_gamma_PT',
    'compute_gamma_EE',
    'compute_triangle_normal',
    # Barrier functions
    'GCPBarrier',
    'gcp_barrier_energy',
    'gcp_barrier_gradient',
    'gcp_barrier_hessian',
    # Handler
    'GCPContactHandler',
]
