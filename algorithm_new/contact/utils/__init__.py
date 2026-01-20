"""
Utility functions for IPC contact handling.

Provides Jacobian computation and SPD projection utilities.
"""

from .jacobian import compute_dtdx_t, compute_d_dtdx
from .spd_projection import compute_contact_subblock, compute_spd_contact_hessian_3x3, extend_spd_contact_hessian_12x12

__all__ = [
    'compute_dtdx_t',
    'compute_d_dtdx',
    'compute_contact_subblock',
    'compute_spd_contact_hessian_3x3',
    'extend_spd_contact_hessian_12x12',
]
