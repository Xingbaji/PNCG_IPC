"""
MAS Preconditioner with BANKSIZE=8 and Contact Support.

Combines the bank-8 optimizations from mas_preconditioner_8 with
the contact functions from mas_preconditioner_contact.

Key features:
1. BANKSIZE=8 optimizations:
   - One-way Gauss-Jordan elimination for matrix inverse (~3x faster)
   - Two-pass symmetric matvec with reduced branching
   - Compact block storage (36 sym blocks vs 136 for BANKSIZE=16)

2. Contact support:
   - SPD Contact Hessian (PPF-Contact-Solver style)
   - Log and Cubic barrier functions
   - Friction support
   - Cross-block contact triplet storage

3. Woodbury low-rank updates for incremental contact changes
"""

from .core import MASPreconditioner8Contact
from .woodbury import WoodburySupport8

# Re-export barrier functions for convenience
from .contact_assembly import (
    barrier_E_log,
    barrier_g_log,
    barrier_H_log,
    barrier_E_cubic,
    barrier_g_cubic,
    barrier_H_cubic,
    barrier_curvature_log,
    barrier_curvature_cubic,
)

__all__ = [
    'MASPreconditioner8Contact',
    'WoodburySupport8',
    # Barrier functions
    'barrier_E_log',
    'barrier_g_log',
    'barrier_H_log',
    'barrier_E_cubic',
    'barrier_g_cubic',
    'barrier_H_cubic',
    'barrier_curvature_log',
    'barrier_curvature_cubic',
]
