"""
MAS Preconditioner Small - Simplified version for fast compilation.

This is a minimal implementation that only includes:
- ARAP elastic Hessian (with eigenvalue filter)
- IC(0) block inversion
- Banded local solve

Removed features:
- Multiple elastic types (only ARAP filter)
- Multiple inversion methods (only IC)
- IPC contact handling
- METIS reordering
- Woodbury update
- Multiple solve methods (only banded)
"""

from .core import MASPreconditionerSmall

__all__ = ['MASPreconditionerSmall']
