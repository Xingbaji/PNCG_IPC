"""
MAS Preconditioner with Contact Support.

Extends the MAS preconditioner to handle contact Hessian contributions
using the Woodbury rank-update formula for efficient updates.
"""

from .contact_assembly import (
    compute_contact_subblock,
    compute_contact_subblock_spd,
    ContactAssembler,
)
from .core import MASPreconditionerContact

__all__ = [
    'compute_contact_subblock',
    'compute_contact_subblock_spd',
    'ContactAssembler',
    'MASPreconditionerContact',
]
