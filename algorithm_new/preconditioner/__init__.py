"""Preconditioner module with registry and implementations."""

from .registry import PreconditionerRegistry
from .diagonal import DiagonalPreconditioner
from .mas import MASPreconditioner, create_mas_preconditioner
from .mas_contact import MASPreconditionerContact, ContactAssembler

__all__ = [
    'PreconditionerRegistry',
    'DiagonalPreconditioner',
    'MASPreconditioner',
    'MASPreconditionerContact',
    'ContactAssembler',
    'create_mas_preconditioner',
]
