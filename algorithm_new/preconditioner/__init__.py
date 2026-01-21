"""Preconditioner module with registry and implementations."""

from .registry import PreconditionerRegistry
from .diagonal import DiagonalPreconditioner
from .mas import MASPreconditioner, create_mas_preconditioner
from .mas_contact import MASPreconditionerContact, ContactAssembler
from .mas_8_contact import MASPreconditioner8Contact, WoodburySupport8

__all__ = [
    'PreconditionerRegistry',
    'DiagonalPreconditioner',
    # BANKSIZE=16 implementations
    'MASPreconditioner',
    'MASPreconditionerContact',
    'ContactAssembler',
    'create_mas_preconditioner',
    # BANKSIZE=8 implementations
    'MASPreconditioner8Contact',
    'WoodburySupport8',
]
