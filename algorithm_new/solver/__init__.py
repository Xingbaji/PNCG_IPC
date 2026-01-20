"""Solver module with builder API and preset configurations."""

from .solver import Solver
from .solver_builder import SolverBuilder
from .presets import (
    create_collision_free_solver,
    create_ipc_solver,
    create_mas_ipc_solver,
)

__all__ = [
    'Solver',
    'SolverBuilder',
    'create_collision_free_solver',
    'create_ipc_solver',
    'create_mas_ipc_solver',
]
