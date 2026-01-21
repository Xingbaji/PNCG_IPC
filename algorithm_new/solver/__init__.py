"""Solver module with builder API and preset configurations."""

from .solver import Solver
from .solver_builder import SolverBuilder
from .presets import (
    create_collision_free_solver,
    create_ipc_solver,
    create_mas_ipc_solver,
    create_mas_collision_free_solver,
    create_gcp_solver,
    create_mas8_ipc_solver,
    create_abd_ipc_solver,
)

__all__ = [
    'Solver',
    'SolverBuilder',
    # Basic presets
    'create_collision_free_solver',
    'create_ipc_solver',
    'create_mas_ipc_solver',
    'create_mas_collision_free_solver',
    # New presets
    'create_gcp_solver',
    'create_mas8_ipc_solver',
    'create_abd_ipc_solver',
]
