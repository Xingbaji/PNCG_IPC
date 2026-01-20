"""
Body dynamics module for ABD (Affine Body Dynamics).

Provides ABD system for managing affine bodies with 12D state representation.
"""

from .jacobian import ABDJacobian
from .mass import ABDDyadicMass
from .shape_energy import ABDShapeEnergy
from .abd_system import ABDSystem, BodyBoundaryType

__all__ = [
    # Jacobian utilities
    'ABDJacobian',
    # Mass utilities
    'ABDDyadicMass',
    # Shape energy
    'ABDShapeEnergy',
    # ABD System
    'ABDSystem',
    'BodyBoundaryType',
]
