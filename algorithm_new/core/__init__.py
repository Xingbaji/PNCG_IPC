"""Core infrastructure for the modular solver architecture."""

from .protocols import (
    GradientContributor,
    HessianContributor,
    Preconditioner,
    CollisionDetector,
    ContactHandler,
    StepSizeComputer,
)
from .precision import PrecisionType, PrecisionConfig, get_precision_config

__all__ = [
    'GradientContributor',
    'HessianContributor',
    'Preconditioner',
    'CollisionDetector',
    'ContactHandler',
    'StepSizeComputer',
    'PrecisionType',
    'PrecisionConfig',
    'get_precision_config',
]
