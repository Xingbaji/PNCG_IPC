"""
Precision configuration for f32/f64 support.

This module provides a factory pattern for creating precision-specific
Taichi types and configurations.
"""

from typing import Literal
from dataclasses import dataclass
import numpy as np
import taichi as ti

# Type alias for precision selection
PrecisionType = Literal['f32', 'f64']


@dataclass
class PrecisionConfig:
    """
    Precision configuration containing Taichi and NumPy types.

    Attributes:
        float_type: Taichi float type (ti.f32 or ti.f64)
        numpy_type: NumPy dtype (np.float32 or np.float64)
        vec3: Taichi 3D vector type
        mat3: Taichi 3x3 matrix type
        name: Precision name string
    """
    float_type: type
    numpy_type: type
    vec3: type
    mat3: type
    name: str

    def scalar_field(self, shape=()) -> ti.field:
        """Create a scalar field with this precision."""
        return ti.field(dtype=self.float_type, shape=shape)

    def vector_field(self, n: int, shape=()) -> ti.field:
        """Create a vector field with this precision."""
        return ti.Vector.field(n, dtype=self.float_type, shape=shape)

    def matrix_field(self, n: int, m: int, shape=()) -> ti.field:
        """Create a matrix field with this precision."""
        return ti.Matrix.field(n, m, dtype=self.float_type, shape=shape)


# Cached precision configs
_PRECISION_CONFIGS = {}


def get_precision_config(precision: PrecisionType) -> PrecisionConfig:
    """
    Get precision configuration for the specified precision type.

    Args:
        precision: Either 'f32' or 'f64'

    Returns:
        PrecisionConfig with appropriate types

    Example:
        cfg = get_precision_config('f64')
        scalar = cfg.scalar_field(shape=(100,))
        vec = cfg.vector_field(3, shape=(100,))
    """
    if precision in _PRECISION_CONFIGS:
        return _PRECISION_CONFIGS[precision]

    if precision == 'f32':
        config = PrecisionConfig(
            float_type=ti.f32,
            numpy_type=np.float32,
            vec3=ti.types.vector(3, ti.f32),
            mat3=ti.types.matrix(3, 3, ti.f32),
            name='f32',
        )
    elif precision == 'f64':
        config = PrecisionConfig(
            float_type=ti.f64,
            numpy_type=np.float64,
            vec3=ti.types.vector(3, ti.f64),
            mat3=ti.types.matrix(3, 3, ti.f64),
            name='f64',
        )
    else:
        raise ValueError(f"Unknown precision type: {precision}. Use 'f32' or 'f64'.")

    _PRECISION_CONFIGS[precision] = config
    return config


def get_default_precision() -> PrecisionType:
    """
    Get the default precision based on Taichi's default_fp setting.

    Returns:
        'f64' if Taichi was initialized with default_fp=ti.f64, else 'f32'
    """
    # Check Taichi's current default float type
    try:
        # This is a heuristic - Taichi doesn't expose default_fp directly
        test_field = ti.field(float, shape=())
        dtype = test_field.dtype
        if dtype == ti.f64:
            return 'f64'
    except:
        pass
    return 'f32'


class PrecisionMixin:
    """
    Mixin class to add precision support to Taichi data-oriented classes.

    Usage:
        @ti.data_oriented
        class MyModule(PrecisionMixin):
            def __init__(self, precision='f32'):
                self.init_precision(precision)
                # Now use self.cfg for precision-specific types
                self.data = self.cfg.vector_field(3, shape=(100,))
    """

    def init_precision(self, precision: PrecisionType):
        """Initialize precision configuration."""
        self._precision = precision
        self._cfg = get_precision_config(precision)

    @property
    def precision(self) -> PrecisionType:
        """Get the precision type."""
        return self._precision

    @property
    def cfg(self) -> PrecisionConfig:
        """Get the precision configuration."""
        return self._cfg
