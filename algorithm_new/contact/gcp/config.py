"""
GCP (Geometric Contact Potential) configuration.

Provides configuration dataclass for GCP contact handling.
"""

from dataclasses import dataclass


@dataclass
class GCPConfig:
    """
    Configuration for Geometric Contact Potential.

    GCP provides an alternative to IPC with:
    - Automatic filtering of adjacent elements via directional factor
    - C2 smooth mollification of barrier function
    - Adaptive per-primitive epsilon support

    Attributes:
        epsilon_target: Maximum detection distance (similar to dHat in IPC)
        adaptive_epsilon: If True, compute per-primitive epsilon based on geometry
        alpha: Smooth step transition parameter (controls mollification width)
        kappa: Barrier stiffness
        min_epsilon: Minimum epsilon to avoid numerical issues
    """

    epsilon_target: float = 0.1
    adaptive_epsilon: bool = True
    alpha: float = 0.1
    kappa: float = 1e4
    min_epsilon: float = 1e-4

    def __post_init__(self):
        """Validate configuration values."""
        assert self.epsilon_target > 0, "epsilon_target must be positive"
        assert 0 < self.alpha < 1, "alpha must be in (0, 1)"
        assert self.kappa > 0, "kappa must be positive"
        assert self.min_epsilon > 0, "min_epsilon must be positive"
        assert self.min_epsilon < self.epsilon_target, "min_epsilon must be less than epsilon_target"

    @classmethod
    def default(cls) -> 'GCPConfig':
        """Create default configuration."""
        return cls()

    @classmethod
    def stiff(cls) -> 'GCPConfig':
        """Create configuration for stiff materials."""
        return cls(
            epsilon_target=0.05,
            kappa=1e6,
            alpha=0.05,
        )

    @classmethod
    def soft(cls) -> 'GCPConfig':
        """Create configuration for soft materials."""
        return cls(
            epsilon_target=0.2,
            kappa=1e3,
            alpha=0.2,
        )
