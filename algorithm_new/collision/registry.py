"""
Collision detector registry for runtime selection.

Allows registering and creating collision detectors by name.
"""

from typing import Dict, Type, Any, Optional
from ..core.protocols import CollisionDetector


class CollisionDetectorRegistry:
    """
    Registry for collision detector implementations.

    Allows runtime selection and configuration of collision detectors.

    Usage:
        # Register a collision detector
        @CollisionDetectorRegistry.register('bvh')
        class BVHCollisionDetector:
            ...

        # Create instance
        detector = CollisionDetectorRegistry.create('bvh', precision='f32')
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a collision detector class.

        Args:
            name: Name to register the detector under

        Returns:
            Decorator function
        """
        def decorator(detector_class: Type):
            cls._registry[name] = detector_class
            return detector_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create a collision detector instance by name.

        Args:
            name: Registered detector name
            **kwargs: Arguments to pass to detector constructor

        Returns:
            CollisionDetector instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown collision detector: {name}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List all registered collision detector names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a collision detector name is registered."""
        return name in cls._registry

    @classmethod
    def get_class(cls, name: str) -> Optional[Type]:
        """Get the class for a registered collision detector."""
        return cls._registry.get(name)
