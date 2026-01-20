"""
Preconditioner registry for runtime selection.

Allows registering and creating preconditioners by name.
"""

from typing import Dict, Type, Any, Optional
from ..core.protocols import Preconditioner


class PreconditionerRegistry:
    """
    Registry for preconditioner implementations.

    Allows runtime selection and configuration of preconditioners.

    Usage:
        # Register a preconditioner
        @PreconditionerRegistry.register('my_precond')
        class MyPreconditioner:
            ...

        # Create instance
        precond = PreconditionerRegistry.create('my_precond', mesh=mesh)
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a preconditioner class.

        Args:
            name: Name to register the preconditioner under

        Returns:
            Decorator function
        """
        def decorator(precond_class: Type):
            cls._registry[name] = precond_class
            return precond_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create a preconditioner instance by name.

        Args:
            name: Registered preconditioner name
            **kwargs: Arguments to pass to preconditioner constructor

        Returns:
            Preconditioner instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown preconditioner: {name}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List all registered preconditioner names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a preconditioner name is registered."""
        return name in cls._registry

    @classmethod
    def get_class(cls, name: str) -> Optional[Type]:
        """Get the class for a registered preconditioner."""
        return cls._registry.get(name)
