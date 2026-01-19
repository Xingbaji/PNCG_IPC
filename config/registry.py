"""
Demo configuration registry for PNCG_IPC.

This module provides a singleton registry pattern for managing demo configurations,
allowing configs to be registered and retrieved by name.
"""

from typing import Dict, Callable, List, Union

from config.base import SimulationConfig


class DemoRegistry:
    """
    Singleton registry for demo configurations.

    Supports both static configs and factory functions for dynamic generation.

    Example:
        # Register a static config
        DemoRegistry.register("my_demo", config)

        # Register a factory function
        @register_demo("my_demo")
        def my_demo_config():
            return SimulationConfig(...)

        # Retrieve a config
        config = DemoRegistry.get("my_demo")

        # List all available configs
        print(DemoRegistry.list())
    """
    _configs: Dict[str, SimulationConfig] = {}
    _factories: Dict[str, Callable[[], SimulationConfig]] = {}

    @classmethod
    def register(cls, name: str, config_or_factory: Union[SimulationConfig, Callable[[], SimulationConfig]]) -> None:
        """
        Register a configuration or factory function.

        Args:
            name: Unique identifier for the demo
            config_or_factory: Either a SimulationConfig instance or a
                              callable that returns one

        Raises:
            ValueError: If config validation fails (for static configs)
        """
        if callable(config_or_factory) and not isinstance(config_or_factory, SimulationConfig):
            cls._factories[name] = config_or_factory
        else:
            # Validate static configs at registration time
            config_or_factory.validate()
            cls._configs[name] = config_or_factory

    @classmethod
    def get(cls, name: str) -> SimulationConfig:
        """
        Get configuration by name.

        Args:
            name: Demo identifier

        Returns:
            SimulationConfig for the requested demo

        Raises:
            KeyError: If demo name is not registered
            ValueError: If factory-generated config fails validation
        """
        if name in cls._configs:
            return cls._configs[name]

        if name in cls._factories:
            config = cls._factories[name]()
            config.validate()
            return config

        available = cls.list()
        raise KeyError(
            f"Demo '{name}' not found in registry. "
            f"Available demos: {available}"
        )

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a demo name is registered."""
        return name in cls._configs or name in cls._factories

    @classmethod
    def list(cls) -> List[str]:
        """Get sorted list of all registered demo names."""
        return sorted(set(cls._configs.keys()) | set(cls._factories.keys()))

    @classmethod
    def clear(cls) -> None:
        """Clear all registered configs (useful for testing)."""
        cls._configs.clear()
        cls._factories.clear()


def register_demo(name: str):
    """
    Decorator for registering demo configurations.

    Can be used with both functions and static config objects.

    Example:
        @register_demo("cube")
        def cube_config():
            return SimulationConfig(
                name="cube",
                material=MaterialConfig(E=1e4),
                ...
            )
    """
    def decorator(func_or_config: Union[SimulationConfig, Callable[[], SimulationConfig]]):
        DemoRegistry.register(name, func_or_config)
        return func_or_config
    return decorator
