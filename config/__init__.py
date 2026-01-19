"""
PNCG_IPC Configuration System.

This module provides a type-safe, extensible configuration system for
simulation demos. It replaces the monolithic if-elif chain in model_loading.py
with a clean registry pattern.

Usage:
    from config import SimulationConfig, DemoRegistry, register_demo

    # Get an existing config
    config = DemoRegistry.get("cube")
    print(f"Material mu: {config.material.mu}")

    # List available demos
    print(DemoRegistry.list())

    # Register a new config
    @register_demo("my_demo")
    def my_config():
        return SimulationConfig(
            name="my_demo",
            material=MaterialConfig(E=1e5, nu=0.3),
            scene=SceneConfig(
                meshes=[MeshInstance(path="../model/mesh/cube/cube.node")]
            )
        )

Example with all parameters:
    config = SimulationConfig(
        name="my_simulation",
        material=MaterialConfig(
            E=1e4,           # Young's modulus
            nu=0.4,          # Poisson's ratio
            density=1000.0,  # kg/m^3
            elastic_type="ARAP_SPD"
        ),
        solver=SolverConfig(
            epsilon=1e-5,    # Convergence tolerance
            iter_max=50,     # Max iterations
            dt=0.04          # Time step
        ),
        scene=SceneConfig(
            meshes=[
                MeshInstance(
                    path="../model/mesh/cube/cube.node",
                    translation=[0.0, 1.0, 0.0],
                    rotation=[0.0, 0.0, 0.0],
                    scale=[1.0, 1.0, 1.0]
                )
            ],
            ground_height=0.0,
            gravity=-9.8
        ),
        ipc=IPCConfig(
            enabled=True,
            dHat=0.01,
            kappa=1.0,
            ground_barrier=True
        )
    )
"""

from config.base import (
    SimulationConfig,
    MaterialConfig,
    SolverConfig,
    SceneConfig,
    MeshInstance,
    IPCConfig,
)
from config.registry import DemoRegistry, register_demo

# Auto-import demo configs to register them
from config.demos import examples  # noqa: F401

__all__ = [
    # Core config classes
    'SimulationConfig',
    'MaterialConfig',
    'SolverConfig',
    'SceneConfig',
    'MeshInstance',
    'IPCConfig',
    # Registry
    'DemoRegistry',
    'register_demo',
]
