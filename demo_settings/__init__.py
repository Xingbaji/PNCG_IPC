"""
Demo Settings - Simplified YAML-based demo configuration system.

Usage:
    from demo_settings import load_demo, list_demos

    # Load a demo config
    config = load_demo('cube_40')

    # List available demos
    demos = list_demos()
    demos_by_category = list_demos(by_category=True)
"""

from demo_settings.loader import (
    load_demo,
    load_demo_config,
    list_demos,
    get_demo_path,
    DemoConfig,
    MaterialConfig,
    SolverConfig,
    SceneConfig,
    IPCConfig,
    MeshConfig,
)

__all__ = [
    'load_demo',
    'load_demo_config',
    'list_demos',
    'get_demo_path',
    'DemoConfig',
    'MaterialConfig',
    'SolverConfig',
    'SceneConfig',
    'IPCConfig',
    'MeshConfig',
]
