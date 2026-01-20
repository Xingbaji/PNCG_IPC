"""
YAML-based demo configuration loader.

This module provides a clean, declarative way to define demo configurations
using YAML files instead of hardcoded Python dictionaries.
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path


# ============================================================================
# Custom YAML Loader with proper scientific notation support
# ============================================================================

class ScientificLoader(yaml.SafeLoader):
    """YAML loader that properly handles scientific notation."""
    pass


# Add constructor for scientific notation
_scientific_pattern = re.compile(
    r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$'
)


def _scientific_constructor(loader, node):
    """Convert scientific notation strings to floats."""
    value = loader.construct_scalar(node)
    if _scientific_pattern.match(value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


# Register the constructor for strings that look like numbers
ScientificLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    _scientific_pattern,
    list('-+0123456789.')
)


def yaml_load(stream):
    """Load YAML with proper scientific notation support."""
    return yaml.load(stream, Loader=ScientificLoader)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class MaterialConfig:
    """Material properties for elastic simulation."""
    E: float = 1e4              # Young's modulus (Pa)
    nu: float = 0.4             # Poisson's ratio
    density: float = 1000.0     # Mass density (kg/m^3)
    elastic_type: str = "ARAP_SPD"  # Constitutive model

    @property
    def mu(self) -> float:
        """Lame's first parameter (shear modulus)."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def la(self) -> float:
        """Lame's second parameter."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class SolverConfig:
    """Solver parameters."""
    epsilon: float = 1e-5       # Convergence tolerance
    iter_max: int = 50          # Maximum iterations
    dt: float = 0.04            # Time step (seconds)


@dataclass
class IPCConfig:
    """IPC (contact) parameters."""
    enabled: bool = False
    dHat: float = 0.01          # Distance threshold
    kappa: float = 1.0          # Barrier stiffness
    ground_barrier: bool = False
    barrier_type: str = "log"   # "log" or "cubic"
    adaptive_kappa: bool = False
    cache_kappa: bool = True


@dataclass
class MeshConfig:
    """Single mesh instance configuration."""
    path: str                   # Path to mesh file (.node)
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class SceneConfig:
    """Scene configuration."""
    meshes: List[MeshConfig] = field(default_factory=list)
    ground_height: float = 0.0
    gravity: float = -9.8
    camera_position: Optional[List[float]] = None
    camera_lookat: Optional[List[float]] = None


@dataclass
class DemoConfig:
    """Complete demo configuration."""
    name: str
    material: MaterialConfig = field(default_factory=MaterialConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    dirichlet_path: Optional[str] = None
    use_mas: bool = False
    use_metis: bool = False

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy demo_dict format for backward compatibility."""
        result = {
            'E': self.material.E,
            'nu': self.material.nu,
            'density': self.material.density,
            'elastic_type': self.material.elastic_type,
            'gravity': self.scene.gravity,
            'dt': self.solver.dt,
            'epsilon': self.solver.epsilon,
            'iter_max': self.solver.iter_max,
            'height': self.scene.ground_height,
            'model_paths': [m.path for m in self.scene.meshes],
            'rotations': [m.rotation for m in self.scene.meshes],
            'scales': [m.scale for m in self.scene.meshes],
            'translations': [m.translation for m in self.scene.meshes],
        }

        # IPC parameters
        if self.ipc.enabled:
            result.update({
                'kappa': self.ipc.kappa,
                'dHat': self.ipc.dHat,
                'ground_barrier': int(self.ipc.ground_barrier),
                'barrier_type': self.ipc.barrier_type,
                'adaptive_kappa': self.ipc.adaptive_kappa,
                'cache_kappa': self.ipc.cache_kappa,
            })

        # Camera
        if self.scene.camera_position:
            result['camera_position'] = self.scene.camera_position
        if self.scene.camera_lookat:
            result['camera_lookat'] = self.scene.camera_lookat

        # Dirichlet BC
        if self.dirichlet_path:
            result['dirichlet_path'] = self.dirichlet_path

        # MAS options
        if self.use_mas:
            result['use_mas'] = True
        if self.use_metis:
            result['use_metis'] = True

        return result


# ============================================================================
# YAML Loader
# ============================================================================

def _get_demo_settings_dir() -> Path:
    """Get the demo_settings directory path."""
    return Path(__file__).parent


def _parse_mesh_config(mesh_data: Union[str, Dict]) -> MeshConfig:
    """Parse mesh configuration from YAML data."""
    if isinstance(mesh_data, str):
        return MeshConfig(path=mesh_data)
    return MeshConfig(
        path=mesh_data['path'],
        translation=mesh_data.get('translation', [0.0, 0.0, 0.0]),
        rotation=mesh_data.get('rotation', [0.0, 0.0, 0.0]),
        scale=mesh_data.get('scale', [1.0, 1.0, 1.0]),
    )


def _expand_mesh_grid(mesh_data: Dict) -> List[MeshConfig]:
    """
    Expand a mesh grid specification into individual mesh configs.

    Supports:
    - grid: {rows: 4, cols: 2, spacing: [1.5, 1.5, 0]}
    - count: 8  (with base translation offset)
    - repeat: 4 (simple repetition with explicit translations)
    """
    path = mesh_data['path']
    meshes = []

    if 'grid' in mesh_data:
        grid = mesh_data['grid']
        rows = grid.get('rows', 1)
        cols = grid.get('cols', 1)
        layers = grid.get('layers', 1)
        spacing = grid.get('spacing', [1.0, 1.0, 1.0])
        base_translation = mesh_data.get('translation', [0.0, 0.0, 0.0])
        base_rotation = mesh_data.get('rotation', [0.0, 0.0, 0.0])
        base_scale = mesh_data.get('scale', [1.0, 1.0, 1.0])

        for layer in range(layers):
            for row in range(rows):
                for col in range(cols):
                    trans = [
                        base_translation[0] + col * spacing[0],
                        base_translation[1] + row * spacing[1],
                        base_translation[2] + layer * spacing[2],
                    ]
                    meshes.append(MeshConfig(
                        path=path,
                        translation=trans,
                        rotation=base_rotation.copy(),
                        scale=base_scale.copy(),
                    ))

    elif 'count' in mesh_data:
        count = mesh_data['count']
        base_rotation = mesh_data.get('rotation', [0.0, 0.0, 0.0])
        base_scale = mesh_data.get('scale', [1.0, 1.0, 1.0])
        translations = mesh_data.get('translations', [[0.0, 0.0, 0.0]] * count)

        for i in range(count):
            trans = translations[i] if i < len(translations) else [0.0, 0.0, 0.0]
            meshes.append(MeshConfig(
                path=path,
                translation=trans,
                rotation=base_rotation.copy(),
                scale=base_scale.copy(),
            ))

    else:
        meshes.append(_parse_mesh_config(mesh_data))

    return meshes


def _parse_config_from_yaml(data: Dict, name: str) -> DemoConfig:
    """Parse DemoConfig from YAML data."""
    # Material
    mat_data = data.get('material', {})
    material = MaterialConfig(
        E=mat_data.get('E', 1e4),
        nu=mat_data.get('nu', 0.4),
        density=mat_data.get('density', 1000.0),
        elastic_type=mat_data.get('elastic_type', 'ARAP_SPD'),
    )

    # Solver
    solver_data = data.get('solver', {})
    solver = SolverConfig(
        epsilon=solver_data.get('epsilon', 1e-5),
        iter_max=solver_data.get('iter_max', 50),
        dt=solver_data.get('dt', 0.04),
    )

    # IPC
    ipc_data = data.get('ipc', {})
    ipc = IPCConfig(
        enabled=ipc_data.get('enabled', False),
        dHat=ipc_data.get('dHat', 0.01),
        kappa=ipc_data.get('kappa', 1.0),
        ground_barrier=ipc_data.get('ground_barrier', False),
        barrier_type=ipc_data.get('barrier_type', 'log'),
        adaptive_kappa=ipc_data.get('adaptive_kappa', False),
        cache_kappa=ipc_data.get('cache_kappa', True),
    )

    # Scene & Meshes
    scene_data = data.get('scene', {})
    meshes_data = scene_data.get('meshes', [])
    meshes = []
    for mesh_data in meshes_data:
        if isinstance(mesh_data, dict) and ('grid' in mesh_data or 'count' in mesh_data):
            meshes.extend(_expand_mesh_grid(mesh_data))
        else:
            meshes.append(_parse_mesh_config(mesh_data))

    scene = SceneConfig(
        meshes=meshes,
        ground_height=scene_data.get('ground_height', 0.0),
        gravity=scene_data.get('gravity', -9.8),
        camera_position=scene_data.get('camera_position'),
        camera_lookat=scene_data.get('camera_lookat'),
    )

    return DemoConfig(
        name=name,
        material=material,
        solver=solver,
        scene=scene,
        ipc=ipc,
        dirichlet_path=data.get('dirichlet_path'),
        use_mas=data.get('use_mas', False),
        use_metis=data.get('use_metis', False),
    )


def get_demo_path(demo_name: str) -> Optional[Path]:
    """
    Find the YAML file path for a demo name.

    Searches all subdirectories of demo_settings for {demo_name}.yaml
    """
    settings_dir = _get_demo_settings_dir()
    for yaml_file in settings_dir.rglob('*.yaml'):
        if yaml_file.stem == demo_name:
            return yaml_file
    return None


def load_demo_config(demo_name: str) -> DemoConfig:
    """
    Load a demo configuration by name.

    Args:
        demo_name: Name of the demo (without .yaml extension)

    Returns:
        DemoConfig object

    Raises:
        FileNotFoundError: If demo YAML file not found
    """
    yaml_path = get_demo_path(demo_name)
    if yaml_path is None:
        available = list_demos()
        raise FileNotFoundError(
            f"Demo '{demo_name}' not found. Available demos: {available}"
        )

    with open(yaml_path, 'r') as f:
        data = yaml_load(f)

    return _parse_config_from_yaml(data, demo_name)


def load_demo(demo_name: str) -> Dict[str, Any]:
    """
    Load a demo configuration and return legacy dict format.

    This is the main entry point for backward compatibility.

    Args:
        demo_name: Name of the demo

    Returns:
        Legacy demo_dict format
    """
    config = load_demo_config(demo_name)
    return config.to_legacy_dict()


def list_demos(by_category: bool = False) -> Union[List[str], Dict[str, List[str]]]:
    """
    List all available demos.

    Args:
        by_category: If True, return dict grouped by category folder

    Returns:
        List of demo names, or dict of {category: [demo_names]}
    """
    settings_dir = _get_demo_settings_dir()
    demos = {}

    for yaml_file in settings_dir.rglob('*.yaml'):
        category = yaml_file.parent.name
        if category == 'demo_settings':
            category = 'root'
        demo_name = yaml_file.stem

        if category not in demos:
            demos[category] = []
        demos[category].append(demo_name)

    # Sort within categories
    for category in demos:
        demos[category].sort()

    if by_category:
        return demos
    else:
        all_demos = []
        for demo_list in demos.values():
            all_demos.extend(demo_list)
        return sorted(all_demos)
