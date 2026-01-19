"""
Core configuration dataclasses for PNCG_IPC simulations.

This module provides type-safe configuration classes with sensible defaults
and validation for physics simulation parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MaterialConfig:
    """Material properties for elastic simulation."""
    E: float = 1e4              # Young's modulus (Pa)
    nu: float = 0.4             # Poisson's ratio (dimensionless, 0 < nu < 0.5)
    density: float = 1000.0     # Mass density (kg/m^3)
    elastic_type: str = "ARAP_SPD"  # Constitutive model (SPD-projected Hessian)

    @property
    def mu(self) -> float:
        """Compute Lame's first parameter (shear modulus)."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def la(self) -> float:
        """Compute Lame's second parameter."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class SolverConfig:
    """Solver parameters for the PNCG optimizer."""
    epsilon: float = 1e-5       # Convergence tolerance
    iter_max: int = 50          # Maximum iterations per timestep
    dt: float = 0.04            # Time step size (seconds)


@dataclass
class IPCConfig:
    """Incremental Potential Contact (IPC) parameters."""
    enabled: bool = False       # Whether IPC is active
    dHat: float = 0.01          # Distance threshold for contact detection
    kappa: float = 1.0          # Barrier stiffness
    ground_barrier: bool = False  # Enable ground plane barrier
    barrier_type: str = "log"   # Barrier function type: "log" or "cubic"
    adaptive_kappa: bool = False  # Enable adaptive barrier stiffness


@dataclass
class MeshInstance:
    """A single mesh instance with transform parameters."""
    path: str                   # Path to mesh file (.node format)
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Euler angles (degrees)
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class SceneConfig:
    """Scene setup including meshes and environment."""
    meshes: List[MeshInstance] = field(default_factory=list)
    ground_height: float = 0.0  # Ground plane height
    gravity: float = -9.8       # Gravitational acceleration (m/s^2)
    camera_position: Optional[List[float]] = None  # Auto-computed if None
    camera_lookat: Optional[List[float]] = None    # Auto-computed if None


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    This is the main configuration class that combines all sub-configurations
    into a single, validated structure.

    Example:
        config = SimulationConfig(
            name="my_demo",
            material=MaterialConfig(E=1e5, nu=0.3),
            solver=SolverConfig(dt=0.01),
            scene=SceneConfig(
                meshes=[MeshInstance(path="../model/mesh/cube/cube.node")],
                gravity=-9.8
            )
        )
    """
    name: str
    material: MaterialConfig = field(default_factory=MaterialConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    dirichlet_path: Optional[str] = None  # Path to Dirichlet BC mask (.npy)
    use_mas: bool = False  # Enable MAS preconditioner

    def validate(self) -> None:
        """
        Validate configuration and raise helpful errors.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.scene.meshes:
            raise ValueError(f"Config '{self.name}': No meshes defined in scene")

        for i, mesh in enumerate(self.scene.meshes):
            if not mesh.path:
                raise ValueError(f"Config '{self.name}': Mesh {i} has no path")

        if self.material.nu <= 0 or self.material.nu >= 0.5:
            raise ValueError(
                f"Config '{self.name}': Poisson's ratio must be in (0, 0.5), "
                f"got {self.material.nu}"
            )

        if self.ipc.enabled and self.ipc.dHat <= 0:
            raise ValueError(
                f"Config '{self.name}': IPC dHat must be positive, "
                f"got {self.ipc.dHat}"
            )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy demo_dict format for backward compatibility.

        This allows new configs to work with the existing model_loading system
        without requiring changes to solver classes.
        """
        result: Dict[str, Any] = {
            'E': self.material.E,
            'nu': self.material.nu,
            'density': self.material.density,
            'gravity': self.scene.gravity,
            'dt': self.solver.dt,
            'epsilon': self.solver.epsilon,
            'iter_max': self.solver.iter_max,
            'height': self.scene.ground_height,
            'elastic_type': self.material.elastic_type,
            'model_paths': [m.path for m in self.scene.meshes],
            'rotations': [m.rotation for m in self.scene.meshes],
            'scales': [m.scale for m in self.scene.meshes],
            'translations': [m.translation for m in self.scene.meshes],
        }

        # Add IPC params only if enabled
        if self.ipc.enabled:
            result.update({
                'kappa': self.ipc.kappa,
                'dHat': self.ipc.dHat,
                'ground_barrier': int(self.ipc.ground_barrier),
                'barrier_type': self.ipc.barrier_type,
                'adaptive_kappa': self.ipc.adaptive_kappa,
            })

        # Add camera if specified
        if self.scene.camera_position is not None:
            result['camera_position'] = self.scene.camera_position
        if self.scene.camera_lookat is not None:
            result['camera_lookat'] = self.scene.camera_lookat

        # Add Dirichlet path if specified
        if self.dirichlet_path is not None:
            result['dirichlet_path'] = self.dirichlet_path

        # Add MAS flag
        if self.use_mas:
            result['use_mas'] = True

        return result
