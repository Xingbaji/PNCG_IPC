"""
Preset solver configurations for common use cases.

Provides convenience functions for creating pre-configured solvers.
"""

from typing import Any, Optional, List, Dict
from ..core.precision import PrecisionType
from ..contact.gcp import GCPConfig
from ..collision import SurfaceData
from .solver_builder import SolverBuilder
from .solver import Solver


def create_collision_free_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e4,
    nu: float = 0.3,
    precision: PrecisionType = 'f32',
    dt: float = 0.04,
    gravity: float = -9.8,
    epsilon: float = 1e-5,
    iter_max: int = 50,
    preconditioner: str = 'diagonal',
    **kwargs
) -> Solver:
    """
    Create a solver for collision-free elastic simulation.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64'
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        preconditioner: Preconditioner type ('diagonal' or 'mas')
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner(preconditioner, **kwargs)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_ipc_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e4,
    nu: float = 0.3,
    precision: PrecisionType = 'f32',
    dt: float = 0.04,
    gravity: float = -9.8,
    epsilon: float = 1e-5,
    iter_max: int = 50,
    kappa: float = 1e4,
    dHat: float = 0.01,
    ground_y: float = 0.0,
    preconditioner: str = 'diagonal',
    **kwargs
) -> Solver:
    """
    Create a solver with IPC contact handling.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64'
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        kappa: Barrier stiffness
        dHat: Activation distance
        ground_y: Ground plane y-coordinate
        preconditioner: Preconditioner type
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_collision_detection()
        .with_ipc_contact(kappa=kappa, dHat=dHat)
        .with_preconditioner(preconditioner, **kwargs)
        .with_ground_barrier(ground_y=ground_y)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_mas_ipc_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e6,
    nu: float = 0.4,
    precision: PrecisionType = 'f64',
    dt: float = 0.01,
    gravity: float = -9.8,
    epsilon: float = 1e-6,
    iter_max: int = 100,
    kappa: float = 1e4,
    dHat: float = 0.01,
    ground_y: float = 0.0,
    metis_reordered: bool = True,
    **kwargs
) -> Solver:
    """
    Create a solver with MAS preconditioner and IPC contact.

    This configuration is optimized for stiff materials with contact.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64' (f64 recommended for stiff)
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        kappa: Barrier stiffness
        dHat: Activation distance
        ground_y: Ground plane y-coordinate
        metis_reordered: Whether mesh uses METIS reordering
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_collision_detection()
        .with_ipc_contact(kappa=kappa, dHat=dHat)
        .with_preconditioner('mas', metis_reordered=metis_reordered, **kwargs)
        .with_ground_barrier(ground_y=ground_y)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_mas_collision_free_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e5,
    nu: float = 0.3,
    precision: PrecisionType = 'f32',
    dt: float = 0.04,
    gravity: float = -9.8,
    epsilon: float = 1e-5,
    iter_max: int = 50,
    metis_reordered: bool = True,
    **kwargs
) -> Solver:
    """
    Create a collision-free solver with MAS preconditioner.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64'
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        metis_reordered: Whether mesh uses METIS reordering
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_preconditioner('mas', metis_reordered=metis_reordered, **kwargs)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_gcp_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e5,
    nu: float = 0.3,
    precision: PrecisionType = 'f32',
    dt: float = 0.04,
    gravity: float = -9.8,
    epsilon: float = 1e-5,
    iter_max: int = 50,
    gcp_config: Optional[GCPConfig] = None,
    epsilon_target: float = 0.1,
    kappa: float = 1e4,
    ground_y: float = 0.0,
    surface_data: Optional[SurfaceData] = None,
    preconditioner: str = 'diagonal',
    **kwargs
) -> Solver:
    """
    Create a solver with GCP (Geometric Contact Potential) handling.

    GCP provides automatic filtering of adjacent elements via the
    directional factor and uses smooth barrier functions.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64'
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        gcp_config: GCPConfig object (if None, uses epsilon_target/kappa)
        epsilon_target: Target detection distance
        kappa: Barrier stiffness
        ground_y: Ground plane y-coordinate
        surface_data: Pre-computed surface data for collision detection
        preconditioner: Preconditioner type
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    builder = (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_collision_detection(surface_data=surface_data))

    if gcp_config is not None:
        builder = builder.with_gcp_contact(config=gcp_config)
    else:
        builder = builder.with_gcp_contact(epsilon_target=epsilon_target, kappa=kappa)

    return (builder
        .with_preconditioner(preconditioner, **kwargs)
        .with_ground_barrier(ground_y=ground_y)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_mas8_ipc_solver(
    mesh: Any,
    density: float = 1000.0,
    E: float = 1e6,
    nu: float = 0.4,
    precision: PrecisionType = 'f64',
    dt: float = 0.01,
    gravity: float = -9.8,
    epsilon: float = 1e-6,
    iter_max: int = 100,
    kappa: float = 1e4,
    dHat: float = 0.01,
    ground_y: float = 0.0,
    surface_data: Optional[SurfaceData] = None,
    metis_reordered: bool = True,
    **kwargs
) -> Solver:
    """
    Create a solver with MAS-8 preconditioner and IPC contact.

    Uses BANKSIZE=8 MAS preconditioner with optimizations:
    - One-way Gauss-Jordan elimination (~3x faster)
    - Compact block storage (36 sym blocks vs 136)

    Recommended for stiff materials with contact.

    Args:
        mesh: MeshTaichi mesh object
        density: Material density (kg/m^3)
        E: Young's modulus
        nu: Poisson's ratio
        precision: 'f32' or 'f64' (f64 recommended for stiff)
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        kappa: Barrier stiffness
        dHat: Activation distance
        ground_y: Ground plane y-coordinate
        surface_data: Pre-computed surface data for collision detection
        metis_reordered: Whether mesh uses METIS reordering
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_collision_detection(surface_data=surface_data)
        .with_ipc_contact(kappa=kappa, dHat=dHat)
        .with_preconditioner('mas8_contact', metis_reordered=metis_reordered, **kwargs)
        .with_ground_barrier(ground_y=ground_y)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())


def create_abd_ipc_solver(
    mesh: Any,
    body_configs: List[Dict[str, Any]],
    density: float = 1000.0,
    E: float = 1e6,
    nu: float = 0.4,
    precision: PrecisionType = 'f64',
    dt: float = 0.01,
    gravity: float = -9.8,
    epsilon: float = 1e-6,
    iter_max: int = 100,
    kappa: float = 1e4,
    dHat: float = 0.01,
    ground_y: float = 0.0,
    surface_data: Optional[SurfaceData] = None,
    preconditioner: str = 'mas',
    metis_reordered: bool = True,
    **kwargs
) -> Solver:
    """
    Create a solver with ABD bodies and IPC contact.

    Supports hybrid ABD-FEM simulation with contact handling.

    Args:
        mesh: MeshTaichi mesh object (for FEM part)
        body_configs: List of ABD body configurations
            Each config: {'vertices': array, 'density': float, 'E': float, 'nu': float}
        density: Material density for FEM (kg/m^3)
        E: Young's modulus for FEM
        nu: Poisson's ratio for FEM
        precision: 'f32' or 'f64'
        dt: Time step
        gravity: Gravity acceleration
        epsilon: Convergence tolerance
        iter_max: Maximum iterations
        kappa: Barrier stiffness
        dHat: Activation distance
        ground_y: Ground plane y-coordinate
        surface_data: Pre-computed surface data for collision detection
        preconditioner: Preconditioner type
        metis_reordered: Whether mesh uses METIS reordering
        **kwargs: Additional parameters

    Returns:
        Configured Solver instance
    """
    return (SolverBuilder()
        .with_precision(precision)
        .with_mesh(mesh, density=density, E=E, nu=nu)
        .with_elastic_model('ARAP_filter')
        .with_collision_detection(surface_data=surface_data)
        .with_ipc_contact(kappa=kappa, dHat=dHat)
        .with_abd_bodies(body_configs=body_configs)
        .with_preconditioner(preconditioner, metis_reordered=metis_reordered, **kwargs)
        .with_ground_barrier(ground_y=ground_y)
        .with_solver_params(dt=dt, epsilon=epsilon, iter_max=iter_max, gravity=gravity)
        .build())
