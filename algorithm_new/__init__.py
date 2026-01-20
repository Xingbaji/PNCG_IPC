"""
Modular PNCG Solver Architecture

This package provides a composition-based solver framework for physics simulation:
- Flexible module combination (collision, IPC, GCP, MAS, ABD)
- f32/f64 precision selection
- Builder pattern for solver configuration

Example:
    from algorithm_new import SolverBuilder

    solver = (SolverBuilder()
        .with_precision('f64')
        .with_mesh('model.node', density=1000, E=1e5, nu=0.3)
        .with_collision_detection()
        .with_ipc_contact(kappa=1e4, dHat=0.01)
        .with_preconditioner('mas')
        .build())

    solver.step()

Modules:
    - core: Precision config and protocols
    - collision: BVH-based collision detection (LBVH)
    - contact: IPC/GCP barrier functions and contact handlers
    - body: ABD affine body dynamics
    - preconditioner: MAS preconditioner with contact support
    - solver: SolverBuilder and presets
"""

# Core
from .core.precision import PrecisionType, get_precision_config

# Collision detection
from .collision import (
    CollisionDetectorRegistry,
    ContactPairStorage,
    BVHCollisionDetector,
    SurfaceData,
)

# Contact handling (IPC, GCP)
from .contact import (
    LogBarrier,
    CubicBarrier,
    GCPConfig,
    GCPBarrier,
    GCPContactHandler,
    IPCContactHandler,
    GroundContactHandler,
    CCDStepSizeComputer,
    compute_adaptive_kappa,
)

# Body dynamics (ABD)
from .body import (
    ABDJacobian,
    ABDDyadicMass,
    ABDShapeEnergy,
    ABDSystem,
    BodyBoundaryType,
)

# Preconditioner
from .preconditioner import (
    PreconditionerRegistry,
    DiagonalPreconditioner,
    MASPreconditioner,
    MASPreconditionerContact,
    ContactAssembler,
    create_mas_preconditioner,
)

# Optimizer
from .optimizer import (
    PNCGOptimizer,
    SubdomainCCD,
)

# Solver
from .solver.solver_builder import SolverBuilder
from .solver.solver import Solver
from .solver.presets import (
    create_collision_free_solver,
    create_ipc_solver,
    create_mas_ipc_solver,
)

__all__ = [
    # Core
    'PrecisionType',
    'get_precision_config',
    # Collision detection
    'CollisionDetectorRegistry',
    'ContactPairStorage',
    'BVHCollisionDetector',
    'SurfaceData',
    # Contact handling
    'LogBarrier',
    'CubicBarrier',
    'GCPConfig',
    'GCPBarrier',
    'GCPContactHandler',
    'IPCContactHandler',
    'GroundContactHandler',
    'CCDStepSizeComputer',
    'compute_adaptive_kappa',
    # Body dynamics
    'ABDJacobian',
    'ABDDyadicMass',
    'ABDShapeEnergy',
    'ABDSystem',
    'BodyBoundaryType',
    # Preconditioner
    'PreconditionerRegistry',
    'DiagonalPreconditioner',
    'MASPreconditioner',
    'MASPreconditionerContact',
    'ContactAssembler',
    'create_mas_preconditioner',
    # Optimizer
    'PNCGOptimizer',
    'SubdomainCCD',
    # Solver
    'SolverBuilder',
    'Solver',
    'create_collision_free_solver',
    'create_ipc_solver',
    'create_mas_ipc_solver',
]
