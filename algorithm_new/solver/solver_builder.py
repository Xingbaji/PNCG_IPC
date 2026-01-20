"""
SolverBuilder: Fluent API for constructing solvers.

Provides a builder pattern for configuring and creating solvers
with various module combinations.
"""

from typing import Optional, Any, Dict
from ..core.precision import PrecisionType
from ..mesh.mesh_system import MeshSystem
from ..optimizer.pncg_optimizer import PNCGOptimizer
from ..preconditioner.registry import PreconditionerRegistry
from .solver import Solver


class SolverBuilder:
    """
    Fluent builder for creating solver configurations.

    Example:
        solver = (SolverBuilder()
            .with_precision('f64')
            .with_mesh(mesh, density=1000, E=1e5, nu=0.3)
            .with_elastic_model('ARAP_filter')
            .with_preconditioner('mas')
            .with_solver_params(dt=0.01, epsilon=1e-6)
            .build())
    """

    def __init__(self):
        """Initialize builder with default configuration."""
        self._precision: PrecisionType = 'f32'
        self._mesh = None
        self._mesh_config: Dict[str, Any] = {}
        self._elastic_type = 'ARAP_filter'
        self._collision_enabled = False
        self._ipc_config: Optional[Dict[str, Any]] = None
        self._preconditioner_name = 'diagonal'
        self._preconditioner_kwargs: Dict[str, Any] = {}
        self._ground_y: Optional[float] = None
        self._solver_config: Dict[str, Any] = {
            'dt': 0.04,
            'epsilon': 1e-5,
            'iter_max': 50,
            'gravity': -9.8,
        }

    def with_precision(self, precision: PrecisionType) -> 'SolverBuilder':
        """
        Set computation precision.

        Args:
            precision: 'f32' or 'f64'

        Returns:
            self for chaining
        """
        self._precision = precision
        return self

    def with_mesh(
        self,
        mesh: Any,
        density: float,
        E: float,
        nu: float,
        **kwargs
    ) -> 'SolverBuilder':
        """
        Configure mesh and material properties.

        Args:
            mesh: MeshTaichi mesh object (already loaded)
            density: Material density (kg/m^3)
            E: Young's modulus
            nu: Poisson's ratio
            **kwargs: Additional mesh parameters

        Returns:
            self for chaining
        """
        # Convert E, nu to Lame parameters
        mu = E / (2.0 * (1.0 + nu))
        la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        self._mesh = mesh
        self._mesh_config = {
            'density': density,
            'mu': mu,
            'la': la,
            **kwargs
        }
        return self

    def with_mesh_lame(
        self,
        mesh: Any,
        density: float,
        mu: float,
        la: float,
        **kwargs
    ) -> 'SolverBuilder':
        """
        Configure mesh with Lame parameters directly.

        Args:
            mesh: MeshTaichi mesh object (already loaded)
            density: Material density (kg/m^3)
            mu: Lame's first parameter
            la: Lame's second parameter
            **kwargs: Additional mesh parameters

        Returns:
            self for chaining
        """
        self._mesh = mesh
        self._mesh_config = {
            'density': density,
            'mu': mu,
            'la': la,
            **kwargs
        }
        return self

    def with_elastic_model(self, elastic_type: str) -> 'SolverBuilder':
        """
        Set elastic constitutive model.

        Args:
            elastic_type: One of 'ARAP', 'ARAP_filter', 'SNH', 'FCR'

        Returns:
            self for chaining
        """
        self._elastic_type = elastic_type
        return self

    def with_collision_detection(self) -> 'SolverBuilder':
        """
        Enable BVH-based collision detection.

        Returns:
            self for chaining
        """
        self._collision_enabled = True
        return self

    def with_ipc_contact(
        self,
        kappa: float,
        dHat: float,
        barrier_type: str = 'log'
    ) -> 'SolverBuilder':
        """
        Enable IPC contact handling.

        Args:
            kappa: Barrier stiffness
            dHat: Activation distance
            barrier_type: 'log' or 'cubic'

        Returns:
            self for chaining
        """
        self._ipc_config = {
            'kappa': kappa,
            'dHat': dHat,
            'barrier_type': barrier_type,
        }
        return self

    def with_preconditioner(self, name: str, **kwargs) -> 'SolverBuilder':
        """
        Set preconditioner type.

        Args:
            name: Preconditioner name ('diagonal', 'mas')
            **kwargs: Preconditioner-specific arguments

        Returns:
            self for chaining
        """
        self._preconditioner_name = name
        self._preconditioner_kwargs = kwargs
        return self

    def with_ground_barrier(self, ground_y: float) -> 'SolverBuilder':
        """
        Enable ground plane barrier.

        Args:
            ground_y: Ground plane y-coordinate

        Returns:
            self for chaining
        """
        self._ground_y = ground_y
        return self

    def with_solver_params(
        self,
        dt: Optional[float] = None,
        epsilon: Optional[float] = None,
        iter_max: Optional[int] = None,
        gravity: Optional[float] = None,
    ) -> 'SolverBuilder':
        """
        Configure solver parameters.

        Args:
            dt: Time step
            epsilon: Convergence tolerance
            iter_max: Maximum iterations per step
            gravity: Gravity acceleration

        Returns:
            self for chaining
        """
        if dt is not None:
            self._solver_config['dt'] = dt
        if epsilon is not None:
            self._solver_config['epsilon'] = epsilon
        if iter_max is not None:
            self._solver_config['iter_max'] = iter_max
        if gravity is not None:
            self._solver_config['gravity'] = gravity
        return self

    def build(self) -> Solver:
        """
        Build the configured solver.

        Returns:
            Configured Solver instance

        Raises:
            ValueError: If required configuration is missing
        """
        # Validate configuration
        if self._mesh is None:
            raise ValueError("Mesh must be configured. Call with_mesh() first.")

        print(f'\n[SolverBuilder] Building solver...')
        print(f'  Precision: {self._precision}')
        print(f'  Elastic model: {self._elastic_type}')
        print(f'  Preconditioner: {self._preconditioner_name}')
        print(f'  Collision: {"enabled" if self._collision_enabled else "disabled"}')
        print(f'  IPC Contact: {"enabled" if self._ipc_config else "disabled"}')
        print(f'  Ground: {self._ground_y if self._ground_y is not None else "disabled"}')

        # Create mesh system
        mesh_system = MeshSystem(
            mesh=self._mesh,
            density=self._mesh_config['density'],
            mu=self._mesh_config['mu'],
            la=self._mesh_config['la'],
            elastic_type=self._elastic_type,
            precision=self._precision,
            dt=self._solver_config['dt'],
            gravity=self._solver_config['gravity'],
        )

        # Create optimizer
        optimizer = PNCGOptimizer(
            mesh=self._mesh,
            mesh_system=mesh_system,
            dt=self._solver_config['dt'],
            epsilon=self._solver_config['epsilon'],
            iter_max=self._solver_config['iter_max'],
            precision=self._precision,
        )

        # Create preconditioner
        preconditioner = PreconditionerRegistry.create(
            self._preconditioner_name,
            mesh=self._mesh,
            precision=self._precision,
            **self._preconditioner_kwargs
        )

        # TODO: Create collision detector if enabled
        collision_detector = None
        if self._collision_enabled:
            print('  [Warning] Collision detection not yet implemented in new architecture')

        # TODO: Create contact handler if configured
        contact_handler = None
        if self._ipc_config is not None:
            print('  [Warning] IPC contact not yet implemented in new architecture')

        # Assemble solver
        solver = Solver(
            mesh_system=mesh_system,
            optimizer=optimizer,
            collision_detector=collision_detector,
            contact_handler=contact_handler,
            preconditioner=preconditioner,
            ground_y=self._ground_y,
        )

        print(f'[SolverBuilder] Build complete.\n')
        return solver
