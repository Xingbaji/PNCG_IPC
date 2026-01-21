"""
SolverBuilder: Fluent API for constructing solvers.

Provides a builder pattern for configuring and creating solvers
with various module combinations.
"""

from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from ..core.precision import PrecisionType
from ..mesh.mesh_system import MeshSystem
from ..optimizer.pncg_optimizer import PNCGOptimizer
from ..preconditioner.registry import PreconditionerRegistry
from ..collision.registry import CollisionDetectorRegistry
from ..collision import SurfaceData
from ..contact import IPCContactHandler, GroundContactHandler, CCDStepSizeComputer
from ..contact.gcp import GCPConfig, GCPContactHandler
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

        # Collision detection
        self._collision_enabled = False
        self._collision_detector_type = 'bvh_final'
        self._collision_kwargs: Dict[str, Any] = {}
        self._surface_data: Optional[SurfaceData] = None

        # Contact handling (IPC or GCP)
        self._ipc_config: Optional[Dict[str, Any]] = None
        self._gcp_config: Optional[GCPConfig] = None

        # Preconditioner
        self._preconditioner_name = 'diagonal'
        self._preconditioner_kwargs: Dict[str, Any] = {}

        # Ground plane
        self._ground_y: Optional[float] = None

        # ABD bodies
        self._abd_config: Optional[Dict[str, Any]] = None

        # Solver parameters
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

    def with_collision_detection(
        self,
        detector_type: str = 'bvh_final',
        surface_data: Optional[SurfaceData] = None,
        **kwargs
    ) -> 'SolverBuilder':
        """
        Enable BVH-based collision detection.

        Args:
            detector_type: Type of collision detector ('bvh', 'bvh_final')
            surface_data: Pre-computed surface data (will be computed if None)
            **kwargs: Additional arguments for the collision detector

        Returns:
            self for chaining
        """
        self._collision_enabled = True
        self._collision_detector_type = detector_type
        self._surface_data = surface_data
        self._collision_kwargs = kwargs
        return self

    def with_ipc_contact(
        self,
        kappa: float,
        dHat: float,
        barrier_type: str = 'log',
        max_contacts: int = 2**18,
    ) -> 'SolverBuilder':
        """
        Enable IPC contact handling.

        Args:
            kappa: Barrier stiffness
            dHat: Activation distance
            barrier_type: 'log' or 'cubic'
            max_contacts: Maximum contact pairs

        Returns:
            self for chaining
        """
        self._ipc_config = {
            'kappa': kappa,
            'dHat': dHat,
            'barrier_type': barrier_type,
            'max_contacts': max_contacts,
        }
        # IPC requires collision detection
        if not self._collision_enabled:
            self._collision_enabled = True
        return self

    def with_gcp_contact(
        self,
        config: Optional[GCPConfig] = None,
        epsilon_target: float = 0.1,
        adaptive_epsilon: bool = True,
        alpha: float = 0.1,
        kappa: float = 1e4,
    ) -> 'SolverBuilder':
        """
        Enable GCP (Geometric Contact Potential) handling.

        GCP provides automatic filtering of adjacent elements via
        the directional factor and uses smooth barrier functions.

        Args:
            config: GCPConfig object (if provided, other args ignored)
            epsilon_target: Target detection distance
            adaptive_epsilon: Whether to adapt epsilon
            alpha: Smoothing parameter
            kappa: Barrier stiffness

        Returns:
            self for chaining
        """
        if config is not None:
            self._gcp_config = config
        else:
            self._gcp_config = GCPConfig(
                epsilon_target=epsilon_target,
                adaptive_epsilon=adaptive_epsilon,
                alpha=alpha,
                kappa=kappa,
            )
        # GCP requires collision detection
        if not self._collision_enabled:
            self._collision_enabled = True
        return self

    def with_abd_bodies(
        self,
        max_bodies: int = 64,
        body_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> 'SolverBuilder':
        """
        Enable ABD (Affine Body Dynamics) bodies.

        Args:
            max_bodies: Maximum number of ABD bodies
            body_configs: List of body configurations
                Each config: {'vertices': array, 'density': float, 'E': float, 'nu': float}

        Returns:
            self for chaining
        """
        self._abd_config = {
            'max_bodies': max_bodies,
            'body_configs': body_configs or [],
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

        contact_type = 'IPC' if self._ipc_config else ('GCP' if self._gcp_config else 'none')
        print(f'\n[SolverBuilder] Building solver...')
        print(f'  Precision: {self._precision}')
        print(f'  Elastic model: {self._elastic_type}')
        print(f'  Preconditioner: {self._preconditioner_name}')
        print(f'  Collision: {"enabled (" + self._collision_detector_type + ")" if self._collision_enabled else "disabled"}')
        print(f'  Contact: {contact_type}')
        print(f'  Ground: {self._ground_y if self._ground_y is not None else "disabled"}')
        print(f'  ABD: {"enabled" if self._abd_config else "disabled"}')

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

        # Create collision detector if enabled
        collision_detector = None
        if self._collision_enabled:
            collision_detector = self._build_collision_detector(mesh_system)

        # Create contact handler if configured
        contact_handler = None
        if self._ipc_config is not None:
            contact_handler = self._build_ipc_handler()
        elif self._gcp_config is not None:
            contact_handler = self._build_gcp_handler()

        # Create ground handler if configured
        ground_handler = None
        if self._ground_y is not None and (self._ipc_config or self._gcp_config):
            kappa = self._ipc_config.get('kappa', 1e4) if self._ipc_config else self._gcp_config.kappa
            dHat = self._ipc_config.get('dHat', 0.01) if self._ipc_config else self._gcp_config.epsilon_target
            ground_handler = GroundContactHandler(
                ground_y=self._ground_y,
                kappa=kappa,
                dHat=dHat,
                precision=self._precision,
            )

        # Create CCD step size computer if contacts enabled
        ccd_step_size = None
        if collision_detector is not None:
            ccd_step_size = CCDStepSizeComputer(
                precision=self._precision,
                ground_y=self._ground_y,
            )

        # Create ABD system if configured
        abd_system = None
        if self._abd_config is not None:
            abd_system = self._build_abd_system()

        # Assemble solver
        solver = Solver(
            mesh_system=mesh_system,
            optimizer=optimizer,
            collision_detector=collision_detector,
            contact_handler=contact_handler,
            ground_handler=ground_handler,
            ccd_step_size=ccd_step_size,
            preconditioner=preconditioner,
            abd_system=abd_system,
            ground_y=self._ground_y,
            ipc_config=self._ipc_config,
            gcp_config=self._gcp_config,
        )

        print(f'[SolverBuilder] Build complete.\n')
        return solver

    def _build_collision_detector(self, mesh_system: MeshSystem):
        """Build collision detector from configuration."""
        # Get or compute surface data
        surface_data = self._surface_data
        if surface_data is None:
            # Extract surface data from mesh
            surface_data = self._extract_surface_data(mesh_system)

        # Create collision detector
        detector = CollisionDetectorRegistry.create(
            self._collision_detector_type,
            precision=self._precision,
            **self._collision_kwargs
        )

        # Initialize with surface data
        detector.init(self._mesh, surface_data)
        return detector

    def _extract_surface_data(self, mesh_system: MeshSystem) -> SurfaceData:
        """Extract surface mesh data for collision detection."""
        # This is a placeholder - actual implementation depends on mesh format
        # For now, create a minimal SurfaceData that can be populated later
        import warnings
        warnings.warn(
            "Surface data not provided. Collision detection requires "
            "boundary_points, boundary_edges, boundary_triangles. "
            "Call with_collision_detection(surface_data=...) with proper data."
        )
        return SurfaceData(
            boundary_points=None,
            boundary_edges=None,
            boundary_triangles=None,
            n_boundary_points=0,
            n_boundary_edges=0,
            n_boundary_triangles=0,
        )

    def _build_ipc_handler(self) -> IPCContactHandler:
        """Build IPC contact handler from configuration."""
        return IPCContactHandler(
            max_contacts=self._ipc_config.get('max_contacts', 2**18),
            barrier_type=self._ipc_config.get('barrier_type', 'log'),
            precision=self._precision,
        )

    def _build_gcp_handler(self) -> GCPContactHandler:
        """Build GCP contact handler from configuration."""
        return GCPContactHandler(
            config=self._gcp_config,
            precision=self._precision,
        )

    def _build_abd_system(self):
        """Build ABD system from configuration."""
        from ..body import ABDSystem

        abd = ABDSystem(
            max_bodies=self._abd_config.get('max_bodies', 64),
            precision=self._precision,
        )

        # Add configured bodies
        for body_config in self._abd_config.get('body_configs', []):
            abd.add_body(
                vertices=body_config['vertices'],
                density=body_config.get('density', 1000.0),
                E=body_config.get('E', 1e6),
                nu=body_config.get('nu', 0.4),
            )

        return abd
