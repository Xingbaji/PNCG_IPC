"""
Solver: High-level simulation controller.

Composes mesh system, optimizer, and optional modules (collision, contact)
into a complete simulation system.
"""

import taichi as ti
import time
from typing import Optional, Any, Dict
from ..core.precision import PrecisionType


@ti.data_oriented
class Solver:
    """
    High-level simulation controller.

    Composes:
    - MeshSystem: geometry and elastic energy
    - PNCGOptimizer: optimization loop
    - Optional modules: collision detection, contact handling

    Usage:
        solver = Solver(mesh_system, optimizer, ...)
        for frame in range(100):
            solver.step()
    """

    def __init__(
        self,
        mesh_system: Any,
        optimizer: Any,
        collision_detector: Optional[Any] = None,
        contact_handler: Optional[Any] = None,
        ground_handler: Optional[Any] = None,
        ccd_step_size: Optional[Any] = None,
        preconditioner: Optional[Any] = None,
        abd_system: Optional[Any] = None,
        ground_y: Optional[float] = None,
        ipc_config: Optional[Dict[str, Any]] = None,
        gcp_config: Optional[Any] = None,
    ):
        """
        Initialize the solver.

        Args:
            mesh_system: MeshSystem instance
            optimizer: PNCGOptimizer instance
            collision_detector: Optional CollisionDetector instance
            contact_handler: Optional ContactHandler instance (IPC or GCP)
            ground_handler: Optional GroundContactHandler instance
            ccd_step_size: Optional CCDStepSizeComputer instance
            preconditioner: Optional Preconditioner instance
            abd_system: Optional ABDSystem instance
            ground_y: Optional ground plane y-coordinate
            ipc_config: IPC configuration dict (kappa, dHat, barrier_type)
            gcp_config: GCP configuration object
        """
        self.mesh_system = mesh_system
        self.optimizer = optimizer
        self.collision_detector = collision_detector
        self.contact_handler = contact_handler
        self.ground_handler = ground_handler
        self.ccd_step_size = ccd_step_size
        self.preconditioner = preconditioner
        self.abd_system = abd_system
        self.ground_y = ground_y
        self.ipc_config = ipc_config or {}
        self.gcp_config = gcp_config

        self.mesh = mesh_system.mesh
        self.frame = 0

        # Contact parameters from config
        self.kappa = self.ipc_config.get('kappa', 1e4) if ipc_config else (gcp_config.kappa if gcp_config else 1e4)
        self.dHat = self.ipc_config.get('dHat', 0.01) if ipc_config else (gcp_config.epsilon_target if gcp_config else 0.01)

        # Set preconditioner on optimizer
        if preconditioner is not None:
            optimizer.set_preconditioner(preconditioner)

        contact_type = 'IPC' if ipc_config else ('GCP' if gcp_config else 'none')
        print(f'[Solver] Initialized:')
        print(f'  - MeshSystem: {mesh_system.n_verts} verts, {mesh_system.n_cells} cells')
        print(f'  - Collision: {"enabled" if collision_detector else "disabled"}')
        print(f'  - Contact: {contact_type}')
        print(f'  - Ground: {ground_y if ground_y is not None else "disabled"}')
        print(f'  - ABD: {"enabled (" + str(abd_system.n_bodies) + " bodies)" if abd_system else "disabled"}')
        print(f'  - Preconditioner: {type(preconditioner).__name__ if preconditioner else "diagonal"}')

    @property
    def dt(self) -> float:
        """Time step."""
        return self.mesh_system.dt

    @property
    def n_verts(self) -> int:
        """Number of vertices."""
        return self.mesh_system.n_verts

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.mesh_system.n_cells

    def step(self, verbose: bool = False) -> int:
        """
        Advance simulation by one time step.

        Args:
            verbose: If True, print iteration details

        Returns:
            Number of iterations taken
        """
        t_start = time.perf_counter()

        # 1. Find collisions if enabled
        n_contacts = 0
        if self.collision_detector is not None:
            n_contacts = self.collision_detector.find_contacts(self.mesh, dHat=self.dHat)

            # Set contacts on handler
            if self.contact_handler is not None and n_contacts > 0:
                self.contact_handler.set_contacts(
                    self.collision_detector._storage,
                    n_contacts
                )

        # 2. Run optimizer step with contact awareness
        # The optimizer should use contact_handler for gradient/Hessian computation
        n_iters = self.optimizer.step(
            verbose=verbose,
            contact_handler=self.contact_handler,
            ground_handler=self.ground_handler,
            ccd_step_size=self.ccd_step_size,
            n_contacts=n_contacts,
            kappa=self.kappa,
            dHat=self.dHat,
        )

        t_total = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f'Frame {self.frame}: {n_iters} iters, {n_contacts} contacts, {t_total:.2f}ms')

        self.frame += 1
        return n_iters

    def step_simple(self, verbose: bool = False) -> int:
        """
        Simple step without collision/contact (collision-free mode).

        Args:
            verbose: If True, print iteration details

        Returns:
            Number of iterations taken
        """
        t_start = time.perf_counter()

        # Run optimizer step (collision-free)
        n_iters = self.optimizer.step(verbose=verbose)

        t_total = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f'Frame {self.frame}: {n_iters} iters, {t_total:.2f}ms')

        self.frame += 1
        return n_iters

    def run(self, n_frames: int, verbose: bool = False):
        """
        Run simulation for multiple frames.

        Args:
            n_frames: Number of frames to simulate
            verbose: If True, print details for each frame
        """
        total_iters = 0
        t_start = time.perf_counter()

        for f in range(n_frames):
            n_iters = self.step(verbose=verbose)
            total_iters += n_iters

        t_total = time.perf_counter() - t_start

        print(f'\n[Solver] Completed {n_frames} frames:')
        print(f'  Total time: {t_total:.3f}s')
        print(f'  Avg time per frame: {t_total/n_frames*1000:.2f}ms')
        print(f'  Total iterations: {total_iters}')
        print(f'  Avg iters per frame: {total_iters/n_frames:.1f}')

    def get_positions(self) -> Any:
        """Get current vertex positions as numpy array."""
        import numpy as np
        positions = np.zeros((self.n_verts, 3), dtype=np.float32)
        self.mesh.verts.x.to_numpy(positions)
        return positions

    def get_velocities(self) -> Any:
        """Get current vertex velocities as numpy array."""
        import numpy as np
        velocities = np.zeros((self.n_verts, 3), dtype=np.float32)
        self.mesh.verts.v.to_numpy(velocities)
        return velocities

    def set_positions(self, positions: Any):
        """Set vertex positions from numpy array."""
        self.mesh.verts.x.from_numpy(positions)

    def set_velocities(self, velocities: Any):
        """Set vertex velocities from numpy array."""
        self.mesh.verts.v.from_numpy(velocities)
