"""
Solver: High-level simulation controller.

Composes mesh system, optimizer, and optional modules (collision, contact)
into a complete simulation system.
"""

import taichi as ti
import time
from typing import Optional, Any
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
        preconditioner: Optional[Any] = None,
        ground_y: Optional[float] = None,
    ):
        """
        Initialize the solver.

        Args:
            mesh_system: MeshSystem instance
            optimizer: PNCGOptimizer instance
            collision_detector: Optional CollisionDetector instance
            contact_handler: Optional ContactHandler instance
            preconditioner: Optional Preconditioner instance
            ground_y: Optional ground plane y-coordinate
        """
        self.mesh_system = mesh_system
        self.optimizer = optimizer
        self.collision_detector = collision_detector
        self.contact_handler = contact_handler
        self.preconditioner = preconditioner
        self.ground_y = ground_y

        self.mesh = mesh_system.mesh
        self.frame = 0

        # Set preconditioner on optimizer
        if preconditioner is not None:
            optimizer.set_preconditioner(preconditioner)

        print(f'[Solver] Initialized:')
        print(f'  - MeshSystem: {mesh_system.n_verts} verts, {mesh_system.n_cells} cells')
        print(f'  - Collision: {"enabled" if collision_detector else "disabled"}')
        print(f'  - Contact: {"enabled" if contact_handler else "disabled"}')
        print(f'  - Preconditioner: {type(preconditioner).__name__ if preconditioner else "diagonal"}')
        print(f'  - Ground: {ground_y if ground_y is not None else "disabled"}')

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

        # Run optimizer step
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
