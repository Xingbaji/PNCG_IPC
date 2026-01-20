"""
Protocol definitions for modular solver components.

These protocols define the interfaces that modules must implement
to participate in the PNCG optimization framework.
"""

from typing import Protocol, runtime_checkable, Any
import taichi as ti


@runtime_checkable
class GradientContributor(Protocol):
    """
    Protocol for modules that contribute to gradient computation.

    Implementers add their gradient contribution to mesh.verts.grad
    using ti.atomic_add for thread-safe accumulation.
    """

    def add_gradient(self, mesh: Any, dt: float, mu: float, la: float) -> None:
        """
        Add gradient contribution to mesh.verts.grad.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            mu: Lame's first parameter
            la: Lame's second parameter
        """
        ...

    def add_diagonal_hessian(self, mesh: Any, dt: float, mu: float, la: float) -> None:
        """
        Add diagonal Hessian contribution to mesh.verts.diagH.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            mu: Lame's first parameter
            la: Lame's second parameter
        """
        ...


@runtime_checkable
class HessianContributor(Protocol):
    """
    Protocol for modules that contribute to Hessian-vector products.

    Used for computing p^T H p in line search and 2D subspace minimization.
    """

    def add_pHp(self, mesh: Any, dt: float, mu: float, la: float) -> float:
        """
        Compute and return p^T H p contribution.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            mu: Lame's first parameter
            la: Lame's second parameter

        Returns:
            The scalar value p^T H p for this module's contribution
        """
        ...


@runtime_checkable
class Preconditioner(Protocol):
    """
    Protocol for preconditioner implementations.

    Preconditioners transform the gradient to improve convergence:
    z = M^{-1} g, where M approximates H.
    """

    def rebuild(self, solver: Any) -> None:
        """
        Rebuild preconditioner with current solver state.

        Called when Hessian structure changes significantly
        (e.g., contact topology changes, CG restart).

        Args:
            solver: The solver instance for accessing mesh and state
        """
        ...

    def apply(self) -> None:
        """
        Apply preconditioner: z = M^{-1} g.

        Reads from mesh.verts.grad, writes to mesh.verts.z.
        """
        ...

    def hessian_matvec(self, v: Any, result: Any) -> None:
        """
        Compute Hessian-vector product: result = H * v.

        Args:
            v: Input vector field
            result: Output vector field
        """
        ...


@runtime_checkable
class CollisionDetector(Protocol):
    """
    Protocol for collision detection modules.

    Detects potential contact pairs using broad-phase (BVH)
    and narrow-phase (distance computation) algorithms.
    """

    def init(self, mesh: Any, surface_data: Any) -> None:
        """
        Initialize collision detection structures.

        Args:
            mesh: MeshTaichi mesh object
            surface_data: Boundary triangles and edges
        """
        ...

    def find_contacts(self, mesh: Any, dHat: float) -> int:
        """
        Find contact pairs within distance threshold.

        Args:
            mesh: MeshTaichi mesh object
            dHat: Distance threshold for contact detection

        Returns:
            Number of contact pairs found
        """
        ...

    @property
    def contact_pairs(self) -> Any:
        """Access to contact pair data array."""
        ...

    @property
    def n_contacts(self) -> int:
        """Number of active contacts."""
        ...


@runtime_checkable
class ContactHandler(Protocol):
    """
    Protocol for contact force computation.

    Computes IPC barrier gradients and Hessians for detected contacts.
    """

    def set_contacts(self, contact_pairs: Any, n_contacts: int) -> None:
        """
        Set the current contact pairs to process.

        Args:
            contact_pairs: Contact pair data from CollisionDetector
            n_contacts: Number of active contacts
        """
        ...

    def add_gradient(self, mesh: Any, dt: float, kappa: float, dHat: float) -> None:
        """
        Add contact gradient contribution to mesh.verts.grad.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            kappa: Barrier stiffness
            dHat: Activation distance
        """
        ...

    def add_diagonal_hessian(self, mesh: Any, dt: float, kappa: float, dHat: float) -> None:
        """
        Add contact diagonal Hessian contribution.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            kappa: Barrier stiffness
            dHat: Activation distance
        """
        ...

    def add_pHp(self, mesh: Any, dt: float, kappa: float, dHat: float) -> float:
        """
        Add contact p^T H p contribution.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            kappa: Barrier stiffness
            dHat: Activation distance

        Returns:
            The scalar value p^T H p for contact contribution
        """
        ...


@runtime_checkable
class StepSizeComputer(Protocol):
    """
    Protocol for step size computation strategies.

    Computes safe step sizes using CCD or other methods.
    """

    def compute_safe_step(
        self,
        mesh: Any,
        contact_pairs: Any,
        n_contacts: int,
        dHat: float,
        ground_y: float
    ) -> float:
        """
        Compute maximum safe step size.

        Args:
            mesh: MeshTaichi mesh object
            contact_pairs: Contact pair data
            n_contacts: Number of contacts
            dHat: Distance threshold
            ground_y: Ground plane y-coordinate

        Returns:
            Maximum safe step size alpha in (0, 1]
        """
        ...


@runtime_checkable
class EnergyComputer(Protocol):
    """
    Protocol for modules that contribute to total energy computation.

    Used for convergence checking and line search.
    """

    def compute_energy(self, mesh: Any, dt: float, mu: float, la: float) -> float:
        """
        Compute energy contribution.

        Args:
            mesh: MeshTaichi mesh object
            dt: Time step
            mu: Lame's first parameter
            la: Lame's second parameter

        Returns:
            Energy value for this module's contribution
        """
        ...
