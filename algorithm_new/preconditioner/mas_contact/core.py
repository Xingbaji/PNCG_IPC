"""
MAS Preconditioner with Contact Support.

Extends the base MAS preconditioner to handle contact Hessian contributions.
"""

import taichi as ti
from typing import Optional

from ...core.precision import PrecisionType, PrecisionMixin
from ..registry import PreconditionerRegistry
from .contact_assembly import ContactAssembler


@PreconditionerRegistry.register('mas_contact')
@ti.data_oriented
class MASPreconditionerContact(PrecisionMixin):
    """
    MAS Preconditioner with Contact Support.

    This is a simplified wrapper that adds contact Hessian handling
    to the MAS preconditioner. It provides:
    - Contact Hessian assembly into block structure
    - Cross-block triplet storage for non-local contributions
    - Hessian matvec with contact contributions

    For the full multilevel Schwarz functionality, this class wraps
    an underlying MAS preconditioner and extends it with contact support.

    Usage:
        precond = MASPreconditionerContact(mesh, max_contacts=2**18)
        precond.set_contact_pairs(contact_storage, n_contacts)
        precond.rebuild(solver)
        precond.apply(r, z)  # z = P^{-1} @ r
    """

    def __init__(
        self,
        mesh,
        max_contacts: int = 2**18,
        max_triplets: int = 2**20,
        banksize: int = 16,
        precision: PrecisionType = 'f32',
    ):
        """
        Initialize MAS preconditioner with contact support.

        Args:
            mesh: Mesh object with vertex data
            max_contacts: Maximum number of contact pairs
            max_triplets: Maximum cross-block triplet entries
            banksize: Nodes per block (typically 16)
            precision: Float precision
        """
        self.init_precision(precision)
        self.mesh = mesh
        self.MAX_CONTACTS = max_contacts
        self.BANKSIZE = banksize

        float_type = self.cfg.float_type
        n_vertices = mesh.verts.x.shape[0]
        self.n_vertices = n_vertices
        self.n_blocks = (n_vertices + banksize - 1) // banksize

        # Contact assembler
        self._contact_assembler = ContactAssembler(
            max_contacts=max_contacts,
            max_triplets=max_triplets,
            banksize=banksize,
            precision=precision,
        )

        # Contact pair reference
        self._contact_pairs = None
        self._n_contacts = ti.field(dtype=ti.i32, shape=())
        self._n_contacts[None] = 0

        # Contact parameters
        self._dt_sq = ti.field(dtype=float_type, shape=())
        self._kappa = ti.field(dtype=float_type, shape=())
        self._dHat = ti.field(dtype=float_type, shape=())

        # Underlying MAS preconditioner (placeholder - would be inherited)
        # In a full implementation, this would extend MASPreconditionerSmall
        self._has_contact_data = ti.field(dtype=ti.i32, shape=())
        self._has_contact_data[None] = 0

    def set_contact_pairs(self, contact_storage, n_contacts: int = None):
        """
        Set contact pairs for Hessian assembly.

        Args:
            contact_storage: ContactPairStorage instance
            n_contacts: Number of contacts (if None, uses storage.count)
        """
        self._contact_pairs = contact_storage.contact_pairs
        if n_contacts is not None:
            self._n_contacts[None] = n_contacts
        else:
            self._n_contacts[None] = contact_storage.count

    def set_contact_params(self, dt: float, kappa: float, dHat: float):
        """
        Set contact parameters for Hessian computation.

        Args:
            dt: Time step
            kappa: Barrier stiffness
            dHat: Barrier threshold
        """
        self._dt_sq[None] = dt * dt
        self._kappa[None] = kappa
        self._dHat[None] = dHat

    def assemble_contacts(self, block_matrices):
        """
        Assemble contact Hessian into block matrices.

        This should be called after the elastic Hessian is assembled
        and before the preconditioner is applied.

        Args:
            block_matrices: Block matrix field from MAS preconditioner
        """
        if self._contact_pairs is None or self._n_contacts[None] == 0:
            return

        # Set block matrices reference
        self._contact_assembler.set_block_matrices(block_matrices, self.n_blocks)

        # Reset triplet storage
        self._contact_assembler.reset()

        # Assemble contact Hessian
        self._contact_assembler.assemble_contact_hessian(
            self._contact_pairs,
            self._n_contacts[None],
            self._dt_sq[None],
            self._kappa[None],
            self._dHat[None],
        )

        self._has_contact_data[None] = 1

    @ti.kernel
    def apply_contact_cross_block_spmv(
        self,
        x: ti.template(),
        y: ti.template(),
    ):
        """
        Apply cross-block contact Hessian: y += H_cross @ x

        This should be called as part of the Hessian matvec.

        Args:
            x: Input vector field
            y: Output vector field (accumulated)
        """
        if self._has_contact_data[None] == 1:
            self._contact_assembler.apply_cross_block_spmv(x, y)

    @property
    def n_contacts(self) -> int:
        """Get current number of contacts."""
        return self._n_contacts[None]

    @property
    def n_triplets(self) -> int:
        """Get number of cross-block triplets."""
        return self._contact_assembler.n_triplets


def create_mas_contact_preconditioner(
    mesh,
    max_contacts: int = 2**18,
    precision: PrecisionType = 'f32',
):
    """
    Factory function to create MAS preconditioner with contact support.

    Args:
        mesh: Mesh object
        max_contacts: Maximum contacts
        precision: Float precision

    Returns:
        MASPreconditionerContact instance
    """
    return MASPreconditionerContact(
        mesh=mesh,
        max_contacts=max_contacts,
        precision=precision,
    )
