"""
Block Diagonal Preconditioner for FEM and ABD systems.

This module implements a simple but effective block diagonal preconditioner
that computes the diagonal blocks of the Hessian matrix:

For FEM:
    - Each vertex has a 3x3 diagonal block: H_ii = m*I + dt^2 * Σ_e ∂²Ψ/∂x_i²
    - The preconditioner inverts each 3x3 block independently

For ABD:
    - Each body has a 12x12 diagonal block: H_body = M + dt^2 * κv * H_shape
    - The preconditioner inverts each 12x12 block independently

This is simpler than MAS but still provides good convergence for many problems.

Reference: pncg_base_collision_free.py (scalar diagonal version)
"""

import taichi as ti
import numpy as np

from math_utils.matrix_util import compute_dFdx
from math_utils.elastic_util import compute_d2PsidF2_ARAP_filter

# ABD constants
ABD_DOF = 12


@ti.data_oriented
class BlockDiagPreconditioner:
    """
    Block diagonal preconditioner for FEM vertices.

    For each vertex i, computes and stores the 3x3 diagonal block H_ii
    and its inverse. Applies z = H_ii^{-1} @ g for preconditioning.
    """

    def __init__(self, mesh):
        """
        Initialize block diagonal preconditioner.

        Args:
            mesh: MeshTaichi mesh object
        """
        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        # Store 3x3 diagonal blocks and their inverses
        self.diag_blocks = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_verts)
        self.diag_blocks_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_verts)

        # State flags
        self.assembled = False

        print(f"[BlockDiag] Initialized: {self.n_verts} verts, {self.n_cells} cells")

    @ti.kernel
    def _clear_diag_blocks(self):
        """Zero out all diagonal blocks."""
        for i in range(self.n_verts):
            self.diag_blocks[i] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _add_inertia_contribution(self):
        """Add mass matrix to diagonal blocks: H_ii += m * I."""
        for vert in self.mesh.verts:
            idx = vert.id
            m = vert.m
            for d in ti.static(range(3)):
                self.diag_blocks[idx][d, d] += m

    @ti.kernel
    def _add_elastic_contribution_arap(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """
        Add ARAP elastic Hessian diagonal contribution.

        For each element e containing vertex i:
            H_ii += dt^2 * W_e * (dFdx_i)^T @ d2PsidF2 @ dFdx_i
        """
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt

            # Compute deformation gradient
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            # Compute element Hessian
            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Extract diagonal 3x3 blocks for each vertex
            for i in ti.static(range(4)):
                v_id = c.verts[i].id
                # Extract 3x3 diagonal block H_e[3i:3i+3, 3i:3i+3]
                diag_block = ti.Matrix.zero(ti.f32, 3, 3)
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        diag_block[di, dj] = H_e[i * 3 + di, i * 3 + dj]

                # Atomic add to global diagonal block
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        ti.atomic_add(self.diag_blocks[v_id][di, dj], diag_block[di, dj])

    @ti.func
    def _invert_3x3(self, A: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(3, 3, ti.f32):
        """Invert a 3x3 matrix using cofactor expansion."""
        det = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
               A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
               A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

        inv = ti.Matrix.zero(ti.f32, 3, 3)

        if ti.abs(det) > 1e-12:
            inv_det = 1.0 / det

            inv[0, 0] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) * inv_det
            inv[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * inv_det
            inv[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * inv_det

            inv[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) * inv_det
            inv[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * inv_det
            inv[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) * inv_det

            inv[2, 0] = (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) * inv_det
            inv[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) * inv_det
            inv[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) * inv_det
        else:
            # Singular matrix: use pseudo-inverse via diagonal
            for d in ti.static(range(3)):
                if ti.abs(A[d, d]) > 1e-12:
                    inv[d, d] = 1.0 / A[d, d]
                else:
                    inv[d, d] = 1.0

        return inv

    @ti.kernel
    def _invert_diag_blocks(self):
        """Invert all 3x3 diagonal blocks."""
        for i in range(self.n_verts):
            H = self.diag_blocks[i]

            # Regularize: ensure positive definiteness
            for d in ti.static(range(3)):
                if H[d, d] < 1e-6:
                    H[d, d] = 1e-6

            self.diag_blocks_inv[i] = self._invert_3x3(H)

    def assemble(self, solver):
        """
        Assemble diagonal blocks from solver parameters.

        Args:
            solver: Solver object with mu, la, dt attributes
        """
        self._clear_diag_blocks()
        self._add_inertia_contribution()
        self._add_elastic_contribution_arap(solver.mu, solver.la, solver.dt)
        self._invert_diag_blocks()
        self.assembled = True

    @ti.kernel
    def apply(self):
        """
        Apply preconditioner: z = H_ii^{-1} @ grad.

        Reads from mesh.verts.grad, writes to mesh.verts.z.
        """
        for vert in self.mesh.verts:
            idx = vert.id
            g = vert.grad
            H_inv = self.diag_blocks_inv[idx]

            # z = H_inv @ g
            z = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    z[i] += H_inv[i, j] * g[j]

            vert.z = z

    def rebuild(self, solver):
        """Rebuild preconditioner (alias for assemble)."""
        self.assemble(solver)


@ti.data_oriented
class BlockDiagPreconditionerABD:
    """
    Block diagonal preconditioner for hybrid FEM-ABD system.

    For FEM: 3x3 diagonal blocks per vertex
    For ABD: 12x12 diagonal blocks per body
    """

    def __init__(self, mesh, abd_system=None):
        """
        Initialize block diagonal preconditioner with ABD support.

        Args:
            mesh: MeshTaichi mesh object
            abd_system: ABDSystem instance (optional)
        """
        self.mesh = mesh
        self.abd_system = abd_system
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        # ABD info
        self.n_abd_bodies = abd_system.n_bodies if abd_system else 0
        self.max_abd_bodies = abd_system.max_bodies if abd_system else 64

        # FEM: 3x3 diagonal blocks
        self.fem_diag_blocks = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_verts)
        self.fem_diag_blocks_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_verts)

        # ABD: 12x12 diagonal blocks
        self.abd_diag_blocks = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                shape=self.max_abd_bodies)
        self.abd_diag_blocks_inv = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                    shape=self.max_abd_bodies)

        # State flags
        self.assembled = False

        print(f"[BlockDiag-ABD] Initialized: {self.n_verts} FEM verts, "
              f"{self.n_abd_bodies} ABD bodies")

    def update_abd_system(self, abd_system):
        """Update ABD system reference."""
        self.abd_system = abd_system
        self.n_abd_bodies = abd_system.n_bodies if abd_system else 0

    # ========================================================================
    # FEM Assembly
    # ========================================================================

    @ti.kernel
    def _clear_fem_diag_blocks(self):
        """Zero out FEM diagonal blocks."""
        for i in range(self.n_verts):
            self.fem_diag_blocks[i] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _add_fem_inertia(self):
        """Add mass matrix to FEM diagonal blocks."""
        for vert in self.mesh.verts:
            idx = vert.id
            m = vert.m
            for d in ti.static(range(3)):
                self.fem_diag_blocks[idx][d, d] += m

    @ti.kernel
    def _add_fem_elastic_arap(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """Add ARAP elastic Hessian diagonal contribution."""
        for c in self.mesh.cells:
            W = c.W
            para = W * dt * dt

            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B

            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            for i in ti.static(range(4)):
                v_id = c.verts[i].id
                for di in ti.static(range(3)):
                    for dj in ti.static(range(3)):
                        ti.atomic_add(self.fem_diag_blocks[v_id][di, dj],
                                      H_e[i * 3 + di, i * 3 + dj])

    @ti.func
    def _invert_3x3(self, A: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(3, 3, ti.f32):
        """Invert a 3x3 matrix."""
        det = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
               A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
               A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

        inv = ti.Matrix.zero(ti.f32, 3, 3)

        if ti.abs(det) > 1e-12:
            inv_det = 1.0 / det
            inv[0, 0] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) * inv_det
            inv[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * inv_det
            inv[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * inv_det
            inv[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) * inv_det
            inv[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * inv_det
            inv[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) * inv_det
            inv[2, 0] = (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) * inv_det
            inv[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) * inv_det
            inv[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) * inv_det
        else:
            for d in ti.static(range(3)):
                if ti.abs(A[d, d]) > 1e-12:
                    inv[d, d] = 1.0 / A[d, d]
                else:
                    inv[d, d] = 1.0

        return inv

    @ti.kernel
    def _invert_fem_diag_blocks(self):
        """Invert all FEM 3x3 diagonal blocks."""
        for i in range(self.n_verts):
            H = self.fem_diag_blocks[i]
            for d in ti.static(range(3)):
                if H[d, d] < 1e-6:
                    H[d, d] = 1e-6
            self.fem_diag_blocks_inv[i] = self._invert_3x3(H)

    # ========================================================================
    # ABD Assembly
    # ========================================================================

    @ti.kernel
    def _clear_abd_diag_blocks(self):
        """Zero out ABD diagonal blocks."""
        for i in range(self.max_abd_bodies):
            self.abd_diag_blocks[i] = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

    @ti.kernel
    def _assemble_abd_blocks(self, dt: ti.f32):
        """
        Assemble ABD 12x12 diagonal blocks.

        H_body = M_body + dt^2 * kappa * volume * H_shape
        """
        from algorithm.abd_system import ABDShapeEnergy, BodyBoundaryType

        for body_id in range(self.n_abd_bodies):
            if self.abd_system.boundary_type[body_id] == BodyBoundaryType.FIXED:
                # Fixed body: identity block
                for i in ti.static(range(ABD_DOF)):
                    self.abd_diag_blocks[body_id][i, i] = 1.0
            else:
                # Add mass matrix
                M = self.abd_system.abd_mass[body_id]
                for i in ti.static(range(ABD_DOF)):
                    for j in ti.static(range(ABD_DOF)):
                        self.abd_diag_blocks[body_id][i, j] = M[i, j]

                # Add shape energy Hessian
                q = self.abd_system.q[body_id]
                kappa = self.abd_system.body_kappa[body_id]
                volume = self.abd_system.body_volume[body_id]
                scale = dt * dt * kappa * volume

                H_shape = ABDShapeEnergy.compute_hessian(q)
                H_shape = ABDShapeEnergy.make_positive_definite(H_shape)

                # Add to affine DOFs (3-11)
                for i in ti.static(range(9)):
                    for j in ti.static(range(9)):
                        self.abd_diag_blocks[body_id][3 + i, 3 + j] += scale * H_shape[i, j]

    @ti.kernel
    def _invert_abd_diag_blocks(self):
        """Invert ABD 12x12 blocks using Cholesky decomposition."""
        for body_id in range(self.n_abd_bodies):
            H = self.abd_diag_blocks[body_id]

            # Cholesky decomposition
            L = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

            for i in ti.static(range(ABD_DOF)):
                for j in range(i + 1):
                    sum_val = H[i, j]
                    for k in range(j):
                        sum_val -= L[i, k] * L[j, k]

                    if i == j:
                        if sum_val > 1e-12:
                            L[i, j] = ti.sqrt(sum_val)
                        else:
                            L[i, j] = 1e-3
                    else:
                        if ti.abs(L[j, j]) > 1e-12:
                            L[i, j] = sum_val / L[j, j]
                        else:
                            L[i, j] = 0.0

            # Inversion via forward/backward substitution
            H_inv = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

            for col in ti.static(range(ABD_DOF)):
                # Forward substitution
                y = ti.Vector.zero(ti.f32, ABD_DOF)
                for i in ti.static(range(ABD_DOF)):
                    sum_val = 1.0 if i == col else 0.0
                    for k in range(i):
                        sum_val -= L[i, k] * y[k]
                    if ti.abs(L[i, i]) > 1e-12:
                        y[i] = sum_val / L[i, i]

                # Backward substitution
                x = ti.Vector.zero(ti.f32, ABD_DOF)
                for i_rev in ti.static(range(ABD_DOF)):
                    i = ABD_DOF - 1 - i_rev
                    sum_val = y[i]
                    for k in range(i + 1, ABD_DOF):
                        sum_val -= L[k, i] * x[k]
                    if ti.abs(L[i, i]) > 1e-12:
                        x[i] = sum_val / L[i, i]

                for i in ti.static(range(ABD_DOF)):
                    H_inv[i, col] = x[i]

            self.abd_diag_blocks_inv[body_id] = H_inv

    # ========================================================================
    # High-level API
    # ========================================================================

    def assemble(self, solver):
        """
        Assemble all diagonal blocks.

        Args:
            solver: Solver object with mu, la, dt attributes
        """
        # FEM
        self._clear_fem_diag_blocks()
        self._add_fem_inertia()
        self._add_fem_elastic_arap(solver.mu, solver.la, solver.dt)
        self._invert_fem_diag_blocks()

        # ABD
        if self.abd_system and self.n_abd_bodies > 0:
            self.n_abd_bodies = self.abd_system.n_bodies
            self._clear_abd_diag_blocks()
            self._assemble_abd_blocks(solver.dt)
            self._invert_abd_diag_blocks()

        self.assembled = True

    @ti.kernel
    def _apply_fem(self):
        """Apply FEM preconditioner: z = H_ii^{-1} @ grad."""
        for vert in self.mesh.verts:
            idx = vert.id
            g = vert.grad
            H_inv = self.fem_diag_blocks_inv[idx]

            z = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    z[i] += H_inv[i, j] * g[j]

            vert.z = z

    @ti.kernel
    def _apply_abd(self):
        """Apply ABD preconditioner: dq = H_body^{-1} @ grad_q."""
        for body_id in range(self.n_abd_bodies):
            g = self.abd_system.grad_q[body_id]
            H_inv = self.abd_diag_blocks_inv[body_id]

            z = ti.Vector.zero(ti.f32, ABD_DOF)
            for i in ti.static(range(ABD_DOF)):
                for j in ti.static(range(ABD_DOF)):
                    z[i] += H_inv[i, j] * g[j]

            self.abd_system.dq[body_id] = z

    def apply(self):
        """Apply preconditioner to both FEM and ABD systems."""
        self._apply_fem()
        if self.abd_system and self.n_abd_bodies > 0:
            self._apply_abd()

    def rebuild(self, solver):
        """Rebuild preconditioner."""
        self.assemble(solver)

    def get_stats(self):
        """Get preconditioner statistics."""
        return {
            'n_fem_verts': self.n_verts,
            'n_abd_bodies': self.n_abd_bodies,
            'type': 'block_diagonal',
        }


@ti.data_oriented
class ScalarDiagPreconditioner:
    """
    Scalar diagonal preconditioner (simplest version).

    For each vertex i, stores only the diagonal entries diagH[i] ∈ ℝ³.
    This is equivalent to pncg_base_collision_free.py's diagH.

    Preconditioner: z_i = grad_i / diagH_i (component-wise division)
    """

    def __init__(self, mesh):
        """
        Initialize scalar diagonal preconditioner.

        Args:
            mesh: MeshTaichi mesh object
        """
        self.mesh = mesh
        self.n_verts = len(mesh.verts)
        self.n_cells = len(mesh.cells)

        # diagH is already stored in mesh.verts.diagH (assumed)
        # This class provides the assembly and apply methods

        print(f"[ScalarDiag] Initialized: {self.n_verts} verts")

    @ti.kernel
    def assemble(self, mu: ti.f32, la: ti.f32, dt: ti.f32):
        """
        Assemble diagonal preconditioner.

        This is the same computation as pncg_base_collision_free.compute_grad_and_diagH
        but separated for modularity.
        """
        # Initialize with mass
        for vert in self.mesh.verts:
            vert.diagH = vert.m * ti.Vector.one(ti.f32, 3)

        # Add elastic contribution
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * dt * dt

            # Compute diagonal of d2Psidx2
            dFdx = compute_dFdx(B)
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, mu, la)
            temp = d2PsidF2 @ dFdx
            H_e = dFdx.transpose() @ temp
            H_e = para * H_e

            # Extract diagonal
            for i in ti.static(range(4)):
                diagH_d2Psidx2 = ti.Vector([H_e[i * 3, i * 3],
                                            H_e[i * 3 + 1, i * 3 + 1],
                                            H_e[i * 3 + 2, i * 3 + 2]])
                diagH_d2Psidx2 = ti.max(diagH_d2Psidx2, 0.0)
                c.verts[i].diagH += diagH_d2Psidx2

    @ti.kernel
    def apply(self):
        """Apply scalar diagonal preconditioner: z = grad / diagH."""
        for vert in self.mesh.verts:
            vert.z = vert.grad / vert.diagH

    def rebuild(self, solver):
        """Rebuild preconditioner."""
        self.assemble(solver.mu, solver.la, solver.dt)
