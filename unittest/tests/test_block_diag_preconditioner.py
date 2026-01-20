"""
Unit tests for Block Diagonal Preconditioner.

Tests:
- FEM block diagonal assembly
- FEM block inversion
- ABD block assembly and inversion
- Preconditioner application

Note: These tests use simplified versions that work without MeshTaichi.
The actual implementation in block_diag_preconditioner.py uses MeshTaichi
iteration patterns that require real mesh objects.
"""

import pytest
import numpy as np
import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f32)


@ti.data_oriented
class MockMesh:
    """Mock mesh for testing."""

    def __init__(self, n_verts, n_cells):
        self.n_verts = n_verts
        self.n_cells = n_cells

        # Vertex fields
        self.verts = MockVertexContainer(n_verts)

        # Cell fields
        self.cells = MockCellContainer(n_cells, self.verts)


@ti.data_oriented
class MockVertexContainer:
    """Mock vertex container."""

    def __init__(self, n_verts):
        self._size = n_verts
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.m = ti.field(dtype=ti.f32, shape=n_verts)
        self.grad = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.diagH = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

        # Initialize with default values
        self._init_verts(n_verts)

    @ti.kernel
    def _init_verts(self, n_verts: ti.i32):
        for i in range(n_verts):
            self.x[i] = ti.Vector([i * 0.1, 0.0, 0.0], dt=ti.f32)
            self.m[i] = 1.0
            self.grad[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.z[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.diagH[i] = ti.Vector([1.0, 1.0, 1.0], dt=ti.f32)

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size


@ti.data_oriented
class MockCellContainer:
    """Mock cell container."""

    def __init__(self, n_cells, verts):
        self._size = n_cells
        self.verts = verts

        # Cell data
        self.W = ti.field(dtype=ti.f32, shape=n_cells)
        self.B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)

        # Initialize
        self._init_cells(n_cells)

    @ti.kernel
    def _init_cells(self, n_cells: ti.i32):
        for i in range(n_cells):
            self.W[i] = 1.0
            self.B[i] = ti.Matrix.identity(ti.f32, 3)

    def __len__(self):
        return self._size


@ti.data_oriented
class TestableBlockDiagPreconditioner:
    """
    Testable version of BlockDiagPreconditioner.

    Uses array-based indexing instead of MeshTaichi iteration.
    """

    def __init__(self, n_verts, mass_field):
        self.n_verts = n_verts
        self.mass_field = mass_field

        # Store 3x3 diagonal blocks and their inverses
        self.diag_blocks = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_verts)
        self.diag_blocks_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_verts)

        self.assembled = False

    @ti.kernel
    def _clear_diag_blocks(self):
        """Zero out all diagonal blocks."""
        for i in range(self.n_verts):
            self.diag_blocks[i] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _add_inertia_contribution(self):
        """Add mass matrix to diagonal blocks: H_ii += m * I."""
        for i in range(self.n_verts):
            m = self.mass_field[i]
            for d in ti.static(range(3)):
                self.diag_blocks[i][d, d] += m

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

    @ti.kernel
    def apply(self, grad_field: ti.template(), z_field: ti.template()):
        """Apply preconditioner: z = H_ii^{-1} @ grad."""
        for i in range(self.n_verts):
            g = grad_field[i]
            H_inv = self.diag_blocks_inv[i]

            z = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            for di in ti.static(range(3)):
                for dj in ti.static(range(3)):
                    z[di] += H_inv[di, dj] * g[dj]

            z_field[i] = z


class TestBlockDiagPreconditioner:
    """Test BlockDiagPreconditioner class (using testable version)."""

    def test_init(self):
        """Test initialization."""
        from algorithm.block_diag_preconditioner import BlockDiagPreconditioner

        mesh = MockMesh(n_verts=32, n_cells=50)
        precond = BlockDiagPreconditioner(mesh)

        assert precond.n_verts == 32
        assert precond.n_cells == 50
        assert not precond.assembled

    def test_inertia_contribution(self):
        """Test that inertia contribution adds mass to diagonal."""
        mesh = MockMesh(n_verts=16, n_cells=20)

        # Set specific mass
        for i in range(16):
            mesh.verts.m[i] = 2.0

        precond = TestableBlockDiagPreconditioner(16, mesh.verts.m)
        precond._clear_diag_blocks()
        precond._add_inertia_contribution()

        # Check that diagonal has mass
        H = precond.diag_blocks[0].to_numpy()
        expected_diag = 2.0
        assert np.isclose(H[0, 0], expected_diag)
        assert np.isclose(H[1, 1], expected_diag)
        assert np.isclose(H[2, 2], expected_diag)

    def test_block_inversion(self):
        """Test that H @ H^{-1} = I for diagonal blocks."""
        mesh = MockMesh(n_verts=8, n_cells=10)

        # Set specific mass
        for i in range(8):
            mesh.verts.m[i] = float(i + 1)

        precond = TestableBlockDiagPreconditioner(8, mesh.verts.m)
        precond._clear_diag_blocks()
        precond._add_inertia_contribution()
        precond._invert_diag_blocks()

        # Check H @ H_inv = I for first vertex
        H = precond.diag_blocks[0].to_numpy()
        H_inv = precond.diag_blocks_inv[0].to_numpy()
        product = H @ H_inv

        I = np.eye(3)
        assert np.allclose(product, I, atol=1e-5), f"H @ H_inv != I: {product}"


# ABD DOF constant
ABD_DOF = 12


@ti.data_oriented
class TestableBlockDiagPreconditionerABD:
    """
    Testable version of BlockDiagPreconditionerABD.

    Simplified for testing without full ABD system.
    """

    def __init__(self, n_verts, mass_field, n_abd_bodies=0, max_abd_bodies=8):
        self.n_verts = n_verts
        self.mass_field = mass_field
        self.n_abd_bodies = n_abd_bodies
        self.max_abd_bodies = max_abd_bodies

        # FEM: 3x3 diagonal blocks
        self.fem_diag_blocks = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_verts)
        self.fem_diag_blocks_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_verts)

        # ABD: 12x12 diagonal blocks
        self.abd_diag_blocks = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                shape=max_abd_bodies)
        self.abd_diag_blocks_inv = ti.Matrix.field(ABD_DOF, ABD_DOF, dtype=ti.f32,
                                                    shape=max_abd_bodies)

        # ABD buffers
        self.abd_grad = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=max_abd_bodies)
        self.abd_z = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=max_abd_bodies)

        self.assembled = False

    @ti.kernel
    def _clear_abd_diag_blocks(self):
        """Zero out ABD diagonal blocks."""
        for i in range(self.max_abd_bodies):
            self.abd_diag_blocks[i] = ti.Matrix.zero(ti.f32, ABD_DOF, ABD_DOF)

    @ti.kernel
    def _set_abd_mass_block(self, body_id: ti.i32, total_mass: ti.f32):
        """Set ABD block as mass matrix (simplified)."""
        # Set translation mass
        for d in ti.static(range(3)):
            self.abd_diag_blocks[body_id][d, d] = total_mass

        # Set rotation/affine inertia (simplified: use identity scaled by mass)
        for d in ti.static(range(3, ABD_DOF)):
            self.abd_diag_blocks[body_id][d, d] = total_mass * 0.1  # Simplified inertia

    @ti.kernel
    def _add_abd_shape_hessian(self, body_id: ti.i32, dt: ti.f32, kappa: ti.f32, volume: ti.f32):
        """Add shape energy Hessian (simplified: diagonal contribution)."""
        scale = dt * dt * kappa * volume

        # Add to affine DOFs (3-11)
        for d in ti.static(range(9)):
            self.abd_diag_blocks[body_id][3 + d, 3 + d] += scale

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

    @ti.kernel
    def _apply_abd(self):
        """Apply ABD preconditioner: z = H_body^{-1} @ grad."""
        for body_id in range(self.n_abd_bodies):
            g = self.abd_grad[body_id]
            H_inv = self.abd_diag_blocks_inv[body_id]

            z = ti.Vector.zero(ti.f32, ABD_DOF)
            for i in ti.static(range(ABD_DOF)):
                for j in ti.static(range(ABD_DOF)):
                    z[i] += H_inv[i, j] * g[j]

            self.abd_z[body_id] = z


class TestBlockDiagPreconditionerABD:
    """Test BlockDiagPreconditionerABD class (using testable version)."""

    def test_init_without_abd(self):
        """Test initialization without ABD system."""
        from algorithm.block_diag_preconditioner import BlockDiagPreconditionerABD

        mesh = MockMesh(n_verts=32, n_cells=50)
        precond = BlockDiagPreconditionerABD(mesh, abd_system=None)

        assert precond.n_verts == 32
        assert precond.n_abd_bodies == 0

    def test_init_with_abd(self):
        """Test initialization with ABD system."""
        from algorithm.block_diag_preconditioner import BlockDiagPreconditionerABD
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)

        # Create ABD system
        abd = ABDSystem(max_bodies=8)
        point_ids = np.arange(4)
        rest_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        masses = np.ones(4, dtype=np.float32)
        abd.add_body(point_ids, rest_pos, masses, volume=0.5, kappa_shape=1e5)

        precond = BlockDiagPreconditionerABD(mesh, abd_system=abd)

        assert precond.n_verts == 32
        assert precond.n_abd_bodies == 1

    def test_abd_block_contains_mass(self):
        """Test that ABD block contains mass matrix (testable version)."""
        mesh = MockMesh(n_verts=32, n_cells=50)

        precond = TestableBlockDiagPreconditionerABD(32, mesh.verts.m, n_abd_bodies=1)
        precond._clear_abd_diag_blocks()

        # Set mass block with total mass = 4.0
        precond._set_abd_mass_block(0, 4.0)

        # Check block
        H = precond.abd_diag_blocks[0].to_numpy()

        # Should be symmetric
        assert np.allclose(H, H.T, atol=1e-10)

        # Translation diagonal should have total mass
        assert np.isclose(H[0, 0], 4.0, rtol=1e-5)
        assert np.isclose(H[1, 1], 4.0, rtol=1e-5)
        assert np.isclose(H[2, 2], 4.0, rtol=1e-5)

    def test_abd_block_inversion(self):
        """Test ABD block inversion: H @ H^{-1} = I."""
        mesh = MockMesh(n_verts=32, n_cells=50)

        precond = TestableBlockDiagPreconditionerABD(32, mesh.verts.m, n_abd_bodies=1)
        precond._clear_abd_diag_blocks()

        # Set mass block with total mass = 4.0 and add shape Hessian
        precond._set_abd_mass_block(0, 4.0)
        precond._add_abd_shape_hessian(0, 0.01, 1e4, 0.5)
        precond._invert_abd_diag_blocks()

        H = precond.abd_diag_blocks[0].to_numpy()
        H_inv = precond.abd_diag_blocks_inv[0].to_numpy()

        product = H @ H_inv
        I = np.eye(ABD_DOF)

        assert np.allclose(product, I, atol=1e-5), f"Max error: {np.max(np.abs(product - I))}"

    def test_abd_solve(self):
        """Test ABD preconditioner solve: H @ z = r."""
        mesh = MockMesh(n_verts=32, n_cells=50)

        precond = TestableBlockDiagPreconditionerABD(32, mesh.verts.m, n_abd_bodies=1)
        precond._clear_abd_diag_blocks()

        # Set mass block and shape Hessian
        precond._set_abd_mass_block(0, 4.0)
        precond._add_abd_shape_hessian(0, 0.01, 1e4, 0.5)
        precond._invert_abd_diag_blocks()

        # Set gradient
        test_grad = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
                             0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        precond.abd_grad[0] = test_grad

        # Apply preconditioner
        precond._apply_abd()

        # Get result
        z = precond.abd_z[0].to_numpy()

        # Verify: H @ z should equal grad
        H = precond.abd_diag_blocks[0].to_numpy()
        Hz = H @ z

        assert np.allclose(Hz, test_grad, rtol=1e-4), f"Max error: {np.max(np.abs(Hz - test_grad))}"


class TestScalarDiagPreconditioner:
    """Test ScalarDiagPreconditioner class."""

    def test_init(self):
        """Test initialization."""
        from algorithm.block_diag_preconditioner import ScalarDiagPreconditioner

        mesh = MockMesh(n_verts=32, n_cells=50)
        precond = ScalarDiagPreconditioner(mesh)

        assert precond.n_verts == 32


class TestPreconditionerComparison:
    """Test that different preconditioners produce valid results."""

    def test_block_vs_scalar_positive_definite(self):
        """Test that block preconditioner produces positive z·g."""
        mesh = MockMesh(n_verts=16, n_cells=20)

        # Set specific mass
        for i in range(16):
            mesh.verts.m[i] = float(i + 1)

        # Set random gradient
        np.random.seed(42)
        for i in range(16):
            mesh.verts.grad[i] = np.random.randn(3).astype(np.float32)

        # Use testable block diagonal preconditioner
        precond = TestableBlockDiagPreconditioner(16, mesh.verts.m)
        precond._clear_diag_blocks()
        precond._add_inertia_contribution()
        precond._invert_diag_blocks()
        precond.apply(mesh.verts.grad, mesh.verts.z)

        # Compute z·g
        zTg = 0.0
        for i in range(16):
            z = np.array([mesh.verts.z[i][j] for j in range(3)])
            g = np.array([mesh.verts.grad[i][j] for j in range(3)])
            zTg += np.dot(z, g)

        # Should be positive (since H is positive definite)
        assert zTg > 0, f"z·g should be positive, got {zTg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
