"""
Unit tests for MAS Preconditioner with ABD integration.

Tests:
- Preconditioner initialization with ABD system
- ABD block matrix assembly (inertia + shape)
- ABD block inversion
- Apply preconditioner to FEM + ABD gradients
- Hessian matvec for ABD
"""

import pytest
import numpy as np
import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f32)


class MockMesh:
    """Mock mesh for testing without MeshTaichi."""

    def __init__(self, n_verts, n_cells):
        self.n_verts = n_verts
        self.n_cells = n_cells

        # Vertex fields
        self.verts = MockVertexContainer(n_verts)

        # Cell fields
        self.cells = MockCellContainer(n_cells, self.verts)


class MockVertexContainer:
    """Mock vertex container."""

    def __init__(self, n_verts):
        self._size = n_verts
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.m = ti.field(dtype=ti.f32, shape=n_verts)
        self.grad = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

        # Initialize with default values
        for i in range(n_verts):
            self.x[i] = [i * 0.1, 0.0, 0.0]
            self.m[i] = 1.0
            self.grad[i] = [0.0, 0.0, 0.0]
            self.z[i] = [0.0, 0.0, 0.0]

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

        # Initialize with identity matrices
        self._init_cells(n_cells)

    @ti.kernel
    def _init_cells(self, n_cells: ti.i32):
        for i in range(n_cells):
            self.W[i] = 1.0
            self.B[i] = ti.Matrix.identity(ti.f32, 3)

    def __len__(self):
        return self._size


class TestMASPreconditionerABDInit:
    """Test MASPreconditionerABD initialization."""

    def test_init_without_abd(self):
        """Test initialization without ABD system."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD

        mesh = MockMesh(n_verts=64, n_cells=100)
        precond = MASPreconditionerABD(mesh, abd_system=None)

        assert precond.n_verts == 64
        assert precond.n_abd_bodies == 0
        assert precond.n_fem_parts == 4  # 64 / 16

    def test_init_with_abd(self):
        """Test initialization with ABD system."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=64, n_cells=100)

        # Create ABD system with one body
        abd = ABDSystem(max_bodies=8)
        point_ids = np.arange(4)
        rest_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        masses = np.ones(4)
        abd.add_body(point_ids, rest_pos, masses, volume=1.0, kappa_shape=1e6)

        precond = MASPreconditionerABD(mesh, abd_system=abd)

        assert precond.n_verts == 64
        assert precond.n_abd_bodies == 1
        assert precond.abd_system is abd


class TestABDBlockAssembly:
    """Test ABD block matrix assembly."""

    def test_abd_block_size(self):
        """Test ABD block matrices are 12x12."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD, ABD_DOF
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body
        point_ids = np.arange(4)
        rest_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        masses = np.ones(4) * 0.25
        abd.add_body(point_ids, rest_pos, masses, volume=0.1, kappa_shape=1e5)

        precond = MASPreconditionerABD(mesh, abd_system=abd)

        # ABD block should be 12x12
        block_shape = precond.abd_block_matrices.shape
        assert block_shape == (4,)  # max_bodies
        assert ABD_DOF == 12

    def test_abd_mass_matrix_in_block(self):
        """Test that ABD block contains mass matrix."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body with known mass
        point_ids = np.arange(4)
        rest_pos = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5]
        ])
        masses = np.array([1.0, 1.0, 1.0, 1.0])  # Total mass = 4
        abd.add_body(point_ids, rest_pos, masses, volume=0.5, kappa_shape=0.0)  # No shape energy

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        # Create mock solver
        class MockSolver:
            dt = 0.01
            mu = 1e5
            la = 1e5

        precond._clear_abd_block_matrices()
        precond._assemble_abd_inertia_and_shape(0.01)

        # Check block matrix
        H = precond.abd_block_matrices[0].to_numpy()

        # Mass matrix should be symmetric
        assert np.allclose(H, H.T, atol=1e-10)

        # Diagonal translation blocks should have total mass
        assert np.isclose(H[0, 0], 4.0, rtol=1e-5)  # m
        assert np.isclose(H[1, 1], 4.0, rtol=1e-5)  # m
        assert np.isclose(H[2, 2], 4.0, rtol=1e-5)  # m


class TestABDBlockInversion:
    """Test ABD block inversion."""

    def test_abd_block_inverse_identity(self):
        """Test that H @ H^{-1} = I for ABD blocks."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD, ABD_DOF
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body
        point_ids = np.arange(4)
        rest_pos = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5]
        ])
        masses = np.ones(4)
        abd.add_body(point_ids, rest_pos, masses, volume=0.5, kappa_shape=1e4)

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        # Assemble and invert
        precond._clear_abd_block_matrices()
        precond._assemble_abd_inertia_and_shape(0.01)
        precond._invert_abd_blocks()

        H = precond.abd_block_matrices[0].to_numpy()
        H_inv = precond.abd_block_inverse[0].to_numpy()

        # Check H @ H_inv = I
        product = H @ H_inv
        I = np.eye(ABD_DOF)

        assert np.allclose(product, I, atol=1e-6), f"Max error: {np.max(np.abs(product - I))}"


class TestABDApply:
    """Test applying preconditioner to ABD system."""

    def test_abd_solve(self):
        """Test ABD block solve: z = H^{-1} @ r."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD, ABD_DOF
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body
        point_ids = np.arange(4)
        rest_pos = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5]
        ])
        masses = np.ones(4)
        abd.add_body(point_ids, rest_pos, masses, volume=0.5, kappa_shape=1e4)

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        # Assemble and invert
        precond._clear_abd_block_matrices()
        precond._assemble_abd_inertia_and_shape(0.01)
        precond._invert_abd_blocks()

        # Set ABD gradient
        test_grad = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        abd.grad_q[0] = test_grad

        # Apply ABD solve
        precond._clear_abd_buffers()
        precond._build_abd_r()
        precond._abd_local_solve()

        # Get result
        z = precond.abd_z[0].to_numpy()

        # Verify: H @ z should equal r
        H = precond.abd_block_matrices[0].to_numpy()
        Hz = H @ z

        assert np.allclose(Hz, test_grad, rtol=1e-5), f"Max error: {np.max(np.abs(Hz - test_grad))}"


class TestABDHessianMatvec:
    """Test ABD Hessian matrix-vector product."""

    def test_abd_hessian_matvec(self):
        """Test ABD Hessian @ v computation."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD, ABD_DOF
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body
        point_ids = np.arange(4)
        rest_pos = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5]
        ])
        masses = np.ones(4)
        abd.add_body(point_ids, rest_pos, masses, volume=0.5, kappa_shape=1e4)

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        # Assemble
        precond._clear_abd_block_matrices()
        precond._assemble_abd_inertia_and_shape(0.01)
        precond.matrices_assembled = True

        # Create input vector
        v = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=4)
        result = ti.Vector.field(ABD_DOF, dtype=ti.f32, shape=4)

        test_v = np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        v[0] = test_v

        # Compute H @ v
        precond.abd_hessian_matvec(v, result)

        # Verify against direct matrix multiply
        H = precond.abd_block_matrices[0].to_numpy()
        expected = H @ test_v
        actual = result[0].to_numpy()

        assert np.allclose(actual, expected, rtol=1e-10)


class TestHybridSystem:
    """Test hybrid FEM + ABD preconditioner."""

    def test_hybrid_stats(self):
        """Test preconditioner statistics for hybrid system."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=128, n_cells=200)
        abd = ABDSystem(max_bodies=8)

        # Add two bodies
        for i in range(2):
            point_ids = np.arange(4) + i * 4
            rest_pos = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]) + np.array([i * 2.0, 0.0, 0.0])
            masses = np.ones(4)
            abd.add_body(point_ids, rest_pos, masses, volume=0.5)

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        stats = precond.get_stats()

        assert stats['n_fem_verts'] == 128
        assert stats['n_abd_bodies'] == 2
        assert stats['n_fem_blocks'] > 0
        assert stats['level_num'] >= 1

    def test_update_abd_system(self):
        """Test updating ABD system reference."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=64, n_cells=100)

        # Start without ABD
        precond = MASPreconditionerABD(mesh, abd_system=None)
        assert precond.n_abd_bodies == 0

        # Add ABD system
        abd = ABDSystem(max_bodies=8)
        point_ids = np.arange(4)
        rest_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        masses = np.ones(4)
        abd.add_body(point_ids, rest_pos, masses, volume=0.5)
        abd.add_body(point_ids + 4, rest_pos + np.array([2.0, 0.0, 0.0]), masses, volume=0.5)

        precond.update_abd_system(abd)
        assert precond.n_abd_bodies == 2


class TestShapeEnergyHessian:
    """Test shape energy Hessian in ABD blocks."""

    def test_shape_hessian_positive_definite(self):
        """Test that ABD block with shape energy is positive definite."""
        from algorithm.mas_preconditioner_abd import MASPreconditionerABD, ABD_DOF
        from algorithm.abd_system import ABDSystem

        mesh = MockMesh(n_verts=32, n_cells=50)
        abd = ABDSystem(max_bodies=4)

        # Add a body with non-trivial shape stiffness
        point_ids = np.arange(8)
        rest_pos = np.random.randn(8, 3)
        rest_pos -= rest_pos.mean(axis=0)  # Center at origin
        masses = np.ones(8)
        abd.add_body(point_ids, rest_pos, masses, volume=1.0, kappa_shape=1e6)

        precond = MASPreconditionerABD(mesh, abd_system=abd)
        precond.build_hierarchy()

        # Assemble with small dt to make shape energy dominant
        precond._clear_abd_block_matrices()
        precond._assemble_abd_inertia_and_shape(0.01)

        H = precond.abd_block_matrices[0].to_numpy()

        # Check symmetry
        assert np.allclose(H, H.T, atol=1e-10)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > 0), f"Min eigenvalue: {eigenvalues.min()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
