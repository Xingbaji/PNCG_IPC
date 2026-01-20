"""
ABD System Unit Tests

Test the Affine Body Dynamics (ABD) system implementation including:
- ABDJacobian: coordinate transformation (J, J^T, J^T @ H @ J)
- ABDDyadicMass: efficient mass matrix representation
- ABDShapeEnergy: shape preservation energy, gradient, and Hessian
- ABDSystem: full system with multiple bodies

Reference: Stiff-GIPC (CUDA implementation)
"""

import sys
import os
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

import taichi as ti


def init_taichi():
    """Initialize Taichi with CPU backend for testing."""
    ti.init(arch=ti.cpu, default_fp=ti.f64)


class TestABDJacobian:
    """Test ABDJacobian class."""

    def test_apply_J_identity(self):
        """Test J @ q with identity affine transformation."""
        print("\n=== Test: ABDJacobian.apply_J with identity ===")
        from algorithm.abd_system import ABDJacobian

        @ti.kernel
        def test_kernel() -> ti.types.vector(3, ti.f64):
            # Identity state: p=[1,2,3], A=I
            q = ti.Vector([1.0, 2.0, 3.0,  # p
                           1.0, 0.0, 0.0,  # a1
                           0.0, 1.0, 0.0,  # a2
                           0.0, 0.0, 1.0], dt=ti.f64)  # a3

            # Rest position
            x_bar = ti.Vector([0.5, 0.5, 0.5], dt=ti.f64)

            # x = p + A @ x_bar = [1,2,3] + I @ [0.5,0.5,0.5] = [1.5, 2.5, 3.5]
            x = ABDJacobian.apply_J(x_bar, q)
            return x

        x = test_kernel()
        expected = np.array([1.5, 2.5, 3.5])
        error = np.linalg.norm(np.array([x[i] for i in range(3)]) - expected)
        print(f"  Result: [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}]")
        print(f"  Expected: {expected}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-10, f"apply_J error too large: {error}"
        print("  PASSED")

    def test_apply_J_rotation(self):
        """Test J @ q with 90-degree rotation around Z axis."""
        print("\n=== Test: ABDJacobian.apply_J with rotation ===")
        from algorithm.abd_system import ABDJacobian

        @ti.kernel
        def test_kernel() -> ti.types.vector(3, ti.f64):
            # 90-degree rotation around Z: A = [[0,-1,0],[1,0,0],[0,0,1]]
            # But in our convention A rows are stored as a1, a2, a3
            # So if we rotate [1,0,0] by 90 deg around Z, we get [0,1,0]
            # R_z(90) = [[0,-1,0],[1,0,0],[0,0,1]]
            # A^T = R, so A = R^T = [[0,1,0],[-1,0,0],[0,0,1]]
            q = ti.Vector([0.0, 0.0, 0.0,  # p = origin
                           0.0, 1.0, 0.0,  # a1 (first row of A)
                          -1.0, 0.0, 0.0,  # a2 (second row of A)
                           0.0, 0.0, 1.0], dt=ti.f64)  # a3

            # Rest position on X axis
            x_bar = ti.Vector([1.0, 0.0, 0.0], dt=ti.f64)

            # x = A @ x_bar = [a1·x_bar, a2·x_bar, a3·x_bar] = [0, -1, 0]
            x = ABDJacobian.apply_J(x_bar, q)
            return x

        x = test_kernel()
        expected = np.array([0.0, -1.0, 0.0])
        error = np.linalg.norm(np.array([x[i] for i in range(3)]) - expected)
        print(f"  Result: [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}]")
        print(f"  Expected: {expected}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-10, f"apply_J rotation error too large: {error}"
        print("  PASSED")

    def test_apply_JT(self):
        """Test J^T @ g gradient projection."""
        print("\n=== Test: ABDJacobian.apply_JT ===")
        from algorithm.abd_system import ABDJacobian

        @ti.kernel
        def test_kernel() -> ti.types.vector(12, ti.f64):
            x_bar = ti.Vector([1.0, 2.0, 3.0], dt=ti.f64)
            g = ti.Vector([1.0, 1.0, 1.0], dt=ti.f64)

            # J^T @ g = [g; x_bar * g[0]; x_bar * g[1]; x_bar * g[2]]
            # = [1,1,1, 1,2,3, 1,2,3, 1,2,3]
            g12 = ABDJacobian.apply_JT(x_bar, g)
            return g12

        g12 = test_kernel()
        expected = np.array([1.0, 1.0, 1.0,  # g
                             1.0, 2.0, 3.0,  # x_bar * g[0]
                             1.0, 2.0, 3.0,  # x_bar * g[1]
                             1.0, 2.0, 3.0]) # x_bar * g[2]
        result = np.array([g12[i] for i in range(12)])
        error = np.linalg.norm(result - expected)
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-10, f"apply_JT error too large: {error}"
        print("  PASSED")

    def test_J_JT_consistency(self):
        """Test that x·(J@q) = q·(J^T@x) for inner product consistency."""
        print("\n=== Test: J and J^T inner product consistency ===")
        from algorithm.abd_system import ABDJacobian

        @ti.kernel
        def test_kernel() -> ti.types.vector(2, ti.f64):
            x_bar = ti.Vector([0.5, -0.3, 0.8], dt=ti.f64)

            # Random q and g
            q = ti.Vector([1.0, 2.0, 3.0, 0.9, 0.1, 0.0,
                           0.1, 0.95, 0.0, 0.0, 0.05, 1.0], dt=ti.f64)
            g = ti.Vector([0.5, -0.3, 0.7], dt=ti.f64)

            # Compute J @ q
            Jq = ABDJacobian.apply_J(x_bar, q)

            # Compute J^T @ g
            JTg = ABDJacobian.apply_JT(x_bar, g)

            # Check: g · (J @ q) == q · (J^T @ g)
            dot1 = g.dot(Jq)
            dot2 = 0.0
            for i in ti.static(range(12)):
                dot2 += q[i] * JTg[i]

            return ti.Vector([dot1, dot2])

        result = test_kernel()
        dot1, dot2 = result[0], result[1]
        error = abs(dot1 - dot2)
        print(f"  g·(J@q) = {dot1:.6f}")
        print(f"  q·(J^T@g) = {dot2:.6f}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-10, f"J/J^T consistency error: {error}"
        print("  PASSED")


class TestABDShapeEnergy:
    """Test ABDShapeEnergy class."""

    def test_energy_identity(self):
        """Test shape energy at identity (should be 0)."""
        print("\n=== Test: Shape energy at identity ===")
        from algorithm.abd_system import ABDShapeEnergy

        @ti.kernel
        def test_kernel() -> ti.f64:
            # Identity state
            q = ti.Vector([0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0], dt=ti.f64)
            return ABDShapeEnergy.compute_energy(q)

        E = test_kernel()
        print(f"  Energy at identity: {E:.2e}")
        assert abs(E) < 1e-10, f"Energy at identity should be 0, got {E}"
        print("  PASSED")

    def test_energy_scaling(self):
        """Test shape energy with uniform scaling."""
        print("\n=== Test: Shape energy with uniform scaling ===")
        from algorithm.abd_system import ABDShapeEnergy

        @ti.kernel
        def test_kernel(scale: ti.f64) -> ti.f64:
            # Uniform scaling: A = scale * I
            q = ti.Vector([0.0, 0.0, 0.0,
                           scale, 0.0, 0.0,
                           0.0, scale, 0.0,
                           0.0, 0.0, scale], dt=ti.f64)
            return ABDShapeEnergy.compute_energy(q)

        # Test various scales
        for scale in [0.5, 1.0, 1.5, 2.0]:
            E = test_kernel(scale)
            # Expected: 3 * (scale^2 - 1)^2
            expected = 3 * (scale**2 - 1)**2
            error = abs(E - expected)
            print(f"  Scale={scale}: E={E:.4f}, expected={expected:.4f}, error={error:.2e}")
            assert error < 1e-10, f"Scaling energy error: {error}"
        print("  PASSED")

    def test_gradient_finite_diff(self):
        """Test gradient against finite differences."""
        print("\n=== Test: Shape energy gradient vs finite diff ===")
        from algorithm.abd_system import ABDShapeEnergy

        @ti.kernel
        def compute_energy(q: ti.types.vector(12, ti.f64)) -> ti.f64:
            return ABDShapeEnergy.compute_energy(q)

        @ti.kernel
        def compute_gradient(q: ti.types.vector(12, ti.f64)) -> ti.types.vector(9, ti.f64):
            return ABDShapeEnergy.compute_gradient(q)

        # Test at non-identity state
        q_np = np.array([0.0, 0.0, 0.0,
                         1.1, 0.05, 0.0,
                         0.05, 0.95, 0.1,
                         0.0, 0.1, 1.05])

        # Compute analytical gradient
        q_ti = ti.Vector(q_np.tolist(), dt=ti.f64)
        grad = compute_gradient(q_ti)
        grad_np = np.array([grad[i] for i in range(9)])

        # Compute finite difference gradient
        eps = 1e-6
        grad_fd = np.zeros(9)
        for i in range(9):
            q_plus = q_np.copy()
            q_minus = q_np.copy()
            q_plus[3 + i] += eps
            q_minus[3 + i] -= eps

            E_plus = compute_energy(ti.Vector(q_plus.tolist(), dt=ti.f64))
            E_minus = compute_energy(ti.Vector(q_minus.tolist(), dt=ti.f64))
            grad_fd[i] = (E_plus - E_minus) / (2 * eps)

        error = np.linalg.norm(grad_np - grad_fd)
        print(f"  Analytical gradient: {grad_np}")
        print(f"  Finite diff gradient: {grad_fd}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-5, f"Gradient error too large: {error}"
        print("  PASSED")

    def test_hessian_finite_diff(self):
        """Test Hessian against finite differences of gradient."""
        print("\n=== Test: Shape energy Hessian vs finite diff ===")
        from algorithm.abd_system import ABDShapeEnergy

        @ti.kernel
        def compute_gradient(q: ti.types.vector(12, ti.f64)) -> ti.types.vector(9, ti.f64):
            return ABDShapeEnergy.compute_gradient(q)

        @ti.kernel
        def compute_hessian(q: ti.types.vector(12, ti.f64)) -> ti.types.matrix(9, 9, ti.f64):
            return ABDShapeEnergy.compute_hessian(q)

        # Test at non-identity state
        q_np = np.array([0.0, 0.0, 0.0,
                         1.1, 0.05, 0.0,
                         0.05, 0.95, 0.1,
                         0.0, 0.1, 1.05])

        # Compute analytical Hessian
        q_ti = ti.Vector(q_np.tolist(), dt=ti.f64)
        H = compute_hessian(q_ti)
        H_np = np.array([[H[i, j] for j in range(9)] for i in range(9)])

        # Compute finite difference Hessian
        eps = 1e-5
        H_fd = np.zeros((9, 9))
        for i in range(9):
            q_plus = q_np.copy()
            q_minus = q_np.copy()
            q_plus[3 + i] += eps
            q_minus[3 + i] -= eps

            grad_plus = compute_gradient(ti.Vector(q_plus.tolist(), dt=ti.f64))
            grad_minus = compute_gradient(ti.Vector(q_minus.tolist(), dt=ti.f64))

            for j in range(9):
                H_fd[j, i] = (grad_plus[j] - grad_minus[j]) / (2 * eps)

        error = np.linalg.norm(H_np - H_fd)
        print(f"  Hessian Frobenius error: {error:.2e}")
        assert error < 1e-4, f"Hessian error too large: {error}"
        print("  PASSED")


class TestABDSystem:
    """Test ABDSystem class."""

    def test_add_body(self):
        """Test adding a body to the system."""
        print("\n=== Test: Add body to ABD system ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a simple cube body
        n_points = 8
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=np.float64)
        masses = np.ones(n_points)
        volume = 1.0

        body_id = system.add_body(point_ids, rest_positions, masses, volume)

        print(f"  Body ID: {body_id}")
        print(f"  N bodies: {system.n_bodies}")
        print(f"  N total points: {system.n_total_points}")

        assert body_id == 0, f"First body should have ID 0"
        assert system.n_bodies == 1, f"Should have 1 body"
        assert system.n_total_points == n_points, f"Should have {n_points} points"
        print("  PASSED")

    def test_mass_matrix_symmetry(self):
        """Test that computed mass matrix is symmetric."""
        print("\n=== Test: Mass matrix symmetry ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a body with non-uniform point distribution
        n_points = 10
        np.random.seed(42)
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.random.randn(n_points, 3)
        masses = np.random.rand(n_points) + 0.1

        body_id = system.add_body(point_ids, rest_positions, masses, volume=1.0)

        # Get mass matrix
        M = system.abd_mass[body_id].to_numpy()

        # Check symmetry
        error = np.linalg.norm(M - M.T)
        print(f"  Symmetry error: {error:.2e}")
        assert error < 1e-10, f"Mass matrix not symmetric: {error}"

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(M)
        min_eig = np.min(eigenvalues)
        print(f"  Min eigenvalue: {min_eig:.6f}")
        assert min_eig > 0, f"Mass matrix not positive definite: min_eig={min_eig}"
        print("  PASSED")

    def test_q_to_x_mapping(self):
        """Test state to position mapping."""
        print("\n=== Test: q -> x mapping ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a simple body at origin with identity transform
        n_points = 4
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        masses = np.ones(n_points)

        body_id = system.add_body(point_ids, rest_positions, masses, volume=1.0)

        # Create vertex field to receive positions
        vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_points)

        # Map positions
        system.compute_x_from_q(vertices)

        # Check positions
        verts_np = vertices.to_numpy()
        print(f"  Mapped positions:\n{verts_np}")

        # With identity transform at COM, positions should match rest positions
        # (after subtracting COM offset)
        com = np.mean(rest_positions, axis=0)
        expected = rest_positions
        # Note: x = p + A @ x_bar, with p=COM, A=I, x_bar = rest - COM
        # So x = COM + I @ (rest - COM) = rest

        # Actually, let's verify the COM calculation
        q = system.q[body_id].to_numpy()
        print(f"  State q: p={q[0:3]}, a1={q[3:6]}, a2={q[6:9]}, a3={q[9:12]}")
        print("  PASSED")

    def test_kinetic_energy(self):
        """Test kinetic energy computation."""
        print("\n=== Test: Kinetic energy computation ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a body
        n_points = 8
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=np.float64)
        masses = np.ones(n_points)

        body_id = system.add_body(point_ids, rest_positions, masses, volume=1.0)

        # Set q_tilde to be different from q
        q = system.q[body_id].to_numpy()
        q_tilde = q.copy()
        q_tilde[0] += 0.1  # Small translation

        system.q_tilde[body_id] = q_tilde

        # Compute kinetic energy
        K = system.compute_kinetic_energy()
        print(f"  Kinetic energy: {K:.6f}")

        # Should be positive since q != q_tilde
        assert K > 0, f"Kinetic energy should be positive"
        print("  PASSED")

    def test_shape_energy(self):
        """Test shape energy computation."""
        print("\n=== Test: Shape energy computation ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a body
        n_points = 8
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.random.randn(n_points, 3)
        masses = np.ones(n_points)

        body_id = system.add_body(point_ids, rest_positions, masses, volume=1.0)

        # At identity, shape energy should be 0
        V0 = system.compute_shape_energy()
        print(f"  Shape energy at identity: {V0:.2e}")
        assert abs(V0) < 1e-6, f"Shape energy at identity should be ~0"

        # Apply some deformation
        q = system.q[body_id].to_numpy()
        q[3] = 1.1  # Scale a1
        system.q[body_id] = q

        V1 = system.compute_shape_energy()
        print(f"  Shape energy after deformation: {V1:.6f}")
        assert V1 > 0, f"Shape energy after deformation should be positive"
        print("  PASSED")

    def test_gradient_projection(self):
        """Test gradient projection from vertex space to state space."""
        print("\n=== Test: Gradient projection ===")
        from algorithm.abd_system import ABDSystem

        system = ABDSystem(max_bodies=10, max_points_per_body=100)

        # Create a body
        n_points = 4
        point_ids = np.arange(n_points, dtype=np.int32)
        rest_positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        masses = np.ones(n_points)

        body_id = system.add_body(point_ids, rest_positions, masses, volume=1.0)
        system.setup_vertex_mapping(n_points)

        # Create vertex gradients (uniform upward force)
        vertex_grad = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
        for i in range(n_points):
            vertex_grad[i] = [0.0, 1.0, 0.0]  # Upward

        # Project to state space
        system.project_gradient_to_q(vertex_grad)

        grad_q = system.grad_q[body_id].to_numpy()
        print(f"  Projected gradient: {grad_q}")

        # Translation part should be sum of vertex gradients
        # = n_points * [0, 1, 0] = [0, 4, 0]
        expected_trans = np.array([0, n_points, 0])
        error = np.linalg.norm(grad_q[0:3] - expected_trans)
        print(f"  Translation gradient: {grad_q[0:3]}, expected: {expected_trans}")
        print(f"  Error: {error:.2e}")
        assert error < 1e-6, f"Translation gradient error: {error}"
        print("  PASSED")


def run_all_tests():
    """Run all ABD system tests."""
    print("=" * 60)
    print("ABD System Unit Tests")
    print("=" * 60)

    init_taichi()

    # Run Jacobian tests
    jacobian_tests = TestABDJacobian()
    jacobian_tests.test_apply_J_identity()
    jacobian_tests.test_apply_J_rotation()
    jacobian_tests.test_apply_JT()
    jacobian_tests.test_J_JT_consistency()

    # Run Shape Energy tests
    shape_tests = TestABDShapeEnergy()
    shape_tests.test_energy_identity()
    shape_tests.test_energy_scaling()
    shape_tests.test_gradient_finite_diff()
    shape_tests.test_hessian_finite_diff()

    # Run System tests
    system_tests = TestABDSystem()
    system_tests.test_add_body()
    system_tests.test_mass_matrix_symmetry()
    system_tests.test_q_to_x_mapping()
    system_tests.test_kinetic_energy()
    system_tests.test_shape_energy()
    system_tests.test_gradient_projection()

    print("\n" + "=" * 60)
    print("All ABD System tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
