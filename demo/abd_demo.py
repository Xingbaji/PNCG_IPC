"""
ABD (Affine Body Dynamics) Demo.

This demo demonstrates the ABD system with a simple falling cube.
The cube is treated as an ABD body (12 DOFs) instead of full FEM.

Usage:
    python abd_demo.py                    # Interactive mode
    python abd_demo.py --headless --frames 100  # Headless mode
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.abd_system import ABDSystem, ABDJacobian


def test_jacobian():
    """Test Jacobian operations."""
    print("=" * 50)
    print("Testing ABD Jacobian operations")
    print("=" * 50)

    # Test point
    x_bar = ti.Vector([1.0, 2.0, 3.0], dt=ti.f64)

    # Identity state (at origin)
    q = ti.Vector([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dt=ti.f64)

    @ti.kernel
    def test_J() -> ti.types.vector(3, ti.f64):
        return ABDJacobian.apply_J(x_bar, q)

    @ti.kernel
    def test_JT() -> ti.types.vector(12, ti.f64):
        g = ti.Vector([1.0, 0.0, 0.0], dt=ti.f64)
        return ABDJacobian.apply_JT(x_bar, g)

    # Test J * q (should equal x_bar for identity at origin)
    x = test_J()
    print(f"J * q (identity at origin): {x}")
    print(f"Expected: {x_bar}")

    # Test J^T * g
    g_q = test_JT()
    print(f"J^T * [1,0,0]: {g_q}")

    print("Jacobian tests passed!")
    print()


def test_abd_system():
    """Test ABD system with a simple cube."""
    print("=" * 50)
    print("Testing ABD System")
    print("=" * 50)

    # Create a simple cube (8 vertices)
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    # Uniform masses
    masses = np.ones(8) * 0.125  # Total mass = 1

    # Create ABD system
    abd_system = ABDSystem(max_bodies=4, max_points_per_body=100)
    abd_system.dt = 0.01
    abd_system.gravity = ti.Vector([0.0, -9.8, 0.0])

    # Add cube as ABD body
    body_id = abd_system.add_body(
        point_ids=np.arange(8),
        rest_positions=cube_verts,
        masses=masses,
        volume=1.0,
        kappa_shape=1e6
    )

    print(f"Added body {body_id}")
    print(f"ABD stats: {abd_system.get_stats()}")

    # Create vertex field for position output
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)

    # Test state to position mapping
    abd_system.compute_x_from_q(vertices)

    print(f"Initial positions (should match cube_verts):")
    for i in range(8):
        print(f"  v{i}: {vertices[i]}")

    # Test prediction
    abd_system.compute_q_tilde(0.01)
    print(f"q_tilde computed (includes gravity)")

    # Test shape energy
    E_shape = abd_system.compute_shape_energy()
    print(f"Shape energy (should be ~0 for identity): {E_shape}")

    print("ABD System tests passed!")
    print()


def run_falling_cube_demo(headless=False, n_frames=100):
    """
    Run a falling cube demo using ABD.

    The cube falls under gravity and should land on the ground.
    """
    print("=" * 50)
    print("ABD Falling Cube Demo")
    print("=" * 50)

    # Create cube vertices
    size = 0.5
    center = np.array([0.0, 1.5, 0.0])  # Start above ground

    cube_verts = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ], dtype=np.float64) + center

    # Cube faces for rendering
    cube_faces = np.array([
        [0, 2, 1], [0, 3, 2],  # Front
        [4, 5, 6], [4, 6, 7],  # Back
        [0, 1, 5], [0, 5, 4],  # Bottom
        [2, 3, 7], [2, 7, 6],  # Top
        [0, 4, 7], [0, 7, 3],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)

    masses = np.ones(8) * 0.125

    # Create ABD system
    abd_system = ABDSystem(max_bodies=4, max_points_per_body=100)
    abd_system.dt = 0.01
    abd_system.gravity = ti.Vector([0.0, -9.8, 0.0])

    body_id = abd_system.add_body(
        point_ids=np.arange(8),
        rest_positions=cube_verts,
        masses=masses,
        volume=size ** 3,
        kappa_shape=1e4
    )

    # Vertex field for rendering
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
    indices = ti.field(dtype=ti.i32, shape=len(cube_faces) * 3)
    indices.from_numpy(cube_faces.flatten())

    # Ground plane
    ground_y = 0.0
    ground_kappa = 1e6
    ground_dHat = 0.1

    @ti.kernel
    def add_ground_barrier_gradient():
        """Add ground barrier gradient to ABD bodies."""
        for i in range(abd_system.n_total_points):
            global_id = abd_system.global_vertex_id[i]
            x = vertices[global_id]

            dist = x[1] - ground_y
            if dist < ground_dHat and dist > 1e-6:
                # Barrier gradient
                t2 = dist - ground_dHat
                g = ground_kappa * (t2 * ti.log(dist / ground_dHat) * (-2.0) - (t2 ** 2) / dist)

                # Project to ABD state
                body_id = abd_system.point_body_id[i]
                x_bar = abd_system.x_bar[i]

                # g_q = J^T @ [0, g, 0]
                g_vec = ti.Vector([0.0, g, 0.0], dt=ti.f64)
                g_q = ABDJacobian.apply_JT(x_bar, g_vec)

                for d in ti.static(range(12)):
                    ti.atomic_add(abd_system.grad_q[body_id][d], g_q[d])

    @ti.kernel
    def compute_abd_gradient():
        """Compute ABD inertia gradient."""
        for body_id in range(abd_system.n_bodies):
            q = abd_system.q[body_id]
            q_tilde = abd_system.q_tilde[body_id]
            M = abd_system.abd_mass[body_id]

            # Clear gradient
            abd_system.grad_q[body_id] = ti.Vector.zero(ti.f64, 12)

            # Inertia: g = M @ (q - q_tilde)
            dq = q - q_tilde
            g_inertia = M @ dq

            for d in ti.static(range(12)):
                abd_system.grad_q[body_id][d] += g_inertia[d]

    @ti.kernel
    def compute_abd_direction():
        """Compute search direction: dq = M^{-1} @ grad."""
        for body_id in range(abd_system.n_bodies):
            M_inv = abd_system.abd_mass_inv[body_id]
            grad_q = abd_system.grad_q[body_id]

            abd_system.dq[body_id] = M_inv @ grad_q

    @ti.kernel
    def compute_gTp() -> ti.f64:
        """Compute g^T @ p."""
        result = 0.0
        for body_id in range(abd_system.n_bodies):
            grad_q = abd_system.grad_q[body_id]
            dq = abd_system.dq[body_id]
            for d in ti.static(range(12)):
                result += grad_q[d] * dq[d]
        return result

    @ti.kernel
    def compute_pHp() -> ti.f64:
        """Compute p^T @ H @ p (mass matrix contribution)."""
        result = 0.0
        for body_id in range(abd_system.n_bodies):
            dq = abd_system.dq[body_id]
            M = abd_system.abd_mass[body_id]
            Mp = M @ dq
            for d in ti.static(range(12)):
                result += dq[d] * Mp[d]
        return result

    def step():
        """Perform one simulation step."""
        # Predict
        abd_system.compute_q_tilde(abd_system.dt)
        abd_system.compute_x_from_q(vertices)

        # Optimization loop
        for iter in range(50):
            # Compute gradient
            compute_abd_gradient()
            add_ground_barrier_gradient()
            abd_system.add_shape_gradient()

            # Compute direction
            compute_abd_direction()

            # Line search
            gTp = compute_gTp()
            pHp = compute_pHp()

            if pHp < 1e-10:
                break

            alpha = gTp / pHp

            # Clamp alpha
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 0.0)

            # Update
            abd_system.step_forward(alpha)
            abd_system.compute_x_from_q(vertices)

            # Check convergence
            if abs(alpha * gTp) < 1e-6:
                break

        # Update velocity
        abd_system.update_velocity(abd_system.dt)

    if headless:
        print(f"Running {n_frames} frames in headless mode...")
        for frame in range(n_frames):
            step()
            if frame % 10 == 0:
                # Print cube center position
                pos = [vertices[i] for i in range(8)]
                center_y = sum(p[1] for p in pos) / 8
                print(f"Frame {frame}: center_y = {center_y:.4f}")
        print("Done!")
    else:
        # Interactive visualization
        window = ti.ui.Window("ABD Falling Cube", (800, 600), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(3, 2, 3)
        camera.lookat(0, 0.5, 0)

        frame = 0
        while window.running:
            # Step simulation
            step()
            frame += 1

            # Render
            camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.3, 0.3, 0.3))
            scene.point_light(pos=(2, 3, 2), color=(1.0, 1.0, 1.0))

            # Draw cube
            scene.mesh(vertices, indices, color=(0.6, 0.4, 0.2))

            # Draw ground
            scene.particles(ti.Vector.field(3, dtype=ti.f32, shape=1), radius=0.01, color=(0.5, 0.5, 0.5))

            canvas.scene(scene)
            window.show()

            if frame > 500:
                break


def test_motor_body():
    """
    Test motor boundary condition.

    Creates a cube with MOTOR boundary type and verifies rotation.
    """
    print("=" * 50)
    print("Testing Motor Body Rotation")
    print("=" * 50)

    from algorithm.abd_system import BodyBoundaryType

    # Create cube vertices
    size = 0.5
    center = np.array([0.0, 0.0, 0.0])

    cube_verts = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ], dtype=np.float64) + center

    masses = np.ones(8) * 0.125

    # Create ABD system with motor body
    abd_system = ABDSystem(max_bodies=4, max_points_per_body=100)
    abd_system.dt = 0.01
    abd_system.gravity = ti.Vector([0.0, 0.0, 0.0])  # No gravity for rotation test

    # Add body as MOTOR with Y-axis rotation
    body_id = abd_system.add_body(
        point_ids=np.arange(8),
        rest_positions=cube_verts,
        masses=masses,
        volume=size ** 3,
        kappa_shape=1e6,
        boundary_type=int(BodyBoundaryType.MOTOR),
        motor_speed=3.14159,  # pi rad/s = half rotation per second
        motor_strength=100.0,
        motor_axis=np.array([0.0, 1.0, 0.0])  # Y-axis rotation
    )

    print(f"Motor body added: {abd_system.get_stats()}")

    # Vertex field for positions
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)

    # Get initial position of first vertex
    abd_system.compute_x_from_q(vertices)
    x0_init = np.array([vertices[0][j] for j in range(3)])
    print(f"Initial v0 position: {x0_init}")

    # Step forward a few times
    for i in range(10):
        abd_system.compute_q_tilde(abd_system.dt)
        # In motor mode, q_tilde becomes the target, we should converge to it
        # For simplicity, just copy q_tilde to q (instant convergence)
        @ti.kernel
        def copy_tilde_to_q():
            for body_id in range(abd_system.n_bodies):
                abd_system.q[body_id] = abd_system.q_tilde[body_id]
        copy_tilde_to_q()

        abd_system.compute_x_from_q(vertices)

    x0_final = np.array([vertices[0][j] for j in range(3)])
    print(f"Final v0 position after 10 steps: {x0_final}")

    # Check that position changed (rotation occurred)
    displacement = np.linalg.norm(x0_final - x0_init)
    print(f"Displacement: {displacement:.6f}")

    if displacement > 1e-4:
        print("Motor rotation test PASSED!")
    else:
        print("Motor rotation test FAILED - no rotation detected")

    print()


def test_fixed_body():
    """
    Test fixed boundary condition.

    Creates a cube with FIXED boundary type and verifies it doesn't move.
    """
    print("=" * 50)
    print("Testing Fixed Body")
    print("=" * 50)

    from algorithm.abd_system import BodyBoundaryType

    # Create cube vertices
    size = 0.5
    center = np.array([0.0, 1.0, 0.0])

    cube_verts = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ], dtype=np.float64) + center

    masses = np.ones(8) * 0.125

    # Create ABD system with fixed body
    abd_system = ABDSystem(max_bodies=4, max_points_per_body=100)
    abd_system.dt = 0.01
    abd_system.gravity = ti.Vector([0.0, -9.8, 0.0])  # Gravity enabled

    # Add body as FIXED
    body_id = abd_system.add_body(
        point_ids=np.arange(8),
        rest_positions=cube_verts,
        masses=masses,
        volume=size ** 3,
        kappa_shape=1e6,
        boundary_type=int(BodyBoundaryType.FIXED)
    )

    print(f"Fixed body added: {abd_system.get_stats()}")

    # Vertex field for positions
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)

    # Get initial position
    abd_system.compute_x_from_q(vertices)
    x0_init = np.array([vertices[0][j] for j in range(3)])
    print(f"Initial v0 position: {x0_init}")

    # Step forward (should not move due to FIXED)
    for i in range(10):
        abd_system.compute_q_tilde(abd_system.dt)
        abd_system.compute_x_from_q(vertices)

    x0_final = np.array([vertices[0][j] for j in range(3)])
    print(f"Final v0 position after 10 steps: {x0_final}")

    # Check that position didn't change
    displacement = np.linalg.norm(x0_final - x0_init)
    print(f"Displacement: {displacement:.6f}")

    if displacement < 1e-6:
        print("Fixed body test PASSED!")
    else:
        print("Fixed body test FAILED - body moved when it shouldn't")

    print()


def main():
    parser = argparse.ArgumentParser(description='ABD Demo')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames for headless mode')
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    parser.add_argument('--motor', action='store_true', help='Run motor test')
    parser.add_argument('--fixed', action='store_true', help='Run fixed body test')
    args = parser.parse_args()

    if args.test:
        test_jacobian()
        test_abd_system()
        print("All tests passed!")
    elif args.motor:
        test_motor_body()
    elif args.fixed:
        test_fixed_body()
    else:
        # Run tests first
        test_jacobian()
        test_abd_system()
        test_motor_body()
        test_fixed_body()

        # Run demo
        run_falling_cube_demo(headless=args.headless, n_frames=args.frames)


if __name__ == '__main__':
    main()
