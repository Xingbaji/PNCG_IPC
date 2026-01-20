"""
Hybrid ABD-FEM PNCG IPC solver.

This module extends the PNCG-IPC solver to support mixed Affine Body Dynamics (ABD)
and Finite Element Method (FEM) bodies. ABD bodies use a 12D reduced coordinate
representation for efficient simulation of rigid/nearly-rigid objects.

Key features:
- Mixed ABD-FEM simulation
- Three contact types: FEM-FEM, ABD-FEM, ABD-ABD
- Gradient/Hessian transformation for ABD bodies
- Compatible with existing MAS preconditioner

Reference: Stiff-GIPC (CUDA implementation)
"""

import time
import numpy as np
import taichi as ti

from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.abd_system import ABDSystem, ABDJacobian, BodyBoundaryType
from util.model_loading import model_loading


@ti.data_oriented
class pncg_abd_ipc_deformer(pncg_ipc_deformer):
    """
    Hybrid ABD-FEM PNCG IPC solver.

    Extends the base PNCG-IPC solver to support:
    - ABD bodies (12D reduced coordinates)
    - FEM bodies (full vertex DOFs)
    - Mixed contact handling
    """

    def __init__(self, demo='cube_0'):
        """
        Initialize hybrid solver.

        Args:
            demo: Demo configuration name
        """
        # Initialize base class (FEM solver)
        super().__init__(demo)

        # Load ABD configuration
        model = model_loading(demo=demo)
        self.use_abd = getattr(model, 'use_abd', False)

        if self.use_abd:
            print('[ABD-IPC] Initializing ABD system...')

            # Initialize ABD system
            abd_config = getattr(model, 'abd_config', {})
            max_bodies = abd_config.get('max_bodies', 64)
            max_points = abd_config.get('max_points_per_body', 10000)

            self.abd_system = ABDSystem(max_bodies=max_bodies,
                                        max_points_per_body=max_points)
            self.abd_system.dt = self.dt
            self.abd_system.gravity = ti.Vector([0.0, self.gravity, 0.0])

            # Track which vertices are ABD vs FEM
            self.is_abd_vertex = ti.field(dtype=ti.i32, shape=self.n_verts)
            self.is_abd_vertex.fill(0)

            # ABD body definitions from config
            abd_bodies = abd_config.get('bodies', [])
            self._setup_abd_bodies(abd_bodies)

            print(f'[ABD-IPC] ABD system initialized: {self.abd_system.n_bodies} bodies')
        else:
            self.abd_system = None

    def _setup_abd_bodies(self, body_configs: list):
        """
        Setup ABD bodies from configuration.

        Args:
            body_configs: List of body configurations, each with:
                - vertex_ids: List of vertex IDs for this body
                - kappa_shape: Shape stiffness (optional)
                - boundary_type: 0=FREE, 1=FIXED, 2=MOTOR (optional)
                - motor_speed: Angular velocity for MOTOR type (optional)
                - motor_strength: Motor torque scaling (optional)
                - motor_axis: Rotation axis for MOTOR type (optional)
        """
        if not body_configs:
            return

        positions_np = self.mesh.verts.x.to_numpy()
        masses_np = self.mesh.verts.m.to_numpy()

        for config in body_configs:
            vertex_ids = np.array(config['vertex_ids'], dtype=np.int32)
            kappa_shape = config.get('kappa_shape', 1e6)
            boundary_type = config.get('boundary_type', BodyBoundaryType.FREE)
            motor_speed = config.get('motor_speed', 0.0)
            motor_strength = config.get('motor_strength', 10.0)
            motor_axis = config.get('motor_axis', np.array([0.0, 1.0, 0.0]))

            # Get positions and masses for this body
            rest_positions = positions_np[vertex_ids]
            masses = masses_np[vertex_ids]

            # Estimate volume from convex hull (simplified)
            volume = self._estimate_volume(rest_positions)

            # Add body to ABD system with boundary conditions
            body_id = self.abd_system.add_body(
                point_ids=vertex_ids,
                rest_positions=rest_positions,
                masses=masses,
                volume=volume,
                kappa_shape=kappa_shape,
                boundary_type=int(boundary_type),
                motor_speed=motor_speed,
                motor_strength=motor_strength,
                motor_axis=motor_axis
            )

            # Mark vertices as ABD
            for vid in vertex_ids:
                self.is_abd_vertex[vid] = 1

        # Setup vertex mapping
        self.abd_system.setup_vertex_mapping(self.n_verts)

    def _estimate_volume(self, positions: np.ndarray) -> float:
        """Estimate volume from point cloud using bounding box."""
        min_p = np.min(positions, axis=0)
        max_p = np.max(positions, axis=0)
        extents = max_p - min_p
        return np.prod(extents)  # Bounding box volume as estimate

    @ti.kernel
    def assign_xn_xhat_hybrid(self):
        """
        Assign x_n and x_hat for hybrid simulation.

        For FEM vertices: standard inertia
        For ABD vertices: positions from ABD state
        """
        # FEM vertices
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                vert.x_n = vert.x
                vert.x_hat = vert.x + self.dt * vert.v
                vert.x_hat[1] += self.gravity * self.dt ** 2

        # ABD vertices are handled by ABD system

    def step(self):
        """
        Perform one simulation step with hybrid ABD-FEM.

        Returns:
            Number of iterations
        """
        print('Frame', self.frame)

        if self.use_abd and self.abd_system is not None:
            return self._step_hybrid()
        else:
            return super().step()

    def _step_hybrid(self):
        """
        Hybrid ABD-FEM step.

        1. Compute ABD predicted state (q_tilde)
        2. Standard FEM prediction (x_hat)
        3. Unified optimization loop
        4. Update velocities
        """
        # ABD: Compute predicted state
        self.abd_system.compute_q_tilde(self.dt)

        # FEM: Standard prediction
        self.assign_xn_xhat_hybrid()

        # Sync ABD positions to mesh
        self.abd_system.compute_x_from_q(self.mesh.verts.x)

        # Optimization loop
        for iter in range(self.iter_max):
            # Find contacts (uses current positions)
            self.find_cnts(PRINT=False)

            # Compute gradients
            self._compute_grad_hybrid()

            # Compute search direction
            if iter == 0:
                self._compute_init_p_hybrid()
            else:
                self._compute_DK_direction_hybrid()

            # Line search
            alpha, gTp, pHp = self._line_search_hybrid()

            # Check step size
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha_int = alpha
                alpha = 0.5 * self.dHat / p_max
                print('alpha clamped', alpha, 'alpha init', alpha_int)

            # Update positions
            self._update_hybrid(alpha)

            # Check convergence
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E

            if delta_E < self.epsilon * delta_E_init:
                print('converge at iter', iter, 'rate', delta_E / delta_E_init,
                      'delta_E', delta_E, 'alpha', alpha)
                break
            else:
                print('iter', iter, 'rate', delta_E / delta_E_init,
                      'delta_E', delta_E, 'alpha', alpha)

        # Update velocities
        self._update_velocity_hybrid()

        self.frame += 1
        return iter

    def _compute_grad_hybrid(self):
        """
        Compute gradient for hybrid ABD-FEM system.

        1. Compute standard FEM gradient (inertia + elastic + contact)
        2. Project contact gradients to ABD state space
        3. Add ABD inertia gradient
        4. Add ABD shape energy gradient
        5. Add motor constraint gradient (for MOTOR bodies)
        """
        # Standard gradient computation (affects all vertices)
        self.compute_grad_and_diagH()

        if self.ground_barrier == 1:
            self.add_grad_and_diagH_ground_barrier()

        # Project gradients to ABD state space
        self.abd_system.project_gradient_to_q(self.mesh.verts.grad)

        # Add ABD inertia gradient
        self.abd_system.add_inertia_gradient()

        # Add ABD shape energy gradient
        self.abd_system.add_shape_gradient()

        # Add motor constraint gradient for MOTOR bodies
        self.abd_system.add_motor_constraint_gradient()

    def _compute_init_p_hybrid(self):
        """Compute initial search direction for hybrid system."""
        # FEM: p = -grad / diagH
        self._compute_init_p_fem()

        # ABD: p = -M^{-1} @ grad_q
        self._compute_init_p_abd()

    @ti.kernel
    def _compute_init_p_fem(self):
        """Compute initial search direction for FEM vertices."""
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                vert.p = -vert.grad / vert.diagH

    @ti.kernel
    def _compute_init_p_abd(self):
        """Compute initial search direction for ABD bodies."""
        for body_id in range(self.abd_system.n_bodies):
            M_inv = self.abd_system.abd_mass_inv[body_id]
            grad_q = self.abd_system.grad_q[body_id]

            self.abd_system.dq[body_id] = M_inv @ grad_q

        # Map ABD dq to vertex p
        for i in range(self.abd_system.n_total_points):
            body_id = self.abd_system.point_body_id[i]
            x_bar = self.abd_system.x_bar[i]
            global_id = self.abd_system.global_vertex_id[i]
            dq = self.abd_system.dq[body_id]

            # p = -J @ dq (negative because dq is descent direction)
            p = ABDJacobian.apply_J(x_bar, dq)
            self.mesh.verts.p[global_id] = -ti.cast(p, ti.f32)

    def _compute_DK_direction_hybrid(self):
        """Compute Dai-Kai direction for hybrid system."""
        # FEM vertices
        self._compute_DK_direction_fem()

        # ABD bodies (simplified: just use gradient descent for now)
        self._compute_init_p_abd()

    @ti.kernel
    def _compute_DK_direction_fem(self):
        """Compute Dai-Kai direction for FEM vertices only."""
        g_p = 0.0  # g^{\top} p
        g_Py = 0.0  # g^{\top} P y
        y_p = 0.0
        y_Py = 0.0

        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p)

        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                y = vert.grad - vert.grad_prev
                Py = y / vert.diagH
                y_p += y.dot(vert.p)
                g_Py += vert.grad.dot(Py)
                y_Py += y.dot(Py)
                g_p += vert.grad.dot(vert.p)

        # Avoid division by zero
        if ti.abs(y_p) < 1e-10:
            y_p = 1e-10

        beta = (g_Py - y_Py * g_p / y_p) / y_p

        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                vert.p = -vert.grad / vert.diagH + beta * vert.p

    def _line_search_hybrid(self):
        """
        Compute line search for hybrid system.

        Includes shape energy Hessian and motor constraint Hessian
        contributions for ABD bodies.

        Returns:
            (alpha, gTp, pHp)
        """
        gTp = self._compute_gTp_hybrid()
        pHp = self._compute_pHp_hybrid()

        if self.ground_barrier == 1:
            pHp += self.add_pHp_ground_barrier()

        # Add ABD shape energy Hessian contribution
        pHp += self.abd_system.compute_shape_hessian_contribution(self.dt)

        # Add motor constraint Hessian contribution
        pHp += self.abd_system.compute_motor_hessian_contribution()

        alpha = -gTp / max(pHp, 1e-10)

        # Apply CCD for ABD bodies
        if self.abd_system is not None and self.abd_system.n_bodies > 0:
            alpha_ccd = self.abd_system.compute_ccd_step_size(
                ground_y=0.0, dHat=self.dHat)
            alpha = min(alpha, alpha_ccd)

        return alpha, gTp, pHp

    @ti.kernel
    def _compute_gTp_hybrid(self) -> float:
        """Compute g^T @ p for hybrid system."""
        result = 0.0

        # FEM contribution
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                result += vert.grad.dot(vert.p)

        # ABD contribution: grad_q^T @ dq
        for body_id in range(self.abd_system.n_bodies):
            grad_q = self.abd_system.grad_q[body_id]
            dq = self.abd_system.dq[body_id]
            for d in ti.static(range(12)):
                result += grad_q[d] * (-dq[d])  # Note: p = -dq

        return result

    @ti.kernel
    def _compute_pHp_hybrid(self) -> float:
        """Compute p^T @ H @ p for hybrid system."""
        result = 0.0

        # FEM: Inertia
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                result += vert.p.norm_sqr() * vert.m

        # FEM: Elastic (standard computation)
        for c in self.mesh.cells:
            # Check if any vertex is ABD
            all_fem = True
            for i in ti.static(range(4)):
                if self.is_abd_vertex[c.verts[i].id] == 1:
                    all_fem = False

            if all_fem:
                Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
                B = c.B
                F = Ds @ B
                d = ti.Vector.zero(float, 12)
                d[0:3] = c.verts[0].p
                d[3:6] = c.verts[1].p
                d[6:9] = c.verts[2].p
                d[9:12] = c.verts[3].p
                tmp = self.compute_p_d2Psidx2_p(F, B, d, self.mu, self.la)
                result += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        # ABD: Mass matrix contribution
        for body_id in range(self.abd_system.n_bodies):
            dq = self.abd_system.dq[body_id]
            M = self.abd_system.abd_mass[body_id]

            # p^T @ M @ p
            Mp = M @ dq
            for d in ti.static(range(12)):
                result += dq[d] * Mp[d]

        # IPC contact contribution (standard - uses vertex p)
        # Note: This works because ABD vertex p is set from J @ dq
        for k, j in self.cid:
            pair = self.cid[k, j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist

            bg = self.get_barrier_g(dist)
            bH = self.get_barrier_H(dist)

            para1 = bg / dist
            para0 = (bH - para1) / dist2

            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]

            from algorithm.collision_detection_bvh import compute_dtdx_t, compute_d_dtdx
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)
            p_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * p_dtdx.norm_sqr()
            result += ti.max(pHp_0 + pHp_1, 0.0)

        return result

    def _update_hybrid(self, alpha: float):
        """
        Update positions for hybrid system.

        Args:
            alpha: Step size
        """
        # Save ABD state for line search rollback
        self.abd_system.copy_q_to_temp()

        # FEM vertices
        self._update_fem_vertices(alpha)

        # ABD bodies
        self.abd_system.step_forward(alpha)

        # Sync ABD positions to mesh
        self.abd_system.compute_x_from_q(self.mesh.verts.x)

    @ti.kernel
    def _update_fem_vertices(self, alpha: float):
        """Update FEM vertex positions."""
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                vert.x = vert.x + alpha * vert.p

    def _update_velocity_hybrid(self):
        """Update velocities after optimization."""
        # FEM vertices
        self._update_fem_velocity()

        # ABD bodies
        self.abd_system.update_velocity(self.dt)

    @ti.kernel
    def _update_fem_velocity(self):
        """Update FEM vertex velocities."""
        for vert in self.mesh.verts:
            vid = vert.id
            if self.is_abd_vertex[vid] == 0:
                vert.v = (vert.x - vert.x_n) / self.dt

    # Override visual methods to support ABD
    def visual(self):
        """Visualization with ABD support."""
        if self.use_abd and self.abd_system is not None:
            # Ensure ABD positions are synced
            self.abd_system.compute_x_from_q(self.mesh.verts.x)
        super().visual()

    def run_headless(self, n_frames=300):
        """Headless run with ABD support."""
        print(f"Running hybrid ABD-FEM in headless mode for {n_frames} frames...")
        for i in range(n_frames):
            self.step()
        print("Headless run finished.")

    def get_abd_stats(self) -> dict:
        """Get ABD system statistics."""
        if self.abd_system is not None:
            return self.abd_system.get_stats()
        return {}


def create_abd_body_from_mesh_region(mesh, vertex_ids: list, kappa_shape: float = 1e6) -> dict:
    """
    Helper to create ABD body configuration from mesh region.

    Args:
        mesh: MeshTaichi mesh
        vertex_ids: List of vertex IDs to include
        kappa_shape: Shape preservation stiffness

    Returns:
        ABD body configuration dict
    """
    return {
        'vertex_ids': vertex_ids,
        'kappa_shape': kappa_shape
    }
