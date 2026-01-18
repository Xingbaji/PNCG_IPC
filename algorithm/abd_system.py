"""
Affine Body Dynamics (ABD) System for PNCG-IPC solver.

This module implements ABD from Stiff-GIPC, adapted for Taichi/MeshTaichi.
ABD represents deformable bodies using a compact 12D state vector:
    q = [p; a1; a2; a3]^T
where p is the center of mass and A = [a1 a2 a3]^T is the affine transformation.

All vertex positions are computed as: x_i = p + A * x̄_i
where x̄_i is the rest-frame position relative to center of mass.

Reference: Stiff-GIPC (CUDA implementation)
"""

import taichi as ti
import numpy as np

# ABD constants
ABD_STATE_DIM = 12  # 3 (position) + 9 (affine matrix)


@ti.data_oriented
class ABDJacobian:
    """
    Jacobian matrix for ABD transformation.

    The Jacobian J is a 3x12 matrix that maps the 12D state to 3D position:
        x = J * q

    For a point with rest position x̄ = (x̄, ȳ, z̄), the Jacobian is:
        J = [I₃ | x̄*I₃ | ȳ*I₃ | z̄*I₃]

    where I₃ is the 3x3 identity matrix.
    """

    @staticmethod
    @ti.func
    def compute_jacobian(x_bar: ti.types.vector(3, ti.f64)) -> ti.types.matrix(3, 12, ti.f64):
        """
        Compute 3x12 Jacobian matrix for a point.

        Args:
            x_bar: Rest position relative to center of mass (3D vector)

        Returns:
            J: 3x12 Jacobian matrix
        """
        J = ti.Matrix.zero(ti.f64, 3, 12)

        # First block: identity (translation part)
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[2, 2] = 1.0

        # Second block: x̄ * I₃ (first column of A)
        J[0, 3] = x_bar[0]
        J[1, 4] = x_bar[0]
        J[2, 5] = x_bar[0]

        # Third block: ȳ * I₃ (second column of A)
        J[0, 6] = x_bar[1]
        J[1, 7] = x_bar[1]
        J[2, 8] = x_bar[1]

        # Fourth block: z̄ * I₃ (third column of A)
        J[0, 9] = x_bar[2]
        J[1, 10] = x_bar[2]
        J[2, 11] = x_bar[2]

        return J

    @staticmethod
    @ti.func
    def apply_J(x_bar: ti.types.vector(3, ti.f64),
                q: ti.types.vector(12, ti.f64)) -> ti.types.vector(3, ti.f64):
        """
        Apply Jacobian: x = J * q (state to position).

        More efficient than explicit matrix multiplication.

        Args:
            x_bar: Rest position relative to center of mass
            q: 12D state vector [p; a1; a2; a3]

        Returns:
            x: 3D world position
        """
        # Extract components
        p = ti.Vector([q[0], q[1], q[2]])

        # A is stored column-major: [a1; a2; a3] where each ai is 3D
        # q[3:6] = a1, q[6:9] = a2, q[9:12] = a3
        # A @ x_bar = x_bar[0]*a1 + x_bar[1]*a2 + x_bar[2]*a3
        Ax = ti.Vector([
            q[3] * x_bar[0] + q[6] * x_bar[1] + q[9] * x_bar[2],
            q[4] * x_bar[0] + q[7] * x_bar[1] + q[10] * x_bar[2],
            q[5] * x_bar[0] + q[8] * x_bar[1] + q[11] * x_bar[2]
        ])

        return p + Ax

    @staticmethod
    @ti.func
    def apply_JT(x_bar: ti.types.vector(3, ti.f64),
                 g: ti.types.vector(3, ti.f64)) -> ti.types.vector(12, ti.f64):
        """
        Apply Jacobian transpose: g_q = J^T * g (gradient projection).

        Projects a 3D gradient to 12D state space.

        Args:
            x_bar: Rest position relative to center of mass
            g: 3D gradient in world space

        Returns:
            g_q: 12D gradient in state space
        """
        g_q = ti.Vector.zero(ti.f64, 12)

        # Translation part: J^T[:3, :] = I₃
        g_q[0] = g[0]
        g_q[1] = g[1]
        g_q[2] = g[2]

        # Affine part: J^T[3:, :] = [x̄*I₃; ȳ*I₃; z̄*I₃]^T
        # a1 gradient
        g_q[3] = x_bar[0] * g[0]
        g_q[4] = x_bar[0] * g[1]
        g_q[5] = x_bar[0] * g[2]

        # a2 gradient
        g_q[6] = x_bar[1] * g[0]
        g_q[7] = x_bar[1] * g[1]
        g_q[8] = x_bar[1] * g[2]

        # a3 gradient
        g_q[9] = x_bar[2] * g[0]
        g_q[10] = x_bar[2] * g[1]
        g_q[11] = x_bar[2] * g[2]

        return g_q

    @staticmethod
    @ti.func
    def apply_JT_H_J(x_bar: ti.types.vector(3, ti.f64),
                     H: ti.types.matrix(3, 3, ti.f64)) -> ti.types.matrix(12, 12, ti.f64):
        """
        Compute J^T * H * J for Hessian transformation.

        Args:
            x_bar: Rest position relative to center of mass
            H: 3x3 Hessian in world space

        Returns:
            H_q: 12x12 Hessian in state space
        """
        H_q = ti.Matrix.zero(ti.f64, 12, 12)

        # Build J explicitly for full transformation
        J = ABDJacobian.compute_jacobian(x_bar)

        # Compute J^T @ H @ J
        JT_H = J.transpose() @ H  # 12x3
        H_q = JT_H @ J  # 12x12

        return H_q


@ti.data_oriented
class ABDBody:
    """
    Single ABD body representation.

    Stores the 12D state and provides methods for:
    - State to position mapping
    - Gradient projection
    - Mass matrix computation
    - Shape energy (penalizes deviation from rigid motion)
    """

    def __init__(self, body_id: int, n_points: int, point_ids: np.ndarray,
                 rest_positions: np.ndarray, masses: np.ndarray):
        """
        Initialize an ABD body.

        Args:
            body_id: Unique identifier for this body
            n_points: Number of points in this body
            point_ids: Global vertex IDs belonging to this body
            rest_positions: Rest positions of all points (n_points x 3)
            masses: Mass of each point (n_points,)
        """
        self.body_id = body_id
        self.n_points = n_points

        # Compute center of mass and relative positions
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass
        x_bar = rest_positions - com  # Relative positions

        # Store in Taichi fields
        self.point_ids = ti.field(dtype=ti.i32, shape=n_points)
        self.point_ids.from_numpy(point_ids.astype(np.int32))

        self.x_bar = ti.Vector.field(3, dtype=ti.f64, shape=n_points)
        self.x_bar.from_numpy(x_bar)

        self.point_mass = ti.field(dtype=ti.f64, shape=n_points)
        self.point_mass.from_numpy(masses)

        # State vectors (12D)
        self.q = ti.Vector.field(12, dtype=ti.f64, shape=())  # Current state
        self.q_prev = ti.Vector.field(12, dtype=ti.f64, shape=())  # Previous state
        self.q_tilde = ti.Vector.field(12, dtype=ti.f64, shape=())  # Predicted state
        self.q_v = ti.Vector.field(12, dtype=ti.f64, shape=())  # Velocity
        self.dq = ti.Vector.field(12, dtype=ti.f64, shape=())  # Search direction
        self.grad_q = ti.Vector.field(12, dtype=ti.f64, shape=())  # Gradient

        # Mass matrix (12x12)
        self.abd_mass = ti.Matrix.field(12, 12, dtype=ti.f64, shape=())
        self.abd_mass_inv = ti.Matrix.field(12, 12, dtype=ti.f64, shape=())

        # Properties
        self.total_mass = total_mass
        self.volume = 0.0  # Set from cells
        self.com_rest = com  # Rest center of mass

        # Shape energy stiffness
        self.kappa_shape = 1e6  # Stiffness for shape preservation

        # Initialize state to identity transformation at COM
        self._init_state(com)

        # Compute mass matrix
        self._compute_mass_matrix()

    def _init_state(self, com: np.ndarray):
        """Initialize state to identity transformation at center of mass."""
        q_init = np.zeros(12)
        q_init[0:3] = com  # Position at COM
        # Identity affine matrix (column-major storage)
        q_init[3] = 1.0   # a1 = (1, 0, 0)
        q_init[7] = 1.0   # a2 = (0, 1, 0)
        q_init[11] = 1.0  # a3 = (0, 0, 1)

        self.q.from_numpy(q_init)
        self.q_prev.from_numpy(q_init)
        self.q_tilde.from_numpy(q_init)
        self.q_v.from_numpy(np.zeros(12))

    @ti.kernel
    def _compute_mass_matrix_kernel(self):
        """
        Compute 12x12 ABD mass matrix.

        M = Σ_i m_i * J_i^T @ J_i

        where J_i is the Jacobian for point i.
        """
        M = ti.Matrix.zero(ti.f64, 12, 12)

        for i in range(self.n_points):
            m = self.point_mass[i]
            x_bar = self.x_bar[i]

            # Compute J^T @ J contribution
            # J = [I₃ | x̄*I₃ | ȳ*I₃ | z̄*I₃]
            # J^T @ J is block diagonal with specific structure

            # Translation-translation block (3x3)
            for d in ti.static(range(3)):
                M[d, d] += m

            # Translation-affine cross terms
            for d in ti.static(range(3)):
                # p-a1 coupling
                M[d, 3 + d] += m * x_bar[0]
                M[3 + d, d] += m * x_bar[0]
                # p-a2 coupling
                M[d, 6 + d] += m * x_bar[1]
                M[6 + d, d] += m * x_bar[1]
                # p-a3 coupling
                M[d, 9 + d] += m * x_bar[2]
                M[9 + d, d] += m * x_bar[2]

            # Affine-affine blocks
            for d in ti.static(range(3)):
                # a1-a1 block
                M[3 + d, 3 + d] += m * x_bar[0] * x_bar[0]
                # a2-a2 block
                M[6 + d, 6 + d] += m * x_bar[1] * x_bar[1]
                # a3-a3 block
                M[9 + d, 9 + d] += m * x_bar[2] * x_bar[2]

                # a1-a2 coupling
                M[3 + d, 6 + d] += m * x_bar[0] * x_bar[1]
                M[6 + d, 3 + d] += m * x_bar[0] * x_bar[1]
                # a1-a3 coupling
                M[3 + d, 9 + d] += m * x_bar[0] * x_bar[2]
                M[9 + d, 3 + d] += m * x_bar[0] * x_bar[2]
                # a2-a3 coupling
                M[6 + d, 9 + d] += m * x_bar[1] * x_bar[2]
                M[9 + d, 6 + d] += m * x_bar[1] * x_bar[2]

        self.abd_mass[None] = M

    def _compute_mass_matrix(self):
        """Compute and invert mass matrix."""
        self._compute_mass_matrix_kernel()

        # Invert on CPU (12x12 is small)
        M = self.abd_mass.to_numpy()[0]

        # Add small regularization for numerical stability
        M += np.eye(12) * 1e-10

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print(f"[ABD] Warning: Mass matrix singular for body {self.body_id}, using pseudo-inverse")
            M_inv = np.linalg.pinv(M)

        self.abd_mass_inv.from_numpy(M_inv.reshape(1, 12, 12))


@ti.data_oriented
class ABDSystem:
    """
    Complete ABD system managing multiple ABD bodies.

    Provides:
    - Body management and state storage
    - Batch position/gradient computation
    - Integration with FEM solver
    - Shape energy computation
    """

    def __init__(self, max_bodies: int = 64, max_points_per_body: int = 10000):
        """
        Initialize ABD system.

        Args:
            max_bodies: Maximum number of ABD bodies
            max_points_per_body: Maximum points per body
        """
        self.max_bodies = max_bodies
        self.max_points_per_body = max_points_per_body
        self.max_total_points = max_bodies * max_points_per_body

        # Number of active bodies
        self.n_bodies = 0

        # Body state vectors (batched for GPU efficiency)
        self.q = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)
        self.q_prev = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)
        self.q_tilde = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)
        self.q_v = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)
        self.dq = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)
        self.grad_q = ti.Vector.field(12, dtype=ti.f64, shape=max_bodies)

        # Mass matrices
        self.abd_mass = ti.Matrix.field(12, 12, dtype=ti.f64, shape=max_bodies)
        self.abd_mass_inv = ti.Matrix.field(12, 12, dtype=ti.f64, shape=max_bodies)

        # Body properties
        self.body_volume = ti.field(dtype=ti.f64, shape=max_bodies)
        self.kappa_shape = ti.field(dtype=ti.f64, shape=max_bodies)

        # Point data (flattened storage)
        self.point_body_id = ti.field(dtype=ti.i32, shape=self.max_total_points)
        self.point_local_id = ti.field(dtype=ti.i32, shape=self.max_total_points)
        self.x_bar = ti.Vector.field(3, dtype=ti.f64, shape=self.max_total_points)
        self.point_mass = ti.field(dtype=ti.f64, shape=self.max_total_points)

        # Body point ranges [start, end)
        self.body_point_start = ti.field(dtype=ti.i32, shape=max_bodies + 1)

        # Global vertex ID mapping (abd_point_idx -> global_vertex_id)
        self.global_vertex_id = ti.field(dtype=ti.i32, shape=self.max_total_points)

        # Reverse mapping (global_vertex_id -> abd_point_idx, -1 if not ABD)
        self.vertex_to_abd_point = None  # Set when mesh is known

        # Total ABD points
        self.n_total_points = 0

        # Simulation parameters
        self.dt = 0.01
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        # Initialize
        self._init_fields()

    @ti.kernel
    def _init_fields(self):
        """Initialize fields to zero/identity."""
        for i in range(self.max_bodies):
            # Identity state
            self.q[i] = ti.Vector([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dt=ti.f64)
            self.q_prev[i] = self.q[i]
            self.q_tilde[i] = self.q[i]
            self.q_v[i] = ti.Vector.zero(ti.f64, 12)
            self.dq[i] = ti.Vector.zero(ti.f64, 12)
            self.grad_q[i] = ti.Vector.zero(ti.f64, 12)

            self.abd_mass[i] = ti.Matrix.identity(ti.f64, 12)
            self.abd_mass_inv[i] = ti.Matrix.identity(ti.f64, 12)

            self.body_volume[i] = 1.0
            self.kappa_shape[i] = 1e6

    def add_body(self, point_ids: np.ndarray, rest_positions: np.ndarray,
                 masses: np.ndarray, volume: float = 1.0, kappa_shape: float = 1e6) -> int:
        """
        Add an ABD body to the system.

        Args:
            point_ids: Global vertex IDs belonging to this body
            rest_positions: Rest positions (n_points x 3)
            masses: Point masses (n_points,)
            volume: Body volume (for shape energy scaling)
            kappa_shape: Shape preservation stiffness

        Returns:
            body_id: Assigned body ID
        """
        if self.n_bodies >= self.max_bodies:
            raise RuntimeError(f"Maximum number of ABD bodies ({self.max_bodies}) exceeded")

        n_points = len(point_ids)
        if self.n_total_points + n_points > self.max_total_points:
            raise RuntimeError(f"Maximum total ABD points ({self.max_total_points}) exceeded")

        body_id = self.n_bodies

        # Compute center of mass
        total_mass = np.sum(masses)
        com = np.sum(rest_positions * masses[:, np.newaxis], axis=0) / total_mass
        x_bar = rest_positions - com

        # Store point data
        start_idx = self.n_total_points
        end_idx = start_idx + n_points

        self.body_point_start[body_id] = start_idx
        self.body_point_start[body_id + 1] = end_idx

        # Upload point data
        for i, (pid, xb, m) in enumerate(zip(point_ids, x_bar, masses)):
            idx = start_idx + i
            self.point_body_id[idx] = body_id
            self.point_local_id[idx] = i
            self.x_bar[idx] = xb.tolist()
            self.point_mass[idx] = m
            self.global_vertex_id[idx] = int(pid)

        # Initialize state (identity at COM)
        q_init = np.zeros(12)
        q_init[0:3] = com
        q_init[3] = 1.0  # Identity affine
        q_init[7] = 1.0
        q_init[11] = 1.0

        self.q[body_id] = q_init.tolist()
        self.q_prev[body_id] = q_init.tolist()
        self.q_tilde[body_id] = q_init.tolist()

        # Compute mass matrix
        self._compute_body_mass_matrix(body_id, start_idx, end_idx)

        # Set properties
        self.body_volume[body_id] = volume
        self.kappa_shape[body_id] = kappa_shape

        self.n_bodies += 1
        self.n_total_points = end_idx

        print(f"[ABD] Added body {body_id}: {n_points} points, COM = {com}")

        return body_id

    def _compute_body_mass_matrix(self, body_id: int, start_idx: int, end_idx: int):
        """Compute mass matrix for a single body."""
        n_points = end_idx - start_idx

        # Gather data
        x_bar_np = np.zeros((n_points, 3))
        masses_np = np.zeros(n_points)

        for i in range(n_points):
            idx = start_idx + i
            x_bar_np[i] = [self.x_bar[idx][j] for j in range(3)]
            masses_np[i] = self.point_mass[idx]

        # Compute M = Σ m_i J_i^T J_i
        M = np.zeros((12, 12))

        for i in range(n_points):
            m = masses_np[i]
            xb = x_bar_np[i]

            # Build J (3x12)
            J = np.zeros((3, 12))
            J[0, 0] = J[1, 1] = J[2, 2] = 1.0
            J[0, 3] = J[1, 4] = J[2, 5] = xb[0]
            J[0, 6] = J[1, 7] = J[2, 8] = xb[1]
            J[0, 9] = J[1, 10] = J[2, 11] = xb[2]

            M += m * J.T @ J

        # Add regularization
        M += np.eye(12) * 1e-10

        # Invert
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # Store
        self.abd_mass[body_id] = M.tolist()
        self.abd_mass_inv[body_id] = M_inv.tolist()

    def setup_vertex_mapping(self, n_total_verts: int):
        """
        Setup reverse mapping from global vertex ID to ABD point index.

        Args:
            n_total_verts: Total number of vertices in the mesh
        """
        self.vertex_to_abd_point = ti.field(dtype=ti.i32, shape=n_total_verts)
        self.vertex_to_abd_point.fill(-1)  # -1 means not an ABD point

        # Fill mapping
        for i in range(self.n_total_points):
            global_id = self.global_vertex_id[i]
            self.vertex_to_abd_point[global_id] = i

    @ti.kernel
    def compute_x_from_q(self, vertices: ti.template()):
        """
        Compute vertex positions from ABD states.

        Args:
            vertices: Vertex position field to update (mesh.verts.x or similar)
        """
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            q = self.q[body_id]
            global_id = self.global_vertex_id[i]

            # x = p + A @ x_bar
            x = ABDJacobian.apply_J(x_bar, q)

            vertices[global_id] = ti.cast(x, ti.f32)

    @ti.kernel
    def compute_q_tilde(self, dt: ti.f64):
        """
        Compute predicted state: q̃ = q + dt*q_v + dt²*M⁻¹*f_ext

        Args:
            dt: Time step
        """
        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            q_v = self.q_v[body_id]
            M_inv = self.abd_mass_inv[body_id]

            # External force (gravity on translation part)
            f_ext = ti.Vector.zero(ti.f64, 12)

            # Compute total mass for gravity
            total_mass = 0.0
            start = self.body_point_start[body_id]
            end = self.body_point_start[body_id + 1]
            for i in range(start, end):
                total_mass += self.point_mass[i]

            # Gravity acts on translation DOFs
            f_ext[0] = total_mass * self.gravity[0]
            f_ext[1] = total_mass * self.gravity[1]
            f_ext[2] = total_mass * self.gravity[2]

            # q̃ = q + dt*v + dt²*M⁻¹*f
            q_tilde = q + dt * q_v + dt * dt * (M_inv @ f_ext)

            self.q_tilde[body_id] = q_tilde
            self.q_prev[body_id] = q

    @ti.kernel
    def project_gradient_to_q(self, grad_x: ti.template()):
        """
        Project vertex gradients to ABD state gradient.

        g_q = Σ_i J_i^T @ g_x[i]

        Args:
            grad_x: Per-vertex gradient field (mesh.verts.grad)
        """
        # Clear gradients
        for body_id in range(self.n_bodies):
            self.grad_q[body_id] = ti.Vector.zero(ti.f64, 12)

        # Accumulate
        for i in range(self.n_total_points):
            body_id = self.point_body_id[i]
            x_bar = self.x_bar[i]
            global_id = self.global_vertex_id[i]

            g_x = ti.cast(grad_x[global_id], ti.f64)
            g_q = ABDJacobian.apply_JT(x_bar, g_x)

            # Atomic add to body gradient
            for d in ti.static(range(12)):
                ti.atomic_add(self.grad_q[body_id][d], g_q[d])

    def compute_shape_energy(self) -> float:
        """
        Compute shape-preserving energy for all ABD bodies.

        V_shape = κ * v * ||A*A^T - I||_F²

        This penalizes non-rigid deformations.

        Returns:
            Total shape energy
        """
        return self._compute_shape_energy_kernel()

    @ti.kernel
    def _compute_shape_energy_kernel(self) -> ti.f64:
        """Kernel for shape energy computation."""
        E_shape = 0.0

        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]

            # Extract A from q (column-major)
            A = ti.Matrix([
                [q[3], q[6], q[9]],
                [q[4], q[7], q[10]],
                [q[5], q[8], q[11]]
            ], dt=ti.f64)

            # C = A @ A^T - I
            AAT = A @ A.transpose()
            C = AAT - ti.Matrix.identity(ti.f64, 3)

            # ||C||_F² = trace(C^T @ C)
            frob_sq = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    frob_sq += C[i, j] * C[i, j]

            E_shape += kappa * vol * frob_sq

        return E_shape

    @ti.kernel
    def add_shape_gradient(self):
        """
        Add shape energy gradient to grad_q.

        ∂V/∂A = 4κv * A * (A^T*A - I)
        """
        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            kappa = self.kappa_shape[body_id]
            vol = self.body_volume[body_id]

            # Extract A
            A = ti.Matrix([
                [q[3], q[6], q[9]],
                [q[4], q[7], q[10]],
                [q[5], q[8], q[11]]
            ], dt=ti.f64)

            # ∂V/∂A = 4κv * A * (A^T*A - I)
            ATA = A.transpose() @ A
            C = ATA - ti.Matrix.identity(ti.f64, 3)
            dVdA = 4.0 * kappa * vol * (A @ C)

            # Add to gradient (column-major storage)
            self.grad_q[body_id][3] += dVdA[0, 0]
            self.grad_q[body_id][4] += dVdA[1, 0]
            self.grad_q[body_id][5] += dVdA[2, 0]

            self.grad_q[body_id][6] += dVdA[0, 1]
            self.grad_q[body_id][7] += dVdA[1, 1]
            self.grad_q[body_id][8] += dVdA[2, 1]

            self.grad_q[body_id][9] += dVdA[0, 2]
            self.grad_q[body_id][10] += dVdA[1, 2]
            self.grad_q[body_id][11] += dVdA[2, 2]

    @ti.kernel
    def update_velocity(self, dt: ti.f64):
        """
        Update velocity after optimization: q_v = (q - q_prev) / dt

        Args:
            dt: Time step
        """
        for body_id in range(self.n_bodies):
            q = self.q[body_id]
            q_prev = self.q_prev[body_id]

            self.q_v[body_id] = (q - q_prev) / dt

    @ti.kernel
    def step_forward(self, alpha: ti.f64):
        """
        Take optimization step: q = q - alpha * dq

        Args:
            alpha: Step size
        """
        for body_id in range(self.n_bodies):
            self.q[body_id] = self.q[body_id] - alpha * self.dq[body_id]

    def get_stats(self) -> dict:
        """Return statistics about the ABD system."""
        return {
            'n_bodies': self.n_bodies,
            'n_total_points': self.n_total_points,
            'max_bodies': self.max_bodies,
        }


# Utility functions for ABD body detection
def detect_rigid_components(mesh_positions: np.ndarray,
                           mesh_cells: np.ndarray,
                           rigidity_threshold: float = 0.1) -> list:
    """
    Detect connected components that could be treated as ABD bodies.

    This is a placeholder for more sophisticated rigid body detection.
    In practice, users would specify which bodies are ABD vs FEM.

    Args:
        mesh_positions: Vertex positions (N x 3)
        mesh_cells: Cell connectivity (M x 4 for tets)
        rigidity_threshold: Threshold for considering a component rigid

    Returns:
        List of (vertex_ids, is_rigid) tuples for each component
    """
    # Simple connected component detection
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_verts = len(mesh_positions)
    n_cells = len(mesh_cells)

    # Build adjacency matrix
    rows = []
    cols = []
    for cell in mesh_cells:
        for i in range(4):
            for j in range(i + 1, 4):
                rows.extend([cell[i], cell[j]])
                cols.extend([cell[j], cell[i]])

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts))

    # Find connected components
    n_components, labels = connected_components(adj, directed=False)

    components = []
    for comp_id in range(n_components):
        vertex_ids = np.where(labels == comp_id)[0]
        # For now, assume all components are deformable (FEM)
        # Users would specify ABD bodies explicitly
        components.append((vertex_ids, False))

    return components
