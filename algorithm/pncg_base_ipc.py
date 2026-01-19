"""
PNCG IPC solver using BVH-based collision detection.
This module replaces spatial hashing with LBVH for better performance.
"""

import time
from algorithm.collision_detection_bvh import *
from util.model_loading import *
from algorithm.mas_preconditioner import MASPreconditioner


@ti.data_oriented
class pncg_ipc_deformer(collision_detection_bvh_module):
    """
    PNCG IPC deformer using LBVH for collision detection.
    Provides the same interface as the original spatial hashing version.
    """

    def __init__(self, demo='cube_0'):
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.adj = model.adj
        self.ground_barrier = model.ground_barrier
        self.frame = 0
        self.SMALL_NUM = 1e-7

        # Initialize model
        self.mesh.verts.place({'x': ti.types.vector(3, float),
                               'v': ti.types.vector(3, float),
                               'm': float,
                               'x_n': ti.types.vector(3, float),
                               'x_hat': ti.types.vector(3, float),
                               'x_prev': ti.types.vector(3, float),
                               'x_init': ti.types.vector(3, float),
                               'grad': ti.types.vector(3, float),
                               'grad_prev': ti.types.vector(3, float),
                               'p': ti.types.vector(3, float),
                               'diagH': ti.types.vector(3, float),
                               })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print('n_verts,n_cells', self.n_verts, self.n_cells)

        # Precompute
        print('precompute!!')
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        # Boundary elements
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size ', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.set_point_lights()

        # IPC parameters
        print('init BVH structures')
        self.kappa = model.kappa
        self.dHat = model.dHat
        self.barrier_type = getattr(model, 'barrier_type', 'log')  # 'log' or 'cubic'
        self.adaptive_kappa = getattr(model, 'adaptive_kappa', False)  # Enable adaptive kappa
        self.cache_kappa = getattr(model, 'cache_kappa', True)  # Cache kappa at iter 0 (only when adaptive_kappa=True)
        self.init_bvh()  # Initialize BVH instead of hash grid
        print('dHat:', self.dHat, 'kappa:', self.kappa, 'barrier_type:', self.barrier_type,
              'adaptive_kappa:', self.adaptive_kappa, 'cache_kappa:', self.cache_kappa)

        self.config = model.dict
        self.config['dHat'] = self.dHat
        self.config['kappa'] = self.kappa
        self.config['barrier_type'] = self.barrier_type
        self.config['adaptive_kappa'] = self.adaptive_kappa
        self.config['cache_kappa'] = self.cache_kappa

        # MAS preconditioner setup
        self.use_mas_preconditioner = getattr(model, 'use_mas', False)
        if self.use_mas_preconditioner:
            print('Initializing MAS preconditioner...')
            # Add z field for preconditioned gradient
            self.mesh.verts.place({'z': ti.types.vector(3, float)})
            self.mas_preconditioner = MASPreconditioner(self.n_verts, self.n_cells, self.mesh)
            print('MAS preconditioner initialized')
        else:
            self.mas_preconditioner = None

    @ti.func
    def barrier_E(self, d):
        E = -self.kappa * (d - self.dHat) ** 2 * ti.log(d / self.dHat)
        return E

    @ti.func
    def barrier_g(self, d):
        t2 = d - self.dHat
        g = self.kappa * (t2 * ti.log(d / self.dHat) * (-2.0) - (t2 ** 2) / d)
        return g

    @ti.func
    def barrier_H(self, d):
        dHat = self.dHat
        H = self.kappa * ((-2) * ti.log(d / dHat) - 4 + 4 * dHat / d + (d - dHat) ** 2 / d ** 2)
        return H

    # Cubic barrier functions
    @ti.func
    def cubic_barrier_E(self, d):
        """Cubic barrier energy: ψ = -2κ/(3ĝ) * (g - ĝ)³"""
        E = 0.0
        if d < self.dHat:
            y = d - self.dHat  # y = g - ĝ (negative when d < dHat)
            E = -2.0 * self.kappa * (y * y * y) / (3.0 * self.dHat)
        return E

    @ti.func
    def cubic_barrier_g(self, d):
        """Cubic barrier gradient: ∂ψ/∂g = -2κ/ĝ * (g - ĝ)²"""
        g = 0.0
        if d < self.dHat:
            y = d - self.dHat
            g = -2.0 * self.kappa * (y * y) / self.dHat
        return g

    @ti.func
    def cubic_barrier_H(self, d):
        """Cubic barrier Hessian: ∂²ψ/∂g² = 4κ * (1 - g/ĝ)"""
        H = 0.0
        if d < self.dHat:
            H = 4.0 * self.kappa * (1.0 - d / self.dHat)
        return H

    # Adaptive kappa versions of cubic barrier functions
    @ti.func
    def cubic_barrier_E_adaptive(self, d, kappa_adaptive):
        """Cubic barrier energy with adaptive kappa: ψ = -2κ/(3ĝ) * (g - ĝ)³"""
        E = 0.0
        if d < self.dHat:
            y = d - self.dHat
            E = -2.0 * kappa_adaptive * (y * y * y) / (3.0 * self.dHat)
        return E

    @ti.func
    def cubic_barrier_g_adaptive(self, d, kappa_adaptive):
        """Cubic barrier gradient with adaptive kappa: ∂ψ/∂g = -2κ/ĝ * (g - ĝ)²"""
        g = 0.0
        if d < self.dHat:
            y = d - self.dHat
            g = -2.0 * kappa_adaptive * (y * y) / self.dHat
        return g

    @ti.func
    def cubic_barrier_H_adaptive(self, d, kappa_adaptive):
        """Cubic barrier Hessian with adaptive kappa: ∂²ψ/∂g² = 4κ * (1 - g/ĝ)"""
        H = 0.0
        if d < self.dHat:
            H = 4.0 * kappa_adaptive * (1.0 - d / self.dHat)
        return H

    @ti.func
    def compute_adaptive_kappa(self, ids: ti.types.vector(4, ti.u32),
                                cord: ti.types.vector(4, float),
                                t: ti.types.vector(3, float),
                                dist: float) -> float:
        """
        Compute elasticity-inclusive dynamic stiffness: κ̄ = m/g² + n·(H·n)

        Based on the cubic barrier paper (Ando 2024, Equation 4-5):
        - m: average vertex mass
        - g: gap distance
        - n: contact direction (normalized)
        - H: elasticity Hessian (we use diagonal approximation)

        The m/g² term ensures barrier becomes infinitely stiff as gap → 0.
        The n·(H·n) term includes elasticity to resist incoming forces.
        """
        # Compute average mass of involved vertices
        m_avg = 0.0
        for i in ti.static(range(4)):
            m_avg += self.mesh.verts.m[ids[i]] * ti.abs(cord[i])

        # Clamp distance to avoid division by zero
        g_clamped = ti.max(dist, 1e-8)

        # First term: m/g² (inertial contribution)
        kappa_inertia = m_avg / (g_clamped * g_clamped)

        # Second term: n·(H·n) (elasticity contribution)
        # We use the diagonal Hessian approximation for efficiency
        # w_i = cord[i] * t is the extended contact direction for vertex i
        # n·(H·n) ≈ Σ_i (w_i · diagH_i · w_i) / ||w||²
        kappa_elastic = 0.0
        w_norm_sq = 0.0
        for i in ti.static(range(4)):
            w_i = cord[i] * t  # Extended direction for vertex i
            diagH_i = self.mesh.verts.diagH[ids[i]]
            # w_i · (diagH · w_i)
            kappa_elastic += w_i[0] * diagH_i[0] * w_i[0] + \
                            w_i[1] * diagH_i[1] * w_i[1] + \
                            w_i[2] * diagH_i[2] * w_i[2]
            w_norm_sq += w_i.norm_sqr()

        # Normalize by ||w||² to get n·(H·n)
        if w_norm_sq > 1e-10:
            kappa_elastic = kappa_elastic / w_norm_sq

        # Total adaptive kappa: κ = n·(H·n) + m/g²
        kappa_adaptive = kappa_inertia + ti.max(kappa_elastic, 0.0)

        return kappa_adaptive

    # Unified dispatch functions
    @ti.func
    def get_barrier_E(self, d):
        E = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            E = self.cubic_barrier_E(d)
        else:
            E = self.barrier_E(d)
        return E

    @ti.func
    def get_barrier_g(self, d):
        g = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            g = self.cubic_barrier_g(d)
        else:
            g = self.barrier_g(d)
        return g

    @ti.func
    def get_barrier_H(self, d):
        H = 0.0
        if ti.static(self.barrier_type == 'cubic'):
            H = self.cubic_barrier_H(d)
        else:
            H = self.barrier_H(d)
        return H

    @ti.kernel
    def update_x_v2(self, alpha: float):
        for vert in self.mesh.verts:
            vert.x = vert.x_prev + alpha * vert.p

    @ti.kernel
    def compute_E(self) -> float:
        E = 0.0
        for vert in self.mesh.verts:
            E += 0.5 * vert.m * (vert.x - vert.x_hat).norm_sqr()
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            Psi = self.compute_Psi(F, self.mu, self.la)
            E += (self.dt ** 2) * c.W * Psi
        # Use compact array iteration (P0 optimization)
        for i in range(self.n_contacts[None]):
            pair = self.contact_pairs[i]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            # Use adaptive kappa if enabled (cubic barrier only)
            E_ipc = 0.0
            if ti.static(self.adaptive_kappa and self.barrier_type == 'cubic'):
                kappa_local = self.compute_adaptive_kappa(ids, cord, t, dist)
                E_ipc = self.cubic_barrier_E_adaptive(dist, kappa_local)
            else:
                E_ipc = self.get_barrier_E(dist)
            E += E_ipc
        return E

    @ti.kernel
    def compute_grad_and_diagH_inertia_elastic(self):
        """Compute gradient and diagH for inertia and elastic potentials only.
        This is called first to build diagH before IPC computation when using adaptive kappa.
        """
        ti.mesh_local(self.mesh.verts.grad)
        # Inertia potential
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)
        # Elastic potential
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp

    @ti.kernel
    def compute_grad_and_diagH_ipc(self):
        """Compute IPC potential contribution to gradient and diagH.
        Uses adaptive kappa if enabled, computed from current diagH.
        Uses compact array iteration (P0 optimization).
        """
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist ** 2

            # Compute adaptive kappa if enabled (cubic barrier only)
            if ti.static(self.adaptive_kappa and self.barrier_type == 'cubic'):
                kappa_local = self.compute_adaptive_kappa(ids, cord, t, dist)
                bg = self.cubic_barrier_g_adaptive(dist, kappa_local)
                bH = self.cubic_barrier_H_adaptive(dist, kappa_local)
            else:
                bg = self.get_barrier_g(dist)
                bH = self.get_barrier_H(dist)

            para = bg / dist
            para0 = (bH - para) / dist2
            for i in range(4):
                CORD = cord[i]
                ID = ids[i]
                self.mesh.verts.grad[ID] += para * CORD * t
                diag_tmp = CORD * CORD * (para0 * t * t + para * ti.Vector.one(float, 3))
                diag_tmp_spd = ti.max(diag_tmp, 0.0)
                self.mesh.verts.diagH[ID] += diag_tmp_spd

    @ti.kernel
    def compute_grad_and_diagH(self):
        # Inertia potential
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)
        # Elastic potential
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp
        # IPC potential - use compact array iteration (P0 optimization)
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist ** 2

            # Compute barrier gradient and Hessian (adaptive or fixed kappa)
            bg = 0.0
            bH = 0.0
            if ti.static(self.adaptive_kappa and self.barrier_type == 'cubic'):
                kappa_local = self.compute_adaptive_kappa(ids, cord, t, dist)
                bg = self.cubic_barrier_g_adaptive(dist, kappa_local)
                bH = self.cubic_barrier_H_adaptive(dist, kappa_local)
            else:
                bg = self.get_barrier_g(dist)
                bH = self.get_barrier_H(dist)

            para = bg / dist
            para0 = (bH - para) / dist2
            for i in range(4):
                CORD = cord[i]
                ID = ids[i]
                self.mesh.verts.grad[ID] += para * CORD * t
                diag_tmp = CORD * CORD * (para0 * t * t + para * ti.Vector.one(float, 3))
                diag_tmp_spd = ti.max(diag_tmp, 0.0)
                self.mesh.verts.diagH[ID] += diag_tmp_spd

    @ti.kernel
    def compute_pHp(self) -> float:
        ret = 0.0
        for vert in self.mesh.verts:
            ret += vert.p.norm_sqr() * vert.m
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            d = ti.Vector.zero(float, 12)
            d[0:3] = c.verts[0].p
            d[3:6] = c.verts[1].p
            d[6:9] = c.verts[2].p
            d[9:12] = c.verts[3].p
            tmp = self.compute_p_d2Psidx2_p(F, B, d, self.mu, self.la)
            ret += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        # Use compact array iteration (P0 optimization)
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist

            # Use adaptive kappa if enabled (cubic barrier only)
            bg = 0.0
            bH = 0.0
            if ti.static(self.adaptive_kappa and self.barrier_type == 'cubic'):
                kappa_local = self.compute_adaptive_kappa(ids, cord, t, dist)
                bg = self.cubic_barrier_g_adaptive(dist, kappa_local)
                bH = self.cubic_barrier_H_adaptive(dist, kappa_local)
            else:
                bg = self.get_barrier_g(dist)
                bH = self.get_barrier_H(dist)

            para1 = bg / dist
            para0 = (bH - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)
            d_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * d_dtdx.norm_sqr()
            pHp = pHp_0 + pHp_1
            ret += ti.max(pHp, 0.0)
        return ret

    @ti.kernel
    def add_E_ground_barrier(self) -> float:
        E = 0.0
        min_dist = 1e-2 * self.dHat
        for i in range(self.n_boundary_points):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist < min_dist:
                    dist = min_dist
                E += self.get_barrier_E(dist)
        return E

    @ti.kernel
    def add_grad_and_diagH_ground_barrier(self):
        min_dist = 1e-2 * self.dHat
        for i in range(self.n_boundary_points):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    self.mesh.verts.x[p][1] = self.ground + min_dist
                    dist = min_dist
                self.mesh.verts.grad[p][1] += self.get_barrier_g(dist)
                self.mesh.verts.diagH[p][1] += self.get_barrier_H(dist)

    @ti.kernel
    def add_pHp_ground_barrier(self) -> float:
        ret = 0.0
        min_dist = 1e-2 * self.dHat
        for i in range(self.boundary_points.shape[0]):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    dist = min_dist
                p_tmp = self.mesh.verts.p[p][1]
                ret_value = p_tmp * self.get_barrier_H(dist) * p_tmp
                ret += ret_value
        return ret

    def line_search_newton(self):
        gTp = self.compute_gTp()
        pHp = self.compute_pHp()
        if self.ground_barrier == 1:
            pHp_ground = self.add_pHp_ground_barrier()
            pHp += pHp_ground
        alpha = -gTp / pHp
        return alpha, gTp, pHp

    @ti.kernel
    def compute_p_inf_norm(self) -> float:
        p_max = 0.0
        for vert in self.mesh.verts:
            p_norm = vert.p.norm()
            ti.atomic_max(p_max, p_norm)
        return p_max

    def step(self):
        print('Frame', self.frame)
        self.assign_xn_xhat()
        for iter in range(self.iter_max):
            self.find_cnts(PRINT=False)
            self.compute_grad_and_diagH()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()

            # Apply MAS preconditioner if enabled
            if self.use_mas_preconditioner and self.mas_preconditioner is not None:
                if iter == 0:
                    # Full rebuild on first iteration of each frame
                    self.mas_preconditioner.rebuild(self)
                # Apply preconditioner: z = P * grad
                self.mas_preconditioner.apply()
                # Compute search direction using preconditioned gradient
                if iter == 0:
                    self.compute_init_p_mas()
                else:
                    self.compute_DK_direction_mas()
            else:
                # Use diagonal preconditioner (original behavior)
                if iter == 0:
                    self.compute_init_p()
                else:
                    self.compute_DK_direction()

            alpha, gTp, pHp = self.line_search_newton()
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha_int = alpha
                alpha = 0.5 * self.dHat / p_max
                print('alpha clamped', alpha, 'alpha init', alpha_int)
            self.update_x(alpha)
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
                      'gTp', gTp, 'pHp', pHp)
                break
            else:
                print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp,
                      'pHp', pHp)

        self.update_v_and_bound()
        self.frame += 1
        return iter

    @ti.kernel
    def compute_init_p_mas(self):
        """Compute initial search direction using MAS preconditioned gradient."""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_DK_direction_mas(self):
        """Compute Dai-Kai direction using MAS preconditioned gradient."""
        g_p = 0.0  # g^{\top} p
        g_Pz = 0.0  # g^{\top} z (P*y approx)
        y_p = 0.0
        y_z = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.z, self.mesh.verts.p)
        for vert in self.mesh.verts:
            y = vert.grad - vert.grad_prev
            # Using z as P*g, approximate P*y ~ z_new - z_old
            # For simplicity, we use y/diagH as an approximation
            Py = y / vert.diagH
            y_p += y.dot(vert.p)
            g_Pz += vert.grad.dot(Py)
            y_z += y.dot(Py)
            g_p += vert.grad.dot(vert.p)
        beta = (g_Pz - y_z * g_p / y_p) / y_p
        for vert in self.mesh.verts:
            vert.p = -vert.z + beta * vert.p

    @ti.kernel
    def line_search_clamped_newton(self, rate: float) -> (float, float, float):
        """Compute newton line search, then clamp alpha with max displacement 0.5 dHat"""
        pHp = 0.0
        gTp = 0.0
        p_max = 0.0
        for vert in self.mesh.verts:
            gTp += vert.grad.dot(vert.p)
            pHp += vert.p.norm_sqr() * vert.m
            p_norm = vert.p.norm()
            ti.atomic_max(p_max, p_norm)

        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            p = ti.Vector.zero(float, 12)
            p[0:3] = c.verts[0].p
            p[3:6] = c.verts[1].p
            p[6:9] = c.verts[2].p
            p[9:12] = c.verts[3].p
            tmp = self.compute_p_d2Psidx2_p(F, B, p, self.mu, self.la)
            pHp += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        # Use compact array iteration (P0 optimization)
        for idx in range(self.n_contacts[None]):
            pair = self.contact_pairs[idx]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist

            # Use adaptive kappa if enabled (cubic barrier only)
            bg = 0.0
            bH = 0.0
            if ti.static(self.adaptive_kappa and self.barrier_type == 'cubic'):
                kappa_local = self.compute_adaptive_kappa(ids, cord, t, dist)
                bg = self.cubic_barrier_g_adaptive(dist, kappa_local)
                bH = self.cubic_barrier_H_adaptive(dist, kappa_local)
            else:
                bg = self.get_barrier_g(dist)
                bH = self.get_barrier_H(dist)

            para1 = bg / dist
            para0 = (bH - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * (p_tmp.dot(dtdx_t) ** 2)
            p_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * p_dtdx.norm_sqr()
            pHp += ti.max(pHp_0 + pHp_1, 0.0)

        alpha = -gTp / pHp
        if alpha * p_max > rate * self.dHat:
            alpha_int = alpha
            alpha = rate * self.dHat / p_max
            print('alpha clamped', alpha, 'alpha init', alpha_int)
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p
        return (alpha, gTp, pHp)

    def step_dirichlet(self):
        print('Frame', self.frame)
        self.assign_xn_xhat()
        for iter in range(self.iter_max):
            self.find_cnts(PRINT=False)
            self.compute_grad_and_diagH()
            self.dirichlet_grad()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_direction()
            alpha, gTp, pHp = self.line_search_newton()
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha_int = alpha
                alpha = 0.5 * self.dHat / p_max
                print('alpha clamped', alpha, 'alpha init', alpha_int)
            self.update_x(alpha)
            delta_E = -alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
                      'gTp', gTp, 'pHp', pHp)
                break
            else:
                print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp,
                      'pHp', pHp)

        self.update_v_and_bound()
        self.frame += 1
        return iter

    @ti.kernel
    def check_inverse(self) -> int:
        """Check if there's any inversion."""
        ret = 0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            J = F.determinant()
            if J < -1e-6:
                ret = 1
        return ret

    def run_headless(self, n_frames=300):
        print(f"Running in headless mode for {n_frames} frames...")
        for i in range(n_frames):
            self.step()
        print("Headless run finished.")
