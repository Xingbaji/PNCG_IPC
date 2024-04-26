from algorithm.collision_detection_v2 import *
from util.model_loading import *

@ti.data_oriented
class pncg_ipc_deformer(collision_detection_module_v2):
    def     __init__(self, demo = 'cube_0'):
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

        # initialize model
        self.mesh.verts.place({'x': ti.types.vector(3,float),
                               'v': ti.types.vector(3,float),
                               'm': float,
                               'x_n': ti.types.vector(3,float),
                               'x_hat': ti.types.vector(3,float),
                               'x_prev': ti.types.vector(3,float),
                               'x_init': ti.types.vector(3, float),
                               'grad': ti.types.vector(3,float),
                               'grad_prev': ti.types.vector(3,float),
                               'p': ti.types.vector(3,float),
                               'diagH': ti.types.vector(3, float),
                               # 'is_contact': int,
                               # 'is_dirichlet': int,
                               })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print('n_verts,n_cells',self.n_verts,self.n_cells)

        #precompute
        print('precompute!!')
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        # Because MeshTaichi doesn't support parallel boundary edges and elements at the same time
        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size ', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.set_point_lights()
        print('init grids')
        self.kappa = model.kappa
        self.dHat = model.dHat
        self.init_hash_grid()
        print('dHat:', self.dHat, 'kappa ', self.kappa)

        self.config = model.dict
        self.config['dHat'] = self.dHat
        self.config['kappa'] = self.kappa

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
        H = self.kappa * (-2) * ti.log(d / dHat) - 4 + 4 * dHat / d + (d - dHat) ** 2 / d ** 2
        return H

    @ti.kernel
    def update_x_v2(self,alpha:float):
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
        for k,j in self.cid:
            pair = self.cid[k,j]
            dist = pair.b
            E_ipc = self.barrier_E(dist)
            E += E_ipc
        return E

    @ti.kernel
    def compute_grad_and_diagH(self):
        # inertia potential
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)
        # elastic potential
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i +1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp ,0.0)
                c.verts[i].diagH += tmp
        #ipc potential
        for k,j in self.cid:
            pair = self.cid[k,j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist **2
            bg = self.barrier_g(dist)
            para = bg / dist
            para0 = (self.barrier_H(dist) - para) / dist2
            for i in range(4):
                CORD = cord[i]
                ID = ids[i]
                self.mesh.verts.grad[ID] += para * CORD * t
                diag_tmp = CORD * CORD * (para0 * t * t + para * ti.Vector.one(float, 3))
                diag_tmp_spd = ti.max(diag_tmp,0.0)
                self.mesh.verts.diagH[ID] += diag_tmp_spd

    @ti.kernel
    def compute_pHp(self)->float:
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

        for k,j in self.cid:
            pair = self.cid[k,j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist
            bg = self.barrier_g(dist)
            para1 = bg / dist
            para0 = (self.barrier_H(dist) - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * ( p_tmp.dot(dtdx_t) ** 2)
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
                E += self.barrier_E(dist)
        return E

    @ti.kernel
    def add_grad_and_diagH_ground_barrier(self):
        # set the boundary condition with ipc, min_dist is used to prevent generating too large force
        min_dist = 1e-2 * self.dHat
        for i in range(self.n_boundary_points):
            p = self.boundary_points[i]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    self.mesh.verts.x[p][1] = self.ground + min_dist
                    dist = min_dist
                self.mesh.verts.grad[p][1] += self.barrier_g(dist)
                self.mesh.verts.diagH[p][1] += self.barrier_H(dist)


    @ti.kernel
    def add_pHp_ground_barrier(self)->float:
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
                ret_value = p_tmp * self.barrier_H(dist) * p_tmp
                ret += ret_value
        return ret

    def line_search_newton(self):
        gTp = self.compute_gTp()
        pHp = self.compute_pHp()
        if self.ground_barrier == 1:
            pHp_ground = self.add_pHp_ground_barrier()
            pHp += pHp_ground
        alpha = - gTp / pHp
        return alpha, gTp, pHp


    @ti.kernel
    def compute_p_inf_norm(self)->float:
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
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
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
    def line_search_clamped_newton(self,rate:float) -> (float,float,float):
        """ first compute newton line search , then clamp the alpha with max displacement 0.5 dHat"""
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
        for k,j in self.cid:
            pair = self.cid[k,j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist
            bg = self.barrier_g(dist)
            para1 = bg / dist
            para0 = (self.barrier_H(dist) - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * ( p_tmp.dot(dtdx_t) ** 2)
            p_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * p_dtdx.norm_sqr()
            pHp += ti.max(pHp_0+pHp_1, 0.0)
        alpha = - gTp / pHp
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
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
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
    def check_inverse(self)->int:
        # check is there any inversion
        ret = 0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            J = F.determinant()
            if J < -1e-6:
                ret = 1
        return ret