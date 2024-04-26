# My implement of newton PCG solver
# Sometime it would cause nan?

import numpy as np
np.set_printoptions(suppress=True)
from algorithm.base_deformer import *
from util.model_loading import *


@ti.data_oriented
class newton_base_deformer(base_deformer):
    def __init__(self, demo):
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        print('mu', self.mu, 'la', self.la)
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        print('ground', self.ground)
        self.mesh = model.mesh
        self.ground = model.ground
        self.frame = 0
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.SMALL_NUM = 1e-7
        # initialize model
        self.mesh.verts.place({'x': ti.types.vector(3, float),
                               'v': ti.types.vector(3, float),
                               'm': float,
                               'x_n': ti.types.vector(3, float),
                               'x_hat': ti.types.vector(3, float),
                               'x_prev': ti.types.vector(3, float),
                               'x_init': ti.types.vector(3, float),
                               'grad': ti.types.vector(3, float),
                               'delta_x': ti.math.vec3,
                               'diagH': ti.math.vec3,
                               'mul_ans': ti.math.vec3, # Ax
                               'b': ti.math.vec3,
                               'r': ti.math.vec3,
                               'd': ti.math.vec3,
                               'q': ti.math.vec3,
                               'tmp': ti.math.vec3
                               })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.n_verts = self.mesh._vert_position.shape[0]
        self.n_cells = len(self.mesh.cells)
        print('n_verts,n_cells', self.n_verts, self.n_cells)

        # precompute
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)
        self.set_point_lights()
        self.config = model.dict

    @ti.kernel
    def assign_xn_xhat(self):
        ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.dt * self.dt * self.gravity

    @ti.kernel
    def compute_E(self) -> float:
        E = 0.0
        ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            E += 0.5 * vert.m * (vert.x - vert.x_hat).norm_sqr()
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            Psi = compute_Psi_ARAP(F,self.mu,self.la)
            E += (self.dt ** 2) * c.W * Psi
        return E

    @ti.kernel
    def mul_operation(self):
        for vert in self.mesh.verts:
            vert.mul_ans = vert.m * vert.tmp
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            dFdx = compute_dFdx(B)  # 9x12
            para = self.dt ** 2 * c.W
            d2PsidF2 = compute_d2PsidF2_ARAP_filter(F, self.mu, self.la)
            d2Psidx2 = (dFdx.transpose()) @ d2PsidF2 @ dFdx  # 12x12
            d2Psidx2_tmp = para * d2Psidx2
            Tmp = ti.Vector.zero(float, 12)
            Tmp[0:3] = c.verts[0].tmp
            Tmp[3:6] = c.verts[1].tmp
            Tmp[6:9] = c.verts[2].tmp
            Tmp[9:12] = c.verts[3].tmp
            mul_res = d2Psidx2_tmp @ Tmp
            for i in range(4):
                for j in range(3):
                    c.verts[i].mul_ans[j] += mul_res[3*i+j]

    @ti.kernel
    def compute_grad_and_diagH(self):
        for vert in self.mesh.verts:
            vert.grad = vert.m * (vert.x - vert.x_hat)
            vert.diagH = vert.m * ti.Vector.one(float, 3)
        # ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.x, self.mesh.verts.diagH)
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B # 3x3
            para = c.W * self.dt ** 2
            dPsidx = para * compute_dPsidx_ARAP(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * compute_diag_d2Psidx2_ARAP_filter(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i +1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp ,0.0)
                c.verts[i].diagH += tmp

    @ti.kernel
    def compute_delta_x_inf_norm(self) -> float:
        ret = 0.0
        for vert in self.mesh.verts:
            ti.atomic_max(ret, (ti.abs(vert.delta_x)).max())
        return ret

    @ti.kernel
    def compute_v(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt

    @ti.kernel
    def update_x(self,alpha:float):
        for vert in self.mesh.verts:
            vert.x = vert.x_prev + alpha * vert.delta_x

    @ti.kernel
    def update_and_bound(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt
            if vert.x[1] < self.ground:
                vert.x[1] = self.ground
                if vert.v[1] < 0.0:
                    vert.v[1] = 0.0

    @ti.kernel
    def assign_b(self):
        for vert in self.mesh.verts:
            vert.b = - vert.grad

    @ti.kernel
    def assign_r(self):
        # r = b - Ax
        for vert in self.mesh.verts:
            vert.r = vert.b - vert.mul_ans

    @ti.kernel
    def assign_d(self):
        for vert in self.mesh.verts:
            vert.d = vert.r / vert.diagH

    @ti.kernel
    def compute_r_dot_d(self)->float:
        ret = 0.0
        for vert in self.mesh.verts:
            ret += vert.r.dot(vert.d)
        return ret

    @ti.kernel
    def compute_q_dot_d(self)->float:
        ret = 0.0
        for vert in self.mesh.verts:
            vert.q = vert.mul_ans
            ret += vert.q.dot(vert.d)
        return ret

    @ti.kernel
    def assign_tmp_as_delta_x(self):
        for vert in self.mesh.verts:
            vert.tmp = vert.delta_x

    @ti.kernel
    def assign_tmp_as_d(self):
        for vert in self.mesh.verts:
            vert.tmp = vert.d

    @ti.kernel
    def update_delta_x(self,alpha:float):
        for vert in self.mesh.verts:
            vert.delta_x += alpha * vert.d

    @ti.kernel
    def update_r(self,alpha:float):
        for vert in self.mesh.verts:
            vert.r -= alpha * vert.q

    @ti.kernel
    def assign_tmp_as_s(self):
        for vert in self.mesh.verts:
            vert.tmp = vert.r / vert.diagH

    @ti.kernel
    def compute_r_dot_s(self)->float:
        ret = 0.0
        for vert in self.mesh.verts:
            ret += vert.r.dot(vert.tmp)
        return ret

    @ti.kernel
    def update_d(self,beta:float):
        for vert in self.mesh.verts:
            vert.d = vert.tmp + beta * vert.d


    def pcg_newton_solver(self):
        epsilon = 1e-6
        iter_max = 1000
        self.compute_grad_and_diagH()
        self.assign_b()  # b = - grad
        self.assign_tmp_as_delta_x()
        self.mul_operation() # Ax , x in pcg is the delta_x in the newton method
        self.assign_r() # r = b - Ax
        self.assign_d() # d = M_inv r
        sig_new = self.compute_r_dot_d()
        sig_0 = sig_new
        for i in range(iter_max):
            self.assign_tmp_as_d()
            self.mul_operation() # q = A @ d
            q_d = self.compute_q_dot_d()
            alpha = sig_new / q_d
            self.update_delta_x(alpha) # x = x + alpha d

            # if i % 50 == 0 and i > 1:
            #     # restart
            #     self.assign_tmp_as_delta_x()
            #     self.mul_operation() # Ax
            #     self.assign_r() # r = b - Ax
            # else:
            self.update_r(alpha) # r = r - alpha q

            self.assign_tmp_as_s() # s = M_inv r
            sig_old = sig_new
            sig_new = self.compute_r_dot_s()
            beta = sig_new / sig_old
            self.update_d(beta)

            if sig_new < epsilon**2 * sig_0:
                break
        print('pcg finish at iter', i, 'rate', sig_new / sig_0, 'converage', i<(iter_max-1))
        return i

    def step(self):
        self.frame += 1
        self.assign_xn_xhat()
        self.iter_max = 50
        for iter in range(self.iter_max):
            self.pcg_newton_solver() # delta_x
            delta_x_norm = self.compute_delta_x_inf_norm()
            E0 = self.compute_E()
            self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
            alpha = 1.0
            self.update_x(alpha)
            E = self.compute_E()
            while E > E0 + 1e-7:
                alpha = alpha * 0.5
                self.update_x(alpha)
                E = self.compute_E()
            print("iter " , iter, "[Step size", alpha, 'delta_x_norm', delta_x_norm, "]")
            if alpha * delta_x_norm < 1e-2 * self.dt:
                print('iter finish with iter', iter, delta_x_norm)
                break
        self.update_and_bound()



if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)#, device_memory_fraction=0.9)#,kernel_profiler=True),advanced_optimization=True,fast_math=True)
    demos = ['armadillo_drop_collision_free','Patrick_Star','cube','cube_10','cube_20','cube_40','banana']
    demo = demos[-1]

    deformer = newton_base_deformer(demo = demo)
    print('init finish')
    deformer.visual()