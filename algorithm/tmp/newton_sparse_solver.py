# Taichi 官方的matrix solver的实现
import numpy as np
np.set_printoptions(suppress=True)
from algorithm.pncg_base_collision_free import *
from util.model_loading import *


# from math_utils.cubic_roots import *


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
        self.b = ti.ndarray(float, 3 * self.n_verts)
        self.MatrixBuilder = ti.linalg.SparseMatrixBuilder(3 * self.n_verts, 3 * self.n_verts, max_num_triplets=10000000)
        elastic = model.elastic_type
        if elastic == 'ARAP':
            self.compute_d2PsidF2 = compute_d2PsidF2_ARAP
        elif elastic == 'ARAP_filter':
            self.compute_d2PsidF2 = compute_d2PsidF2_ARAP_filter
        elif elastic == 'SNH':
            self.compute_d2PsidF2 = compute_d2PsidF2_SNH
        elif elastic == 'FCR':
            self.compute_d2PsidF2 = compute_d2PsidF2_FCR
        elif elastic == 'FCR_filter':
            self.compute_d2PsidF2 = compute_d2PsidF2_FCR_filter
        elif elastic == 'NH':
            self.compute_d2PsidF2 = compute_d2PsidF2_NH


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
            Psi = self.compute_Psi(F, self.mu, self.la)
            E += (self.dt ** 2) * c.W * Psi
        return E

    @ti.kernel
    def assemble_Elastic(self, A: ti.types.sparse_matrix_builder()):
        # ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            id = vert.id
            mass = vert.m
            A[3 * id, 3 * id] += mass
            A[3 * id + 1, 3 * id + 1] += mass
            A[3 * id + 2, 3 * id + 2] += mass
            vert.grad = vert.m * (vert.x - vert.x_hat)
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            dFdx = compute_dFdx(B)  # 9x12
            para = self.dt ** 2 * c.W
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            d2PsidF2 = self.compute_d2PsidF2(F, self.mu, self.la)
            d2Psidx2 = (dFdx.transpose()) @ d2PsidF2 @ dFdx  # 12x12
            # d2Psidx2 = spd_matrix(d2Psidx2)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)

            A_tmp = para * d2Psidx2
            ids = ti.Vector([c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id], ti.i32)
            for o,p,m,n in ti.ndrange(3,3,4,4):
                A[3 * ids[m] + o, 3 * ids[n] + p] += A_tmp[3 * m + o, 3 * n + p]

    @ti.kernel
    def assemble_A(self, A: ti.types.sparse_matrix_builder()):
        # ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            id = vert.id
            mass = vert.m
            A[3 * id, 3 * id] += mass
            A[3 * id + 1, 3 * id + 1] += mass
            A[3 * id + 2, 3 * id + 2] += mass
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            dFdx = compute_dFdx(B)  # 9x12
            para = self.dt ** 2 * c.W
            d2PsidF2 = self.compute_d2PsidF2(F, self.mu, self.la)
            d2Psidx2 = (dFdx.transpose()) @ d2PsidF2 @ dFdx  # 12x12
            # d2Psidx2 = spd_matrix(d2Psidx2)
            A_tmp = para * d2Psidx2
            ids = ti.Vector([c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id], ti.i32)
            for m in ti.static(range(4)):  # m 行
                for n in ti.static(range(4)):  # n 列
                    for o in ti.static(range(3)):
                        for p in ti.static(range(3)):
                            # A[3 * c.verts[m].id + o, 3 * c.verts[n].id + p] += A_tmp[3 * m + o, 3 * n + p]
                            A[3 * ids[m] + o, 3 * ids[n] + p] += A_tmp[3 * m + o, 3 * n + p]
            # for o,p,m,n in ti.ndrange(3,3,4,4):
            #     A[3 * ids[m] + o, 3 * ids[n] + p] += A_tmp[3 * m + o, 3 * n + p]

    @ti.kernel
    def compute_grad(self):
        # b = - grad
        for vert in self.mesh.verts:
            vert.grad = vert.m * (vert.x - vert.x_hat)
        # ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.x, self.mesh.verts.diagH)
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B # 3x3
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i +1], dPsidx[3 * i + 2]], float)

    @ti.kernel
    def compute_delta_x_inf_norm(self, delta_x:ti.types.ndarray() ) -> float:
        ret = 0.0
        for i in delta_x:
            ti.atomic_max(ret, ti.abs(delta_x[i]))
        return ret

    @ti.kernel
    def compute_v(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt


    @ti.kernel
    def update_2(self, delta_x_: ti.types.ndarray()):
        for vert in self.mesh.verts:
            id = vert.id
            delta_x = ti.Vector([delta_x_[3 * id], delta_x_[3 * id + 1], delta_x_[3 * id + 2]], ti.f32)
            vert.x += delta_x
            vert.v = delta_x / self.dt
            if vert.x[1] < 0.0:
                vert.x[1] = 0.0
                if vert.v[1] < 0.0:
                    vert.v[1] = 0.0

    @ti.kernel
    def update_x(self, alpha: float, delta_x_: ti.types.ndarray()):
        for vert in self.mesh.verts:
            id = vert.id
            delta_x = ti.Vector([delta_x_[3 * id], delta_x_[3 * id + 1], delta_x_[3 * id + 2]], ti.f32)
            vert.x = vert.x_prev + alpha * delta_x
    @ti.kernel
    def assign_b(self, b_: ti.types.ndarray()):
        for vert in self.mesh.verts:
            id = vert.id
            b_[3 * id] = -vert.grad[0]
            b_[3 * id + 1] = -vert.grad[1]
            b_[3 * id + 2] = -vert.grad[2]

    @ti.kernel
    def copy_to_mesh(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.n_verts):
            source[i][0] = des[3 * i]
            source[i][1] = des[3 * i + 1]
            source[i][2] = des[3 * i + 2]

    @ti.kernel
    def update_v_and_bound(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt
            if vert.x[1] < self.ground:
                vert.x[1] = self.ground
                if vert.v[1] < 0.0:
                    vert.v[1] = 0.0

    def step(self):
        self.frame += 1
        self.assign_xn_xhat()
        self.iter_max = 50
        for iter in range(self.iter_max):
            # self.compute_grad()
            self.assemble_Elastic(self.MatrixBuilder)
            A = self.MatrixBuilder.build()
            self.assign_b(self.b) # b = - grad
            solver = ti.linalg.SparseSolver(solver_type="LDLT")
            solver.analyze_pattern(A)
            solver.factorize(A)
            delta_x = solver.solve(self.b)
            delta_x_norm = self.compute_delta_x_inf_norm(delta_x)
            E0 = self.compute_E()
            self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
            alpha = 1.0
            self.update_x(alpha, delta_x)
            E = self.compute_E()
            while E > E0:
                alpha = alpha * 0.5
                self.update_x(alpha, delta_x)
                E = self.compute_E()
            print("iter " , iter, "[Step size after line search: ", alpha, 'delta_x_norm', delta_x_norm, "]")
            print(delta_x[0])
            if delta_x_norm < 1e-2 * self.dt:
                print('iter finish with iter', iter, delta_x_norm)
                break
        self.update_v_and_bound()

    def step2(self):
        self.frame += 1
        self.assign_xn_xhat()
        self.compute_grad()
        self.assemble_A(self.MatrixBuilder)
        # self.assemble_Elastic(self.MatrixBuilder)
        A = self.MatrixBuilder.build()
        self.assign_b(self.b) # b = - grad
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        delta_x = solver.solve(self.b)
        # self.copy_to_mesh(delta_x, self.mesh.verts.delta_x)
        self.update_2(delta_x)

    def step_sparse_iterative_solver(self):
        self.frame += 1
        self.assign_xn_xhat()
        self.iter_max = 10
        for iter in range(self.iter_max):
            self.assemble_Elastic(self.MatrixBuilder)
            # print('matrix assemble finish')
            A = self.MatrixBuilder.build()
            self.assign_b(self.b, self.mesh.verts.grad)  # b = - grad

            # sparse iterative solver
            solver = ti.linalg.SparseCG(A=A,b=self.b)
            delta_x, exit_code = solver.solve()
            print(exit_code)

            self.copy_to_mesh(delta_x, self.mesh.verts.delta_x)
            delta_x_norm = self.compute_delta_x_inf_norm()
            E0 = self.compute_E()
            self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
            alpha = 1.0
            self.update_x(alpha)
            E = self.compute_E()
            while E > E0:
                alpha = alpha * 0.5
                self.update_x(alpha)
                E = self.compute_E()
            print("iter ", iter, "[Step size after line search: ", alpha, "]")
            if delta_x_norm < 1e-3 * (self.dt ** 2):
                print('iter finish with iter', iter, delta_x_norm)
                break
        self.update_v_and_bound()


if __name__ == '__main__':
    ti.init(arch=ti.cpu, default_fp=ti.f32) # it does not work on GPU ?
    demos = ['cube','cube_10','banana']
    demo = demos[2]

    deformer = newton_base_deformer(demo = demo)
    print('init finish')
    # for i in range(10):
    #     deformer.step()
    deformer.visual()