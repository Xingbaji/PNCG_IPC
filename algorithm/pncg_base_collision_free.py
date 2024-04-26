from algorithm.base_deformer import *
from util.model_loading import *

@ti.data_oriented
class pncg_base_deformer(base_deformer):
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
                               'grad_prev': ti.types.vector(3, float),
                               'p': ti.types.vector(3, float),
                               })
        self.mesh.verts.place({'diagH': ti.types.vector(3, float)})
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
    def compute_E(self) -> float:
        """Compute energy"""
        E = 0.0
        for vert in self.mesh.verts:
            E += 0.5 * vert.m * (vert.x - vert.x_hat).norm_sqr()
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            Psi = self.compute_Psi(F, self.mu, self.la)
            E += (self.dt ** 2) * c.W * Psi
        return E

    @ti.kernel
    def compute_grad_and_diagH(self):
        for vert in self.mesh.verts:
            vert.grad_prev = vert.grad
            vert.grad = vert.m * (vert.x - vert.x_hat)
            vert.diagH = vert.m * ti.Vector.one(float, 3)
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B  # 3x3
            para = c.W * self.dt ** 2
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp

    @ti.kernel
    def compute_pHp(self) -> float:
        ret = 0.0
        for vert in self.mesh.verts:
            ret += vert.p.norm_sqr() * vert.m

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
            ret += c.W * self.dt ** 2 * ti.max(tmp, 0.0)  # clamp negetive value to 0, similar to spd projection
        return ret

    @ti.kernel
    def assign_xn_xhat(self):
        ti.mesh_local(self.mesh.verts.x) # somewhere mesh_local would raise error, why?
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.dt * self.dt * self.gravity

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        ret = 0.0
        for vert in self.mesh.verts:
            ti.atomic_max(ret, (ti.abs(vert.grad)).max())
        return ret

    @ti.kernel
    def update_v(self):
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt

    @ti.kernel
    def update_x(self, alpha: float):
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p

    @ti.kernel
    def update_v_and_bound(self):
        """if object hits ground, set x to be ground and v to be 0. It would cause nan for NH material"""
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt
            if vert.x[1] < self.ground:
                vert.x[1] = self.ground
                if vert.v[1] < 0.0:
                    vert.v[1] = 0.0

    @ti.kernel
    def compute_init_p(self):
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH

    @ti.kernel
    def compute_gTp(self) -> float:
        gTp = 0.0
        for vert in self.mesh.verts:
            gTp += vert.grad.dot(vert.p)
        return gTp

    @ti.kernel
    def compute_DK_direction(self):
        g_p = 0.0  # g^{\top} p
        g_Py = 0.0
        y_p = 0.0
        y_Py = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p)
        for vert in self.mesh.verts:
            y = vert.grad - vert.grad_prev
            Py = y / vert.diagH
            y_p += y.dot(vert.p)
            g_Py += vert.grad.dot(Py)
            y_Py += y.dot(Py)
            g_p += vert.grad.dot(vert.p)
        beta = (g_Py - y_Py * g_p / y_p) / y_p
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH + beta * vert.p

    @ti.kernel
    def compute_FR_direction(self):
        g_P_g = 0.0
        gk_P_gk = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p)
        for vert in self.mesh.verts:
            g_P_g += vert.grad.dot(vert.grad / vert.diagH)
            gk_P_gk += vert.grad_prev.dot(vert.grad_prev / vert.diagH)
        beta = g_P_g / gk_P_gk
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH + beta * vert.p

    @ti.kernel
    def compute_CD_direction(self):
        g_k_p = 0.0
        gkp1_P_gkp1 = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p)
        for vert in self.mesh.verts:
            g_k_p += vert.grad_prev.dot(vert.p)
            gkp1_P_gkp1 += vert.grad.dot(vert.grad / vert.diagH)
        beta = - gkp1_P_gkp1 / g_k_p
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH + beta * vert.p

    @ti.kernel
    def compute_PR_direction(self):
        g_k_p = 0.0
        gkp1_P_y = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p)
        for vert in self.mesh.verts:
            g_k_p += vert.grad_prev.dot(vert.p)
            y = vert.grad - vert.grad_prev
            gkp1_P_y += vert.grad.dot(y)
        beta = - gkp1_P_y / g_k_p
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH + beta * vert.p

    @ti.kernel
    def compute_DK_plus_direction(self):
        g_Py = 0.0
        y_p = 0.0
        y_Py = 0.0
        g_p = 0.0
        p_norm = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p, self.mesh.verts.grad_prev)
        for vert in self.mesh.verts:
            diagH = vert.diagH
            grad = vert.grad
            y = grad - vert.grad_prev
            Py = y / diagH
            y_p += y.dot(vert.p)
            g_Py += grad.dot(Py)
            y_Py += y.dot(Py)
            g_p += grad.dot(vert.p)
            p_norm += vert.p.norm_sqr()
        beta = (g_Py - y_Py * g_p / y_p) / y_p
        # print('beta', beta)
        beta_plus = ti.max(beta, 0.01 * g_p / p_norm)
        # beta_plus = ti.max(beta, 0.0)
        if beta_plus > beta:
            print('beta', beta)
            print('beta_plus', beta_plus)
        for vert in self.mesh.verts:
            vert.p = - vert.grad / vert.diagH + beta_plus * vert.p

    @ti.kernel
    def compute_BFGS_direction(self, alpha: float):
        g_Py = 0.0
        y_p = 0.0
        y_Py = 0.0
        g_p = 0.0
        ti.mesh_local(self.mesh.verts.grad, self.mesh.verts.diagH, self.mesh.verts.p, self.mesh.verts.grad_prev)
        for vert in self.mesh.verts:
            diagH = vert.diagH
            grad = vert.grad
            y = grad - vert.grad_prev
            Py = y / diagH
            y_p += y.dot(vert.p)
            g_Py += grad.dot(Py)
            y_Py += y.dot(Py)
            g_p += grad.dot(vert.p)
        beta = ( g_Py - (alpha + y_Py / y_p) * g_p ) / y_p
        gamma = g_p / y_p
        for vert in self.mesh.verts:
            Py = (vert.grad - vert.grad_prev) / vert.diagH
            vert.p = - vert.grad / vert.diagH + beta * vert.p + gamma * Py

    @ti.kernel
    def compute_newton_alpha(self) -> (float,float,float):
        """ Compute step size and update x
        pHp: p^{\top} H p; gTp: grad^{top} p"""
        pHp = 0.0
        gTp = 0.0
        for vert in self.mesh.verts:
            gTp += vert.grad.dot(vert.p)
            pHp += vert.p.norm_sqr() * vert.m

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
        alpha = - gTp / pHp
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p
        return (alpha, gTp, pHp)

    @ti.kernel
    def dirichlet_grad(self):
        # set the gradient of dirichelet vertice to be 0
        for vert in self.mesh.verts: #S(grad)
            if vert.is_dirichlet == 1:
                vert.grad = ti.Vector([0, 0, 0])

    def init_dirichlet(self):
        print('init dirichlet')
        self.mesh.verts.place({'is_dirichlet': ti.i32})
        path = self.dict['model_paths'][0].rsplit('/',1)[0] + '/is_dirichlet.npy'
        dirichlet_np = np.load(path)
        print('load np')
        print(dirichlet_np.shape, self.n_verts)
        self.mesh.verts.is_dirichlet.from_numpy(dirichlet_np)
        print('dirichlet init finish')

    def step(self):
        print('Frame', self.frame)
        self.assign_xn_xhat()
        self.compute_grad_and_diagH()
        self.compute_init_p()
        alpha, gTp, pHp = self.compute_newton_alpha()
        delta_E = - alpha * gTp - 0.5 * alpha**2 * pHp
        delta_E_init = delta_E
        for iter in range(self.iter_max):
            if delta_E < self.epsilon * delta_E_init:
                break
            else:
                print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp, 'pHp',
                      pHp)
            self.compute_grad_and_diagH()
            self.compute_DK_direction()
            alpha, gTp, pHp = self.compute_newton_alpha()
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
        print('finish at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp',
              gTp, 'pHp', pHp)
        self.update_v_and_bound()
        self.frame += 1
        return iter