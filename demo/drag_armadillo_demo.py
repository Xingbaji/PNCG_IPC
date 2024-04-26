import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_collision_free import *
class DK_dirichlet(pncg_base_deformer):
    def init_dirichlet(self):
        print('init dirichlet')
        self.mesh.verts.place({'is_dirichlet': ti.i32})
        path = '../model/mesh/Armadillo13K/is_dirichlet.npy'
        dirichlet_np = np.load(path)
        print('load np')
        print(dirichlet_np.shape, self.n_verts)
        self.mesh.verts.is_dirichlet.from_numpy(dirichlet_np)
        print('init finish')

    def get_dirichlet_points(self):
        x_np = self.mesh.get_position_as_numpy()
        x_min = np.min(x_np[:,1])
        x_max = np.max(x_np[:,1])
        points_lower = (x_np[:,1] - x_min) < 1.5
        points_higher = (x_max - x_np[:,1]) < 1.5
        dirichlet_points = np.logical_or(points_lower,points_higher)
        dirichlet_points = np.int32(dirichlet_points)
        print(np.sum(points_lower),np.sum(points_higher),np.sum(dirichlet_points))
        path = self.dict['model_paths'][0].rsplit('/',1)[0] + '/is_dirichlet.npy'
        np.save(path,dirichlet_points)
    @ti.kernel
    def assign_xn_xhat_2(self, force: ti.f32):
        ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[2] += self.dt * self.dt * force / vert.m

    def step(self):
        print('Frame', self.frame)
        if self.frame < 80:
            self.assign_xn_xhat_2(5000.0)
        else:
            self.assign_xn_xhat()
        self.compute_grad_and_diagH()
        self.dirichlet_grad()
        self.compute_init_p()
        alpha, gTp, pHp = self.compute_newton_alpha()
        delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
        delta_E_init = delta_E
        for iter in range(self.iter_max):
            if delta_E < self.epsilon * delta_E_init:
                break

            self.compute_grad_and_diagH()
            self.dirichlet_grad()
            self.compute_DK_direction()
            alpha, gTp, pHp = self.compute_newton_alpha()
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
        print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp',
              gTp, 'pHp', pHp)
        self.update_v_and_bound()
        self.frame += 1
        return iter

if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    demo = 'armadillo_collision_free'
    deformer = DK_dirichlet(demo = demo)
    deformer.get_dirichlet_points()
    deformer.init_dirichlet()
    deformer.visual()

