import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_ipc import *

@ti.data_oriented
class twist_demo(pncg_ipc_deformer):
    @ti.kernel
    def init_dirichlet_assign_speed(self,speed_angle:float):
        speed = speed_angle / 180.0
        ti.mesh_local(self.mesh.verts.x)
        for vert in self.mesh.verts:
            if vert.is_dirichlet == 1:
                x_id = vert.x
                a = x_id[0]
                b = x_id[1]
                c = x_id[2]
                angle = ti.atan2(b, c)
                angle += speed * ti.math.pi * (1 if a < 0 else -1)
                radius = ti.sqrt(b * b + c * c)
                vert.x[1] = radius * ti.sin(angle)
                vert.x[2] = radius * ti.cos(angle)
                vert.x_hat = vert.x
            else:
                vert.x_hat = vert.x + vert.v * self.dt
            vert.x_n = vert.x

    def step(self):
        print('Frame', self.frame)
        self.init_dirichlet_assign_speed(2.88)
        for iter in range(self.iter_max):
            self.find_cnts_iter(iter)
            self.compute_grad_and_diagH()
            self.dirichlet_grad()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_direction()
            alpha, gTp, pHp = self.line_search_clamped_newton(1.0)
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init and iter > 20:
                # print('converage at iter', iter, '!')
                break
        print('finish at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp,
              'pHp', pHp)
        # print('dcd collision', self.check_dcd(),'inverse', self.check_inverse())
        self.update_v()
        self.frame += 1
        return iter

if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    demos = ['twist_mat150','twist_rods']
    ipc_deformer = twist_demo(demo=demos[1])
    ipc_deformer.visual()