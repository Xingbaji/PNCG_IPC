import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_collision_free import *
@ti.data_oriented
class cubic_demos(pncg_base_deformer):
    """unittest of cubic squeeze rotate stretch"""
    def get_dirichlet_points(self):
        x_np = self.mesh.get_position_as_numpy()
        x_min = np.min(x_np[:,1])
        x_max = np.max(x_np[:,1])
        points_lower = x_np[:,1] == x_min
        points_higher = x_np[:,1] == x_max
        dirichlet_points = np.logical_or(points_lower,points_higher)
        dirichlet_points = np.int32(dirichlet_points)
        print(np.sum(points_lower),np.sum(points_higher),np.sum(dirichlet_points))
        path = self.dict['model_paths'][0].rsplit('/',1)[0] + '/is_dirichlet.npy'
        np.save(path,dirichlet_points)

    @ti.kernel
    def assign_xn_xhat_squeeze(self, speed:ti.f32):
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            if vert.is_dirichlet == 1:
                if vert.x[1] == 0.25:
                    # lower
                    vert.x_hat = vert.x
                else:
                    vert.x[1] -= speed * self.dt
                    vert.x_hat = vert.x
            else:
                vert.x_hat = vert.x + self.dt * vert.v

    @ti.kernel
    def assign_xn_xhat_stretch(self, speed:ti.f32):
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            if vert.is_dirichlet == 1:
                if vert.x[1] == 0.25:
                    # lower
                    vert.x_hat = vert.x
                else:
                    vert.x[1] += speed * self.dt
                    vert.x_hat = vert.x
            else:
                vert.x_hat = vert.x + self.dt * vert.v

    @ti.kernel
    def assign_xn_xhat_rotate(self, speed:ti.f32, frame:ti.i32):
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            if vert.is_dirichlet == 1:
                if vert.x[1] == 0.25:
                    # lower
                    vert.x_hat = vert.x
                else:
                    angle = frame * speed * ti.math.pi
                    # print('frame',self.frame,'angle',angle,ti.cos(angle),ti.sin(angle))
                    rotate_matrix = ti.Matrix([[ti.cos(angle), 0, ti.sin(angle)], [0, 1, 0], [-ti.sin(angle), 0, ti.cos(angle)]],float)
                    center_point = ti.Vector([0.5,vert.x[1],-0.5],float)
                    p_c = vert.x - center_point
                    p_rot = rotate_matrix @ p_c
                    p_final = p_rot + center_point
                    # p_final[1] += speed * 30
                    # print(vert.x, p_rot,p_final)
                    vert.x = p_final
                    vert.x_hat = vert.x
            else:
                vert.x_hat = vert.x + self.dt * vert.v

    def step(self):
        print('Frame', self.frame)
        if self.frame < 70:
            self.assign_xn_xhat_squeeze(0.05)
        elif self.frame < 500:
            self.assign_xn_xhat_stretch(0.05)
        else:
            self.assign_xn_xhat_rotate(1e-5,self.frame-500)
        self.compute_grad_and_diagH()
        self.dirichlet_grad()
        self.compute_init_p()
        alpha, gTp, pHp = self.compute_newton_alpha()
        delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
        delta_E_init = delta_E
        for iter in range(self.iter_max):
            if delta_E < self.epsilon * delta_E_init:
                break
            # else:
            #     print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp, 'pHp',
            #           pHp)
            self.compute_grad_and_diagH()
            self.dirichlet_grad()
            self.compute_DK_direction()
            alpha, gTp, pHp = self.compute_newton_alpha()
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
        print('finish at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp',
              gTp, 'pHp', pHp)
        self.update_v_and_bound()
        self.frame += 1
        return iter

if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)#, device_memory_fraction=0.9)#,kernel_profiler=True),advanced_optimization=True,fast_math=True)
    demos = ['cube','cube_10','cube_20','cube_40']
    demo = demos[3]
    deformer = cubic_demos(demo = demo)
    print(deformer.dict)
    deformer.get_dirichlet_points()
    deformer.init_dirichlet()
    print('init finish')
    deformer.visual()
    # deformer.save_new(1000,SAVE_OBJ=False)
