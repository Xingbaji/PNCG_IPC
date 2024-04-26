import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_ipc import *
@ti.data_oriented
class unittest_demos(pncg_ipc_deformer):
    def init_dirichlet(self):
        self.mesh.verts.place({'is_dirichlet': ti.i32})
        self.per_vertex_color =  ti.Vector.field(3, dtype=float,shape=self.n_verts)
        self.set_pertex_color()
        obj_down = self.demo.split('_')[1]
        if obj_down == 'cube':
            self.fix_obj_down(8)
        elif obj_down == 'spike':
            self.fix_obj_down(5)
        elif obj_down == 'wedge':
            self.fix_obj_down(6)
        elif obj_down == 'crack':
            self.fix_obj_down(40)
        elif obj_down == 'edge':
            self.fix_obj_down(15)
        elif obj_down == 'cliff':
            self.fix_obj_down(8)
        elif obj_down == 'cubes':
            self.fix_obj_down(24)

    @ti.kernel
    def set_pertex_color(self):
        for i in range(self.n_verts):
            self.per_vertex_color[i] = ti.Vector([0.1,0.2,0.5],float)

    @ti.kernel
    def fix_obj_down(self,n:ti.i32):
        for i in range(n):
            self.mesh.verts.is_dirichlet[i] = 1
            self.per_vertex_color[i] = ti.Vector([0.5,0.5,0.5],float)

    @ti.kernel
    def dirichlet_grad(self):
        for vert in self.mesh.verts:
            if vert.is_dirichlet == 1:
                vert.grad = ti.Vector([0, 0, 0])


    def step(self):
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
            alpha, gTp, pHp = self.line_search_clamped_newton(0.5)
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                break
            # else:
            #     print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp,
            #           'pHp', pHp)
        print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
              'gTp', gTp, 'pHp', pHp)

        self.update_v_and_bound()
        self.frame += 1
        return iter



    def visual(self):
        print('precombile')
        self.step()
        print('precombile finish')
        window = ti.ui.Window("unittests", (1920, 1080), vsync=True)
        dir = '../demo_results/final/' + self.demo + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        canvas = window.get_canvas()
        canvas.set_background_color(color=(1.0, 1.0, 1.0))
        # scene = ti.ui.Scene()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        camera.fov(75)

        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    self.restart()
                if window.event.key == 'b':
                    break
            camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((1.0,1.0,1.0))
            scene.mesh_instance(self.mesh.verts.x, self.indices, show_wireframe=False, per_vertex_color=self.per_vertex_color)
            scene.lines(self.mesh.verts.x,3.0,self.boundary_edges,color=(0.0,0.0,0.0))
            canvas.scene(scene)
            window.show()
            self.step()

if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32,default_ip=ti.i32)#, device_memory_fraction=0.9)#, kernel_profiler=True)
    demos = ['unittest_wedge_wedge', 'unittest_wedge_spike', 'unittest_spike_spike',
             'unittest_crack_spike','unittest_crack_wedge','unittest_edge_spike',
             'unittest_cube_spike2','unittest_cube_wedge','unittest_edge_cube','unittest_cliff_cube']
    for demo in demos:
        ipc_deformer = unittest_demos(demo=demo)
        ipc_deformer.init_dirichlet()
        print('init finish')
        ipc_deformer.visual()
        # ipc_deformer.save_results()
