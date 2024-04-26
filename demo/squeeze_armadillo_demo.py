import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_ipc import *

@ti.data_oriented
class squeeze_armadillo_demo(pncg_ipc_deformer):
    def init_visual(self):
        self.boundary_barriers = ti.Vector.field(1, dtype=ti.f32, shape=(6,)) # left, right, down, up, back, front
        self.per_vertex_color =  ti.Vector.field(3, dtype=ti.f32,shape=self.n_verts)
        self.lines = ti.Vector.field(3, dtype=ti.f32,shape=24)
        self.set_barrier_init_value(5.0) # set boundary barrier
        self.set_lines()
        self.set_color() # set color of each object
        self.frame = 0

    @ti.kernel
    def set_color(self):
        n = 13054
        for i in range(n):
            self.per_vertex_color[i] = ti.Vector([0.473, 0.643, 0.8],ti.f32)
        for i in range(n, 2 * n):
            self.per_vertex_color[i] = ti.Vector([0.233, 0.8, 0.409],ti.f32)
        for i in range(2 * n, 3 * n):
            self.per_vertex_color[i] = ti.Vector([0.8, 0.466, 0.565], ti.f32)
        for i in range(3 * n , 4 * n):
            self.per_vertex_color[i] = ti.Vector([0.8, 0.8 ,0.8], ti.f32)
    @ti.kernel
    def set_lines(self):
        N = 0
        ti.loop_config(serialize=True)
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                for k in ti.static(range(2)):
                    self.lines[N] = ti.Vector(
                        [self.boundary_barriers[i], self.boundary_barriers[2 + j], self.boundary_barriers[4 + k]],
                        ti.f32)
                    N+=1
        ti.loop_config(serialize=True)
        for i in ti.static(range(2)):
            for k in ti.static(range(2)):
                for j in ti.static(range(2)):
                    self.lines[N] = ti.Vector(
                        [self.boundary_barriers[i], self.boundary_barriers[2 + j], self.boundary_barriers[4 + k]],
                        ti.f32)
                    N+=1
        ti.loop_config(serialize=True)
        for k in ti.static(range(2)):
            for j in ti.static(range(2)):
                for i in ti.static(range(2)):
                    # self.lines[i*4+j*2+k] = ti.Vector([self.boundary_barriers[i], self.boundary_barriers[2+j], self.boundary_barriers[4+k]], ti.f32)
                    self.lines[N] = ti.Vector(
                        [self.boundary_barriers[i], self.boundary_barriers[2 + j], self.boundary_barriers[4 + k]],
                        ti.f32)
                    N+=1

    def set_barrier_init_value(self, value):
        node_np = self.mesh.verts.x.to_numpy()
        node_box = ti.Vector([np.min(node_np[:, 0]), np.max(node_np[:, 0]), np.min(node_np[:, 1]), np.max(node_np[:, 1]),
                              np.min(node_np[:, 2]), np.max(node_np[:, 2])], ti.f32)
        self.boundary_barriers[0][0] = node_box[0] - value
        self.boundary_barriers[1][0] = node_box[1] + value
        self.boundary_barriers[2][0] = node_box[2] - value
        self.boundary_barriers[3][0] = node_box[3] + value
        self.boundary_barriers[4][0] = node_box[4] - value
        self.boundary_barriers[5][0] = node_box[5] + value

    def update_boundary(self, speed):
        self.boundary_barriers[0][0] += speed
        self.boundary_barriers[1][0] -= speed
        self.boundary_barriers[3][0] -= speed
        self.boundary_barriers[4][0] += speed
        self.boundary_barriers[5][0] -= speed

    @ti.kernel
    def add_grad_and_diagH_barriers(self):
        min_dist = 1e-3 * self.dHat
        for vert in self.mesh.verts:
            x_a0 = vert.x
            dist_left = x_a0[0] - self.boundary_barriers[0][0]
            dist_right = self.boundary_barriers[1][0] - x_a0[0]
            dist_down = x_a0[1] - self.boundary_barriers[2][0]
            dist_up = self.boundary_barriers[3][0] - x_a0[1]
            dist_back = x_a0[2] - self.boundary_barriers[4][0]
            dist_front = self.boundary_barriers[5][0] - x_a0[2]

            if dist_left < self.dHat:
                if dist_left < min_dist:
                    vert.x[0] = self.boundary_barriers[0][0] + min_dist
                    dist_left = min_dist
                vert.grad[0] += self.barrier_g(dist_left)
                vert.diagH[0] += self.barrier_H(dist_left)
            elif dist_right < self.dHat:
                if dist_right < min_dist:
                    vert.x[0] = self.boundary_barriers[1][0] - min_dist
                    dist_right = min_dist
                vert.grad[0] -= self.barrier_g(dist_right)
                vert.diagH[0] += self.barrier_H(dist_right)
            if dist_down < self.dHat:
                if dist_down < min_dist:
                    vert.x[1] = self.boundary_barriers[2][0] + min_dist
                    dist_down = min_dist
                vert.grad[1] += self.barrier_g(dist_down)
                vert.diagH[1] += self.barrier_H(dist_down)
            elif dist_up < self.dHat:
                if dist_up < min_dist:
                    vert.x[1] = self.boundary_barriers[3][0] - min_dist
                    dist_up = min_dist
                vert.grad[1] -= self.barrier_g(dist_up)
                vert.diagH[1] += self.barrier_H(dist_up)
            if dist_back < self.dHat:
                if dist_back < min_dist:
                    vert.x[2] = self.boundary_barriers[4][0] + min_dist
                    dist_back = min_dist
                vert.grad[2] += self.barrier_g(dist_back)
                vert.diagH[2] += self.barrier_H(dist_back)
            elif dist_front < self.dHat:
                if dist_front < min_dist:
                    vert.x[2] = self.boundary_barriers[5][0] - min_dist
                    dist_front = min_dist
                vert.grad[2] -= self.barrier_g(dist_front)
                vert.diagH[2] += self.barrier_H(dist_front)

    def step(self):
        print('Frame', self.frame)
        if self.frame < 450:
            self.update_boundary(0.2)
        elif self.frame < 550:
            self.update_boundary(0.1)
        else:
            self.set_barrier_init_value(500)
        self.assign_xn_xhat()
        for iter in range(self.iter_max):
            self.find_cnts()
            self.compute_grad_and_diagH()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_barriers()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_direction()
            alpha, gTp, pHp = self.line_search_clamped_newton(0.5)
            delta_E = - alpha * gTp - 0.5 * alpha ** 2 * pHp
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                # print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
                #       'gTp', gTp, 'pHp', pHp)
                break
            # else:
            #     print('iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha, 'gTp', gTp,
            #           'pHp', pHp)
        print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
              'gTp', gTp, 'pHp', pHp)
        self.update_v()
        self.frame += 1
        return iter

    def visual(self):
        window = ti.ui.Window("Visualization", (800, 600), vsync=True)
        dir = '../demo_results/final/' + self.demo + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        camera.fov(75)
        canvas.set_background_color(color=(0.2, 0.2, 0.2))
        point_light_color = (0.5,0.5,0.5)
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    self.restart()
                if window.event.key == 'b':
                    break
            self.set_lines()
            camera.track_user_inputs(window, movement_speed=1.0, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.1, 0.1, 0.1))
            for light_pos in self.point_lights:
                scene.point_light(pos=light_pos, color=point_light_color)
            scene.mesh_instance(self.mesh.verts.x, self.indices, show_wireframe=False, per_vertex_color=self.per_vertex_color)
            scene.lines(self.lines, color = (1.0, 1.0, 1.0), width = 1.0)

            canvas.scene(scene)
            window.show()
            self.step()

if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)#, device_memory_fraction=0.9)#, kernel_profiler=True)
    demo = 'squeeze_four_armadillo'
    ipc_deformer = squeeze_armadillo_demo(demo=demo)
    ipc_deformer.init_visual()
    print('init finish')
    ipc_deformer.find_cnts(PRINT=True)
    ipc_deformer.visual()