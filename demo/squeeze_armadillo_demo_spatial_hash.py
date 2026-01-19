import sys
import os
import time
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_ipc_spatial_hash import *

@ti.data_oriented
class squeeze_armadillo_demo_spatial_hash(pncg_ipc_deformer):
    def __init__(self, demo='cube_0'):
        super().__init__(demo)
        self.time_log = False  # Enable/disable timing

    def init_visual(self):
        self.boundary_barriers = ti.Vector.field(1, dtype=ti.f32, shape=(6,)) # left, right, down, up, back, front
        self.per_vertex_color =  ti.Vector.field(3, dtype=ti.f32,shape=self.n_verts)
        self.lines = ti.Vector.field(3, dtype=ti.f32,shape=24)
        self.set_barrier_init_value(5.0) # set boundary barrier

    @ti.kernel
    def set_barrier_init_value(self,value:float):
        for i in range(6):
            self.boundary_barriers[i][0] = value

    @ti.kernel
    def update_boundary(self,value:float):
        for i in ti.static(range(6)):
            if i == 2:
                self.boundary_barriers[i][0] += value
            if i == 3:
                self.boundary_barriers[i][0] -= value

    @ti.kernel
    def init_lines(self):
        # draw the boundary cube
        for i in range(24):
            self.lines[i] = ti.Vector([0.0, 0.0, 0.0], float)
        left = self.boundary_barriers[0][0]
        right = self.boundary_barriers[1][0]
        down = self.boundary_barriers[2][0]
        up = self.boundary_barriers[3][0]
        back = self.boundary_barriers[4][0]
        front = self.boundary_barriers[5][0]
        self.lines[0] = ti.Vector([left, down, back], float)
        self.lines[1] = ti.Vector([right, down, back], float)
        self.lines[2] = ti.Vector([left, up, back], float)
        self.lines[3] = ti.Vector([right, up, back], float)
        self.lines[4] = ti.Vector([left, down, front], float)
        self.lines[5] = ti.Vector([right, down, front], float)
        self.lines[6] = ti.Vector([left, up, front], float)
        self.lines[7] = ti.Vector([right, up, front], float)
        self.lines[8] = ti.Vector([left, down, back], float)
        self.lines[9] = ti.Vector([left, up, back], float)
        self.lines[10] = ti.Vector([right, down, back], float)
        self.lines[11] = ti.Vector([right, up, back], float)
        self.lines[12] = ti.Vector([left, down, front], float)
        self.lines[13] = ti.Vector([left, up, front], float)
        self.lines[14] = ti.Vector([right, down, front], float)
        self.lines[15] = ti.Vector([right, up, front], float)
        self.lines[16] = ti.Vector([left, down, back], float)
        self.lines[17] = ti.Vector([left, down, front], float)
        self.lines[18] = ti.Vector([right, down, back], float)
        self.lines[19] = ti.Vector([right, down, front], float)
        self.lines[20] = ti.Vector([left, up, back], float)
        self.lines[21] = ti.Vector([left, up, front], float)
        self.lines[22] = ti.Vector([right, up, back], float)
        self.lines[23] = ti.Vector([right, up, front], float)

    @ti.kernel
    def add_grad_and_diagH_barriers(self):
        for vert in self.mesh.verts:
            x = vert.x
            min_dist = 1e-2 * self.dHat
            left = self.boundary_barriers[0][0]
            right = self.boundary_barriers[1][0]
            down = self.boundary_barriers[2][0]
            up = self.boundary_barriers[3][0]
            back = self.boundary_barriers[4][0]
            front = self.boundary_barriers[5][0]
            dist_left = x[0] - left
            if dist_left < self.dHat:
                if dist_left < min_dist:
                    dist_left = min_dist
                vert.grad[0] += self.barrier_g(dist_left)
                vert.diagH[0] += self.barrier_H(dist_left)
            dist_right = right - x[0]
            if dist_right < self.dHat:
                if dist_right < min_dist:
                    dist_right = min_dist
                vert.grad[0] -= self.barrier_g(dist_right)
                vert.diagH[0] += self.barrier_H(dist_right)
            dist_down = x[1] - down
            if dist_down < self.dHat:
                if dist_down < min_dist:
                    dist_down = min_dist
                vert.grad[1] += self.barrier_g(dist_down)
                vert.diagH[1] += self.barrier_H(dist_down)
            dist_up = up - x[1]
            if dist_up < self.dHat:
                if dist_up < min_dist:
                    dist_up = min_dist
                vert.grad[1] -= self.barrier_g(dist_up)
                vert.diagH[1] += self.barrier_H(dist_up)
            dist_back = x[2] - back
            if dist_back < self.dHat:
                if dist_back < min_dist:
                    dist_back = min_dist
                vert.grad[2] += self.barrier_g(dist_back)
                vert.diagH[2] += self.barrier_H(dist_back)
            dist_front = front - x[2]
            if dist_front < self.dHat:
                if dist_front < min_dist:
                    dist_front = min_dist
                vert.grad[2] -= self.barrier_g(dist_front)
                vert.diagH[2] += self.barrier_H(dist_front)

    def step(self):
        if self.time_log:
            ti.sync()
            t_frame_start = time.perf_counter()

        print('Frame', self.frame)
        if self.frame < 450:
            self.update_boundary(0.2)
        elif self.frame < 550:
            self.update_boundary(0.1)
        else:
            self.set_barrier_init_value(500)
        self.assign_xn_xhat()

        for iter in range(self.iter_max):
            self.find_cnts(TIME_LOG=self.time_log)
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
                break

        print('converage at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E, 'alpha', alpha,
              'gTp', gTp, 'pHp', pHp)
        self.update_v()

        if self.time_log:
            ti.sync()
            t_frame_end = time.perf_counter()
            print(f"[Frame Time] total: {(t_frame_end - t_frame_start)*1000:.2f}ms")

        self.frame += 1
        return iter

    def visual(self):
        window = ti.ui.Window("Visualization", (800, 600), vsync=True)
        dir = '../demo_results/final/' + self.demo + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(*self.camera_position)
        camera.lookat(*self.camera_lookat)

        while window.running:
            self.init_lines()
            camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            scene.mesh(self.mesh.verts.x, self.indices, per_vertex_color=self.per_vertex_color)
            scene.lines(self.lines, color = (1.0, 1.0, 1.0), width = 1.0)

            canvas.scene(scene)
            window.show()
            self.step()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Squeeze Armadillo Demo (Spatial Hash)')
    parser.add_argument('--headless', action='store_true', help='run without GUI')
    parser.add_argument('--frames', type=int, default=100, help='number of frames to run in headless mode')
    parser.add_argument('--time-log', action='store_true', help='enable timing log for performance')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)
    demo = 'squeeze_four_armadillo'
    ipc_deformer = squeeze_armadillo_demo_spatial_hash(demo=demo)
    ipc_deformer.time_log = args.time_log  # Enable timing if requested
    ipc_deformer.init_visual()
    print('init finish')
    ipc_deformer.find_cnts(PRINT=True, TIME_LOG=args.time_log)

    if args.headless:
        ipc_deformer.run_headless(args.frames)
    else:
        ipc_deformer.visual()
