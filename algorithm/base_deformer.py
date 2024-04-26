import imageio
from math_utils.elastic_util import *
from util.logger import *
from util.model_loading import *

@ti.data_oriented
class base_deformer:
    def __init__(self, demo):
        #set physical paramters
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.mu, self.la = model.mu, model.la
        self.dict = model.dict
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.frame = 0
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.mesh = model.mesh

        self.mesh.verts.place({'x': ti.math.vec3,
                          'v': ti.math.vec3,
                          'm': float})

        self.mesh.cells.place({'B': ti.math.mat3,
                          'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])
        print('load mesh finish')
        self.precompute()
        print('precompute finish')
        self.indices = ti.field(ti.u32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        print('init indice finish')
        self.assign_elastic_type(model.elastic_type)
        self.set_point_lights()

    def assign_elastic_type(self,elastic):
        #set elastic type
        if elastic=='ARAP': # as rigig as possible
            self.compute_Psi = compute_Psi_ARAP
            self.compute_dPsidx = compute_dPsidx_ARAP # dPsidx = \frac{\partial \PSi}{\partial x}
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP # d2Psidx2 is \frac{\partial^2 \PSi}{\partial x^2}
            self.compute_p_d2Psidx2_p = compute_pHp_ARAP
        elif elastic == 'SNH': # stable neo-hookean
            self.compute_Psi = compute_Psi_SNH
            self.compute_dPsidx = compute_dPsidx_SNH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_SNH
            self.compute_p_d2Psidx2_p = compute_pHp_SNH
        elif elastic=='ARAP_filter':
            # ARAP_filter: filter the eigen value to make the hessian matrix spd
            self.compute_Psi = compute_Psi_ARAP
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_filter
            self.compute_p_d2Psidx2_p = compute_pHp_ARAP_filter
        elif elastic=='FCR': # fixed corotated
            self.compute_Psi = compute_Psi_FCR
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR
            self.compute_p_d2Psidx2_p = compute_pHp_FCR
        elif elastic=='FCR_filter': # fixed corotated with filterd ARAP eigen value
            self.compute_Psi = compute_Psi_FCR
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR_filter
            self.compute_p_d2Psidx2_p = compute_pHp_FCR_filter
        elif elastic == 'NH': # neo-hookean
            self.compute_Psi = compute_Psi_NH
            self.compute_dPsidx = compute_dPsidx_NH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_NH
            self.compute_p_d2Psidx2_p = compute_pHp_NH
        else:
            print('Wrong elastic type')

    def step(self):
        pass

    @ti.kernel
    def precompute(self):
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1,4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4.0

    @ti.kernel
    def init_indices(self):
        for c in self.mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    self.indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


    def save_results(self, n_frames=300, SAVE_OBJ=False, SAVE_PNG=True):
        import json
        import time
        window = ti.ui.Window("visual fem", (800, 600), show_window=False)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        camera.fov(75)
        dir = '../demo_results/final/' + self.demo + '/'
        img_dir = dir+'imgs/'
        obj_dir = dir+'objs/'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            os.makedirs(obj_dir)
        with open(dir + 'config.json', 'w') as f:
            json.dump(self.config, f)
        writer = imageio.get_writer(os.path.join(dir, self.demo+'.gif'), duration=10)
        canvas.set_background_color(color=(0.2, 0.2, 0.2))
        point_light_color = (1.0, 0.9, 0.8)  # Warm white color
        diffuse_color = (0.6, 0.4, 0.2)
        Times = []
        Iters = []
        self.step()
        with Logger(dir + f'log.txt'):
            for i in range(n_frames):
                tic = time.time()
                step_iters = self.step()
                toc = time.time()
                step_time = toc - tic
                print('Time', step_time, 'FPS', 1.0 / step_time)
                Times.append(step_time)
                Iters.append(step_iters)
                if SAVE_OBJ == True:
                    self.save_obj(obj_dir)
                if SAVE_PNG == True:
                    scene.set_camera(camera)
                    scene.ambient_light((0.1, 0.1, 0.1))
                    for light_pos in self.point_lights:
                        scene.point_light(pos=light_pos, color=point_light_color)
                    scene.mesh_instance(self.mesh.verts.x, self.indices, show_wireframe=False,
                                        color=diffuse_color)
                    canvas.scene(scene)
                    filename = f'{self.frame:05d}.png'
                    save_path = os.path.join(img_dir, filename)
                    window.save_image(save_path)
                    writer.append_data(imageio.v3.imread(save_path))
                    print(filename, 'saved')
            average_time = sum(Times) / len(Times)
            print('average time', average_time, 'average FPS', 1.0 / average_time)
            print('times', Times)
            FPS = [1.0 / t for t in Times]
            print('FPS', FPS)
            print('Iters', Iters)
        writer.close()
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(Times, label='Time')
        plt.savefig(dir + f'times.png')
        plt.close()
        plt.figure()
        plt.plot(Iters, label='Iters')
        plt.savefig(dir + f'iters.png')
        plt.close()
        plt.figure()
        plt.plot(FPS, label='FPS')
        plt.savefig(dir + f'FPS.png')
        plt.close()

    def convert_pngs2gif(self,path):
        writer = imageio.get_writer(os.path.join(path, self.demo+'.gif'), duration=10)
        for file in os.listdir(path):
            if file.endswith('.png'):
                writer.append_data(imageio.v3.imread(file))
        writer.close()
        print('gifs saved')

    def save_obj(self,obj_dir, rate = 1):
        if self.frame % rate == 0:
            x_np = self.mesh.verts.x.to_numpy()
            boundary_triangles_np = self.boundary_triangles.to_numpy()
            directory = obj_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            f = open(directory + f'{self.frame:03d}.obj', 'w')
            for i in range(self.n_verts):
                f.write('v %.6f %.6f %.6f\n' % (x_np[i, 0] , x_np[i, 1] , x_np[i, 2] ))
            for [p0, p1, p2] in boundary_triangles_np:
                f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
            f.close()
            v_np = self.mesh.verts.v.to_numpy()
            np.save(directory + f'x{self.frame:03d}.npy', x_np)
            np.save(directory + f'v{self.frame:03d}.npy', v_np)

    def load_frame(self, path, frame):
        self.frame = frame
        print(f'x{frame:03d}.npy')
        x_np = np.load(path + f'x{frame:03d}.npy')
        v_np = np.load(path + f'v{frame:03d}.npy')
        self.mesh.verts.x.from_numpy(x_np)
        self.mesh.verts.v.from_numpy(v_np)

    def restart(self):
        self.mesh.verts.x.copy_from(self.mesh.verts.x_init)
        self.mesh.verts.v.fill(0)
        self.frame = 0

    @ti.kernel
    def init_v(self,v:float):
        for vert in self.mesh.verts:
            vert.v[1] = v

    def set_point_lights(self, num_lights=6, light_distance=1.5):
        node_np = self.mesh.get_position_as_numpy()
        node_center = np.mean(node_np, axis=0)
        node_extents = np.max(node_np, axis=0) - np.min(node_np, axis=0)
        max_extent = np.max(node_extents)

        self.point_lights = []
        for i in range(num_lights):
            # Calculate spherical coordinates
            theta = np.pi * (1.0 + 2 * i / num_lights)
            phi = np.arccos(-1.0 + 2 * i / num_lights)

            # Convert to Cartesian coordinates
            x = light_distance * max_extent * np.sin(phi) * np.cos(theta)
            y = light_distance * max_extent * np.sin(phi) * np.sin(theta)
            z = light_distance * max_extent * np.cos(phi)

            # Offset from the center of the mesh
            light_position = node_center + np.array([x, y, z])
            self.point_lights.append(light_position)


    def visual(self):
        print('precombile')
        self.step()
        print('precombile finish')
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
        point_light_color = (1.0, 0.9, 0.8)  # Warm white color
        diffuse_color = (0.6, 0.4, 0.2)
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    self.restart()
                if window.event.key == 'b':
                    break
            camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.1, 0.1, 0.1))
            for light_pos in self.point_lights:
                scene.point_light(pos=light_pos, color=point_light_color)
            scene.mesh_instance(self.mesh.verts.x, self.indices, show_wireframe=False,
                                 color=diffuse_color)
            canvas.scene(scene)
            window.show()
            self.step()

