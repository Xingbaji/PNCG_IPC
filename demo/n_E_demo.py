import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
from algorithm.pncg_base_ipc import *

@ti.data_oriented
class pncg_index(pncg_ipc_deformer):
    def set_index(self):
        # initialize model
        self.mesh.verts.place({'index': ti.i32})
        self.per_vertex_color =  ti.Vector.field(3, dtype=float,shape=self.n_verts)
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)
        self.init_index()

        self.frame = 0
        self.CNT = 0

    @ti.kernel
    def init_index(self):
        # setting index for each vertex
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            vert.index = index

    @ti.kernel
    def compute_DK_index(self):
        # compute DK direction for each object
        g_Py_index = ti.Vector.zero(float, self.N_object)
        y_p_index = ti.Vector.zero(float, self.N_object)
        y_Py_index = ti.Vector.zero(float, self.N_object)
        g_p_index = ti.Vector.zero(float, self.N_object)
        beta_index = ti.Vector.zero(float, self.N_object)
        for vert in self.mesh.verts:
            index = vert.index
            y = vert.grad - vert.grad_prev
            Py = y / vert.diagH
            y_p_index[index] += y.dot(vert.p)
            g_Py_index[index] += vert.grad.dot(Py)
            y_Py_index[index] += y.dot(Py)
            g_p_index[index] += vert.grad.dot(vert.p)

        for i in range(self.N_object):
            beta_index[i] = (g_Py_index[i] - y_Py_index[i] * g_p_index[i] / y_p_index[i]) / y_p_index[i]
        for vert in self.mesh.verts:
            index = vert.index
            vert.p = - vert.grad / vert.diagH + beta_index[index] * vert.p

    @ti.kernel
    def compute_alpha_index_and_update_x(self)->float:
        gTp_index = ti.Vector.zero(float, self.N_object)
        pHp_index = ti.Vector.zero(float, self.N_object)
        alpha_index = ti.Vector.zero(float, self.N_object)
        p_max_index = ti.Vector.zero(float, self.N_object)
        for vert in self.mesh.verts:
            index = vert.index
            gTp_index[index] += vert.grad.dot(vert.p)

        for vert in self.mesh.verts:
            index = vert.index
            pHp_index[index] += vert.p.norm_sqr() * vert.m

        for c in self.mesh.cells:
            index = c.verts[0].index
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            p = ti.Vector.zero(float, 12)
            p[0:3] = c.verts[0].p
            p[3:6] = c.verts[1].p
            p[6:9] = c.verts[2].p
            p[9:12] = c.verts[3].p
            tmp = self.compute_p_d2Psidx2_p(F, B, p, self.mu, self.la)
            pHp_index[index] += c.W * self.dt ** 2 * ti.max(tmp, 0.0)

        for k,j in self.cid:
            pair = self.cid[k,j]
            ids = pair.a
            dist = pair.b
            cord = pair.c
            t = pair.d
            dist2 = dist * dist
            bg = self.barrier_g(dist)
            para1 = bg / dist
            para0 = (self.barrier_H(dist) - para1) / dist2
            p_tmp = ti.Vector.zero(float, 12)
            p_tmp[0:3] = self.mesh.verts.p[ids[0]]
            p_tmp[3:6] = self.mesh.verts.p[ids[1]]
            p_tmp[6:9] = self.mesh.verts.p[ids[2]]
            p_tmp[9:12] = self.mesh.verts.p[ids[3]]
            dtdx_t = compute_dtdx_t(t, cord)
            pHp_0 = para0 * ( p_tmp.dot(dtdx_t) ** 2 )
            d_dtdx = compute_d_dtdx(p_tmp, cord)
            pHp_1 = para1 * d_dtdx.norm_sqr()
            pHp = ti.max(pHp_0 + pHp_1,0.0)
            index0 = self.mesh.verts.index[ids[0]]
            index1 = self.mesh.verts.index[ids[2]]
            pHp_index[index0] += pHp * 0.5 # 平分pHp
            pHp_index[index1] += pHp * 0.5

        min_dist = 1e-2 * self.dHat
        for i in range(self.boundary_points.shape[0]):
            p = self.boundary_points[i]
            index = self.mesh.verts.index[p]
            x_a0 = self.mesh.verts.x[p]
            dist = x_a0[1] - self.ground
            if dist < self.dHat:
                if dist <= min_dist:
                    dist = min_dist
                p_tmp = self.mesh.verts.p[p][1]
                ret_value = p_tmp * self.barrier_H(dist) * p_tmp
                pHp_index[index] += ret_value

        for vert in self.mesh.verts:
            index = vert.index
            d_norm = vert.p.norm()
            ti.atomic_max(p_max_index[index], d_norm)
        # print('p_max', p_max_index)
        Delta_E = 0.0
        for i in range(self.N_object):
            alpha_i = - gTp_index[i] / pHp_index[i]
            if alpha_i * p_max_index[i] > 0.5 * self.dHat:
                alpha_index[i] = 0.5 * self.dHat / p_max_index[i]
            else:
                alpha_index[i] = alpha_i
            Delta_E -= (alpha_index[i] * gTp_index[i] + 0.5 * alpha_index[i]**2 * pHp_index[i])

        for vert in self.mesh.verts:
            index = vert.index
            alpha = alpha_index[index]
            vert.x += alpha * vert.p
        return Delta_E

    def step(self):
        print('Frame', self.frame)
        self.assign_xn_xhat()
        for iter in range(self.iter_max):
            # print('iter', iter)
            self.find_cnts()
            self.compute_grad_and_diagH()
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_index()
            delta_E = self.compute_alpha_index_and_update_x()
            if iter == 0:
                delta_E_init = delta_E
            if delta_E < self.epsilon * delta_E_init:
                break
        print('finish at iter', iter, 'rate', delta_E / delta_E_init, 'delta_E', delta_E)
        self.update_v_and_bound()
        self.frame += 1
        return iter




if __name__ == '__main__':
    ti.init(arch=ti.gpu, default_fp=ti.f32)#, device_memory_fraction=0.9)#, kernel_profiler=True)
    demo = 'eight_E_drop_demo_contact'
    ipc_deformer = pncg_index(demo=demo)
    ipc_deformer.set_index()
    ipc_deformer.visual()
    # ipc_deformer.save_new(500, SAVE_OBJ=True)