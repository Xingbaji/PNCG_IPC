from algorithm.pncg_base_collision_free import *
from math_utils.graphic_util import *
from util.model_loading import *

@ti.data_oriented
class collision_detection_module(pncg_base_deformer):
    def __init__(self, demo = 'cube_0'):
        model = model_loading(demo=demo)
        self.demo = demo
        print('demo', self.demo)
        self.mu = ti.field(dtype=ti.f32,shape=())
        self.la = ti.field(dtype=ti.f32,shape=())
        self.mu[None], self.la[None]= model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.adj = model.adj
        self.frame = 0
        self.SMALL_NUM = 1e-6

        # initialize model
        self.mesh.verts.place({'x': ti.types.vector(3,float),
                               'v': ti.types.vector(3,float),
                               'm': float,
                               'x_n': ti.types.vector(3,float),
                               'x_hat': ti.types.vector(3,float),
                               'x_prev': ti.types.vector(3,float),
                               'x_init': ti.types.vector(3, float),
                               'grad': ti.types.vector(3,float),
                               'grad_prev': ti.types.vector(3,float),
                               'diagH': ti.types.vector(3,float),
                               'p': ti.types.vector(3,float),
                               })

        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print('n_verts,n_cells',self.n_verts,self.n_cells)

        #precompute
        self.precompute()
        self.indices = ti.field(ti.i32, shape=len(self.mesh.cells) * 4 * 3)
        self.init_indices()
        self.assign_elastic_type(model.elastic_type)

        self.boundary_points = model.boundary_points
        self.boundary_edges = model.boundary_edges
        self.boundary_triangles = model.boundary_triangles
        self.n_boundary_points = self.boundary_points.shape[0]
        self.n_boundary_edges = self.boundary_edges.shape[0]
        self.n_boundary_triangles = self.boundary_triangles.shape[0]
        print('boundary size ', self.n_boundary_points, self.n_boundary_edges, self.n_boundary_triangles)
        self.dHat = model.dHat

    def init_hash_grid(self):
        # the implement of spatial hashing is from https://github.com/matthias-research/pages
        # Bresenham's line algorithm is used to assign edges to grid. It is not rigorous, but it semms ok.
        # TODO: Xiaolin Wu's line algorithm may be better; Use better spatial hashing techinique
        self.mean_edge_len = self.compute_mean_of_boundary_edges()
        self.spatial_hash_inv_dx = 1.0 / self.mean_edge_len
        print('boundary edge number', self.n_boundary_edges, 'mean edge length', self.mean_edge_len)
        self.table_size = int(3 * self.n_boundary_edges)

        #seperately assign points and edges with two tables
        self.cell_start_E = ti.field(ti.i32)
        ti.root.dense(ti.i, self.table_size+1).place(self.cell_start_E)
        self.cell_entries_E = ti.field(ti.i32)
        ti.root.dense(ti.i, self.table_size).place(self.cell_entries_E)

        self.cell_start_P = ti.field(ti.i32)
        ti.root.dense(ti.i, self.table_size +1).place(self.cell_start_P)
        self.cell_entries_P = ti.field(ti.i32)
        ti.root.dense(ti.i, self.table_size).place(self.cell_entries_P)

        self.MAX_C = 2**21 # Max number of constraints
        # (a:ids, b:dist, c:cord, d:t)
        self.pair = ti.types.struct(a=ti.types.vector(4, ti.u32), b=float, c= ti.types.vector(4, float), d = ti.types.vector(3, float))
        self.cid = self.pair.field()
        self.cid_root = ti.root.bitmasked(ti.ij,(2,self.MAX_C)).place(self.cid) # we only store the PT and EE constraints.
        self.pse_E = ti.algorithms.PrefixSumExecutor(self.table_size + 1)
        self.pse_P = ti.algorithms.PrefixSumExecutor(self.table_size + 1)

        if self.adj == 1:
            # when dHat is too large, there would be constraints at rest pose, so filter some constraints
            self.define_adj_matrix()
            self.attempt_PT = self.attempt_PT_adj
            self.attempt_EE = self.attempt_EE_adj
        else:
            self.attempt_PT = self.attempt_PT_no_adj
            self.attempt_EE = self.attempt_EE_no_adj


    @ti.func
    def hash_coords(self, x):
        h = (x[0] * 92837111) ^ (x[1] * 689287499) ^ (x[2] * 283923481)
        return ti.abs(h) % self.table_size

    @ti.func
    def hash_coords_2(self, x, y):
        h = (x * 92837111) ^ (y * 689287499)
        return ti.abs(h) % self.MAX_C

    @ti.func
    def int_coord(self, coord):
        return ti.floor(coord * self.spatial_hash_inv_dx,ti.i32)

    @ti.func
    def hash_pos(self, x):
        return self.hash_coords(self.int_coord(x))

    @ti.kernel
    def count_cells_points(self):
        # the count of cells
        for i in range(self.n_boundary_points):
            x = self.mesh.verts.x[self.boundary_points[i]]
            hash_id = self.hash_pos(x)
            ti.atomic_add(self.cell_start_P[hash_id],1)

    @ti.kernel
    def count_cells_edges(self):
        # the count of cells. Bresenham's line algorithm is used.
        for p in range(self.n_boundary_edges):
            a0 = self.boundary_edges[p, 0]
            a1 = self.boundary_edges[p, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            a0_ind = self.int_coord(x_a0)
            a1_ind = self.int_coord(x_a1)
            aim_ind = a0_ind
            x0 = a0_ind[0]
            y0 = a0_ind[1]
            z0 = a0_ind[2]
            x1 = a1_ind[0]
            y1 = a1_ind[1]
            z1 = a1_ind[2]
            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0
            x_inc = 1
            if dx < 0:
                x_inc = -1
            l = ti.abs(dx)
            y_inc = 1
            if dy < 0:
                y_inc = -1
            m = ti.abs(dy)
            z_inc = 1
            if dz < 0:
                z_inc = -1
            n = ti.abs(dz)
            dx1 = l * 2
            dy1 = m * 2
            dz1 = n * 2

            if (l >= m) and (l >= n):
                err_1 = dy1 - l
                err_2 = dz1 - l
                for i in range(l):
                    hash_id = self.hash_coords(aim_ind)
                    ti.atomic_add(self.cell_start_E[hash_id], 1)
                    if err_1 > 0:
                        aim_ind[1] += y_inc
                        err_1 -= dx1
                    if err_2 > 0:
                        aim_ind[2] += z_inc
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    aim_ind[0] += x_inc
            elif (m >= l) and (m >= n):
                err_1 = dx1 - m
                err_2 = dz1 - m
                for i in range(m):
                    hash_id = self.hash_coords(aim_ind)
                    ti.atomic_add(self.cell_start_E[hash_id], 1)
                    if (err_1 > 0):
                        aim_ind[0] += x_inc
                        err_1 -= dy1
                    if (err_2 > 0):
                        aim_ind[2] += z_inc
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    aim_ind[1] += y_inc
            else:
                err_1 = dy1 - n
                err_2 = dx1 - n
                for i in range(n):
                    hash_id = self.hash_coords(aim_ind)
                    ti.atomic_add(self.cell_start_E[hash_id], 1)
                    if (err_1 > 0):
                        aim_ind[1] += y_inc
                        err_1 -= dz1
                    if (err_2 > 0):
                        aim_ind[0] += x_inc
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    aim_ind[2] += z_inc
            hash_id = self.hash_coords(aim_ind)
            ti.atomic_add(self.cell_start_E[hash_id], 1)

    @ti.kernel
    def fill_in_cells_points(self):
        for i in range(self.n_boundary_points):
            x_id = self.boundary_points[i]
            x = self.mesh.verts.x[x_id]
            hash_id = self.hash_pos(x)
            id_old = ti.atomic_sub(self.cell_start_P[hash_id],1)
            self.cell_entries_P[id_old-1] = x_id

    @ti.kernel
    def fill_in_cells_edges(self):
        for p in range(self.n_boundary_edges):
            a0 = self.boundary_edges[p, 0]
            a1 = self.boundary_edges[p, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            a0_ind = self.int_coord(x_a0)
            a1_ind = self.int_coord(x_a1)
            aim_ind = a0_ind
            x0 = a0_ind[0]
            y0 = a0_ind[1]
            z0 = a0_ind[2]
            x1 = a1_ind[0]
            y1 = a1_ind[1]
            z1 = a1_ind[2]
            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0
            x_inc = 1
            if dx < 0:
                x_inc = -1
            l = ti.abs(dx)
            y_inc = 1
            if dy < 0:
                y_inc = -1
            m = ti.abs(dy)
            z_inc = 1
            if dz < 0:
                z_inc = -1
            n = ti.abs(dz)
            dx1 = l * 2
            dy1 = m * 2
            dz1 = n * 2

            if (l >= m) and (l >= n):
                err_1 = dy1 - l
                err_2 = dz1 - l
                for i in range(l):
                    hash_id = self.hash_coords(aim_ind)
                    id_old = ti.atomic_sub(self.cell_start_E[hash_id], 1)
                    self.cell_entries_E[id_old - 1] = p
                    if err_1 > 0:
                        aim_ind[1] += y_inc
                        err_1 -= dx1
                    if err_2 > 0:
                        aim_ind[2] += z_inc
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    aim_ind[0] += x_inc
            elif (m >= l) and (m >= n):
                err_1 = dx1 - m
                err_2 = dz1 - m
                for i in range(m):
                    hash_id = self.hash_coords(aim_ind)
                    id_old = ti.atomic_sub(self.cell_start_E[hash_id], 1)
                    self.cell_entries_E[id_old - 1] = p
                    if (err_1 > 0):
                        aim_ind[0] += x_inc
                        err_1 -= dy1
                    if (err_2 > 0):
                        aim_ind[2] += z_inc
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    aim_ind[1] += y_inc
            else:
                err_1 = dy1 - n
                err_2 = dx1 - n
                for i in range(n):
                    hash_id = self.hash_coords(aim_ind)
                    id_old = ti.atomic_sub(self.cell_start_E[hash_id], 1)
                    self.cell_entries_E[id_old - 1] = p
                    if (err_1 > 0):
                        aim_ind[1] += y_inc
                        err_1 -= dz1
                    if (err_2 > 0):
                        aim_ind[0] += x_inc
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    aim_ind[2] += z_inc
            hash_id = self.hash_coords(aim_ind)
            id_old = ti.atomic_sub(self.cell_start_E[hash_id], 1)
            self.cell_entries_E[id_old - 1] = p

    @ti.kernel
    def fill_in_cells_edges_adj(self):
        for i in range(self.n_boundary_edges):
            a0 = self.boundary_edges[i, 0]
            a1 = self.boundary_edges[i, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            lower = ti.floor(ti.min(x_a0, x_a1) * self.spatial_hash_inv_dx,ti.i32)
            upper = ti.floor(ti.max(x_a0, x_a1) * self.spatial_hash_inv_dx,ti.i32)  + 1
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.hash_coords(I)
                id_old = ti.atomic_sub(self.cell_start_E[hash_id],1)
                self.cell_entries_E[id_old-1] = i

    def find_cnts(self,PRINT=False):
        self.cid_root.deactivate_all()
        # EE
        self.cell_start_E.fill(0)
        self.cell_entries_E.fill(-1)
        self.count_cells_edges()
        self.pse_E.run(self.cell_start_E)
        self.fill_in_cells_edges()
        self.find_constraints_EE()
        # PT
        self.cell_start_P.fill(0)
        self.cell_entries_P.fill(-1)
        self.count_cells_points()
        self.pse_P.run(self.cell_start_P)
        self.fill_in_cells_points()
        self.find_constraints_PT()
        if PRINT == True:
            N = self.print_cnts()
            return N

    def find_cnts_iter(self, iter, rate = 10, PRINT=False):
        # It seems to be not necessary to update the grid each iteration,
        self.cid_root.deactivate_all()
        if iter % rate == 0 or iter < rate:
            self.cell_start_E.fill(0)
            self.cell_entries_E.fill(-1)
            self.count_cells_edges()
            self.pse_E.run(self.cell_start_E)
            self.fill_in_cells_edges()

            self.cell_start_P.fill(0)
            self.cell_entries_P.fill(-1)
            self.count_cells_points()
            self.pse_P.run(self.cell_start_P)
            self.fill_in_cells_points()

        self.find_constraints_PT()
        self.find_constraints_EE()
        if PRINT == True:
            self.print_cnts()


    def check_dcd(self):
        self.cell_start_E.fill(0)
        self.cell_entries_E.fill(-1)
        self.count_cells_edges()
        self.pse_E.run(self.cell_start_E)
        self.fill_in_cells_edges()
        ret = self.check_collision_3d()
        return ret

    @ti.kernel
    def check_collision_3d(self) -> ti.i32:
        # check is there any segment intersect with triangle
        result = 0
        for i in range(self.n_boundary_triangles):
            t0 = self.boundary_triangles[i, 0]
            t1 = self.boundary_triangles[i, 1]
            t2 = self.boundary_triangles[i, 2]
            x0 = self.mesh.verts.x[t0]
            x1 = self.mesh.verts.x[t1]
            x2 = self.mesh.verts.x[t2]
            lower = ti.floor( (ti.min(x0, x1, x2) - self.dHat) * self.spatial_hash_inv_dx,ti.i32)
            upper = ti.floor( (ti.max(x0, x1, x2) + self.dHat) * self.spatial_hash_inv_dx,ti.i32) + 1
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.hash_coords(I)
                start = self.cell_start_E[hash_id]
                end = self.cell_start_E[hash_id + 1]
                for k in range(start, end):
                    j = self.cell_entries_E[k]
                    a0 = self.boundary_edges[j, 0]
                    a1 = self.boundary_edges[j, 1]
                    if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
                        x_a0 = self.mesh.verts.x[a0]
                        x_a1 = self.mesh.verts.x[a1]
                        rrr = segment_intersect_triangle_new(x_a0, x_a1, x0, x1, x2)
                        if rrr:
                            result = 1
        return result

    @ti.kernel
    def find_constraints_PT(self):
        for i in range(self.n_boundary_triangles):
            t0 = self.boundary_triangles[i, 0]
            t1 = self.boundary_triangles[i, 1]
            t2 = self.boundary_triangles[i, 2]
            x0 = self.mesh.verts.x[t0]
            x1 = self.mesh.verts.x[t1]
            x2 = self.mesh.verts.x[t2]
            lower = ti.floor( (ti.min(x0, x1, x2) - self.dHat) * self.spatial_hash_inv_dx,ti.i32)
            upper = ti.floor( (ti.max(x0, x1, x2) + self.dHat) * self.spatial_hash_inv_dx,ti.i32) + 1
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.hash_coords(I)
                start = self.cell_start_P[hash_id]
                end = self.cell_start_P[hash_id + 1]
                for j in range(start, end):
                    p = self.cell_entries_P[j]
                    xp = self.mesh.verts.x[p]
                    self.attempt_PT(i, p, t0, t1, t2, xp, x0, x1, x2)
    @ti.func
    def attempt_PT_no_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[0, hash_index] = self.pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_PT_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        hash_adj = self.hash_coords_2(p, triangle_id)
        if p != t0 and p != t1 and p != t2 and self.adj_matrix[hash_adj] == 0 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[0,hash_index] = self.pair(ids, dist, cord, t_pt)

    @ti.kernel
    def find_constraints_EE(self):
        for p in range(self.n_boundary_edges):
            a0 = self.boundary_edges[p, 0]
            a1 = self.boundary_edges[p, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            a0_ind = self.int_coord(x_a0)
            a1_ind = self.int_coord(x_a1)
            aim_ind = a0_ind
            x0 = a0_ind[0]
            y0 = a0_ind[1]
            z0 = a0_ind[2]
            x1 = a1_ind[0]
            y1 = a1_ind[1]
            z1 = a1_ind[2]
            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0
            x_inc = 1
            if dx < 0:
                x_inc = -1
            l = ti.abs(dx)
            y_inc = 1
            if dy < 0:
                y_inc = -1
            m = ti.abs(dy)
            z_inc = 1
            if dz < 0:
                z_inc = -1
            n = ti.abs(dz)
            dx1 = l * 2
            dy1 = m * 2
            dz1 = n * 2

            if (l >= m) and (l >= n):
                err_1 = dy1 - l
                err_2 = dz1 - l
                for i in range(l):
                    hash_id = self.hash_coords(aim_ind)
                    self.attempt_EE(hash_id, p, a0, a1, x_a0, x_a1)
                    if err_1 > 0:
                        aim_ind[1] += y_inc
                        err_1 -= dx1
                    if err_2 > 0:
                        aim_ind[2] += z_inc
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    aim_ind[0] += x_inc
            elif (m >= l) and (m >= n):
                err_1 = dx1 - m
                err_2 = dz1 - m
                for i in range(m):
                    hash_id = self.hash_coords(aim_ind)
                    self.attempt_EE(hash_id, p, a0, a1, x_a0, x_a1)
                    if (err_1 > 0):
                        aim_ind[0] += x_inc
                        err_1 -= dy1
                    if (err_2 > 0):
                        aim_ind[2] += z_inc
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    aim_ind[1] += y_inc
            else:
                err_1 = dy1 - n
                err_2 = dx1 - n
                for i in range(n):
                    hash_id = self.hash_coords(aim_ind)
                    self.attempt_EE(hash_id, p, a0, a1, x_a0, x_a1)
                    if (err_1 > 0):
                        aim_ind[1] += y_inc
                        err_1 -= dz1
                    if (err_2 > 0):
                        aim_ind[0] += x_inc
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    aim_ind[2] += z_inc
            hash_id = self.hash_coords(aim_ind)
            self.attempt_EE(hash_id, p, a0, a1, x_a0, x_a1)

    @ti.func
    def attempt_EE_no_adj(self, hash_id, edge_id_0, a0, a1, x_a0, x_a1):
        start = self.cell_start_E[hash_id]
        end = self.cell_start_E[hash_id + 1]
        for k in range(start, end):
            j = self.cell_entries_E[k]
            if edge_id_0 < j:
                b0 = self.boundary_edges[j, 0]
                b1 = self.boundary_edges[j, 1]
                x_b0 = self.mesh.verts.x[b0]
                x_b1 = self.mesh.verts.x[b1]
                if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                    dist = t_ee.norm()
                    if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM: # if dist is too small, ipc potential would be too large
                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                        hash_index = self.hash_coords_2(edge_id_0, j)
                        self.cid[1, hash_index] = self.pair(ids, dist, cord, t_ee)

    @ti.func
    def attempt_EE_adj(self, hash_id, edge_id_0, a0, a1, x_a0, x_a1):
        start = self.cell_start_E[hash_id]
        end = self.cell_start_E[hash_id + 1]
        for k in range(start, end):
            edge_id_1 = self.cell_entries_E[k]
            hash_adj = self.hash_coords_2(self.n_verts + edge_id_0, edge_id_1)
            if edge_id_0 < edge_id_1 and self.adj_matrix[hash_adj] == 0:
                b0 = self.boundary_edges[edge_id_1, 0]
                b1 = self.boundary_edges[edge_id_1, 1]
                x_b0 = self.mesh.verts.x[b0]
                x_b1 = self.mesh.verts.x[b1]
                if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)  # check sc, tc in [0,1]
                    dist = t_ee.norm()
                    if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                        hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                        self.cid[1,hash_index] = self.pair(ids, dist, cord, t_ee)

    @ti.kernel
    def compute_min_edge_edge_dist(self)->float:
        min_edge_edge_dist = 10.0
        for i in range(self.n_boundary_edges):
            a0 = self.boundary_edges[i, 0]
            a1 = self.boundary_edges[i, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            for j in range(self.n_boundary_edges):
                b0 = self.boundary_edges[j, 0]
                b1 = self.boundary_edges[j, 1]
                x_b0 = self.mesh.verts.x[b0]
                x_b1 = self.mesh.verts.x[b1]
                if self.adj_matrix[a0,b0] == 0 and self.adj_matrix[a0,b1] == 0 and self.adj_matrix[a1,b0] == 0 and self.adj_matrix[a1,b1] == 0:
                    t_ee,_,_ = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                    ti.atomic_min(min_edge_edge_dist, t_ee.norm())
        print('min edge edge dist at start:', min_edge_edge_dist)
        return min_edge_edge_dist

    def define_adj_matrix(self):
        self.adj_matrix = ti.field(bool)
        ti.root.bitmasked(ti.i, 2 ** 24).place(self.adj_matrix)
        self.cell_start_E.fill(0)
        self.cell_entries_E.fill(-1)
        self.count_cells_edges()
        self.pse_E.run(self.cell_start_E)
        self.fill_in_cells_edges()
        self.assign_adj_matrix_EE()
        # PT
        self.cell_start_P.fill(0)
        self.cell_entries_P.fill(-1)
        self.count_cells_points()
        self.pse_P.run(self.cell_start_P)
        self.fill_in_cells_points()
        self.assign_adj_matrix_PT()
        print('adj matrix defined')

    @ti.kernel
    def assign_adj_matrix_PT(self):
        n_pT = 0
        for i in range(self.n_boundary_triangles):
            t0 = self.boundary_triangles[i, 0]
            t1 = self.boundary_triangles[i, 1]
            t2 = self.boundary_triangles[i, 2]
            x0 = self.mesh.verts.x[t0]
            x1 = self.mesh.verts.x[t1]
            x2 = self.mesh.verts.x[t2]
            lower = ti.floor( (ti.min(x0, x1, x2) - self.dHat) * self.spatial_hash_inv_dx,ti.i32)
            upper = ti.floor( (ti.max(x0, x1, x2) + self.dHat) * self.spatial_hash_inv_dx,ti.i32) + 1
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.hash_coords(I)
                start = self.cell_start_P[hash_id]
                end = self.cell_start_P[hash_id + 1]
                # print('start', start, 'end', end, 'size', end-start)
                for j in range(start, end):
                    # n += 1
                    p = self.cell_entries_P[j]
                    xp = self.mesh.verts.x[p]
                    if point_triangle_ccd_broadphase(xp, x0, x1, x2, 1.0 * self.dHat) and p != t0 and p != t1 and p != t2:
                        # cord0, cord1, cord2 = dist3D_Point_Triangle(xp, x0, x1, x2)
                        # xt = cord0 * x0 + cord1 * x1 + cord2 * x2
                        # t_pt = xp - xt
                        # dist = t_pt.norm()
                        # if dist < 1.0 * self.dHat and p != t0 and p != t1 and p != t2:
                        hash_index = self.hash_coords_2(p, i)
                        self.adj_matrix[hash_index] = 1
                        ti.atomic_add(n_pT, 1)
        print('n_pT', n_pT)

        N = 0
        for i in self.adj_matrix:
            N += 1
        print('N all', N)

    @ti.kernel
    def assign_adj_matrix_EE(self):
        n_EE = 0
        for i in range(self.n_boundary_edges):
            a0 = self.boundary_edges[i, 0]
            a1 = self.boundary_edges[i, 1]
            x_a0 = self.mesh.verts.x[a0]
            x_a1 = self.mesh.verts.x[a1]
            lower = ti.floor( (ti.min(x_a0, x_a1) - self.dHat) * self.spatial_hash_inv_dx,ti.i32)
            upper = ti.floor( (ti.max(x_a0, x_a1) + self.dHat) * self.spatial_hash_inv_dx,ti.i32) + 1
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.hash_coords(I)
                start = self.cell_start_E[hash_id]
                end = self.cell_start_E[hash_id + 1]
                for k in range(start, end):
                    j = self.cell_entries_E[k]
                    if i < j:
                        b0 = self.boundary_edges[j, 0]
                        b1 = self.boundary_edges[j, 1]
                        x_b0 = self.mesh.verts.x[b0]
                        x_b1 = self.mesh.verts.x[b1]
                        if edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, 1.0 * self.dHat) and a0 != b0 and a1 != b1 and a0 != b1 and a1 != b0:
                            # removing the adjacent edges, 1.5 or larger is also ok
                            # t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)  # check sc, tc in [0,1]
                            # dist = t_ee.norm()
                            # if dist < 1.0 * self.dHat and a0 != b0 and a1 != b1 and a0 != b1 and a1 != b0:
                            hash_index = self.hash_coords_2(self.n_verts + i, j)
                            self.adj_matrix[hash_index] = 1
                            ti.atomic_add(n_EE, 1)
        print('n_EE', n_EE)
        N = 0
        for i in self.adj_matrix:
            N += 1
        print('N EE true', N)

    @ti.kernel
    def print_cnts(self)->ti.i32:
        N = 0
        min_dist = 1.0
        for k,j in self.cid:
            N += 1
            pair = self.cid[k,j]
            dist = pair.b
            ti.atomic_min(min_dist, dist)
        print('number of cnts', N, 'min dist', min_dist)
        return N

    @ti.kernel
    def compute_mean_of_boundary_edges(self) -> float:
        total = 0.0
        for i in range(self.n_boundary_edges):
            total += (self.mesh.verts.x[self.boundary_edges[i, 0]] - self.mesh.verts.x[
                self.boundary_edges[i, 1]]).norm()
        result = total / ti.cast(self.n_boundary_edges, float)
        print("Mean of boundary edges:", result)
        return result

    @ti.kernel
    def compute_min_of_boundary_edges(self) -> float:
        ret = 10.0
        for i in range(self.n_boundary_edges):
            dist =  (self.mesh.verts.x[self.boundary_edges[i, 0]] - self.mesh.verts.x[
                self.boundary_edges[i, 1]]).norm()
            ti.atomic_min(ret, dist)
        print("Min of boundary edges:", ret)
        return ret
