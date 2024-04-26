from algorithm.collision_detection import *
# collison_detection_v2.py contains PP,PE,PT,EE (P:point, E:edge, T triangle) constriants
# collison_detection.py contains only PT,EE constriants

# TODO: edge: Xiaolin Wu's line algorithm
@ti.data_oriented
class collision_detection_module_v2(collision_detection_module):
    # use PP,PE,PT,EE to store constraints
    def init_hash_grid(self):
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
        self.cid_root = ti.root.bitmasked(ti.ij,(4,self.MAX_C)).place(self.cid) #!
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


    @ti.func
    def attempt_PT_no_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2, type = dist3D_Point_Triangle_type(xp, x0, x1, x2) # 0: PP, 1: PE, 2: PT
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[type, hash_index] = self.pair(ids, dist, cord, t_pt)

    @ti.func
    def attempt_PT_adj(self, triangle_id, p, t0, t1, t2, xp, x0, x1, x2):
        hash_adj = self.hash_coords_2(p, triangle_id)
        if p != t0 and p != t1 and p != t2 and self.adj_matrix[hash_adj] == 0 and point_triangle_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            cord0, cord1, cord2, type = dist3D_Point_Triangle_type(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()
            if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                ids = ti.Vector([p, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], float)
                hash_index = self.hash_coords_2(p, triangle_id)
                self.cid[type,hash_index] = self.pair(ids, dist, cord, t_pt)

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
            edge_id_1 = self.cell_entries_E[k]
            if edge_id_0 < edge_id_1:
                b0 = self.boundary_edges[edge_id_1, 0]
                b1 = self.boundary_edges[edge_id_1, 1]
                x_b0 = self.mesh.verts.x[b0]
                x_b1 = self.mesh.verts.x[b1]
                if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat):
                    t_ee, sc, tc = dist3D_Segment_to_Segment(x_a0, x_a1, x_b0, x_b1)
                    dist = t_ee.norm()
                    if dist < self.dHat and ti.abs(dist) > self.SMALL_NUM:
                        type_index = 0
                        hash_index = 0
                        if sc < self.SMALL_NUM:
                            if tc < self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a0, b0), ti.max(a0, b0))
                            elif tc > 1.0 - self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a0, b1), ti.max(a0, b1))
                            else:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(a0, edge_id_1)
                        elif sc > 1.0 - self.SMALL_NUM:
                            if tc < self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a1, b0), ti.max(a1, b0))
                            elif tc > 1.0 - self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a1, b1), ti.max(a1, b1))
                            else:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(a1, edge_id_1)
                        else:
                            if tc < self.SMALL_NUM:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(b0, edge_id_0)
                            elif tc > 1.0 - self.SMALL_NUM:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(b1, edge_id_0)
                            else:  # EE
                                type_index = 3
                                hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                        self.cid[type_index, hash_index] = self.pair(ids, dist, cord, t_ee)

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
                        type_index = 0
                        hash_index = 0
                        if sc < self.SMALL_NUM:
                            if tc < self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a0, b0), ti.max(a0, b0))
                            elif tc > 1.0 - self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a0, b1), ti.max(a0, b1))
                            else:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(a0, edge_id_1)
                        elif sc > 1.0 - self.SMALL_NUM:
                            if tc < self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a1, b0), ti.max(a1, b0))
                            elif tc > 1.0 - self.SMALL_NUM:  # PP
                                type_index = 0
                                hash_index = self.hash_coords_2(ti.min(a1, b1), ti.max(a1, b1))
                            else:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(a1, edge_id_1)
                        else:
                            if tc < self.SMALL_NUM:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(b0, edge_id_0)
                            elif tc > 1.0 - self.SMALL_NUM:  # PE
                                type_index = 1
                                hash_index = self.hash_coords_2(b1, edge_id_0)
                            else:  # EE
                                type_index = 3
                                hash_index = self.hash_coords_2(edge_id_0, edge_id_1)
                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], float)
                        self.cid[type_index, hash_index] = self.pair(ids, dist, cord, t_ee)
