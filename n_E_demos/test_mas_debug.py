"""Debug MAS preconditioner values."""

import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner import MASPreconditioner


@ti.data_oriented
class TestSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(self.n_verts, self.n_cells, self.mesh, use_metis=False)

    @ti.kernel
    def apply_velocity(self):
        for vert in self.mesh.verts:
            vert.v = ti.Vector([0.0, -1.0, 0.0])

    @ti.kernel
    def grad_stats(self) -> ti.types.vector(3, float):
        g_min = 1e30
        g_max = 0.0
        g_sum = 0.0
        for vert in self.mesh.verts:
            g = vert.grad.norm()
            ti.atomic_min(g_min, g)
            ti.atomic_max(g_max, g)
            g_sum += g
        return ti.Vector([g_min, g_max, g_sum / 8368.0])

    @ti.kernel
    def diagH_stats(self) -> ti.types.vector(3, float):
        h_min = 1e30
        h_max = 0.0
        h_sum = 0.0
        for vert in self.mesh.verts:
            h = vert.diagH.norm()
            ti.atomic_min(h_min, h)
            ti.atomic_max(h_max, h)
            h_sum += h
        return ti.Vector([h_min, h_max, h_sum / 8368.0])

    @ti.kernel
    def z_stats(self) -> ti.types.vector(3, float):
        z_min = 1e30
        z_max = 0.0
        z_sum = 0.0
        for vert in self.mesh.verts:
            z = vert.z.norm()
            ti.atomic_min(z_min, z)
            ti.atomic_max(z_max, z)
            z_sum += z
        return ti.Vector([z_min, z_max, z_sum / 8368.0])

    @ti.kernel
    def compute_gTz(self) -> float:
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.z)
        return result

    @ti.kernel
    def compute_p_neg_z(self):
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_gTp(self) -> float:
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.p)
        return result

    @ti.kernel
    def multi_level_r_stats(self) -> ti.types.vector(3, float):
        """Check multi_level_r values"""
        r_min = 1e30
        r_max = 0.0
        r_sum = 0.0
        count = 0
        for i in range(self.mas.total_nodes_all_levels):
            r = self.mas.multi_level_r[i].norm()
            if r > 1e-12:
                ti.atomic_min(r_min, r)
                ti.atomic_max(r_max, r)
                r_sum += r
                count += 1
        return ti.Vector([r_min, r_max, r_sum / ti.max(count, 1)])

    @ti.kernel
    def multi_level_z_stats(self) -> ti.types.vector(3, float):
        """Check multi_level_z values"""
        z_min = 1e30
        z_max = 0.0
        z_sum = 0.0
        count = 0
        for i in range(self.mas.total_nodes_all_levels):
            z = self.mas.multi_level_z[i].norm()
            if z > 1e-12:
                ti.atomic_min(z_min, z)
                ti.atomic_max(z_max, z)
                z_sum += z
                count += 1
        return ti.Vector([z_min, z_max, z_sum / ti.max(count, 1)])


def main():
    solver = TestSolver()

    # Apply initial velocity
    solver.apply_velocity()

    # Prepare for optimization
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    g = solver.grad_stats()
    print(f'Gradient: min={g[0]:.4e}, max={g[1]:.4e}, avg={g[2]:.4e}')

    h = solver.diagH_stats()
    print(f'DiagH: min={h[0]:.4e}, max={h[1]:.4e}, avg={h[2]:.4e}')

    # Build MAS
    print("\nBuilding MAS hierarchy...")
    solver.mas.build_hierarchy()

    print("Assembling block matrices...")
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)

    print("Inverting block matrices...")
    solver.mas.invert_block_matrices(use_full_inversion=True, use_cholesky=False, use_incomplete=False, use_oneway_gj=False)

    # Apply MAS
    print("\nApplying MAS preconditioner...")
    solver.mas.apply()

    # Check intermediate values
    r = solver.multi_level_r_stats()
    print(f'multi_level_r: min={r[0]:.4e}, max={r[1]:.4e}, avg={r[2]:.4e}')

    mlz = solver.multi_level_z_stats()
    print(f'multi_level_z: min={mlz[0]:.4e}, max={mlz[1]:.4e}, avg={mlz[2]:.4e}')

    z = solver.z_stats()
    print(f'z (mesh.verts): min={z[0]:.4e}, max={z[1]:.4e}, avg={z[2]:.4e}')

    gTz = solver.compute_gTz()
    print(f'\ngTz = {gTz:.4e}')

    # Compute p = -z
    solver.compute_p_neg_z()

    gTp = solver.compute_gTp()
    print(f'gTp = {gTp:.4e} (should be -gTz = {-gTz:.4e})')

    # Now test line search
    print("\nTesting line search...")
    alpha, gTp_ls, pHp = solver.line_search_newton()
    print(f'line_search_newton: alpha={alpha:.4e}, gTp={gTp_ls:.4e}, pHp={pHp:.4e}')

    p_max = solver.compute_p_inf_norm()
    print(f'|p|_inf = {p_max:.4e}')
    print(f'displacement = alpha * |p|_inf = {alpha * p_max:.4e}')


if __name__ == '__main__':
    main()
