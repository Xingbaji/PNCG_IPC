"""
Debug why gTz < 0 in MAS preconditioner.
"""

import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
import numpy as np
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner, BANKSIZE


@ti.data_oriented
class TestSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(self.n_verts, self.n_cells, self.mesh, use_metis=False)

    @ti.kernel
    def apply_velocity(self, vy: float):
        for vert in self.mesh.verts:
            vert.v = ti.Vector([0.0, vy, 0.0])

    @ti.kernel
    def compute_gTz(self) -> float:
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.z)
        return result


def check_spd(mat_np, name):
    """Check if matrix is SPD."""
    eigenvalues = np.linalg.eigvalsh(mat_np)
    is_spd = np.all(eigenvalues > 0)
    print(f"  {name}: eigenvalues min={eigenvalues.min():.4e}, max={eigenvalues.max():.4e}, SPD={is_spd}")
    return is_spd


def check_block_inverse(mas, block_id):
    """Check if block inverse is computed correctly."""
    # Get block matrix
    block_np = np.zeros((BANKSIZE * 3, BANKSIZE * 3))
    inv_block_np = np.zeros((BANKSIZE * 3, BANKSIZE * 3))

    # Reconstruct full matrix from symmetric storage
    for i in range(BANKSIZE):
        for j in range(BANKSIZE):
            if i <= j:
                sym_idx = BANKSIZE * i - i * (i + 1) // 2 + j
            else:
                sym_idx = BANKSIZE * j - j * (j + 1) // 2 + i

            sub_block = mas.block_matrices.to_numpy()[block_id, sym_idx]
            inv_sub_block = mas.inv_block_matrices.to_numpy()[block_id, sym_idx]

            for di in range(3):
                for dj in range(3):
                    if i <= j:
                        block_np[i*3+di, j*3+dj] = sub_block[di, dj]
                        block_np[j*3+dj, i*3+di] = sub_block[di, dj]
                        inv_block_np[i*3+di, j*3+dj] = inv_sub_block[di, dj]
                        inv_block_np[j*3+dj, i*3+di] = inv_sub_block[di, dj]

    # Check inverse quality
    identity_check = block_np @ inv_block_np
    identity_error = np.max(np.abs(identity_check - np.eye(BANKSIZE * 3)))

    print(f"  Block {block_id}: inverse error (|A*A^-1 - I|) = {identity_error:.4e}")

    return block_np, inv_block_np


def main():
    solver = TestSolver()

    # Apply initial velocity
    solver.apply_velocity(-1.0)

    print("="*60)
    print("Frame 0")
    print("="*60)

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    # Build MAS
    if not solver.mas.hierarchy_built:
        solver.mas.build_hierarchy()
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
    solver.mas.invert_block_matrices(use_full_inversion=True, use_cholesky=False,
                                      use_incomplete=False, use_oneway_gj=False)

    # Check some blocks
    print("\nChecking block matrices (Frame 0):")
    for block_id in [0, 100, 200]:
        check_block_inverse(solver.mas, block_id)

    # Apply MAS
    solver.mas.apply()
    gTz = solver.compute_gTz()
    print(f"\ngTz (Frame 0) = {gTz:.4e}")

    # Do one optimization step
    print("\n" + "="*60)
    print("After optimization step")
    print("="*60)

    # Set p = -z and do line search
    @ti.kernel
    def set_p_neg_z():
        for vert in solver.mesh.verts:
            vert.p = -vert.z
    set_p_neg_z()

    alpha, gTp, pHp = solver.line_search_newton()
    p_max = solver.compute_p_inf_norm()
    if alpha * p_max > 0.5 * solver.dHat:
        alpha = 0.5 * solver.dHat / p_max
    solver.update_x(alpha)
    solver.update_v_and_bound()

    print(f"Applied step: alpha={alpha:.4e}")

    # Frame 1
    print("\n" + "="*60)
    print("Frame 1")
    print("="*60)

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    # Check gradient
    @ti.kernel
    def grad_stats() -> ti.types.vector(3, float):
        g_min = 1e30
        g_max = 0.0
        g_sum = 0.0
        for vert in solver.mesh.verts:
            g = vert.grad.norm()
            ti.atomic_min(g_min, g)
            ti.atomic_max(g_max, g)
            g_sum += g
        return ti.Vector([g_min, g_max, g_sum / 8368.0])
    g = grad_stats()
    print(f"Gradient: min={g[0]:.4e}, max={g[1]:.4e}, avg={g[2]:.4e}")

    # Rebuild MAS
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
    solver.mas.invert_block_matrices(use_full_inversion=True, use_cholesky=False,
                                      use_incomplete=False, use_oneway_gj=False)

    # Check blocks again
    print("\nChecking block matrices (Frame 1):")
    for block_id in [0, 100, 200]:
        block_np, inv_block_np = check_block_inverse(solver.mas, block_id)

        # Check SPD
        check_spd(block_np, f"Block[{block_id}]")

    # Apply MAS
    solver.mas.apply()
    gTz = solver.compute_gTz()
    print(f"\ngTz (Frame 1) = {gTz:.4e}")

    # Detailed analysis: check multi_level_z values
    multi_level_z_np = solver.mas.multi_level_z.to_numpy()
    multi_level_r_np = solver.mas.multi_level_r.to_numpy()

    print(f"\nMulti-level analysis:")
    print(f"  multi_level_r: min={np.linalg.norm(multi_level_r_np, axis=1).min():.4e}, max={np.linalg.norm(multi_level_r_np, axis=1).max():.4e}")
    print(f"  multi_level_z: min={np.linalg.norm(multi_level_z_np, axis=1).min():.4e}, max={np.linalg.norm(multi_level_z_np, axis=1).max():.4e}")

    # Check vertex-level contribution to gTz
    @ti.kernel
    def vertex_gTz_analysis():
        pos_count = 0
        neg_count = 0
        max_pos_contrib = 0.0
        max_neg_contrib = 0.0

        for vert in solver.mesh.verts:
            dot = vert.grad.dot(vert.z)
            if dot > 0:
                pos_count += 1
                ti.atomic_max(max_pos_contrib, dot)
            else:
                neg_count += 1
                ti.atomic_min(max_neg_contrib, dot)

        print("Vertex analysis:")
        print("  Positive: ", pos_count, " max=", max_pos_contrib)
        print("  Negative: ", neg_count, " min=", max_neg_contrib)

    vertex_gTz_analysis()


if __name__ == '__main__':
    main()
