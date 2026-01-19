"""
Detailed debug of MAS preconditioner to find gTz < 0 issue.

The issue: MAS produces z such that gTz < 0 after frame 0.
This means z is not in the descent direction.
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
from algorithm.mas_preconditioner import MASPreconditioner


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
    def count_positive_gizi(self) -> ti.types.vector(2, int):
        """Count vertices where g_i dot z_i > 0 vs < 0"""
        pos = 0
        neg = 0
        for vert in self.mesh.verts:
            dot = vert.grad.dot(vert.z)
            if dot > 0:
                pos += 1
            else:
                neg += 1
        return ti.Vector([pos, neg])


def test_single_frame(solver, frame_id: int, rebuild_mas: bool = True):
    """Test a single optimization frame."""
    print(f"\n{'='*60}")
    print(f"Frame {frame_id}")
    print(f"{'='*60}")

    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    g = solver.grad_stats()
    print(f"Gradient: min={g[0]:.4e}, max={g[1]:.4e}, avg={g[2]:.4e}")

    if rebuild_mas:
        print("\nRebuilding MAS...")
        if not solver.mas.hierarchy_built:
            solver.mas.build_hierarchy()
        solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
        solver.mas.invert_block_matrices(
            use_full_inversion=True,
            use_cholesky=False,
            use_incomplete=False,
            use_oneway_gj=False
        )

    print("\nApplying MAS...")
    solver.mas.apply()

    z = solver.z_stats()
    print(f"z: min={z[0]:.4e}, max={z[1]:.4e}, avg={z[2]:.4e}")

    gTz = solver.compute_gTz()
    print(f"gTz = {gTz:.4e}")

    counts = solver.count_positive_gizi()
    print(f"Vertices with g_i.z_i > 0: {counts[0]}, < 0: {counts[1]}")

    # Check block matrices statistics
    check_block_stats(solver.mas)

    return gTz


def check_block_stats(mas):
    """Check block matrix values."""
    # Use numpy to inspect some values
    block_sample = mas.block_matrices.to_numpy()[0, 0]
    inv_block_sample = mas.inv_block_matrices.to_numpy()[0, 0]

    print(f"\nBlock[0,0] diagonal: {np.diag(block_sample)}")
    print(f"InvBlock[0,0] diagonal: {np.diag(inv_block_sample)}")

    # Check if inverse is reasonable
    eigenvalues = np.linalg.eigvalsh(block_sample)
    print(f"Block[0,0] eigenvalues: min={eigenvalues.min():.4e}, max={eigenvalues.max():.4e}")

    eigenvalues_inv = np.linalg.eigvalsh(inv_block_sample)
    print(f"InvBlock[0,0] eigenvalues: min={eigenvalues_inv.min():.4e}, max={eigenvalues_inv.max():.4e}")


def main():
    solver = TestSolver()

    # Apply initial velocity
    solver.apply_velocity(-1.0)

    # Frame 0 - should be positive
    gTz0 = test_single_frame(solver, 0, rebuild_mas=True)

    # Do one optimization step (simplified)
    solver.step_collision_free(max_iter=1, verbose=False)

    # Frame 1 - often negative
    gTz1 = test_single_frame(solver, 1, rebuild_mas=True)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Frame 0: gTz = {gTz0:.4e} {'(OK)' if gTz0 > 0 else '(BAD!)'}")
    print(f"Frame 1: gTz = {gTz1:.4e} {'(OK)' if gTz1 > 0 else '(BAD!)'}")

    if gTz1 < 0:
        print("\nPossible causes:")
        print("1. Hessian approximation is not SPD (indefinite)")
        print("2. Block matrix inversion is numerically unstable")
        print("3. Cross-warp contributions not handled correctly")


if __name__ == '__main__':
    main()
