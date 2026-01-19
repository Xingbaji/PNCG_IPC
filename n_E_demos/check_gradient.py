"""Check gradient computation."""
import sys
sys.path.insert(0, '/root/PNCG_IPC')
sys.path.insert(0, '/root/PNCG_IPC/demo')

import os
os.chdir('/root/PNCG_IPC/demo')

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.pncg_base_ipc import pncg_ipc_deformer
import numpy as np

solver = pncg_ipc_deformer(demo='eight_E_drop_demo_contact')
print(f'gravity: {solver.gravity}')
print(f'ground: {solver.ground}')
print(f'dt: {solver.dt}')
print(f'E: {solver.mu * 2 * (1 + 0.4)}')  # Approx Young's modulus

# Get positions
pos_np = solver.mesh.get_position_as_numpy()
print(f'y_min: {pos_np[:,1].min():.3f}, y_max: {pos_np[:,1].max():.3f}')

# Run one frame to see what happens
solver.assign_xn_xhat()
solver.compute_grad_and_diagH()
if solver.ground_barrier == 1:
    solver.add_grad_and_diagH_ground_barrier()

# Check gradient
@ti.kernel
def check_grad(solver: ti.template()) -> ti.types.vector(4, float):
    g_sum = 0.0
    g_max = 0.0
    for vert in solver.mesh.verts:
        g_norm = vert.grad.norm()
        g_sum += g_norm
        ti.atomic_max(g_max, g_norm)
    return ti.Vector([g_sum, g_max, 0.0, 0.0])

g = check_grad(solver)
print(f'Gradient iter 0: sum={g[0]:.6e}, max={g[1]:.6e}')

# Do one step
solver.compute_init_p()
alpha, gTp, pHp = solver.line_search_newton()
p_max = solver.compute_p_inf_norm()
print(f'alpha={alpha:.6e}, gTp={gTp:.6e}, pHp={pHp:.6e}, p_max={p_max:.6e}')

solver.update_x(alpha)

# Check new gradient
solver.compute_grad_and_diagH()
if solver.ground_barrier == 1:
    solver.add_grad_and_diagH_ground_barrier()
g2 = check_grad(solver)
print(f'Gradient iter 1: sum={g2[0]:.6e}, max={g2[1]:.6e}')

# Check position change
pos_np2 = solver.mesh.get_position_as_numpy()
print(f'Position change: max={np.abs(pos_np2 - pos_np).max():.6e}')
