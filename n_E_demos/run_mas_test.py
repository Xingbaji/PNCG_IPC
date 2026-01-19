"""Quick MAS test script."""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../demo')
import os
os.chdir('../demo')

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False)

from util.model_loading import model_loading
model = model_loading(demo='cube_10')
mesh = model.mesh
print(f'Model loaded: n_verts={len(model.mesh.verts)}')

# Place fields
mesh.verts.place({
    'x': ti.types.vector(3, float),
    'grad': ti.types.vector(3, float),
    'diagH': ti.types.vector(3, float),
    'z': ti.types.vector(3, float),
    'm': float,
})
mesh.cells.place({'B': ti.math.mat3, 'W': float})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

n_verts = len(mesh.verts)
n_cells = len(mesh.cells)


@ti.kernel
def init_data(mesh: ti.template()):
    for vert in mesh.verts:
        vert.m = 1.0
        vert.grad = ti.Vector([0.0, -1.0, 0.0])
        vert.diagH = ti.Vector([1.0, 1.0, 1.0])
        vert.z = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def precompute_cells(mesh: ti.template()):
    for c in mesh.cells:
        Dm = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        c.B = Dm.inverse()
        c.W = ti.abs(Dm.determinant()) / 6.0


@ti.kernel
def compute_z_stats(mesh: ti.template()) -> ti.types.vector(3, float):
    z_norm = 0.0
    z_dot_g = 0.0
    z_count = 0.0
    for vert in mesh.verts:
        z_norm += vert.z.norm_sqr()
        z_dot_g += vert.z.dot(vert.grad)
        z_count += 1.0
    return ti.Vector([ti.sqrt(z_norm), z_dot_g, z_count])


print('Initializing data...')
init_data(mesh)
precompute_cells(mesh)
print('Data initialized')

# Create MAS
print('Creating MAS preconditioner...')
from algorithm.mas_preconditioner import MASPreconditioner
mas = MASPreconditioner(n_verts, n_cells, mesh, use_metis=False)
print(f'MAS created: level_num={mas.level_num}')

# Build hierarchy
print('Building hierarchy...')
import time
t0 = time.time()
mas.build_hierarchy()
ti.sync()
print(f'Hierarchy built in {time.time() - t0:.2f}s')


# Create minimal solver
class MinimalSolver:
    def __init__(self):
        self.mesh = mesh
        self.mu = model.mu
        self.la = model.la
        self.dt = model.dt
        self.elastic_type = model.elastic_type
        self.n_verts = n_verts
        self.n_cells = n_cells


solver = MinimalSolver()
print(f'Solver params: mu={solver.mu:.6f}, la={solver.la:.6f}, dt={solver.dt}')
print(f'elastic_type: {solver.elastic_type}')

# Check mass values
@ti.kernel
def check_mass(mesh: ti.template()) -> ti.types.vector(3, float):
    min_m = 1e10
    max_m = -1e10
    sum_m = 0.0
    for vert in mesh.verts:
        m = vert.m
        ti.atomic_min(min_m, m)
        ti.atomic_max(max_m, m)
        sum_m += m
    return ti.Vector([min_m, max_m, sum_m])

mass_stats = check_mass(mesh)
print(f'Mass stats: min={mass_stats[0]:.6f}, max={mass_stats[1]:.6f}, sum={mass_stats[2]:.6f}')

# Assemble matrices (use_full_hessian=False to simplify debugging)
print('Assembling block matrices (use_full_hessian=False)...')
t0 = time.time()
mas.assemble_block_matrices(solver, use_full_hessian=False)
ti.sync()
print(f'Assembled in {time.time() - t0:.2f}s')

# Invert matrices
print('Inverting block matrices...')
t0 = time.time()
mas.invert_block_matrices(use_full_inversion=True)
ti.sync()
print(f'Inverted in {time.time() - t0:.2f}s')

# Apply preconditioner
print('Applying preconditioner...')
t0 = time.time()
mas.apply(use_full_solve=True)
ti.sync()
print(f'Applied in {time.time() - t0:.2f}s')

# Check result
stats = compute_z_stats(mesh)
print(f'Result: ||z||={stats[0]:.6f}, z.g={stats[1]:.6f}, n={int(stats[2])}')
print(f'z.g should be positive (SPD check): {"PASS" if stats[1] > 0 else "FAIL"}')

# Debug: Check intermediate results
print('\n--- Debug Info ---')
print(f'actual_levels: {mas.actual_levels}')
print(f'level_size[0]: {mas.level_size[0].to_numpy()}')
print(f'level_size[1]: {mas.level_size[1].to_numpy()}')

# Check multi_level_r and multi_level_z
import numpy as np
r_np = mas.multi_level_r.to_numpy()
z_np = mas.multi_level_z.to_numpy()
print(f'multi_level_r[:10]: {r_np[:10]}')
print(f'multi_level_z[:10]: {z_np[:10]}')

# Check block matrices
block_mat = mas.block_matrices.to_numpy()
inv_block_mat = mas.inv_block_matrices.to_numpy()
print(f'block_matrices shape: {block_mat.shape}')
print(f'Block 0 diagonal (Level 0):')
for i in range(min(3, block_mat.shape[1])):
    print(f'  Block [{i},{i}]: diag = {np.diag(block_mat[0, i, :3, :3])}')

print(f'inv_block_matrices shape: {inv_block_mat.shape}')
print(f'Inv Block 0 (Level 0, sym_idx=0):')
print(f'  {inv_block_mat[0, 0, :3, :3]}')

# Verify A * A^{-1} ≈ I
print('\nVerifying A * A^{-1} ≈ I for sym_idx=0 (diagonal block):')
A = block_mat[0, 0, :, :]
A_inv = inv_block_mat[0, 0, :, :]
product = A @ A_inv
print(f'  ||A * A^{-1} - I|| = {np.linalg.norm(product - np.eye(3)):.6e}')

print('Done!')
