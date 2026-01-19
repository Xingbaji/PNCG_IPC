"""Debug MAS assembly - step by step."""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../demo')
import os
os.chdir('../demo')

import taichi as ti
import numpy as np
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


print('Initializing data...')
init_data(mesh)
precompute_cells(mesh)

# Create MAS
from algorithm.mas_preconditioner import MASPreconditioner
mas = MASPreconditioner(n_verts, n_cells, mesh, use_metis=False)
mas.build_hierarchy()
print(f'MAS hierarchy built')


# Step 1: Clear and add inertia only
print('\n--- Step 1: Inertia contribution only ---')
mas._clear_block_matrices()
mas._add_inertia_contribution(model.dt)
ti.sync()

block_mat = mas.block_matrices.to_numpy()
print(f'Block [0, 0] after inertia:')
print(f'  Diagonal: {np.diag(block_mat[0, 0, :, :])}')
print(f'  Full block:')
print(f'  {block_mat[0, 0, :, :]}')
print(f'  Block [0, 1] (off-diag): {block_mat[0, 1, :, :]}')

# Step 2: Add elastic contribution
print('\n--- Step 2: Add elastic contribution ---')
mas.elastic_type = 0  # ARAP
mas._add_elastic_contribution_full(model.mu, model.la, model.dt, 0)
ti.sync()

block_mat = mas.block_matrices.to_numpy()
print(f'Block [0, 0] after elastic:')
print(f'  Diagonal: {np.diag(block_mat[0, 0, :, :])}')
print(f'  Full block:')
print(f'  {block_mat[0, 0, :, :]}')
print(f'  Eigenvalues: {np.linalg.eigvalsh(block_mat[0, 0, :, :])}')

# Check multiple blocks
print('\n--- Multiple blocks check ---')
for i in range(5):
    diag = np.diag(block_mat[0, i, :, :])
    eig = np.linalg.eigvalsh(block_mat[0, i, :, :])
    print(f'  Block [0, {i}]: diag={diag}, min_eig={eig.min():.6f}')

# Step 3: Add regularization
print('\n--- Step 3: Add regularization ---')
mas._add_regularization(1e-6)
ti.sync()

block_mat = mas.block_matrices.to_numpy()
print(f'Block [0, 0] after regularization:')
print(f'  Diagonal: {np.diag(block_mat[0, 0, :, :])}')

# Check first few cells info
print('\n--- Cell info ---')
@ti.kernel
def print_cell_info(mesh: ti.template()):
    for i in range(min(3, n_cells)):
        c = mesh.cells[i]
        v0, v1, v2, v3 = c.verts[0].id, c.verts[1].id, c.verts[2].id, c.verts[3].id
        W = c.W
        print(f'Cell {i}: verts=[{v0}, {v1}, {v2}, {v3}], W={W}')

print_cell_info(mesh)

print('\nDone!')
