"""Quick MAS test script - test full hessian with correct elastic type."""
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

# Set elastic type to integer BEFORE assembly
# 0=ARAP, 1=SNH, 2=FCR
elastic_type_str = model.elastic_type
print(f'elastic_type from model: {elastic_type_str}')

# Map string to int
elastic_type_map = {
    'ARAP': 0, 'ARAP_filter': 0,
    'SNH': 1,
    'FCR': 2, 'FCR_filter': 2,
}
elastic_type_int = elastic_type_map.get(elastic_type_str, 0)
print(f'elastic_type_int: {elastic_type_int}')

# Set it in MAS preconditioner
mas.elastic_type = elastic_type_int

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
        self.elastic_type = elastic_type_int  # Use int!
        self.n_verts = n_verts
        self.n_cells = n_cells


solver = MinimalSolver()
print(f'Solver params: mu={solver.mu:.6f}, la={solver.la:.6f}, dt={solver.dt}')

# Assemble matrices with full hessian
print('Assembling block matrices (use_full_hessian=True)...')
t0 = time.time()
mas.assemble_block_matrices(solver, use_full_hessian=True)
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

# Check some block matrix values
import numpy as np
block_mat = mas.block_matrices.to_numpy()
print(f'\nBlock matrix diagnostics:')
print(f'  Shape: {block_mat.shape}')
print(f'  Block [0, 0] (diagonal entry):')
diag_block = block_mat[0, 0, :, :]
print(f'    {diag_block}')
print(f'    Eigenvalues: {np.linalg.eigvalsh(diag_block)}')

print('Done!')
