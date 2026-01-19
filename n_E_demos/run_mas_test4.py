"""Quick MAS test script - verify ARAP_filter fix."""
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
elastic_type_int = 0  # ARAP
mas.elastic_type = elastic_type_int
print(f'elastic_type_int: {elastic_type_int}')

# Build hierarchy
print('Building hierarchy...')
import time
t0 = time.time()
mas.build_hierarchy()
ti.sync()
print(f'Hierarchy built in {time.time() - t0:.3f}s')


# Create minimal solver
class MinimalSolver:
    def __init__(self):
        self.mesh = mesh
        self.mu = model.mu
        self.la = model.la
        self.dt = model.dt
        self.elastic_type = elastic_type_int
        self.n_verts = n_verts
        self.n_cells = n_cells


solver = MinimalSolver()
print(f'Solver params: mu={solver.mu:.6f}, la={solver.la:.6f}, dt={solver.dt}')

# Assemble matrices with full hessian
print('\n=== Test with FULL HESSIAN (use_full_hessian=True) ===')
t0 = time.time()
mas.assemble_block_matrices(solver, use_full_hessian=True)
ti.sync()
print(f'Assembled in {time.time() - t0:.3f}s')

# Check block matrix SPD
block_mat = mas.block_matrices.to_numpy()
print(f'\nBlock matrix diagnostics:')
print(f'  Shape: {block_mat.shape}')

# BANKSIZE = 16, so diagonal blocks are at sym_idx = _sym_index(lane, lane)
# _sym_index(i, i) = BANKSIZE * i - i*(i+1)/2 + i = 16*i - i*(i+1)/2 + i
# For lane 0: sym_idx = 0
# For lane 1: sym_idx = 16*1 - 1 + 1 = 16
# For lane 2: sym_idx = 16*2 - 3 + 2 = 31
# etc.
def sym_index(row, col):
    r = min(row, col)
    c = max(row, col)
    return 16 * r - r * (r + 1) // 2 + c

print(f'\n  Checking TRUE diagonal blocks (vertex self-blocks):')
all_spd = True
n_blocks = block_mat.shape[0]
n_checked = 0
for warp_id in range(min(3, n_blocks)):  # Check first 3 warps
    for lane in range(16):  # BANKSIZE = 16
        idx = warp_id * 16 + lane
        if idx >= n_verts:
            continue
        sym_idx = sym_index(lane, lane)
        diag_block = block_mat[warp_id, sym_idx, :, :]
        eigs = np.linalg.eigvalsh(diag_block)
        min_eig = eigs.min()
        spd = min_eig > 0
        if not spd:
            all_spd = False
            print(f'    Vertex {idx} (warp={warp_id}, lane={lane}): min_eig={min_eig:.6f} [NOT SPD]')
        n_checked += 1
        if n_checked <= 5 or not spd:
            status = "SPD" if spd else "NOT SPD"
            print(f'    Vertex {idx}: min_eig={min_eig:.6f}, max_eig={eigs.max():.6f} [{status}]')

print(f'\n  Checked {n_checked} diagonal blocks, all SPD: {all_spd}')

if not all_spd:
    print('\n  ERROR: Block matrices are not SPD after ARAP_filter fix!')
    print('  Checking first diagonal block details:')
    diag_block = block_mat[0, 0, :, :]
    print(f'    Full block:\n{diag_block}')
    print(f'    Eigenvalues: {np.linalg.eigvalsh(diag_block)}')
else:
    # Continue with inversion and apply
    print('\nInverting block matrices...')
    t0 = time.time()
    mas.invert_block_matrices(use_full_inversion=True)
    ti.sync()
    print(f'Inverted in {time.time() - t0:.3f}s')

    # Apply preconditioner
    print('Applying preconditioner...')
    t0 = time.time()
    mas.apply(use_full_solve=True)
    ti.sync()
    print(f'Applied in {time.time() - t0:.3f}s')

    # Check result
    stats = compute_z_stats(mesh)
    print(f'\nResult: ||z||={stats[0]:.6f}, z·g={stats[1]:.6f}, n={int(stats[2])}')

    spd_check = stats[1] > 0
    print(f'\nSPD Check (z·g > 0): {"PASS" if spd_check else "FAIL"}')

    if spd_check:
        print('\n✓ MAS preconditioner with full hessian is working correctly!')
    else:
        print('\n✗ MAS preconditioner produced non-SPD result!')

print('\nDone!')
