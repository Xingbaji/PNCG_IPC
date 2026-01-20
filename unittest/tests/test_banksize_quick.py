"""
Quick BANKSIZE performance test using the existing MAS implementation.
"""

import sys
import os
import time
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file_path)
unittest_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(unittest_dir)
demo_dir = os.path.join(project_root, 'demo')

sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(project_root)

import taichi as ti
ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path=".taichi_cache")

import meshtaichi_patcher as Patcher
from util.model_loading import _merge_and_reorder_models
from algorithm.mas_preconditioner_small import MASPreconditionerSmall, BANKSIZE


def run_test():
    # Load mesh
    model_path = 'model/mesh/cube_10/cube_10.node'
    raw_data = Patcher.load_mesh_rawdata(model_path)
    vertices = raw_data[0].astype(np.float32)
    cells = raw_data[3].astype(np.int32)
    n_verts, n_cells = len(vertices), len(cells)
    print(f'Loaded: {n_verts} verts, {n_cells} cells')
    print(f'Testing with BANKSIZE={BANKSIZE}')

    # Create mesh with METIS reordering
    models = [[vertices, None, None, cells]]
    reordered_dict, metis_result = _merge_and_reorder_models(models)
    mesh = Patcher.load_mesh(reordered_dict, relations=['CV'])

    mesh.verts.place({
        'x': ti.types.vector(3, float),
        'v': ti.types.vector(3, float),
        'm': float,
        'x_n': ti.types.vector(3, float),
        'x_hat': ti.types.vector(3, float),
        'grad': ti.types.vector(3, float),
        'z': ti.types.vector(3, float),
    })
    mesh.cells.place({'B': ti.math.mat3, 'W': float})

    # Get reordered vertices
    reordered_verts = reordered_dict[0]
    mesh.verts.x.from_numpy(reordered_verts)
    mesh.verts.v.fill([0.0, 0.0, 0.0])

    # Create preconditioner
    print('Creating MAS preconditioner...')
    mas = MASPreconditionerSmall(mesh)
    mas.build_hierarchy()

    print(f'Hierarchy: {mas.level_num} levels, {mas.total_blocks} blocks')

    # Mock solver
    class MockSolver:
        def __init__(self):
            self.mu = 384615.4
            self.la = 576923.1
            self.dt = 0.01

    solver = MockSolver()

    # Precompute
    @ti.kernel
    def precompute(mesh: ti.template(), density: ti.f32):
        for c in mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += density * c.W / 4.0

    @ti.kernel
    def set_xhat(mesh: ti.template(), dt: ti.f32, gravity: ti.f32):
        for vert in mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + dt * vert.v
            vert.x_hat[1] += gravity * dt * dt

    @ti.kernel
    def set_grad(mesh: ti.template()):
        for vert in mesh.verts:
            vert.grad = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5])

    precompute(mesh, 1000.0)
    set_xhat(mesh, 0.01, -9.8)

    # Benchmark
    n_iter = 5
    warmup = 2

    print('Benchmarking...')

    # Assemble
    for i in range(warmup):
        mas.assemble_block_matrices(solver)
    ti.sync()
    t0 = time.perf_counter()
    for i in range(n_iter):
        mas.assemble_block_matrices(solver)
        ti.sync()
    assemble_time = (time.perf_counter() - t0) / n_iter * 1000

    # Invert
    for i in range(warmup):
        mas.invert_block_matrices()
    ti.sync()
    t0 = time.perf_counter()
    for i in range(n_iter):
        mas.invert_block_matrices()
        ti.sync()
    invert_time = (time.perf_counter() - t0) / n_iter * 1000

    # Apply
    for i in range(warmup):
        set_grad(mesh)
        mas.apply()
    ti.sync()
    t0 = time.perf_counter()
    for i in range(n_iter):
        set_grad(mesh)
        mas.apply()
        ti.sync()
    apply_time = (time.perf_counter() - t0) / n_iter * 1000

    print(f'BANKSIZE={BANKSIZE}: assemble={assemble_time:.3f}ms, invert={invert_time:.3f}ms, apply={apply_time:.3f}ms')
    print('Test passed!')


if __name__ == '__main__':
    run_test()
