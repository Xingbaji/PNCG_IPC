"""MAS Preconditioner Benchmark - test correctness and performance."""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../demo')
import os
os.chdir('../demo')

import taichi as ti
import numpy as np
import time
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False)

from util.model_loading import model_loading


def benchmark_mas(demo_name='cube_10', n_iterations=5):
    """Benchmark MAS preconditioner on a given demo."""
    print(f"\n{'='*70}")
    print(f"MAS Benchmark: {demo_name}")
    print(f"{'='*70}")

    # Load model
    model = model_loading(demo=demo_name)
    mesh = model.mesh
    n_verts = len(mesh.verts)
    n_cells = len(mesh.cells)
    print(f"Model: n_verts={n_verts}, n_cells={n_cells}")

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

    @ti.kernel
    def reset_z(mesh: ti.template()):
        for vert in mesh.verts:
            vert.z = ti.Vector([0.0, 0.0, 0.0])

    init_data(mesh)
    precompute_cells(mesh)

    # Create MAS
    from algorithm.mas_preconditioner import MASPreconditioner
    mas = MASPreconditioner(n_verts, n_cells, mesh, use_metis=False)
    mas.elastic_type = 0  # ARAP

    # Build hierarchy (one-time cost)
    print("\n[1] Building hierarchy...")
    ti.sync()
    t0 = time.perf_counter()
    mas.build_hierarchy()
    ti.sync()
    hierarchy_time = (time.perf_counter() - t0) * 1000
    print(f"    Hierarchy built: {mas.actual_levels} levels, {hierarchy_time:.2f} ms")

    # Create minimal solver
    class MinimalSolver:
        def __init__(self):
            self.mesh = mesh
            self.mu = model.mu
            self.la = model.la
            self.dt = model.dt
            self.elastic_type = 0
            self.n_verts = n_verts
            self.n_cells = n_cells

    solver = MinimalSolver()
    print(f"    Material: mu={solver.mu:.2f}, la={solver.la:.2f}, dt={solver.dt}")

    # First run (includes JIT compilation)
    print("\n[2] First run (includes JIT compilation)...")

    ti.sync()
    t0 = time.perf_counter()
    mas.assemble_block_matrices(solver, use_full_hessian=True)
    ti.sync()
    first_assemble = (time.perf_counter() - t0) * 1000
    print(f"    Assemble: {first_assemble:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.invert_block_matrices(use_full_inversion=True)
    ti.sync()
    first_invert = (time.perf_counter() - t0) * 1000
    print(f"    Invert: {first_invert:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.apply(use_full_solve=True)
    ti.sync()
    first_apply = (time.perf_counter() - t0) * 1000
    print(f"    Apply: {first_apply:.2f} ms")

    stats = compute_z_stats(mesh)
    print(f"    Result: ||z||={stats[0]:.6f}, z·g={stats[1]:.6f}")
    first_spd = stats[1] > 0
    print(f"    SPD check: {'PASS' if first_spd else 'FAIL'}")

    # Subsequent runs (no JIT overhead)
    print(f"\n[3] Benchmark ({n_iterations} iterations, excluding JIT)...")

    assemble_times = []
    invert_times = []
    apply_times = []

    for i in range(n_iterations):
        reset_z(mesh)

        ti.sync()
        t0 = time.perf_counter()
        mas.assemble_block_matrices(solver, use_full_hessian=True)
        ti.sync()
        assemble_times.append((time.perf_counter() - t0) * 1000)

        ti.sync()
        t0 = time.perf_counter()
        mas.invert_block_matrices(use_full_inversion=True)
        ti.sync()
        invert_times.append((time.perf_counter() - t0) * 1000)

        ti.sync()
        t0 = time.perf_counter()
        mas.apply(use_full_solve=True)
        ti.sync()
        apply_times.append((time.perf_counter() - t0) * 1000)

    # Final SPD check
    stats = compute_z_stats(mesh)
    final_spd = stats[1] > 0

    # Statistics
    assemble_avg = np.mean(assemble_times)
    assemble_std = np.std(assemble_times)
    invert_avg = np.mean(invert_times)
    invert_std = np.std(invert_times)
    apply_avg = np.mean(apply_times)
    apply_std = np.std(apply_times)
    total_avg = assemble_avg + invert_avg + apply_avg

    print(f"\n[4] Results:")
    print(f"    Assemble:  {assemble_avg:8.2f} ± {assemble_std:.2f} ms")
    print(f"    Invert:    {invert_avg:8.2f} ± {invert_std:.2f} ms")
    print(f"    Apply:     {apply_avg:8.2f} ± {apply_std:.2f} ms")
    print(f"    ----------------------------------------")
    print(f"    Total:     {total_avg:8.2f} ms per preconditioner application")
    print(f"    SPD check: {'PASS' if final_spd else 'FAIL'}")

    return {
        'demo': demo_name,
        'n_verts': n_verts,
        'n_cells': n_cells,
        'levels': mas.actual_levels,
        'hierarchy_ms': hierarchy_time,
        'assemble_ms': assemble_avg,
        'invert_ms': invert_avg,
        'apply_ms': apply_avg,
        'total_ms': total_avg,
        'spd_pass': final_spd,
    }


if __name__ == '__main__':
    demos = ['cube_10', 'cube_20', 'eight_E_drop_demo_contact']  # Include larger demo
    results = []

    for demo in demos:
        try:
            result = benchmark_mas(demo, n_iterations=5)
            results.append(result)
        except Exception as e:
            print(f"ERROR on {demo}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Demo':<20} {'Verts':<8} {'Cells':<8} {'Lvls':<5} {'Assemble':<10} {'Invert':<10} {'Apply':<10} {'Total':<10} {'SPD':<5}")
    print("-" * 95)
    for r in results:
        print(f"{r['demo']:<20} {r['n_verts']:<8} {r['n_cells']:<8} {r['levels']:<5} "
              f"{r['assemble_ms']:<10.2f} {r['invert_ms']:<10.2f} {r['apply_ms']:<10.2f} "
              f"{r['total_ms']:<10.2f} {'PASS' if r['spd_pass'] else 'FAIL':<5}")
