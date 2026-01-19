"""MAS Preconditioner Benchmark v3 - Optimized for faster JIT compilation."""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../demo')
import os
os.chdir('../demo')

import taichi as ti
import numpy as np
import time

# Initialize Taichi with offline cache ENABLED for faster subsequent runs
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True,
        kernel_profiler=True)

print(f"Taichi version: {ti.__version__}")
print(f"Offline cache enabled for faster JIT on subsequent runs")


def create_simple_tet_mesh(n_per_side=3):
    """Create a simple tetrahedral mesh directly without meshtaichi file I/O.

    This avoids meshtaichi file reading overhead for quick testing.
    Creates a cube subdivided into tetrahedra.

    Args:
        n_per_side: number of vertices per side (default 3 = 27 verts, ~48 tets)

    Returns:
        vertices: (n_verts, 3) array of vertex positions
        cells: (n_cells, 4) array of tet vertex indices
    """
    # Create grid vertices
    verts_per_axis = n_per_side
    vertices = []
    for i in range(verts_per_axis):
        for j in range(verts_per_axis):
            for k in range(verts_per_axis):
                vertices.append([i / (verts_per_axis - 1),
                               j / (verts_per_axis - 1),
                               k / (verts_per_axis - 1)])
    vertices = np.array(vertices, dtype=np.float32)

    # Create tets from grid cubes (5 tets per cube)
    cells = []
    n = verts_per_axis

    def idx(i, j, k):
        return i * n * n + j * n + k

    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                # 8 corners of the cube
                v000 = idx(i, j, k)
                v001 = idx(i, j, k + 1)
                v010 = idx(i, j + 1, k)
                v011 = idx(i, j + 1, k + 1)
                v100 = idx(i + 1, j, k)
                v101 = idx(i + 1, j, k + 1)
                v110 = idx(i + 1, j + 1, k)
                v111 = idx(i + 1, j + 1, k + 1)

                # 5-tet decomposition of a cube
                cells.append([v000, v100, v010, v001])
                cells.append([v100, v110, v010, v111])
                cells.append([v001, v010, v011, v111])
                cells.append([v100, v001, v010, v111])
                cells.append([v001, v100, v101, v111])

    cells = np.array(cells, dtype=np.int32)
    return vertices, cells


def benchmark_mas_simple(n_per_side=4, n_iterations=3, use_warp_reduction=False):
    """Benchmark MAS preconditioner with synthetic mesh (no file I/O).

    Args:
        n_per_side: vertices per side of cube mesh
        n_iterations: number of benchmark iterations
        use_warp_reduction: whether to use P1 warp reduction optimization
    """
    print(f"\n{'='*70}")
    print(f"MAS Benchmark (Simple Mesh): n_per_side={n_per_side}, warp_reduction={use_warp_reduction}")
    print(f"{'='*70}")

    # Create synthetic mesh
    vertices, cells = create_simple_tet_mesh(n_per_side)
    n_verts = len(vertices)
    n_cells = len(cells)
    print(f"Synthetic mesh: n_verts={n_verts}, n_cells={n_cells}")

    # Use ti.field instead of meshtaichi for simplicity
    x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    grad = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    diagH = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    z = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    m = ti.field(dtype=ti.f32, shape=n_verts)

    cell_verts = ti.field(dtype=ti.i32, shape=(n_cells, 4))
    cell_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)
    cell_W = ti.field(dtype=ti.f32, shape=n_cells)

    # Initialize from numpy
    x.from_numpy(vertices)
    cell_verts.from_numpy(cells)

    @ti.kernel
    def init_data():
        for i in range(n_verts):
            m[i] = 1.0
            grad[i] = ti.Vector([0.0, -1.0, 0.0])
            diagH[i] = ti.Vector([1.0, 1.0, 1.0])
            z[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def precompute_cells():
        for c in range(n_cells):
            v0 = cell_verts[c, 0]
            v1 = cell_verts[c, 1]
            v2 = cell_verts[c, 2]
            v3 = cell_verts[c, 3]

            x0 = x[v0]
            Dm = ti.Matrix.cols([x[v1] - x0, x[v2] - x0, x[v3] - x0])
            cell_B[c] = Dm.inverse()
            cell_W[c] = ti.abs(Dm.determinant()) / 6.0

    @ti.kernel
    def compute_z_stats() -> ti.types.vector(3, ti.f64):
        z_norm = ti.cast(0.0, ti.f64)
        z_dot_g = ti.cast(0.0, ti.f64)
        z_count = ti.cast(0.0, ti.f64)
        for i in range(n_verts):
            z_norm += ti.cast(z[i].norm_sqr(), ti.f64)
            z_dot_g += ti.cast(z[i].dot(grad[i]), ti.f64)
            z_count += 1.0
        return ti.Vector([ti.sqrt(z_norm), z_dot_g, z_count], dt=ti.f64)

    @ti.kernel
    def reset_z():
        for i in range(n_verts):
            z[i] = ti.Vector([0.0, 0.0, 0.0])

    print("\n[1] Initializing data...")
    init_data()
    precompute_cells()

    # Create MAS with field-based interface
    print("\n[2] Creating MAS preconditioner...")
    from algorithm.mas_preconditioner import MASPreconditioner
    mas = MASPreconditioner(n_verts, n_cells, mesh=None, use_metis=False)
    mas.elastic_type = 0  # ARAP

    # Set up field accessors for MAS
    mas._x_field = x
    mas._grad_field = grad
    mas._z_field = z
    mas._diagH_field = diagH
    mas._cell_verts = cell_verts
    mas._cell_B = cell_B
    mas._cell_W = cell_W

    # Build hierarchy
    print("\n[3] Building hierarchy...")
    ti.sync()
    t0 = time.perf_counter()

    # Build adjacency from cells
    adj_list = [set() for _ in range(n_verts)]
    for c in range(n_cells):
        v = cells[c]
        for i in range(4):
            for j in range(i + 1, 4):
                adj_list[v[i]].add(v[j])
                adj_list[v[j]].add(v[i])

    # Convert to flat arrays
    neighbor_starts = [0]
    neighbor_list = []
    for i in range(n_verts):
        neighbors = sorted(adj_list[i])
        neighbor_list.extend(neighbors)
        neighbor_starts.append(len(neighbor_list))

    mas._build_hierarchy_from_adjacency(
        np.array(neighbor_list, dtype=np.int32),
        np.array(neighbor_starts, dtype=np.int32)
    )
    ti.sync()
    hierarchy_time = (time.perf_counter() - t0) * 1000
    print(f"    Hierarchy built: {mas.actual_levels} levels, {hierarchy_time:.2f} ms")

    # Create minimal solver proxy
    class SimpleSolverProxy:
        def __init__(self):
            self.mu = 1e4
            self.la = 1e4
            self.dt = 0.04
            self.elastic_type = 0
            self.n_verts = n_verts
            self.n_cells = n_cells

    solver = SimpleSolverProxy()

    # First run (includes JIT compilation)
    print("\n[4] First run (includes JIT compilation)...")
    ti.profiler.clear_kernel_profiler_info()

    ti.sync()
    t0 = time.perf_counter()
    mas.assemble_block_matrices_simple(solver, x, cell_verts, cell_B, cell_W)
    ti.sync()
    first_assemble = (time.perf_counter() - t0) * 1000
    print(f"    Assemble: {first_assemble:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.invert_block_matrices(use_full_inversion=True, use_cholesky=True)
    ti.sync()
    first_invert = (time.perf_counter() - t0) * 1000
    print(f"    Invert: {first_invert:.2f} ms")

    ti.sync()
    t0 = time.perf_counter()
    mas.apply_simple(grad, z, use_warp_reduction=use_warp_reduction)
    ti.sync()
    first_apply = (time.perf_counter() - t0) * 1000
    print(f"    Apply: {first_apply:.2f} ms")

    stats = compute_z_stats()
    print(f"    Result: ||z||={stats[0]:.6f}, z·g={stats[1]:.6f}")
    first_spd = stats[1] > 0
    print(f"    SPD check: {'PASS' if first_spd else 'FAIL'}")

    # Subsequent runs
    print(f"\n[5] Benchmark ({n_iterations} iterations, excluding JIT)...")
    ti.profiler.clear_kernel_profiler_info()

    assemble_times = []
    invert_times = []
    apply_times = []

    for i in range(n_iterations):
        reset_z()

        ti.sync()
        t0 = time.perf_counter()
        mas.assemble_block_matrices_simple(solver, x, cell_verts, cell_B, cell_W)
        ti.sync()
        assemble_times.append((time.perf_counter() - t0) * 1000)

        ti.sync()
        t0 = time.perf_counter()
        mas.invert_block_matrices(use_full_inversion=True, use_cholesky=True)
        ti.sync()
        invert_times.append((time.perf_counter() - t0) * 1000)

        ti.sync()
        t0 = time.perf_counter()
        mas.apply_simple(grad, z, use_warp_reduction=use_warp_reduction)
        ti.sync()
        apply_times.append((time.perf_counter() - t0) * 1000)

    # Final SPD check
    stats = compute_z_stats()
    final_spd = stats[1] > 0

    # Statistics
    assemble_avg = np.mean(assemble_times)
    assemble_std = np.std(assemble_times)
    invert_avg = np.mean(invert_times)
    invert_std = np.std(invert_times)
    apply_avg = np.mean(apply_times)
    apply_std = np.std(apply_times)
    total_avg = assemble_avg + invert_avg + apply_avg

    print(f"\n[6] Results:")
    print(f"    Assemble:  {assemble_avg:8.2f} ± {assemble_std:.2f} ms")
    print(f"    Invert:    {invert_avg:8.2f} ± {invert_std:.2f} ms")
    print(f"    Apply:     {apply_avg:8.2f} ± {apply_std:.2f} ms")
    print(f"    ----------------------------------------")
    print(f"    Total:     {total_avg:8.2f} ms per preconditioner application")
    print(f"    SPD check: {'PASS' if final_spd else 'FAIL'}")

    print(f"\n[7] Kernel Profiler (top kernels):")
    ti.profiler.print_kernel_profiler_info('count')

    return {
        'n_verts': n_verts,
        'n_cells': n_cells,
        'levels': mas.actual_levels,
        'warp_reduction': use_warp_reduction,
        'hierarchy_ms': hierarchy_time,
        'assemble_ms': assemble_avg,
        'invert_ms': invert_avg,
        'apply_ms': apply_avg,
        'total_ms': total_avg,
        'spd_pass': final_spd,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MAS Benchmark v3 (Simple Mesh)')
    parser.add_argument('--size', type=int, default=4, help='Vertices per side (4=64 verts, 5=125 verts)')
    parser.add_argument('--iterations', type=int, default=3, help='Number of benchmark iterations')
    parser.add_argument('--warp-reduction', action='store_true', help='Use P1 warp reduction optimization')
    parser.add_argument('--compare', action='store_true', help='Compare with/without warp reduction')
    args = parser.parse_args()

    if args.compare:
        print("\n" + "="*70)
        print("COMPARISON: Sequential vs Warp Reduction")
        print("="*70)

        result_seq = benchmark_mas_simple(args.size, args.iterations, use_warp_reduction=False)

        print("\n\n--- Running Warp Reduction comparison ---\n")

        result_warp = benchmark_mas_simple(args.size, args.iterations, use_warp_reduction=True)

        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Kernel':<15} {'Sequential':<15} {'Warp Reduce':<15} {'Speedup':<10}")
        print("-" * 55)
        for key in ['assemble_ms', 'invert_ms', 'apply_ms', 'total_ms']:
            seq_val = result_seq[key]
            warp_val = result_warp[key]
            speedup = seq_val / warp_val if warp_val > 0 else 0
            name = key.replace('_ms', '')
            print(f"{name:<15} {seq_val:<15.2f} {warp_val:<15.2f} {speedup:<10.2f}x")

        print(f"\nSPD check: Sequential={'PASS' if result_seq['spd_pass'] else 'FAIL'}, "
              f"Warp={'PASS' if result_warp['spd_pass'] else 'FAIL'}")
    else:
        result = benchmark_mas_simple(args.size, args.iterations, use_warp_reduction=args.warp_reduction)

        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Vertices: {result['n_verts']}, Cells: {result['n_cells']}")
        print(f"Levels: {result['levels']}")
        print(f"Warp reduction: {args.warp_reduction}")
        print(f"Total time: {result['total_ms']:.2f} ms")
        print(f"SPD check: {'PASS' if result['spd_pass'] else 'FAIL'}")
