"""
Test METIS mesh-level reordering.

This tests the new approach where mesh data is reordered BEFORE creating MeshTaichi mesh,
eliminating the need for runtime mapping lookups.

Modes compared:
1. No METIS (baseline)
2. METIS runtime mapping (current approach)
3. METIS pre-reordered (new approach - fastest)
"""

import sys
import os
import time
import builtins

# Suppress print during import
_original_print = builtins.print
def _filtered_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    if any(x in msg.lower() for x in ['error', 'warning', 'fail', 'exception']):
        _original_print(*args, **kwargs)
builtins.print = _filtered_print

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(os.path.join(project_root, 'demo'))

import numpy as np
import taichi as ti
import meshtaichi_patcher as Patcher

# Restore print
builtins.print = _original_print


def test_mesh_reorder_basic():
    """Test basic mesh reordering functionality."""
    print("\n" + "=" * 70)
    print("Test: Basic Mesh Reordering")
    print("=" * 70)

    ti.init(arch=ti.gpu, debug=False)

    from algorithm.mas_preconditioner_small import (
        reorder_mesh_data_metis,
        check_pymetis_available,
        BANKSIZE,
    )

    if not check_pymetis_available():
        print("[SKIP] pymetis not available")
        return True

    # Create simple test mesh
    n_verts = 100
    n_cells = 200

    # Random vertex positions
    np.random.seed(42)
    vertices = np.random.randn(n_verts, 3).astype(np.float64)

    # Random cell connectivity (valid vertex indices)
    cells = np.random.randint(0, n_verts, size=(n_cells, 4)).astype(np.int32)

    # Reorder
    reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(
        vertices, cells, BANKSIZE
    )

    # Verify
    assert reordered_verts.shape == vertices.shape, "Vertex shape mismatch"
    assert reordered_cells.shape == cells.shape, "Cell shape mismatch"
    assert metis_result.is_valid(), "METIS result invalid"

    # Check that all cell vertex IDs are valid
    assert np.all(reordered_cells >= 0), "Negative vertex IDs in cells"
    assert np.all(reordered_cells < n_verts), "Out-of-range vertex IDs in cells"

    print(f"  Vertices: {n_verts}")
    print(f"  Cells: {n_cells}")
    print(f"  Partitions: {metis_result.n_parts}")
    print("  [PASS] Basic reordering works correctly")

    return True


def test_mesh_reorder_cube_20():
    """Test mesh reordering with cube_20 model."""
    print("\n" + "=" * 70)
    print("Test: Mesh Reordering with cube_20")
    print("=" * 70)

    ti.init(arch=ti.gpu, debug=False)

    from algorithm.mas_preconditioner_small import (
        reorder_mesh_data_metis,
        check_pymetis_available,
        BANKSIZE,
    )

    if not check_pymetis_available():
        print("[SKIP] pymetis not available")
        return True

    # Load cube_20 raw data
    # Patcher.load_mesh_rawdata returns dict: {0: vertices, 3: cells, 'face': faces}
    model_path = '../model/mesh/cube_20/cube_20.node'
    raw_data = Patcher.load_mesh_rawdata(model_path)
    vertices = raw_data[0]  # (n_verts, 3) - key 0 for vertices
    cells = raw_data[3]     # (n_cells, 4) - key 3 for tetrahedra

    n_verts = len(vertices)
    n_cells = len(cells)
    print(f"  Original: {n_verts} vertices, {n_cells} cells")

    # Reorder
    t_start = time.perf_counter()
    reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(
        vertices, cells, BANKSIZE
    )
    t_reorder = time.perf_counter() - t_start
    print(f"  Reorder time: {t_reorder*1000:.2f}ms")

    # Verify dimensions
    assert reordered_verts.shape == vertices.shape
    assert reordered_cells.shape == cells.shape
    assert metis_result.is_valid()

    # Verify cell vertex IDs are valid
    assert np.all(reordered_cells >= 0)
    assert np.all(reordered_cells < n_verts)

    # Create mesh with reordered data
    t_start = time.perf_counter()
    mesh = Patcher.load_mesh([(reordered_verts, reordered_cells)], relations=["CV"])
    t_mesh = time.perf_counter() - t_start
    print(f"  Mesh creation time: {t_mesh*1000:.2f}ms")

    # Place vertex attributes
    mesh.verts.place({
        'x': ti.math.vec3,
        'x0': ti.math.vec3,
        'm': ti.f64,
        'grad': ti.math.vec3,
        'z': ti.math.vec3,
    })
    mesh.cells.place({
        'B': ti.math.mat3,
        'W': ti.f64,
    })

    print(f"  Partitions: {metis_result.n_parts}")
    print(f"  BANKSIZE: {BANKSIZE}")
    print("  [PASS] Mesh creation with reordered data successful")

    return True


def test_preconditioner_modes_comparison():
    """Compare all three preconditioner modes."""
    print("\n" + "=" * 70)
    print("Test: Preconditioner Modes Comparison")
    print("=" * 70)

    ti.init(arch=ti.gpu, debug=False)

    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall,
        compute_metis_reorder,
        reorder_mesh_data_metis,
        check_pymetis_available,
        BANKSIZE,
    )

    if not check_pymetis_available():
        print("[SKIP] pymetis not available")
        return True

    # Load cube_20
    # Patcher.load_mesh_rawdata returns dict: {0: vertices, 3: cells, 'face': faces}
    model_path = '../model/mesh/cube_20/cube_20.node'
    raw_data = Patcher.load_mesh_rawdata(model_path)
    vertices = raw_data[0]
    cells = raw_data[3]
    n_verts = len(vertices)
    n_cells = len(cells)

    print(f"  Model: {n_verts} verts, {n_cells} cells")

    # === Mode 1: No METIS ===
    print("\n  Mode 1: No METIS")
    mesh1 = Patcher.load_mesh([(vertices, cells)], relations=["CV"])
    _setup_mesh_fields(mesh1, vertices)
    _precompute_B_W(mesh1)

    mas1 = MASPreconditionerSmall(mesh1)
    mas1.build_hierarchy()
    print(f"    use_metis={mas1.use_metis}, metis_reordered={mas1.metis_reordered}")

    # === Mode 2: METIS runtime mapping ===
    print("\n  Mode 2: METIS Runtime Mapping")
    mesh2 = Patcher.load_mesh([(vertices, cells)], relations=["CV"])
    _setup_mesh_fields(mesh2, vertices)
    _precompute_B_W(mesh2)

    metis_result = compute_metis_reorder(n_verts, cells, BANKSIZE)
    mas2 = MASPreconditionerSmall(mesh2, metis_result=metis_result)
    mas2.build_hierarchy()
    print(f"    use_metis={mas2.use_metis}, metis_reordered={mas2.metis_reordered}")
    print(f"    Has partId_map_real: {mas2.partId_map_real is not None}")

    # === Mode 3: METIS pre-reordered ===
    print("\n  Mode 3: METIS Pre-Reordered")
    reordered_verts, reordered_cells, metis_result3 = reorder_mesh_data_metis(
        vertices, cells, BANKSIZE
    )
    mesh3 = Patcher.load_mesh([(reordered_verts, reordered_cells)], relations=["CV"])
    _setup_mesh_fields(mesh3, reordered_verts)
    _precompute_B_W(mesh3)

    mas3 = MASPreconditionerSmall(mesh3, metis_reordered=True)
    mas3.build_hierarchy()
    print(f"    use_metis={mas3.use_metis}, metis_reordered={mas3.metis_reordered}")
    print(f"    Has partId_map_real: {mas3.partId_map_real is not None}")

    # Verify mode 3 has no mapping tables
    assert mas3.partId_map_real is None, "Pre-reordered mode should not have mapping"
    assert mas3.real_map_partId is None, "Pre-reordered mode should not have mapping"

    print("\n  [PASS] All three modes created successfully")
    return True


def test_preconditioner_benchmark():
    """Benchmark all three modes."""
    print("\n" + "=" * 70)
    print("Test: Performance Benchmark")
    print("=" * 70)

    ti.init(arch=ti.gpu, debug=False)

    from algorithm.mas_preconditioner_small import (
        MASPreconditionerSmall,
        compute_metis_reorder,
        reorder_mesh_data_metis,
        check_pymetis_available,
        BANKSIZE,
    )

    if not check_pymetis_available():
        print("[SKIP] pymetis not available")
        return True

    # Load cube_20
    model_path = '../model/mesh/cube_20/cube_20.node'
    raw_data = Patcher.load_mesh_rawdata(model_path)
    vertices = raw_data[0]
    cells = raw_data[3]
    n_verts = len(vertices)
    n_cells = len(cells)

    print(f"  Model: {n_verts} verts, {n_cells} cells")

    warmup = 3
    n_iterations = 10

    # Create a mock solver for assemble
    class MockSolver:
        def __init__(self):
            self.dt = 0.01
            self.mu = 1000.0
            self.la = 5000.0

    solver = MockSolver()

    results = {}

    # === Mode 1: No METIS ===
    print("\n  Benchmarking Mode 1: No METIS...")
    mesh1 = Patcher.load_mesh([(vertices, cells)], relations=["CV"])
    _setup_mesh_fields(mesh1, vertices)
    _precompute_B_W(mesh1)
    _init_random_grad(mesh1, n_verts)

    mas1 = MASPreconditionerSmall(mesh1)
    mas1.build_hierarchy()

    # Benchmark
    times_assemble = []
    times_apply = []
    for i in range(warmup + n_iterations):
        ti.sync()
        t0 = time.perf_counter()
        mas1.assemble_block_matrices(solver)
        ti.sync()
        t1 = time.perf_counter()
        mas1.invert_block_matrices()
        ti.sync()
        t2 = time.perf_counter()
        mas1.apply()
        ti.sync()
        t3 = time.perf_counter()
        if i >= warmup:
            times_assemble.append(t1 - t0)
            times_apply.append(t3 - t2)

    results['no_metis'] = {
        'assemble': np.mean(times_assemble) * 1000,
        'apply': np.mean(times_apply) * 1000,
    }

    # === Mode 2: METIS runtime mapping ===
    print("  Benchmarking Mode 2: METIS Runtime Mapping...")
    mesh2 = Patcher.load_mesh([(vertices, cells)], relations=["CV"])
    _setup_mesh_fields(mesh2, vertices)
    _precompute_B_W(mesh2)
    _init_random_grad(mesh2, n_verts)

    metis_result = compute_metis_reorder(n_verts, cells, BANKSIZE)
    mas2 = MASPreconditionerSmall(mesh2, metis_result=metis_result)
    mas2.build_hierarchy()

    times_assemble = []
    times_apply = []
    for i in range(warmup + n_iterations):
        ti.sync()
        t0 = time.perf_counter()
        mas2.assemble_block_matrices(solver)
        ti.sync()
        t1 = time.perf_counter()
        mas2.invert_block_matrices()
        ti.sync()
        t2 = time.perf_counter()
        mas2.apply()
        ti.sync()
        t3 = time.perf_counter()
        if i >= warmup:
            times_assemble.append(t1 - t0)
            times_apply.append(t3 - t2)

    results['runtime_mapping'] = {
        'assemble': np.mean(times_assemble) * 1000,
        'apply': np.mean(times_apply) * 1000,
    }

    # === Mode 3: METIS pre-reordered ===
    print("  Benchmarking Mode 3: METIS Pre-Reordered...")
    reordered_verts, reordered_cells, _ = reorder_mesh_data_metis(
        vertices, cells, BANKSIZE
    )
    mesh3 = Patcher.load_mesh([(reordered_verts, reordered_cells)], relations=["CV"])
    _setup_mesh_fields(mesh3, reordered_verts)
    _precompute_B_W(mesh3)
    _init_random_grad(mesh3, n_verts)

    mas3 = MASPreconditionerSmall(mesh3, metis_reordered=True)
    mas3.build_hierarchy()

    times_assemble = []
    times_apply = []
    for i in range(warmup + n_iterations):
        ti.sync()
        t0 = time.perf_counter()
        mas3.assemble_block_matrices(solver)
        ti.sync()
        t1 = time.perf_counter()
        mas3.invert_block_matrices()
        ti.sync()
        t2 = time.perf_counter()
        mas3.apply()
        ti.sync()
        t3 = time.perf_counter()
        if i >= warmup:
            times_assemble.append(t1 - t0)
            times_apply.append(t3 - t2)

    results['pre_reordered'] = {
        'assemble': np.mean(times_assemble) * 1000,
        'apply': np.mean(times_apply) * 1000,
    }

    # Print results
    print("\n  Results (ms):")
    print("  " + "-" * 60)
    print(f"  {'Mode':<25} {'Assemble':>12} {'Apply':>12} {'Total':>12}")
    print("  " + "-" * 60)

    for mode, data in results.items():
        total = data['assemble'] + data['apply']
        print(f"  {mode:<25} {data['assemble']:>12.3f} {data['apply']:>12.3f} {total:>12.3f}")

    print("  " + "-" * 60)

    # Calculate speedups
    baseline = results['no_metis']['assemble'] + results['no_metis']['apply']
    runtime_total = results['runtime_mapping']['assemble'] + results['runtime_mapping']['apply']
    prereorder_total = results['pre_reordered']['assemble'] + results['pre_reordered']['apply']

    print(f"\n  Speedup vs No-METIS:")
    print(f"    Runtime mapping: {baseline / runtime_total:.2f}x")
    print(f"    Pre-reordered: {baseline / prereorder_total:.2f}x")

    if runtime_total > 0:
        print(f"\n  Pre-reordered vs Runtime mapping: {runtime_total / prereorder_total:.2f}x")

    print("\n  [PASS] Benchmark completed")
    return True


def _setup_mesh_fields(mesh, vertices):
    """Setup mesh fields for testing."""
    mesh.verts.place({
        'x': ti.math.vec3,
        'x0': ti.math.vec3,
        'm': ti.f64,
        'grad': ti.math.vec3,
        'z': ti.math.vec3,
    })
    mesh.cells.place({
        'B': ti.math.mat3,
        'W': ti.f64,
    })

    # Initialize positions from numpy array
    verts_ti = ti.Vector.field(3, dtype=ti.f64, shape=len(vertices))
    verts_ti.from_numpy(vertices)

    @ti.kernel
    def init(verts_data: ti.template()):
        for v in mesh.verts:
            mesh.verts.x[v.id] = verts_data[v.id]
            mesh.verts.x0[v.id] = verts_data[v.id]
            mesh.verts.m[v.id] = 0.0
            mesh.verts.grad[v.id] = ti.Vector([0.0, 0.0, 0.0])

    init(verts_ti)


def _precompute_B_W(mesh, density=1000.0):
    """Precompute B matrix and cell volumes."""
    @ti.kernel
    def compute():
        for c in mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += density * c.W / 4.0

    compute()


def _init_random_grad(mesh, n_verts):
    """Initialize random gradient for testing."""
    np.random.seed(42)
    grad_np = np.random.randn(n_verts, 3).astype(np.float64)
    grad_ti = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
    grad_ti.from_numpy(grad_np)

    @ti.kernel
    def set_grad(grad_data: ti.template()):
        for v in mesh.verts:
            mesh.verts.grad[v.id] = grad_data[v.id]

    set_grad(grad_ti)


if __name__ == "__main__":
    all_passed = True

    tests = [
        test_mesh_reorder_basic,
        test_mesh_reorder_cube_20,
        test_preconditioner_modes_comparison,
        test_preconditioner_benchmark,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 70)
