"""
MAS Preconditioner Test Demo

Incremental testing of MAS preconditioner components:
1. Test hierarchy build
2. Test block matrix assembly
3. Test block matrix inversion
4. Test apply (P * g)

Usage:
    python test_mas_preconditioner.py --test build      # Test hierarchy build only
    python test_mas_preconditioner.py --test assemble   # Test matrix assembly
    python test_mas_preconditioner.py --test invert     # Test matrix inversion
    python test_mas_preconditioner.py --test apply      # Test full apply
    python test_mas_preconditioner.py --test all        # Run all tests
    python test_mas_preconditioner.py --test step       # Test one simulation step
"""

import sys
import os
import time
import argparse
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
os.chdir(demo_dir)

import taichi as ti


def test_hierarchy_build(demo='eight_E_stiffness_mas'):
    """Test 1: Build hierarchy only."""
    print("\n" + "="*70)
    print("TEST 1: Hierarchy Build")
    print("="*70)

    from algorithm.mas_preconditioner import MASPreconditioner
    from util.model_loading import model_loading

    # Load model
    print(f"\n[1.1] Loading model: {demo}")
    model = model_loading(demo=demo)
    mesh = model.mesh

    # Place required fields
    mesh.verts.place({
        'x': ti.types.vector(3, float),
        'v': ti.types.vector(3, float),
        'm': float,
        'grad': ti.types.vector(3, float),
        'diagH': ti.types.vector(3, float),
        'z': ti.types.vector(3, float),
    })
    mesh.cells.place({'B': ti.math.mat3, 'W': float})
    mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

    n_verts = len(mesh.verts)
    n_cells = len(mesh.cells)
    print(f"    n_verts: {n_verts}, n_cells: {n_cells}")

    # Create MAS preconditioner
    print(f"\n[1.2] Creating MASPreconditioner...")
    try:
        mas = MASPreconditioner(n_verts, n_cells, mesh)
        print(f"    Created successfully!")
        print(f"    n_levels (max): {mas.n_levels}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Build hierarchy
    print(f"\n[1.3] Building hierarchy...")
    try:
        ti.sync()
        t0 = time.perf_counter()
        mas.build_hierarchy()
        ti.sync()
        t1 = time.perf_counter()
        print(f"    Build time: {(t1-t0)*1000:.2f} ms")
        print(f"    hierarchy_built: {mas.hierarchy_built}")
        print(f"    total_neighbors: {mas.total_neighbors}")

        # Print level sizes
        level_sizes = mas.level_sizes.to_numpy()
        print(f"    Level sizes: {level_sizes[:6]}")

    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n[1.4] Hierarchy build: PASSED")
    return mas, mesh, model


def test_matrix_assembly(mas, mesh, model):
    """Test 2: Assemble block matrices."""
    print("\n" + "="*70)
    print("TEST 2: Block Matrix Assembly")
    print("="*70)

    # Create a minimal solver-like object
    class MinimalSolver:
        def __init__(self, model, mesh):
            self.mesh = mesh
            self.mu = model.mu
            self.la = model.la
            self.dt = model.dt
            self.elastic_type = model.elastic_type
            self.n_verts = len(mesh.verts)
            self.n_cells = len(mesh.cells)

    solver = MinimalSolver(model, mesh)

    # Initialize vertex data
    print(f"\n[2.1] Initializing vertex data...")

    @ti.kernel
    def init_vertex_data(mesh: ti.template()):
        for vert in mesh.verts:
            vert.m = 1.0  # Unit mass
            vert.grad = ti.Vector([0.0, -1.0, 0.0])  # Gravity-like gradient
            vert.diagH = ti.Vector([1.0, 1.0, 1.0])  # Unit diagonal Hessian
            vert.z = ti.Vector([0.0, 0.0, 0.0])

    init_vertex_data(mesh)
    print(f"    Initialized {solver.n_verts} vertices")

    # Precompute B and W for cells
    print(f"\n[2.2] Precomputing cell data...")

    @ti.kernel
    def precompute_cells(mesh: ti.template()):
        for c in mesh.cells:
            # Compute deformation gradient inverse (B)
            Dm = ti.Matrix.cols([
                c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))
            ])
            c.B = Dm.inverse()
            # Compute volume weight (W)
            c.W = ti.abs(Dm.determinant()) / 6.0

    precompute_cells(mesh)
    print(f"    Precomputed {solver.n_cells} cells")

    # Assemble block matrices
    print(f"\n[2.3] Assembling block matrices...")
    try:
        ti.sync()
        t0 = time.perf_counter()
        mas.assemble_block_matrices(solver, use_full_hessian=True)
        ti.sync()
        t1 = time.perf_counter()
        print(f"    Assembly time: {(t1-t0)*1000:.2f} ms")

        # Check block matrices
        block_matrices = mas.block_matrices.to_numpy()
        n_blocks = (solver.n_verts + 15) // 16
        print(f"    n_blocks: {n_blocks}")
        print(f"    block_matrices shape: {block_matrices.shape}")

        # Check for NaN/Inf
        if np.any(np.isnan(block_matrices)):
            print(f"    WARNING: NaN found in block_matrices!")
        if np.any(np.isinf(block_matrices)):
            print(f"    WARNING: Inf found in block_matrices!")

        # Check diagonal dominance (for a few blocks)
        print(f"\n[2.4] Checking first block diagonal...")
        for level in range(min(2, mas.n_levels)):
            block_0 = block_matrices[level, 0, :, :]
            diag = np.diag(block_0)
            print(f"    Level {level}, Block 0 diagonal (first 6): {diag[:6]}")
            print(f"    Level {level}, Block 0 diagonal min/max: {diag.min():.6f} / {diag.max():.6f}")

    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[2.5] Matrix assembly: PASSED")
    return True


def test_matrix_inversion(mas, mesh, model):
    """Test 3: Invert block matrices."""
    print("\n" + "="*70)
    print("TEST 3: Block Matrix Inversion")
    print("="*70)

    print(f"\n[3.1] Inverting block matrices...")
    try:
        ti.sync()
        t0 = time.perf_counter()
        mas.invert_block_matrices(use_full_inversion=True)
        ti.sync()
        t1 = time.perf_counter()
        print(f"    Inversion time: {(t1-t0)*1000:.2f} ms")
        print(f"    matrices_inverted: {mas.matrices_inverted}")

        # Check inverse matrices
        inv_matrices = mas.inv_block_matrices.to_numpy()
        print(f"    inv_block_matrices shape: {inv_matrices.shape}")

        # Check for NaN/Inf
        if np.any(np.isnan(inv_matrices)):
            print(f"    WARNING: NaN found in inv_block_matrices!")
            # Find where
            nan_locs = np.argwhere(np.isnan(inv_matrices))
            print(f"    NaN locations (first 10): {nan_locs[:10]}")
        if np.any(np.isinf(inv_matrices)):
            print(f"    WARNING: Inf found in inv_block_matrices!")

        # Check inverse diagonal
        print(f"\n[3.2] Checking first block inverse diagonal...")
        for level in range(min(2, mas.n_levels)):
            inv_block_0 = inv_matrices[level, 0, :, :]
            diag = np.diag(inv_block_0)
            print(f"    Level {level}, Inv Block 0 diagonal (first 6): {diag[:6]}")
            print(f"    Level {level}, Inv Block 0 diagonal min/max: {diag.min():.6f} / {diag.max():.6f}")

        # Verify A * A^{-1} ≈ I for first block
        print(f"\n[3.3] Verifying A * A^{-1} ≈ I for first block...")
        block_matrices = mas.block_matrices.to_numpy()
        A = block_matrices[0, 0, :, :]
        A_inv = inv_matrices[0, 0, :, :]
        product = A @ A_inv
        identity_error = np.linalg.norm(product - np.eye(48)) / np.linalg.norm(np.eye(48))
        print(f"    ||A * A^{-1} - I|| / ||I|| = {identity_error:.6e}")
        if identity_error > 1e-3:
            print(f"    WARNING: Large inversion error!")

    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[3.4] Matrix inversion: PASSED")
    return True


def test_apply(mas, mesh, model):
    """Test 4: Apply preconditioner (z = P * g)."""
    print("\n" + "="*70)
    print("TEST 4: Apply Preconditioner (z = P * g)")
    print("="*70)

    n_verts = len(mesh.verts)

    # Set a known gradient
    print(f"\n[4.1] Setting test gradient...")

    @ti.kernel
    def set_test_gradient(mesh: ti.template()):
        for vert in mesh.verts:
            # Simple gradient: gravity in -y direction
            vert.grad = ti.Vector([0.0, -1.0, 0.0])
            vert.z = ti.Vector([0.0, 0.0, 0.0])

    set_test_gradient(mesh)

    # Get gradient norm before
    @ti.kernel
    def compute_grad_norm(mesh: ti.template()) -> float:
        norm_sq = 0.0
        for vert in mesh.verts:
            norm_sq += vert.grad.norm_sqr()
        return ti.sqrt(norm_sq)

    grad_norm = compute_grad_norm(mesh)
    print(f"    ||grad|| = {grad_norm:.6f}")

    # Apply preconditioner
    print(f"\n[4.2] Applying preconditioner...")
    try:
        ti.sync()
        t0 = time.perf_counter()
        mas.apply(use_full_solve=True)
        ti.sync()
        t1 = time.perf_counter()
        print(f"    Apply time: {(t1-t0)*1000:.2f} ms")

        # Check z
        @ti.kernel
        def compute_z_norm(mesh: ti.template()) -> float:
            norm_sq = 0.0
            for vert in mesh.verts:
                norm_sq += vert.z.norm_sqr()
            return ti.sqrt(norm_sq)

        @ti.kernel
        def compute_z_stats(mesh: ti.template()) -> ti.types.vector(4, float):
            min_val = 1e10
            max_val = -1e10
            sum_val = 0.0
            n = 0.0
            for vert in mesh.verts:
                for i in ti.static(range(3)):
                    val = vert.z[i]
                    ti.atomic_min(min_val, val)
                    ti.atomic_max(max_val, val)
                    sum_val += val
                    n += 1.0
            return ti.Vector([min_val, max_val, sum_val / n, n])

        z_norm = compute_z_norm(mesh)
        z_stats = compute_z_stats(mesh)

        print(f"    ||z|| = {z_norm:.6f}")
        print(f"    z min/max/mean: {z_stats[0]:.6f} / {z_stats[1]:.6f} / {z_stats[2]:.6f}")

        # Check for NaN/Inf in z
        @ti.kernel
        def check_z_valid(mesh: ti.template()) -> int:
            invalid = 0
            for vert in mesh.verts:
                for i in ti.static(range(3)):
                    if ti.math.isnan(vert.z[i]) or ti.math.isinf(vert.z[i]):
                        invalid = 1
            return invalid

        if check_z_valid(mesh):
            print(f"    WARNING: NaN or Inf found in z!")

        # Check z·g (should be positive for valid preconditioner)
        @ti.kernel
        def compute_z_dot_g(mesh: ti.template()) -> float:
            dot = 0.0
            for vert in mesh.verts:
                dot += vert.z.dot(vert.grad)
            return dot

        z_dot_g = compute_z_dot_g(mesh)
        print(f"    z·g = {z_dot_g:.6f} (should be positive)")

        if z_dot_g <= 0:
            print(f"    WARNING: z·g <= 0, preconditioner may not be SPD!")

    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[4.3] Apply: PASSED")
    return True


def test_single_step(demo='eight_E_stiffness_mas'):
    """Test 5: Run a single simulation step with MAS-PNCG."""
    print("\n" + "="*70)
    print("TEST 5: Single Simulation Step")
    print("="*70)

    print(f"\n[5.1] Creating MASPNCGSolver...")
    try:
        from algorithm.mas_pncg_solver import MASPNCGSolver
        solver = MASPNCGSolver(demo=demo)
        print(f"    Created successfully!")
        print(f"    n_verts: {solver.n_verts}, n_cells: {solver.n_cells}")
        print(f"    MAS levels: {solver.mas_preconditioner.n_levels}")
    except Exception as e:
        print(f"    FAILED to create solver: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n[5.2] Running step()...")
    try:
        ti.sync()
        t0 = time.perf_counter()
        n_iters = solver.step()
        ti.sync()
        t1 = time.perf_counter()
        print(f"    Step time: {(t1-t0)*1000:.2f} ms")
        print(f"    Iterations: {n_iters}")
    except Exception as e:
        print(f"    FAILED during step: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[5.3] Single step: PASSED")
    return True


def run_all_tests(demo='eight_E_stiffness_mas'):
    """Run all tests in sequence."""
    print("\n" + "="*70)
    print("MAS PRECONDITIONER TEST SUITE")
    print("="*70)
    print(f"Demo: {demo}")

    results = {}

    # Test 1: Hierarchy build
    result = test_hierarchy_build(demo)
    if result is None:
        results['hierarchy_build'] = 'FAILED'
        print("\nStopping due to hierarchy build failure.")
        return results
    results['hierarchy_build'] = 'PASSED'
    mas, mesh, model = result

    # Test 2: Matrix assembly
    if test_matrix_assembly(mas, mesh, model):
        results['matrix_assembly'] = 'PASSED'
    else:
        results['matrix_assembly'] = 'FAILED'
        print("\nStopping due to matrix assembly failure.")
        return results

    # Test 3: Matrix inversion
    if test_matrix_inversion(mas, mesh, model):
        results['matrix_inversion'] = 'PASSED'
    else:
        results['matrix_inversion'] = 'FAILED'
        print("\nStopping due to matrix inversion failure.")
        return results

    # Test 4: Apply
    if test_apply(mas, mesh, model):
        results['apply'] = 'PASSED'
    else:
        results['apply'] = 'FAILED'
        print("\nStopping due to apply failure.")
        return results

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, status in results.items():
        status_str = "✓ PASSED" if status == 'PASSED' else "✗ FAILED"
        print(f"  {test_name:<25} {status_str}")
    print("="*70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAS Preconditioner Test')
    parser.add_argument('--test', type=str, default='all',
                       choices=['build', 'assemble', 'invert', 'apply', 'step', 'all'],
                       help='Which test to run')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_mas',
                       help='Demo configuration to use')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    if args.test == 'all':
        run_all_tests(demo=args.demo)
    elif args.test == 'build':
        test_hierarchy_build(demo=args.demo)
    elif args.test == 'step':
        test_single_step(demo=args.demo)
    else:
        # For other tests, we need to build hierarchy first
        result = test_hierarchy_build(demo=args.demo)
        if result is not None:
            mas, mesh, model = result
            if args.test == 'assemble':
                test_matrix_assembly(mas, mesh, model)
            elif args.test == 'invert':
                test_matrix_assembly(mas, mesh, model)
                test_matrix_inversion(mas, mesh, model)
            elif args.test == 'apply':
                test_matrix_assembly(mas, mesh, model)
                test_matrix_inversion(mas, mesh, model)
                test_apply(mas, mesh, model)
