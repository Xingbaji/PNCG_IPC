"""Test new simplified invert_block_matrices API."""

import sys
import os

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
ti.init(arch=ti.cuda, default_fp=ti.f32)

from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner


def test_new_api():
    """Test new method-based API."""
    solver = pncg_ipc_deformer(demo='eight_E_stiffness_test')
    solver.mesh.verts.place({'z': ti.types.vector(3, float)})
    solver.mas = MASPreconditioner(solver.n_verts, solver.n_cells, solver.mesh, use_metis=False)

    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    tests = [
        # (description, method, kwargs)
        ("Default IC with adaptive reg", "ic", {"adaptive_regularization": 0.05}),
        ("Gauss-Jordan", "gauss_jordan", {}),
        ("GJ shorthand", "gj", {}),
        ("Cholesky with reg", "cholesky", {"regularization_epsilon": 5e5}),
        ("Blocked Cholesky with reg", "blocked_cholesky", {"regularization_epsilon": 5e5}),
        ("One-way GJ", "oneway_gj", {}),
        ("Diagonal only", "diagonal", {}),
    ]

    print("\n" + "=" * 60)
    print("Testing New API: invert_block_matrices(method=...)")
    print("=" * 60)

    passed = 0
    for desc, method, kwargs in tests:
        try:
            solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
            solver.mas.invert_block_matrices(method=method, **kwargs)
            print(f"  [PASS] {desc} (method='{method}')")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")

    # Test legacy API backward compatibility
    print("\n" + "-" * 60)
    print("Testing Legacy API Backward Compatibility")
    print("-" * 60)

    legacy_tests = [
        ("IC (legacy)", {"use_incomplete": True, "regularization_epsilon": 5e5}),
        ("Cholesky (legacy)", {"use_cholesky": True, "use_incomplete": False, "regularization_epsilon": 5e5}),
        ("GJ (legacy)", {"use_cholesky": False, "use_incomplete": False}),
        ("Diagonal (legacy)", {"use_full_inversion": False}),
    ]

    for desc, kwargs in legacy_tests:
        try:
            solver.mas.assemble_block_matrices(solver, use_full_hessian=True)
            solver.mas.invert_block_matrices(**kwargs)
            print(f"  [PASS] {desc}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")

    total = len(tests) + len(legacy_tests)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == '__main__':
    success = test_new_api()
    sys.exit(0 if success else 1)
