"""
GCP Initial Collision Test

This test verifies that the Geometric Contact Potential (GCP) correctly handles
large dHat values without causing spurious collisions at the initial state.

Key Tests:
1. With standard IPC + large dHat (without adjacency): Should have many false collisions
2. With GCP + large dHat: Should have zero/minimal collisions due to gamma filtering

Usage:
    python test_gcp_initial_collision.py                  # Run all tests
    python test_gcp_initial_collision.py --dhat 0.1       # Test with specific dHat
    python test_gcp_initial_collision.py --verbose        # Detailed output
"""

import sys
import os
import argparse
import time

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.append(demo_dir)
os.chdir(demo_dir)

import taichi as ti


def test_ipc_large_dhat_no_adj(demo, dhat, verbose=False):
    """
    Test standard IPC with large dHat but WITHOUT adjacency filtering.

    This should show many false collisions because adjacent boundary elements
    are within dHat distance at the initial (rest) configuration.
    """
    print("\n" + "="*60)
    print("Test 1: Standard IPC with large dHat (NO adjacency filtering)")
    print("="*60)

    from algorithm.pncg_base_ipc import pncg_ipc_deformer

    # Create solver without adjacency matrix
    solver = pncg_ipc_deformer(demo=demo)
    solver.dHat = dhat
    solver.adj = 0  # No adjacency filtering!

    # Initialize BVH (this will NOT define adjacency matrix since adj=0)
    print(f"dHat = {dhat}")
    print(f"Adjacency filtering: DISABLED")
    print(f"n_boundary_points = {solver.n_boundary_points}")
    print(f"n_boundary_triangles = {solver.n_boundary_triangles}")
    print(f"n_boundary_edges = {solver.n_boundary_edges}")

    # Find constraints at initial state
    print("\nFinding constraints at initial state...")
    solver.build_bvh()
    solver.cid_root.deactivate_all()
    solver.find_constraints_PT_bvh()
    solver.find_constraints_EE_bvh()

    # Count constraints
    n_constraints = solver.print_cnts()

    print(f"\n>>> RESULT: {n_constraints} constraints detected at rest configuration")

    if n_constraints > 0:
        print(">>> This is EXPECTED - without adjacency filtering, adjacent elements")
        print("    within dHat are incorrectly flagged as collision candidates.")

    return n_constraints


def test_gcp_large_dhat(demo, dhat, verbose=False):
    """
    Test GCP with large dHat.

    GCP uses directional factors (gamma) to automatically filter adjacent elements,
    so even with large dHat, there should be few/no constraints at rest.
    """
    print("\n" + "="*60)
    print("Test 2: GCP with large dHat (gamma filtering)")
    print("="*60)

    from algorithm.collision_detection_bvh import collision_detection_bvh_module
    from algorithm.gcp_contact_potential import GCPModule, GCPConfig
    from util.model_loading import model_loading

    # Load model
    model = model_loading(demo=demo)

    # Create a minimal collision detection module
    class GCPTestModule(collision_detection_bvh_module):
        def __init__(self, demo):
            super().__init__(demo)
            self.dHat = dhat
            self.init_bvh()

    solver = GCPTestModule(demo=demo)

    # Initialize GCP module
    gcp = GCPModule(
        solver.n_boundary_points,
        solver.n_boundary_edges,
        solver.n_boundary_triangles,
        GCPConfig(
            epsilon_target=dhat,
            adaptive_epsilon=True,
            alpha=0.1,
            kappa=1.0
        )
    )

    print(f"dHat (epsilon_target) = {dhat}")
    print(f"Adjacency filtering: NOT NEEDED (gamma filtering instead)")
    print(f"n_boundary_points = {solver.n_boundary_points}")
    print(f"n_boundary_triangles = {solver.n_boundary_triangles}")
    print(f"n_boundary_edges = {solver.n_boundary_edges}")

    # Compute adaptive epsilon from rest configuration
    print("\nComputing adaptive epsilon from rest configuration...")
    gcp.compute_adaptive_epsilon(
        solver.mesh,
        solver.boundary_points,
        solver.boundary_edges,
        solver.boundary_triangles
    )

    # Build BVH
    solver.build_bvh()

    # Find constraints with GCP filtering
    print("Finding constraints with GCP gamma filtering...")
    gcp.find_constraints_gcp(
        solver.mesh,
        solver.boundary_points,
        solver.boundary_edges,
        solver.boundary_triangles,
        solver.bvh_triangles,
        solver.bvh_edges,
        solver.n_verts
    )

    # Count constraints
    n_constraints = gcp.print_constraints_gcp()

    print(f"\n>>> RESULT: {n_constraints} constraints detected at rest configuration")

    if n_constraints == 0:
        print(">>> This is EXPECTED - GCP's gamma filtering correctly identifies")
        print("    that adjacent elements are not approaching contact.")
    else:
        print(">>> Some constraints detected - this may be due to:")
        print("    1. Non-zero gamma for nearly-parallel surfaces")
        print("    2. Actual contact regions in the rest configuration")

    return n_constraints


def test_ipc_with_adjacency(demo, dhat, verbose=False):
    """
    Test standard IPC with large dHat AND adjacency filtering.

    This is the baseline comparison - adjacency matrix explicitly excludes
    adjacent element pairs that are within dHat at rest.
    """
    print("\n" + "="*60)
    print("Test 3: Standard IPC with large dHat (WITH adjacency filtering)")
    print("="*60)

    from algorithm.pncg_base_ipc import pncg_ipc_deformer

    # Create solver with adjacency matrix
    solver = pncg_ipc_deformer(demo=demo)
    solver.dHat = dhat
    solver.adj = 1  # Enable adjacency filtering

    # Initialize BVH with adjacency matrix
    print(f"dHat = {dhat}")
    print(f"Adjacency filtering: ENABLED (define_adj_matrix)")

    # Manually define adjacency matrix since adj=0 in demo config
    # This was not done during init_bvh() since adj=0 at construction time
    print("\nDefining adjacency matrix...")
    solver.define_adj_matrix()

    # Switch to adjacency-aware functions (these were set to no_adj versions at construction)
    solver.attempt_PT = solver.attempt_PT_adj
    solver.attempt_EE = solver.attempt_EE_adj

    # Find constraints at initial state
    print("\nFinding constraints at initial state...")
    solver.build_bvh()
    solver.cid_root.deactivate_all()
    solver.find_constraints_PT_bvh()
    solver.find_constraints_EE_bvh()

    # Count constraints
    n_constraints = solver.print_cnts()

    print(f"\n>>> RESULT: {n_constraints} constraints detected at rest configuration")

    if n_constraints == 0:
        print(">>> This is EXPECTED - adjacency matrix correctly excludes")
        print("    adjacent element pairs from collision detection.")

    return n_constraints


def run_all_tests(demo='eight_E_drop_demo_contact', dhat=0.1, verbose=False):
    """Run all initial collision tests and compare results."""

    print("\n" + "#"*70)
    print("# GCP Initial Collision Test Suite")
    print("#"*70)
    print(f"\nDemo: {demo}")
    print(f"dHat: {dhat}")
    print(f"Verbose: {verbose}")

    results = {}

    # Test 1: IPC without adjacency (expected: many false collisions)
    try:
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f64)
        n1 = test_ipc_large_dhat_no_adj(demo, dhat, verbose)
        results['ipc_no_adj'] = n1
    except Exception as e:
        print(f"Test 1 failed: {e}")
        results['ipc_no_adj'] = -1

    # Test 2: GCP with gamma filtering (expected: zero/minimal collisions)
    try:
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f64)
        n2 = test_gcp_large_dhat(demo, dhat, verbose)
        results['gcp'] = n2
    except Exception as e:
        print(f"Test 2 failed: {e}")
        results['gcp'] = -1

    # Test 3: IPC with adjacency (expected: zero collisions - baseline)
    try:
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f64)
        n3 = test_ipc_with_adjacency(demo, dhat, verbose)
        results['ipc_with_adj'] = n3
    except Exception as e:
        print(f"Test 3 failed: {e}")
        results['ipc_with_adj'] = -1

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    print(f"\n{'Method':<30} {'Constraints at Rest':<20} {'Status'}")
    print("-"*70)

    # IPC without adjacency
    status1 = "EXPECTED (false collisions)" if results['ipc_no_adj'] > 0 else "UNEXPECTED"
    print(f"{'IPC (no adjacency)':<30} {results['ipc_no_adj']:<20} {status1}")

    # GCP
    status2 = "PASS (gamma filtering works)" if results['gcp'] <= 5 else "NEEDS INVESTIGATION"
    print(f"{'GCP (gamma filtering)':<30} {results['gcp']:<20} {status2}")

    # IPC with adjacency
    # Note: IPC with adjacency may still have some constraints from cross-object pairs
    # that happen to be within dHat at rest (e.g., different E letters close together)
    status3 = "OK (cross-object pairs)" if results['ipc_with_adj'] < results['ipc_no_adj'] else "UNEXPECTED"
    print(f"{'IPC (with adjacency)':<30} {results['ipc_with_adj']:<20} {status3}")

    print("-"*70)

    # Comparison
    print("\n>>> KEY INSIGHT:")
    if results['gcp'] <= results['ipc_with_adj'] + 5:
        print("    GCP achieves similar filtering to adjacency matrix WITHOUT")
        print("    precomputing the adjacency structure!")
        print(f"    Reduction from IPC(no_adj): {results['ipc_no_adj']} -> {results['gcp']} constraints")
    else:
        print("    GCP has more constraints than expected. Possible causes:")
        print("    - Alpha parameter may need tuning")
        print("    - Rest configuration has actual contact regions")

    return results


def main():
    parser = argparse.ArgumentParser(description='GCP Initial Collision Test')
    parser.add_argument('--demo', type=str, default='eight_E_drop_demo_contact',
                        help='Demo configuration to test')
    parser.add_argument('--dhat', type=float, default=0.1,
                        help='dHat value to test (default: 0.1, 10x typical IPC)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--test', type=int, choices=[1, 2, 3], default=None,
                        help='Run specific test only (1=IPC no adj, 2=GCP, 3=IPC with adj)')
    args = parser.parse_args()

    if args.test is not None:
        ti.init(arch=ti.gpu, default_fp=ti.f64)

        if args.test == 1:
            test_ipc_large_dhat_no_adj(args.demo, args.dhat, args.verbose)
        elif args.test == 2:
            test_gcp_large_dhat(args.demo, args.dhat, args.verbose)
        elif args.test == 3:
            test_ipc_with_adjacency(args.demo, args.dhat, args.verbose)
    else:
        run_all_tests(args.demo, args.dhat, args.verbose)


if __name__ == '__main__':
    main()
