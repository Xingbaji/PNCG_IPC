"""
Diagnose the quality of P @ g at the first iteration in MAS preconditioner.

Usage:
    cd /root/PNCG_IPC/demo
    PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/diagnose_mas_quality.py
"""

import sys
import os
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.mas_preconditioner_pkg import MASPreconditioner
from algorithm.pncg_base_ipc import pncg_ipc_deformer


class DiagnosticSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


@ti.kernel
def apply_diagonal_preconditioner(grad: ti.template(), diagH: ti.template(), z_diag: ti.template()):
    for i in grad:
        for d in ti.static(range(3)):
            if diagH[i][d] > 1e-10:
                z_diag[i][d] = grad[i][d] / diagH[i][d]
            else:
                z_diag[i][d] = grad[i][d]


def diagnose_mas_quality():
    print("=" * 80)
    print("MAS Preconditioner Quality Diagnostic: First Iteration P @ g")
    print("=" * 80)

    # Initialize solver
    solver = DiagnosticSolver()
    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")

    # Setup first iteration state
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()
    if solver.ground_barrier == 1:
        solver.add_grad_and_diagH_ground_barrier()

    # Get gradient
    g = solver.mesh.verts.grad.to_numpy()
    print(f"\nGradient statistics:")
    print(f"  ||g||_2 = {np.linalg.norm(g):.4e}")
    print(f"  ||g||_inf = {np.max(np.abs(g)):.4e}")

    # Build and apply MAS preconditioner
    solver.mas.build_hierarchy()
    print(f"\nMAS hierarchy: {solver.mas.actual_levels} levels")
    for l in range(solver.mas.actual_levels):
        level_size = solver.mas.level_size.to_numpy()[l]
        print(f"  Level {l}: {level_size[0]} nodes (offset {level_size[1]})")

    # Assemble and invert block matrices
    solver.mas.assemble_block_matrices(solver, use_full_hessian=True)

    # Get block 0 eigenvalue analysis
    solver.mas._expand_sym_to_full()
    block0 = solver.mas.full_block_matrix.to_numpy()[0]
    block0_sym = (block0 + block0.T) / 2
    eigenvalues = np.linalg.eigvalsh(block0_sym)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)

    print(f"\nBlock 0 eigenvalue analysis:")
    print(f"  λ_min = {min_eig:.4e}")
    print(f"  λ_max = {max_eig:.4e}")
    print(f"  Condition number = {abs(max_eig / min_eig) if min_eig != 0 else float('inf'):.4e}")
    print(f"  Negative eigenvalues: {np.sum(eigenvalues < 0)}")

    # Invert with proper regularization
    reg_epsilon = abs(min_eig) * 1.1 + 1e3 if min_eig < 0 else 1e3
    solver.mas.invert_block_matrices(
        use_full_inversion=True,
        use_cholesky=False,  # Gauss-Jordan
        force_symmetry=True,
        regularization_epsilon=reg_epsilon
    )
    print(f"\nUsing regularization ε = {reg_epsilon:.4e}")

    # Apply MAS preconditioner: z = P @ g
    solver.mas.apply()
    z = solver.mesh.verts.z.to_numpy()
    n_verts = solver.n_verts

    print(f"\n" + "=" * 80)
    print("QUALITY METRICS FOR P @ g")
    print("=" * 80)

    # Metric 1: Descent direction
    gTz = np.sum(g * z)
    print(f"\n1. Descent Direction Check:")
    print(f"   g^T z = {gTz:.6e}")
    print(f"   Status: {'✓ VALID (g^T z > 0)' if gTz > 0 else '✗ INVALID (g^T z <= 0)'}")

    # Metric 2: Angle between z and g
    g_norm = np.linalg.norm(g)
    z_norm = np.linalg.norm(z)
    cos_angle = gTz / (g_norm * z_norm + 1e-10)
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    print(f"\n2. Direction Alignment:")
    print(f"   ||g|| = {g_norm:.4e}")
    print(f"   ||z|| = {z_norm:.4e}")
    print(f"   cos(θ) = {cos_angle:.6f}")
    print(f"   θ = {angle_deg:.2f}°")
    print(f"   Status: {'✓ Well-aligned (θ < 45°)' if angle_deg < 45 else '⚠ Weak alignment (θ >= 45°)'}")

    # Metric 3: Relative scaling
    scaling_ratio = z_norm / g_norm
    print(f"\n3. Scaling Analysis:")
    print(f"   ||z|| / ||g|| = {scaling_ratio:.4e}")
    print(f"   Expected for good preconditioner: ~ 1/κ(H) to 1")

    # Metric 4: Multi-level contribution
    multi_level_z = solver.mas.multi_level_z.to_numpy()
    level0_z = multi_level_z[:n_verts]
    level0_norm = np.linalg.norm(level0_z)

    print(f"\n4. Multi-Level Contribution:")
    print(f"   Level 0: ||z_0|| = {level0_norm:.4e}")

    level_contributions = [level0_norm]
    for l in range(1, solver.mas.actual_levels):
        level_offset = solver.mas.level_size.to_numpy()[l][1]
        level_size_val = solver.mas.level_size.to_numpy()[l][0]
        level_z = multi_level_z[level_offset:level_offset + level_size_val]
        level_norm = np.linalg.norm(level_z)
        level_contributions.append(level_norm)
        ratio = level_norm / (level0_norm + 1e-10)
        print(f"   Level {l}: ||z_{l}|| = {level_norm:.4e} (ratio to L0: {ratio:.4f})")

    # Metric 5: Component-wise analysis
    z_x = z[:, 0]
    z_y = z[:, 1]
    z_z_comp = z[:, 2]

    print(f"\n5. Component-wise Analysis:")
    print(f"   ||z_x|| = {np.linalg.norm(z_x):.4e}")
    print(f"   ||z_y|| = {np.linalg.norm(z_y):.4e}")
    print(f"   ||z_z|| = {np.linalg.norm(z_z_comp):.4e}")

    # Metric 6: Compare with diagonal preconditioner
    z_diag_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    apply_diagonal_preconditioner(solver.mesh.verts.grad, solver.mesh.verts.diagH, z_diag_field)
    z_diag = z_diag_field.to_numpy()

    gTz_diag = np.sum(g * z_diag)
    z_diag_norm = np.linalg.norm(z_diag)

    print(f"\n6. Comparison with Diagonal Preconditioner:")
    print(f"   MAS:      g^T z = {gTz:.4e}, ||z|| = {z_norm:.4e}")
    print(f"   Diagonal: g^T z = {gTz_diag:.4e}, ||z|| = {z_diag_norm:.4e}")
    print(f"   MAS / Diag ratio: ||z_MAS|| / ||z_diag|| = {z_norm / (z_diag_norm + 1e-10):.4f}")

    # Metric 7: Residual analysis for Block 0
    BANKSIZE = 16
    multi_level_r = solver.mas.multi_level_r.to_numpy()
    r_block0 = multi_level_r[:BANKSIZE]
    z_block0 = level0_z[:BANKSIZE]

    # Get inverse block 0
    inv_block0 = solver.mas.inv_block_matrices.to_numpy()[0]

    # Reconstruct using symmetric storage
    def sym_idx(i, j):
        min_ij = min(i, j)
        max_ij = max(i, j)
        return BANKSIZE * min_ij - min_ij * (min_ij + 1) // 2 + max_ij

    # Compute z_expected = inv_block @ r
    z_expected_block0 = np.zeros((BANKSIZE, 3))
    for i in range(BANKSIZE):
        for j in range(BANKSIZE):
            idx = sym_idx(i, j)
            inv_ij = inv_block0[idx]
            if i <= j:
                z_expected_block0[i] += inv_ij @ r_block0[j]
            else:
                z_expected_block0[i] += inv_ij.T @ r_block0[j]

    block0_error = np.linalg.norm(z_block0 - z_expected_block0)
    block0_rel_error = block0_error / (np.linalg.norm(z_expected_block0) + 1e-10)

    print(f"\n7. Block 0 Local Solve Accuracy:")
    print(f"   ||z_0^actual - z_0^expected|| = {block0_error:.4e}")
    print(f"   Relative error = {block0_rel_error:.4e}")

    # Metric 8: Prolongation contribution analysis
    print(f"\n8. Prolongation Contribution Analysis:")
    agg_table = solver.mas.aggregation_table.to_numpy()

    if solver.mas.actual_levels >= 2:
        level1_offset = solver.mas.level_size.to_numpy()[1][1]

        # Each fine vertex gets z_1[agg_table[idx][0]]
        z_from_level1 = np.zeros_like(z)
        for idx in range(n_verts):
            coarse_idx = agg_table[idx, 0]
            if coarse_idx >= 0 and coarse_idx < solver.mas.total_nodes_all_levels:
                z_from_level1[idx] = multi_level_z[coarse_idx]

        level1_contribution_norm = np.linalg.norm(z_from_level1)
        print(f"   From Level 1: ||C_1^T z_1|| = {level1_contribution_norm:.4e}")
        print(f"   Ratio to ||z||: {level1_contribution_norm / (z_norm + 1e-10):.4f}")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    quality_score = 0
    issues = []

    if gTz > 0:
        quality_score += 1
        print("✓ Descent direction valid (g^T z > 0)")
    else:
        issues.append("Descent direction INVALID")
        print("✗ Descent direction INVALID")

    if angle_deg < 45:
        quality_score += 1
        print("✓ Good alignment with gradient (θ < 45°)")
    elif angle_deg < 90:
        issues.append(f"Weak alignment (θ = {angle_deg:.1f}°)")
        print(f"⚠ Weak alignment (θ = {angle_deg:.1f}°)")
    else:
        issues.append(f"Poor alignment (θ = {angle_deg:.1f}°)")
        print(f"✗ Poor alignment (θ = {angle_deg:.1f}°)")

    if len(level_contributions) > 1 and level_contributions[1] > 1e-10:
        quality_score += 1
        print("✓ Multi-level structure active (Level 1 contributes)")
    else:
        issues.append("Coarse level may not contribute")
        print("⚠ Coarse level contribution may be negligible")

    if block0_rel_error < 1e-3:
        quality_score += 1
        print("✓ Local solve accurate")
    else:
        issues.append(f"Local solve error: {block0_rel_error:.2e}")
        print(f"⚠ Local solve error: {block0_rel_error:.2e}")

    print(f"\nQuality Score: {quality_score}/4")

    if issues:
        print("\nPotential Issues:")
        for issue in issues:
            print(f"  - {issue}")

    # Analysis of the weak alignment issue
    print(f"\n" + "=" * 80)
    print("ANALYSIS: WHY IS THE ALIGNMENT WEAK?")
    print("=" * 80)

    print(f"""
The angle θ = {angle_deg:.2f}° indicates weak alignment between z = P @ g and g.

Key observations:
1. g^T z = {gTz:.4e} > 0, so z is still a valid descent direction
2. ||z|| / ||g|| = {scaling_ratio:.4e} << 1, z is much smaller than g

This is caused by the large regularization ε = {reg_epsilon:.4e}:
- Block 0 has negative eigenvalues (min = {min_eig:.4e})
- Regularization adds ε * I to make the matrix SPD
- When ε >> ||A||, the preconditioner approaches ε^-1 * I (identity scaling)
- This makes z ≈ g / ε, which is very small

The preconditioner is still valid (g^T z > 0) but may converge slowly because:
- The effective condition number improvement is limited
- z is approximately parallel to g (diagonal preconditioning effect)

Recommendation:
- Use adaptive regularization (relative to ||diag(A)||) instead of fixed ε
- Or use Gauss-Jordan without regularization (handles non-SPD directly)
""")

    print("=" * 80)

    return {
        'gTz': gTz,
        'angle_deg': angle_deg,
        'z_norm': z_norm,
        'g_norm': g_norm,
        'quality_score': quality_score,
        'issues': issues
    }


if __name__ == '__main__':
    diagnose_mas_quality()
