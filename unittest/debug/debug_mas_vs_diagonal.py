"""
Debug script to compare MAS preconditioner vs Diagonal preconditioner.

This script compares:
1. Block matrix assembly - diagonal entries should match diagH
2. Inversion quality - check if P^-1 * H ≈ I for well-conditioned blocks
3. z vector norms and directions
"""

import taichi as ti
import numpy as np
import sys
sys.path.insert(0, '/root/PNCG_IPC')

ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision
from algorithm.mas_preconditioner_small import BANKSIZE


def debug_mas_preconditioner():
    """Debug MAS preconditioner step by step."""

    print("\n" + "="*70)
    print("DEBUG: MAS Preconditioner Analysis")
    print("="*70)

    # Create solvers
    solver_diag = MASPNCGSolverNoCollision(demo='eight_E_freefall', use_mas=False)
    solver_mas = MASPNCGSolverNoCollision(demo='eight_E_freefall', use_mas=True)

    n_verts = solver_mas.n_verts
    print(f"\nMesh: {n_verts} vertices, {solver_mas.n_cells} cells")
    print(f"BANKSIZE: {BANKSIZE}")
    print(f"Number of blocks: {solver_mas.mas_preconditioner.n_parts}")
    print(f"Number of levels: {solver_mas.mas_preconditioner.level_num}")

    # Step 1: Compute gradient and diagH for both
    solver_diag.assign_xn_xhat()
    solver_mas.assign_xn_xhat()

    solver_diag.compute_grad_and_diagH()
    solver_mas.compute_grad_and_diagH()

    # Get gradient and diagH from both
    grad_diag = solver_diag.mesh.verts.grad.to_numpy()
    diagH_diag = solver_diag.mesh.verts.diagH.to_numpy()
    grad_mas = solver_mas.mesh.verts.grad.to_numpy()
    diagH_mas = solver_mas.mesh.verts.diagH.to_numpy()

    print(f"\n--- Gradient Statistics ---")
    print(f"Gradient norm (diag): {np.linalg.norm(grad_diag):.6e}")
    print(f"Gradient norm (mas):  {np.linalg.norm(grad_mas):.6e}")
    print(f"Gradient difference:  {np.linalg.norm(grad_diag - grad_mas):.6e}")

    print(f"\n--- DiagH Statistics ---")
    print(f"DiagH min (diag): {diagH_diag.min():.6e}")
    print(f"DiagH max (diag): {diagH_diag.max():.6e}")
    print(f"DiagH mean (diag): {diagH_diag.mean():.6e}")

    # Step 2: Build MAS hierarchy and assemble matrices
    print("\n--- Building MAS Hierarchy ---")
    solver_mas.mas_preconditioner.rebuild(solver_mas)

    # Step 3: Compare diagonal entries of MAS block matrices with diagH
    print("\n--- Comparing MAS Block Diagonals with diagH ---")

    # Extract diagonal from MAS block matrices
    mas_diag = extract_mas_diagonal(solver_mas.mas_preconditioner, n_verts)

    # Compare with diagH
    diag_diff = np.abs(mas_diag - diagH_mas.flatten())
    rel_diff = diag_diff / (np.abs(diagH_mas.flatten()) + 1e-12)

    print(f"MAS diagonal entries (first 10): {mas_diag[:10]}")
    print(f"diagH entries (first 10): {diagH_mas.flatten()[:10]}")
    print(f"Max absolute difference: {diag_diff.max():.6e}")
    print(f"Max relative difference: {rel_diff.max():.6e}")
    print(f"Mean relative difference: {rel_diff.mean():.6e}")

    # Step 4: Apply preconditioners and compare z
    print("\n--- Applying Preconditioners ---")

    # Diagonal preconditioner
    solver_diag.apply_diagonal_preconditioner()
    z_diag = solver_diag.mesh.verts.z.to_numpy()

    # MAS preconditioner
    solver_mas.mas_preconditioner.apply()
    z_mas = solver_mas.mesh.verts.z.to_numpy()

    print(f"\n--- z Vector Statistics ---")
    print(f"z norm (diag): {np.linalg.norm(z_diag):.6e}")
    print(f"z norm (mas):  {np.linalg.norm(z_mas):.6e}")
    print(f"z ratio (mas/diag): {np.linalg.norm(z_mas) / np.linalg.norm(z_diag):.2f}x")

    # Check z direction alignment
    z_diag_flat = z_diag.flatten()
    z_mas_flat = z_mas.flatten()
    cos_angle = np.dot(z_diag_flat, z_mas_flat) / (np.linalg.norm(z_diag_flat) * np.linalg.norm(z_mas_flat) + 1e-12)
    print(f"z direction cosine (diag vs mas): {cos_angle:.6f}")

    # Check z^T g for both
    grad_flat = grad_mas.flatten()
    zTg_diag = np.dot(z_diag_flat, grad_flat)
    zTg_mas = np.dot(z_mas_flat, grad_flat)
    print(f"\n--- z^T g (should be > 0) ---")
    print(f"z^T g (diag): {zTg_diag:.6e}")
    print(f"z^T g (mas):  {zTg_mas:.6e}")

    # Step 5: Check z^T H z using both methods
    print("\n--- z^T H z Comparison ---")

    # For diagonal: z^T * diagH * z (approximation)
    zTHz_diag_approx = np.sum(z_diag * diagH_diag * z_diag)
    print(f"z^T diagH z (diag approx): {zTHz_diag_approx:.6e}")

    # For MAS: use exact hessian_matvec
    solver_mas._copy_z_to_buffer()
    solver_mas.mas_preconditioner.hessian_matvec(solver_mas.hv_input, solver_mas.hv_output)
    Hv_mas = solver_mas.hv_output.to_numpy()
    zTHz_mas_exact = np.dot(z_mas_flat, Hv_mas.flatten())
    print(f"z^T H z (mas exact): {zTHz_mas_exact:.6e}")

    # Step 6: Check MAS inverse quality
    print("\n--- MAS Inverse Quality Check ---")
    check_inverse_quality(solver_mas.mas_preconditioner, solver_mas, n_verts)

    # Step 7: Check per-block analysis
    print("\n--- Per-Block Analysis ---")
    analyze_blocks(solver_mas.mas_preconditioner, diagH_mas, n_verts)

    return solver_diag, solver_mas


def extract_mas_diagonal(precond, n_verts):
    """Extract diagonal entries from MAS block matrices."""
    block_matrices_np = precond.block_matrices.to_numpy()

    mas_diag = np.zeros(n_verts * 3)

    for i in range(n_verts):
        block_id = i // BANKSIZE
        lane_id = i % BANKSIZE

        # Symmetric storage index for diagonal (row=lane_id, col=lane_id)
        s_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id

        block_3x3 = block_matrices_np[block_id, s_idx]
        for d in range(3):
            mas_diag[i * 3 + d] = block_3x3[d, d]

    return mas_diag


def check_inverse_quality(precond, solver, n_verts):
    """Check if P^-1 * H * v ≈ v for random v."""

    # Create random test vector
    np.random.seed(42)
    v_test = np.random.randn(n_verts, 3).astype(np.float32)

    # Copy to Taichi field
    v_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    result_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    v_field.from_numpy(v_test)

    # Compute H * v using exact matvec
    precond.hessian_matvec(v_field, result_field)
    Hv = result_field.to_numpy()

    # Copy Hv to mesh.verts.grad to apply preconditioner
    solver.mesh.verts.grad.from_numpy(Hv)

    # Apply MAS preconditioner: z = P^-1 * (H * v)
    precond.apply()
    z = solver.mesh.verts.z.to_numpy()

    # Ideally P^-1 * H * v ≈ v
    # So z should be close to v_test

    # Compute relative error
    diff_norm = np.linalg.norm(z - v_test)
    v_norm = np.linalg.norm(v_test)
    rel_error = diff_norm / v_norm

    print(f"||P^-1 * H * v - v|| / ||v|| = {rel_error:.6e}")
    print(f"||v|| = {v_norm:.6e}")
    print(f"||P^-1 * H * v|| = {np.linalg.norm(z):.6e}")

    # Also check if P^-1 ≈ H^-1 by checking H * P^-1 * g ≈ g
    # First, restore grad
    grad_orig = solver.mesh.verts.grad.to_numpy().copy()

    # Apply preconditioner: z = P^-1 * grad
    solver.mesh.verts.grad.from_numpy(grad_orig)
    precond.apply()
    z_from_grad = solver.mesh.verts.z.to_numpy()

    # Compute H * z
    z_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    Hz_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    z_field.from_numpy(z_from_grad)
    precond.hessian_matvec(z_field, Hz_field)
    Hz = Hz_field.to_numpy()

    # Check H * P^-1 * g ≈ g
    diff_norm2 = np.linalg.norm(Hz - grad_orig)
    grad_norm = np.linalg.norm(grad_orig)
    rel_error2 = diff_norm2 / grad_norm

    print(f"\n||H * P^-1 * g - g|| / ||g|| = {rel_error2:.6e}")
    print(f"||g|| = {grad_norm:.6e}")
    print(f"||H * P^-1 * g|| = {np.linalg.norm(Hz):.6e}")


def analyze_blocks(precond, diagH, n_verts):
    """Analyze individual blocks."""
    block_matrices_np = precond.block_matrices.to_numpy()
    inv_block_matrices_np = precond.inv_block_matrices.to_numpy()

    n_blocks = (n_verts + BANKSIZE - 1) // BANKSIZE

    print(f"\nAnalyzing {n_blocks} blocks (first 3 blocks):")

    for block_id in range(min(3, n_blocks)):
        print(f"\n--- Block {block_id} ---")

        # Get block size (may be < BANKSIZE for last block)
        block_start = block_id * BANKSIZE
        block_end = min((block_id + 1) * BANKSIZE, n_verts)
        block_size = block_end - block_start

        # Extract full 48x48 matrix from symmetric storage
        full_block = np.zeros((BANKSIZE * 3, BANKSIZE * 3))
        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                s_idx = BANKSIZE * row - row * (row + 1) // 2 + col
                block_3x3 = block_matrices_np[block_id, s_idx]

                for di in range(3):
                    for dj in range(3):
                        full_block[row * 3 + di, col * 3 + dj] = block_3x3[di, dj]
                        if row != col:
                            full_block[col * 3 + di, row * 3 + dj] = block_3x3[dj, di]

        # Truncate to actual block size
        actual_size = block_size * 3
        full_block = full_block[:actual_size, :actual_size]

        # Get diagH for this block
        diagH_block = diagH[block_start:block_end].flatten()

        # Print diagonal comparison
        mas_diag_block = np.diag(full_block)
        print(f"  Block size: {block_size} vertices, {actual_size} DOFs")
        print(f"  MAS diagonal (first 6): {mas_diag_block[:6]}")
        print(f"  diagH (first 6): {diagH_block[:6]}")

        # Check matrix properties
        eigenvalues = np.linalg.eigvalsh(full_block)
        print(f"  Min eigenvalue: {eigenvalues.min():.6e}")
        print(f"  Max eigenvalue: {eigenvalues.max():.6e}")
        print(f"  Condition number: {eigenvalues.max() / max(eigenvalues.min(), 1e-12):.6e}")

        # Check inverse quality
        full_inv = np.zeros((BANKSIZE * 3, BANKSIZE * 3))
        for row in range(BANKSIZE):
            for col in range(row, BANKSIZE):
                s_idx = BANKSIZE * row - row * (row + 1) // 2 + col
                inv_3x3 = inv_block_matrices_np[block_id, s_idx]

                for di in range(3):
                    for dj in range(3):
                        full_inv[row * 3 + di, col * 3 + dj] = inv_3x3[di, dj]
                        if row != col:
                            full_inv[col * 3 + di, row * 3 + dj] = inv_3x3[dj, di]

        full_inv = full_inv[:actual_size, :actual_size]

        # Check H_block * H_block^-1 ≈ I
        product = full_block @ full_inv
        identity = np.eye(actual_size)
        inv_error = np.linalg.norm(product - identity, 'fro') / np.linalg.norm(identity, 'fro')
        print(f"  ||H * P^-1 - I||_F / ||I||_F = {inv_error:.6e}")


if __name__ == '__main__':
    debug_mas_preconditioner()
