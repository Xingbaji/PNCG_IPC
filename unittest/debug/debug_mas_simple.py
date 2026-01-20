"""
Simplified debug script to compare MAS vs Diagonal preconditioner.
Focus on key metrics without heavy computation.
"""

import taichi as ti
import numpy as np
import sys
sys.path.insert(0, '/root/PNCG_IPC')

ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.mas_pncg_solver_nocolli import MASPNCGSolverNoCollision
from algorithm.mas_preconditioner_small import BANKSIZE


def main():
    print("\n" + "="*70)
    print("DEBUG: MAS vs Diagonal Preconditioner")
    print("="*70)

    # Create MAS solver
    solver = MASPNCGSolverNoCollision(demo='eight_E_freefall', use_mas=True)
    n_verts = solver.n_verts
    precond = solver.mas_preconditioner

    print(f"\nMesh: {n_verts} vertices, {solver.n_cells} cells")
    print(f"BANKSIZE: {BANKSIZE}")
    print(f"Number of blocks: {precond.n_parts}")
    print(f"Number of levels: {precond.level_num}")

    # Initialize simulation
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()

    # Get gradient and diagH
    grad = solver.mesh.verts.grad.to_numpy()
    diagH = solver.mesh.verts.diagH.to_numpy()

    print(f"\n--- Gradient ---")
    print(f"||grad||: {np.linalg.norm(grad):.6e}")
    print(f"grad first 3 verts: {grad[:3]}")

    print(f"\n--- DiagH ---")
    print(f"min: {diagH.min():.6e}, max: {diagH.max():.6e}, mean: {diagH.mean():.6e}")
    print(f"diagH first 3 verts: {diagH[:3]}")

    # Build MAS preconditioner
    print("\n--- Building MAS Preconditioner ---")
    precond.rebuild(solver)

    # Extract MAS diagonal and compare
    block_matrices_np = precond.block_matrices.to_numpy()
    print("\n--- MAS Block Matrix Diagonal vs diagH ---")

    mas_diag = []
    for i in range(min(n_verts, 48)):  # First 3 blocks
        block_id = i // BANKSIZE
        lane_id = i % BANKSIZE
        s_idx = BANKSIZE * lane_id - lane_id * (lane_id + 1) // 2 + lane_id
        block_3x3 = block_matrices_np[block_id, s_idx]
        mas_diag.append([block_3x3[0, 0], block_3x3[1, 1], block_3x3[2, 2]])

    mas_diag = np.array(mas_diag)
    diagH_part = diagH[:len(mas_diag)]

    print(f"MAS diagonal first 5 verts:\n{mas_diag[:5]}")
    print(f"diagH first 5 verts:\n{diagH_part[:5]}")

    diff = np.abs(mas_diag - diagH_part)
    rel_diff = diff / (diagH_part + 1e-12)
    print(f"\nMax abs diff: {diff.max():.6e}")
    print(f"Max rel diff: {rel_diff.max():.6e}")
    print(f"Mean rel diff: {rel_diff.mean():.6e}")

    # Apply diagonal preconditioner
    print("\n--- Applying Diagonal Preconditioner ---")
    solver.apply_diagonal_preconditioner()
    z_diag = solver.mesh.verts.z.to_numpy().copy()

    print(f"||z_diag||: {np.linalg.norm(z_diag):.6e}")
    print(f"z_diag first 3 verts:\n{z_diag[:3]}")

    # Apply MAS preconditioner
    print("\n--- Applying MAS Preconditioner ---")
    precond.apply()
    z_mas = solver.mesh.verts.z.to_numpy()

    print(f"||z_mas||: {np.linalg.norm(z_mas):.6e}")
    print(f"z_mas first 3 verts:\n{z_mas[:3]}")

    # Compare
    ratio = np.linalg.norm(z_mas) / np.linalg.norm(z_diag)
    print(f"\n||z_mas|| / ||z_diag|| = {ratio:.2f}x")

    # Check dot products
    grad_flat = grad.flatten()
    zTg_diag = np.dot(z_diag.flatten(), grad_flat)
    zTg_mas = np.dot(z_mas.flatten(), grad_flat)

    print(f"\nz^T g (diag): {zTg_diag:.6e}")
    print(f"z^T g (mas):  {zTg_mas:.6e}")

    # Check z^T H z using MAS hessian_matvec
    print("\n--- z^T H z ---")

    # Copy z_diag to buffer
    v_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    result_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    # For z_diag
    v_field.from_numpy(z_diag)
    precond.hessian_matvec(v_field, result_field)
    Hv = result_field.to_numpy()
    zTHz_diag = np.dot(z_diag.flatten(), Hv.flatten())
    print(f"z_diag^T H z_diag: {zTHz_diag:.6e}")

    # For z_mas
    v_field.from_numpy(z_mas)
    precond.hessian_matvec(v_field, result_field)
    Hv = result_field.to_numpy()
    zTHz_mas = np.dot(z_mas.flatten(), Hv.flatten())
    print(f"z_mas^T H z_mas: {zTHz_mas:.6e}")

    # Check the multi-level r buffer
    print("\n--- Multi-level r buffer ---")
    multi_r = precond.multi_level_r.to_numpy()
    multi_z = precond.multi_level_z.to_numpy()

    # Level 0
    level0_size = precond.level_size[0][0]
    level0_r = multi_r[:level0_size]
    level0_z = multi_z[:level0_size]
    print(f"Level 0: {level0_size} nodes")
    print(f"  ||r||: {np.linalg.norm(level0_r):.6e}")
    print(f"  ||z||: {np.linalg.norm(level0_z):.6e}")

    for level in range(1, precond.level_num):
        level_size = precond.level_size[level][0]
        level_offset = precond.level_size[level][1]
        level_r = multi_r[level_offset:level_offset+level_size]
        level_z = multi_z[level_offset:level_offset+level_size]
        print(f"Level {level}: {level_size} nodes")
        print(f"  ||r||: {np.linalg.norm(level_r):.6e}")
        print(f"  ||z||: {np.linalg.norm(level_z):.6e}")

    # Check inverse matrices
    print("\n--- Checking Inverse Quality (Block 0) ---")
    inv_block_matrices = precond.inv_block_matrices.to_numpy()

    # Build full block 0 matrix and inverse
    block_full = np.zeros((48, 48))
    inv_full = np.zeros((48, 48))

    for row in range(BANKSIZE):
        for col in range(row, BANKSIZE):
            s_idx = BANKSIZE * row - row * (row + 1) // 2 + col
            block_3x3 = block_matrices_np[0, s_idx]
            inv_3x3 = inv_block_matrices[0, s_idx]

            for di in range(3):
                for dj in range(3):
                    block_full[row*3+di, col*3+dj] = block_3x3[di, dj]
                    inv_full[row*3+di, col*3+dj] = inv_3x3[di, dj]
                    if row != col:
                        block_full[col*3+di, row*3+dj] = block_3x3[dj, di]
                        inv_full[col*3+di, row*3+dj] = inv_3x3[dj, di]

    # Check eigenvalues of block matrix
    eig_block = np.linalg.eigvalsh(block_full)
    print(f"Block 0 eigenvalues: min={eig_block.min():.6e}, max={eig_block.max():.6e}")
    print(f"Block 0 condition number: {eig_block.max() / max(eig_block.min(), 1e-12):.6e}")

    # Check inverse quality
    product = block_full @ inv_full
    identity = np.eye(48)
    inv_error = np.linalg.norm(product - identity, 'fro') / np.linalg.norm(identity, 'fro')
    print(f"||H_block @ P_inv - I||_F / ||I||_F = {inv_error:.6e}")

    # Check if inv_full is good approximation
    # Compare z from direct solve vs inv_full
    r0 = level0_r[:BANKSIZE]  # First block residual
    z0_direct = np.linalg.solve(block_full, r0.flatten())
    z0_inv = inv_full @ r0.flatten()
    z0_diff = np.linalg.norm(z0_direct - z0_inv) / np.linalg.norm(z0_direct)
    print(f"||z_direct - z_inv|| / ||z_direct|| = {z0_diff:.6e}")

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    main()
