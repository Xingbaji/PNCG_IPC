"""
Debug block matrix assembly to find source of negative eigenvalues.
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
from algorithm.mas_preconditioner_pkg.constants import BANKSIZE, BLOCK_DOF


class DebugSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


def analyze_block_diagonal(block):
    """Analyze diagonal entries of a block matrix."""
    BLOCK_DOF = 16 * 3  # 48
    diag = np.array([block[i, i] for i in range(BLOCK_DOF)])
    return {
        'min': np.min(diag),
        'max': np.max(diag),
        'negative_count': np.sum(diag < 0),
        'negative_sum': np.sum(diag[diag < 0]),
        'positive_sum': np.sum(diag[diag > 0])
    }


def main():
    print("=" * 70)
    print("Block Matrix Assembly Debug")
    print("=" * 70)

    solver = DebugSolver()
    solver.mas.build_hierarchy()
    solver.assign_xn_xhat()

    print(f"\nMesh: {solver.n_verts} vertices, {solver.n_cells} cells")
    print(f"Number of blocks (Level 0): {solver.n_verts // BANKSIZE}")

    # Step 1: Inertia only
    print("\n" + "-" * 70)
    print("Step 1: Inertia contribution only")
    print("-" * 70)

    solver.mas._clear_block_matrices()
    solver.mas._add_inertia_contribution(solver.dt)
    solver.mas._expand_sym_to_full()

    block0_inertia = solver.mas.full_block_matrix.to_numpy()[0]
    block0_inertia_sym = (block0_inertia + block0_inertia.T) / 2
    eigvals_inertia = np.linalg.eigvalsh(block0_inertia_sym)

    print(f"Block 0 (Inertia only):")
    print(f"  Min eigenvalue: {np.min(eigvals_inertia):.4e}")
    print(f"  Max eigenvalue: {np.max(eigvals_inertia):.4e}")
    print(f"  Negative count: {np.sum(eigvals_inertia < 0)}")

    diag_inertia = analyze_block_diagonal(block0_inertia)
    print(f"  Diagonal: min={diag_inertia['min']:.4e}, max={diag_inertia['max']:.4e}")

    # Step 2: Add elastic (one cell at a time conceptually)
    print("\n" + "-" * 70)
    print("Step 2: Adding elastic contribution (ARAP_filter)")
    print("-" * 70)

    solver.mas._clear_block_matrices()
    solver.mas._add_inertia_contribution(solver.dt)
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    solver.mas._expand_sym_to_full()

    block0_full = solver.mas.full_block_matrix.to_numpy()[0]
    block0_full_sym = (block0_full + block0_full.T) / 2
    eigvals_full = np.linalg.eigvalsh(block0_full_sym)

    print(f"Block 0 (Inertia + Elastic):")
    print(f"  Min eigenvalue: {np.min(eigvals_full):.4e}")
    print(f"  Max eigenvalue: {np.max(eigvals_full):.4e}")
    print(f"  Negative count: {np.sum(eigvals_full < 0)}")

    diag_full = analyze_block_diagonal(block0_full)
    print(f"  Diagonal: min={diag_full['min']:.4e}, max={diag_full['max']:.4e}")
    print(f"  Negative diagonal entries: {diag_full['negative_count']}")

    # Elastic contribution alone
    elastic_contrib = block0_full - block0_inertia
    elastic_sym = (elastic_contrib + elastic_contrib.T) / 2
    eigvals_elastic = np.linalg.eigvalsh(elastic_sym)

    print(f"\nElastic contribution alone:")
    print(f"  Min eigenvalue: {np.min(eigvals_elastic):.4e}")
    print(f"  Max eigenvalue: {np.max(eigvals_elastic):.4e}")
    print(f"  Negative count: {np.sum(eigvals_elastic < 0)}")

    diag_elastic = analyze_block_diagonal(elastic_contrib)
    print(f"  Diagonal: min={diag_elastic['min']:.4e}, max={diag_elastic['max']:.4e}")
    print(f"  Negative diagonal entries: {diag_elastic['negative_count']}")

    # Key insight: Check which vertices have negative diagonal
    print("\n" + "-" * 70)
    print("Analysis of negative diagonal entries")
    print("-" * 70)

    for i in range(BANKSIZE):
        for d in range(3):
            idx = i * 3 + d
            if elastic_contrib[idx, idx] < 0:
                print(f"  Vertex {i}, dim {d}: elastic diag = {elastic_contrib[idx, idx]:.4e}")
                # How many cells contribute to this vertex in Block 0?

    # Check the sum of diagonal elements
    print(f"\nSum of diagonal elements:")
    print(f"  Elastic: {np.trace(elastic_contrib):.4e}")
    print(f"  Inertia: {np.trace(block0_inertia):.4e}")
    print(f"  Full:    {np.trace(block0_full):.4e}")

    # Check symmetry
    print(f"\nSymmetry check (||A - A^T||):")
    print(f"  Block 0 full: {np.linalg.norm(block0_full - block0_full.T):.4e}")

    # Analyze eigenvalue composition
    print("\n" + "-" * 70)
    print("Eigenvalue analysis")
    print("-" * 70)

    print(f"\nSmallest 10 eigenvalues of full block:")
    sorted_eigvals = np.sort(eigvals_full)
    for i in range(min(10, len(sorted_eigvals))):
        print(f"  λ_{i} = {sorted_eigvals[i]:.4e}")

    # Check if problem is in the sparse structure
    print("\n" + "-" * 70)
    print("Sparse structure analysis")
    print("-" * 70)

    # Count non-zeros in symmetric storage
    sym_block = solver.mas.block_matrices.to_numpy()[0]  # (136, 3, 3)
    nonzero_count = np.sum(np.abs(sym_block) > 1e-10)
    total_entries = sym_block.size
    print(f"  Non-zero entries in sym storage: {nonzero_count}/{total_entries}")

    # Check if there's any pattern to the negative eigenvalues
    eigvecs = np.linalg.eigh(block0_full_sym)[1]
    neg_eigvec = eigvecs[:, 0]  # Eigenvector for smallest eigenvalue

    print(f"\nEigenvector for smallest eigenvalue:")
    print(f"  ||v|| = {np.linalg.norm(neg_eigvec):.4f}")
    print(f"  Max component: {np.max(np.abs(neg_eigvec)):.4f}")

    # Which vertices does this eigenvector primarily affect?
    vert_norms = []
    for i in range(BANKSIZE):
        v_i = neg_eigvec[i*3:(i+1)*3]
        vert_norms.append(np.linalg.norm(v_i))
    vert_norms = np.array(vert_norms)

    print(f"  Vertex contributions to smallest eigenvector:")
    top_verts = np.argsort(vert_norms)[::-1][:5]
    for v in top_verts:
        print(f"    Vertex {v}: ||v_i|| = {vert_norms[v]:.4f}")


if __name__ == '__main__':
    main()
