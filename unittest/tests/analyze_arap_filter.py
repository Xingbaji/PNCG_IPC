"""
Analyze why ARAP_filter produces negative eigenvalues in block matrices.

Usage:
    cd /root/PNCG_IPC/demo
    PYTHONPATH=/root/PNCG_IPC python ../unittest/tests/analyze_arap_filter.py
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


class AnalysisSolver(pncg_ipc_deformer):
    def __init__(self):
        super().__init__(demo='eight_E_stiffness_test')
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )
        # Field to store F matrices
        self.F_field = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_cells)

    @ti.kernel
    def extract_F_matrices(self):
        """Extract deformation gradient F for each cell."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            F = Ds @ c.B
            self.F_field[c.id] = F


def analyze_singular_values(solver):
    """Analyze singular values of F matrices to understand filter behavior."""
    solver.extract_F_matrices()
    F_all = solver.F_field.to_numpy()

    print("\n" + "=" * 70)
    print("SINGULAR VALUE ANALYSIS OF DEFORMATION GRADIENTS")
    print("=" * 70)

    # Compute SVD for each cell
    min_sums = []
    for cell_idx in range(solver.n_cells):
        F = F_all[cell_idx]
        U, s, Vh = np.linalg.svd(F)

        # Compute sums
        sum_01 = s[0] + s[1]  # for lambda2
        sum_02 = s[0] + s[2]  # for lambda1
        sum_12 = s[1] + s[2]  # for lambda0

        min_sums.append(min(sum_01, sum_02, sum_12))

    min_sums = np.array(min_sums)

    print(f"\nNumber of cells: {solver.n_cells}")
    print(f"Cells with min(s_i + s_j) < 2: {np.sum(min_sums < 2)}")
    print(f"Cells with min(s_i + s_j) >= 2: {np.sum(min_sums >= 2)}")
    print(f"Min of min sums: {np.min(min_sums):.6f}")
    print(f"Max of min sums: {np.max(min_sums):.6f}")

    # Find cells with smallest sums (most compressed)
    most_compressed_idx = np.argmin(min_sums)
    F_compressed = F_all[most_compressed_idx]
    U, s, Vh = np.linalg.svd(F_compressed)

    print(f"\nMost compressed cell (idx={most_compressed_idx}):")
    print(f"  Singular values: {s}")
    print(f"  s0 + s1 = {s[0]+s[1]:.4f}")
    print(f"  s0 + s2 = {s[0]+s[2]:.4f}")
    print(f"  s1 + s2 = {s[1]+s[2]:.4f}")

    return min_sums


def analyze_element_hessian(F, mu=1.0, la=1.0):
    """Compute ARAP_filter Hessian and check eigenvalues."""
    U, s, Vh = np.linalg.svd(F)
    V = Vh.T

    s0, s1, s2 = s

    # Compute lambdas (filter version)
    lambda0 = 2.0 / (s1 + s2) if s1 + s2 >= 2.0 else 1.0
    lambda1 = 2.0 / (s0 + s2) if s0 + s2 >= 2.0 else 1.0
    lambda2 = 2.0 / (s0 + s1) if s0 + s1 >= 2.0 else 1.0

    print(f"\n  Lambda values (filtered):")
    print(f"    lambda0 = {lambda0:.4f} (from s1+s2={s1+s2:.4f})")
    print(f"    lambda1 = {lambda1:.4f} (from s0+s2={s0+s2:.4f})")
    print(f"    lambda2 = {lambda2:.4f} (from s0+s1={s0+s1:.4f})")

    # Compute twist mode vectors
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]

    Q0 = np.outer(V1, U2) - np.outer(V2, U1)
    Q1 = np.outer(V2, U0) - np.outer(V0, U2)
    Q2 = np.outer(V1, U0) - np.outer(V0, U1)

    q0 = Q0.flatten()
    q1 = Q1.flatten()
    q2 = Q2.flatten()

    # Hessian = 2μI - μ * Σ λᵢ (qᵢ ⊗ qᵢ)
    H = 2.0 * mu * np.eye(9)
    H -= mu * lambda0 * np.outer(q0, q0)
    H -= mu * lambda1 * np.outer(q1, q1)
    H -= mu * lambda2 * np.outer(q2, q2)

    # Make symmetric
    H = (H + H.T) / 2

    # Eigenvalue analysis
    eigvals = np.linalg.eigvalsh(H)

    print(f"\n  Element Hessian eigenvalues:")
    print(f"    Min: {np.min(eigvals):.6e}")
    print(f"    Max: {np.max(eigvals):.6e}")
    print(f"    Negative count: {np.sum(eigvals < 0)}")
    print(f"    Near-zero count (|λ| < 1e-6): {np.sum(np.abs(eigvals) < 1e-6)}")

    return H, eigvals


def main():
    print("=" * 70)
    print("ARAP_filter Negative Eigenvalue Analysis")
    print("=" * 70)

    solver = AnalysisSolver()
    solver.assign_xn_xhat()

    # Analyze singular values
    min_sums = analyze_singular_values(solver)

    # Check the most compressed cell's element Hessian
    solver.extract_F_matrices()
    F_all = solver.F_field.to_numpy()

    most_compressed_idx = np.argmin(min_sums)
    F_compressed = F_all[most_compressed_idx]

    print("\n" + "=" * 70)
    print("ELEMENT HESSIAN ANALYSIS FOR MOST COMPRESSED CELL")
    print("=" * 70)

    H, eigvals = analyze_element_hessian(F_compressed, solver.mu, solver.la)

    # Check a normal cell (near identity)
    # Find cell with sum closest to 2 (identity-like)
    near_identity_idx = np.argmin(np.abs(min_sums - 2.0))
    F_identity = F_all[near_identity_idx]

    print("\n" + "=" * 70)
    print("ELEMENT HESSIAN ANALYSIS FOR NEAR-IDENTITY CELL")
    print("=" * 70)
    print(f"Cell index: {near_identity_idx}")

    H2, eigvals2 = analyze_element_hessian(F_identity, solver.mu, solver.la)

    # Now check the block matrix composition
    print("\n" + "=" * 70)
    print("BLOCK MATRIX COMPOSITION ANALYSIS")
    print("=" * 70)

    solver.mas.build_hierarchy()

    # Only inertia
    solver.mas._clear_block_matrices()
    solver.mas._add_inertia_contribution(solver.dt)
    solver.mas._expand_sym_to_full()
    block0_inertia = solver.mas.full_block_matrix.to_numpy()[0].copy()

    print(f"\n1. Inertia matrix (Block 0):")
    eigvals_inertia = np.linalg.eigvalsh((block0_inertia + block0_inertia.T) / 2)
    print(f"   Min eigenvalue: {np.min(eigvals_inertia):.4e}")
    print(f"   Max eigenvalue: {np.max(eigvals_inertia):.4e}")

    # Inertia + Elastic
    solver.mas._clear_block_matrices()
    solver.mas._add_inertia_contribution(solver.dt)
    solver.mas._add_elastic_contribution_full(solver.mu, solver.la, solver.dt, 0)
    solver.mas._expand_sym_to_full()
    block0_full = solver.mas.full_block_matrix.to_numpy()[0].copy()

    print(f"\n2. Inertia + Elastic matrix (Block 0):")
    eigvals_full = np.linalg.eigvalsh((block0_full + block0_full.T) / 2)
    print(f"   Min eigenvalue: {np.min(eigvals_full):.4e}")
    print(f"   Max eigenvalue: {np.max(eigvals_full):.4e}")
    print(f"   Negative count: {np.sum(eigvals_full < 0)}")

    # Elastic alone
    elastic_contribution = block0_full - block0_inertia
    print(f"\n3. Elastic contribution alone (Block 0):")
    eigvals_elastic = np.linalg.eigvalsh((elastic_contribution + elastic_contribution.T) / 2)
    print(f"   Min eigenvalue: {np.min(eigvals_elastic):.4e}")
    print(f"   Max eigenvalue: {np.max(eigvals_elastic):.4e}")
    print(f"   Negative count: {np.sum(eigvals_elastic < 0)}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    if np.sum(eigvals_full < 0) > 0:
        print("""
The negative eigenvalues in the block matrix are likely caused by:

1. **Aggregation of semi-definite matrices**: Even though each element
   Hessian from ARAP_filter is PSD (≥0), when multiple elements' Hessians
   are summed into a block, numerical errors can accumulate.

2. **Near-zero eigenvalues becoming negative**: ARAP_filter produces
   eigenvalues = 0 at identity (not > 0). Small numerical errors can
   push these to slightly negative values.

3. **Non-symmetric assembly**: If the symmetric storage or expansion
   has numerical errors, this can introduce asymmetry.

The negative eigenvalues (~-4e5) are much larger than numerical precision,
suggesting a potential bug in the assembly process or the filter itself.
""")

    # Check for assembly asymmetry
    print("\n" + "=" * 70)
    print("SYMMETRY CHECK")
    print("=" * 70)

    asym_error = np.linalg.norm(block0_full - block0_full.T)
    print(f"Asymmetry error ||A - A^T||: {asym_error:.4e}")

    # Check diagonal
    diag = np.diag(block0_full)
    print(f"Diagonal values: min={np.min(diag):.4e}, max={np.max(diag):.4e}")
    print(f"Negative diagonal entries: {np.sum(diag < 0)}")


if __name__ == '__main__':
    main()
