"""
Test if element Hessian H_e is actually symmetric in the real computation.
"""

import sys
import os
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner, BANKSIZE

ti.init(arch=ti.gpu, default_fp=ti.f32)


# Store H_e for inspection
H_e_storage = ti.Matrix.field(12, 12, dtype=ti.f32, shape=(10,))  # Store first 10 elements
H_e_count = ti.field(dtype=ti.i32, shape=())


@ti.data_oriented
class DiagnosticSolver(pncg_ipc_deformer):
    def __init__(self, demo='eight_E_stiffness_test'):
        super().__init__(demo=demo)
        self.mesh.verts.place({'z': ti.types.vector(3, float)})
        self.mas = MASPreconditioner(
            self.n_verts, self.n_cells, self.mesh,
            use_metis=False
        )


@ti.func
def compute_dFdx(B: ti.template()) -> ti.template():
    """Compute derivative of deformation gradient w.r.t. vertex positions."""
    dFdx = ti.Matrix.zero(ti.f32, 9, 12)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            # Vertex 0 contribution (negative sum of others)
            dFdx[i * 3 + j, i] = -(B[j, 0] + B[j, 1] + B[j, 2])
            # Vertices 1, 2, 3 contributions
            for k in ti.static(range(3)):
                dFdx[i * 3 + j, (k + 1) * 3 + i] = B[j, k]
    return dFdx


@ti.kernel
def compute_and_store_He(mesh: ti.template(), mu: ti.f32, dt: ti.f32):
    """Compute H_e for first few elements and store for analysis."""
    H_e_count[None] = 0

    for c in mesh.cells:
        if H_e_count[None] >= 10:
            continue

        idx = ti.atomic_add(H_e_count[None], 1)
        if idx >= 10:
            continue

        W = c.W
        para = W * dt * dt

        # Compute deformation gradient
        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
        F = Ds @ c.B

        # Compute dFdx
        dFdx = compute_dFdx(c.B)

        # For ARAP: d2PsidF2 = 2 * mu * I_9x9
        d2PsidF2 = ti.Matrix.identity(ti.f32, 9) * (2.0 * mu)

        # H_e = dFdx^T @ d2PsidF2 @ dFdx
        temp = d2PsidF2 @ dFdx
        H_e = dFdx.transpose() @ temp
        H_e = para * H_e

        H_e_storage[idx] = H_e


def main():
    print("="*70)
    print("H_e SYMMETRY TEST")
    print("="*70)

    solver = DiagnosticSolver(demo='eight_E_stiffness_test')
    solver.assign_xn_xhat()
    solver.compute_grad_and_diagH()

    # Compute H_e for first few elements
    compute_and_store_He(solver.mesh, solver.mu, solver.dt)

    n_stored = H_e_count[None]
    print(f"\nStored {n_stored} element Hessians")

    He_np = H_e_storage.to_numpy()

    print("\n[Test] Check H_e symmetry for each element:")
    print("-" * 60)

    for idx in range(n_stored):
        H_e = He_np[idx]
        sym_err = np.linalg.norm(H_e - H_e.T)
        rel_sym_err = sym_err / (np.linalg.norm(H_e) + 1e-12)

        # Check eigenvalues
        eigvals = np.linalg.eigvalsh(H_e)
        min_eig = eigvals.min()

        print(f"  Element {idx}:")
        print(f"    ||H_e - H_e^T|| = {sym_err:.6e} (relative: {rel_sym_err:.6e})")
        print(f"    min eigenvalue = {min_eig:.6e}")

        if sym_err > 1e-3:
            print(f"    WARNING: Large symmetry error!")
            # Show some entries
            print(f"    H_e[0,3] = {H_e[0,3]:.6f}, H_e[3,0] = {H_e[3,0]:.6f}")
            print(f"    H_e[0,6] = {H_e[0,6]:.6f}, H_e[6,0] = {H_e[6,0]:.6f}")

    # Check 3x3 diagonal sub-blocks
    print("\n[Test] Check 3x3 diagonal sub-block symmetry:")
    print("-" * 60)

    for idx in range(min(3, n_stored)):
        H_e = He_np[idx]
        print(f"  Element {idx}:")
        for i in range(4):
            sub_block = H_e[i*3:(i+1)*3, i*3:(i+1)*3]
            sub_sym_err = np.linalg.norm(sub_block - sub_block.T)
            print(f"    H_e[{i},{i}] symmetry error: {sub_sym_err:.6e}")


if __name__ == '__main__':
    main()
