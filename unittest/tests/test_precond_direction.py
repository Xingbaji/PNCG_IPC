"""
MAS Preconditioner Direction Accuracy Test at iter=0

This test validates the accuracy of MAS preconditioner's P @ g direction at iter=0.
Compares MAS with diagonal preconditioner using several metrics:

1. Descent direction check: g^T z > 0 (must be positive for descent)
2. Angle with gradient: cos(g, z)
3. Descent quality: g^T z / (|g| |z|)
4. Condition number proxy: |z| / |g|

Ground truth reference: Exact inverse via NumPy solve (H^{-1} g)

Based on test_mas_freefall.py demo setup.

Usage:
    python test_precond_direction.py                  # Quick test
    python test_precond_direction.py --demo cube_freefall_20  # Larger mesh
    python test_precond_direction.py --verbose        # Detailed output
"""

import sys
import os
import time
import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

# Setup paths
current_file_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file_path)
unittest_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(unittest_dir)
demo_dir = os.path.join(project_root, 'demo')

sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

# Suppress MAS verbose output
import builtins
_original_print = builtins.print
_mas_verbose = False

def _filtered_print(*args, **kwargs):
    """Print filter that suppresses MAS/METIS messages unless verbose mode."""
    if args:
        msg = str(args[0])
        if msg.startswith("[MAS]") or msg.startswith("[METIS]"):
            if not _mas_verbose:
                return
    _original_print(*args, **kwargs)

builtins.print = _filtered_print

import taichi as ti
from math_utils.elastic_util import *
from util.model_loading import model_loading
# Use the simplified MAS implementation for testing
from algorithm.mas_preconditioner_small import MASPreconditionerSmall


@ti.data_oriented
class PrecondDirectionTester:
    """
    Tests the accuracy of MAS preconditioner's P @ g direction at iter=0.

    At iter=0 of the first time step:
    1. Computes gradient g
    2. Applies MAS preconditioner: z_mas = P_mas @ g
    3. Applies diagonal preconditioner: z_diag = P_diag @ g
    4. Computes ground truth: z_exact = H^{-1} @ g (via sparse solve)
    5. Compares all three using various metrics
    """

    def __init__(self, demo='cube_freefall_10'):
        """Initialize tester with given demo configuration."""
        # Load model
        model = model_loading(demo=demo)
        self.demo = demo
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.mesh = model.mesh
        self.epsilon = model.epsilon

        # Place vertex fields
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),
            'v': ti.types.vector(3, float),
            'm': float,
            'x_n': ti.types.vector(3, float),
            'x_hat': ti.types.vector(3, float),
            'grad': ti.types.vector(3, float),
            'diagH': ti.types.vector(3, float),
            'z': ti.types.vector(3, float),
        })

        # Place cell fields
        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})

        # Initialize positions
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])

        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        self.ndof = self.n_verts * 3

        print(f"Mesh: {self.n_verts} vertices, {self.n_cells} cells, {self.ndof} DOFs")

        # Precompute mass, B, W
        self.precompute()

        # Assign elastic type
        self.assign_elastic_type(model.elastic_type)

        # Initialize MAS preconditioner (using simplified version)
        print(f"Initializing MAS-Small preconditioner...")
        self.mas = MASPreconditionerSmall(self.mesh)
        print(f"MAS-Small initialized with {self.mas.level_num} levels")

    def assign_elastic_type(self, elastic):
        """Set elastic type functions.

        MAS preconditioner uses integer elastic_type:
        0=ARAP, 1=SNH, 2=FCR, 3=ARAP_SPD, 4=NH_SPD, 5=STVK_SPD
        """
        if elastic == 'ARAP':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP
            self.elastic_type = 0
        elif elastic == 'SNH':
            self.compute_dPsidx = compute_dPsidx_SNH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_SNH
            self.elastic_type = 1
        elif elastic == 'ARAP_filter':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_filter
            self.elastic_type = 0
        elif elastic == 'FCR':
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR
            self.elastic_type = 2
        elif elastic == 'FCR_filter':
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR_filter
            self.elastic_type = 2
        elif elastic == 'NH':
            self.compute_dPsidx = compute_dPsidx_NH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_NH
            self.elastic_type = 1  # Map NH to SNH for MAS assembly
        # SPD-projected Hessian materials (eigenanalysis-based)
        elif elastic == 'ARAP_SPD':
            self.compute_dPsidx = compute_dPsidx_ARAP_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_SPD
            self.elastic_type = 3
        elif elastic == 'NH_SPD':
            self.compute_dPsidx = compute_dPsidx_NH_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_NH_SPD
            self.elastic_type = 4
        elif elastic == 'STVK_SPD':
            self.compute_dPsidx = compute_dPsidx_STVK_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_STVK_SPD
            self.elastic_type = 5
        else:
            print(f'Warning: Unknown elastic type {elastic}, using ARAP_SPD')
            self.compute_dPsidx = compute_dPsidx_ARAP_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_SPD
            self.elastic_type = 3
        self.elastic_type_str = elastic

    @ti.kernel
    def precompute(self):
        """Precompute mass, B matrix, and cell volumes."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4.0

    @ti.kernel
    def init_v(self, vy: float):
        """Initialize velocity."""
        for vert in self.mesh.verts:
            vert.v[1] = vy

    @ti.kernel
    def assign_xn_xhat(self):
        """Assign x_n and x_hat for implicit time integration."""
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.gravity * self.dt * self.dt

    @ti.kernel
    def compute_grad_and_diagH(self):
        """Compute gradient and diagonal Hessian."""
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)

        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2

            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            diagH = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)

            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3*i], dPsidx[3*i+1], dPsidx[3*i+2]], float)
                tmp = ti.Vector([diagH[3*i], diagH[3*i+1], diagH[3*i+2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp

    @ti.kernel
    def apply_diagonal_preconditioner(self):
        """Apply diagonal preconditioner: z = diag(H)^{-1} @ g."""
        for vert in self.mesh.verts:
            for i in ti.static(range(3)):
                if vert.diagH[i] > 1e-10:
                    vert.z[i] = vert.grad[i] / vert.diagH[i]
                else:
                    vert.z[i] = vert.grad[i]

    def extract_cell_verts(self):
        """Extract cell-vertex connectivity using Taichi kernel."""
        # Allocate storage for cell vertices
        cell_verts_field = ti.field(dtype=ti.i32, shape=(self.n_cells, 4))

        @ti.kernel
        def _extract_cell_verts(cell_verts: ti.template()):
            for c in self.mesh.cells:
                cell_verts[c.id, 0] = c.verts[0].id
                cell_verts[c.id, 1] = c.verts[1].id
                cell_verts[c.id, 2] = c.verts[2].id
                cell_verts[c.id, 3] = c.verts[3].id

        _extract_cell_verts(cell_verts_field)
        return cell_verts_field.to_numpy()

    def build_sparse_hessian(self):
        """
        Build full sparse Hessian matrix for ground truth computation.

        H = M + dt^2 * sum_e(H_e)

        Returns:
            scipy.sparse.csr_matrix: Full Hessian matrix
        """
        # Get mesh data
        x_np = self.mesh.verts.x.to_numpy()
        m_np = self.mesh.verts.m.to_numpy()

        # Build COO format
        rows = []
        cols = []
        data = []

        # Mass/inertia terms (diagonal)
        for v in range(self.n_verts):
            for d in range(3):
                idx = v * 3 + d
                rows.append(idx)
                cols.append(idx)
                data.append(m_np[v])

        # Element Hessian contributions
        # Get cell-vertex connectivity (4 vertices per tetrahedral cell)
        cell_verts = self.extract_cell_verts()  # Shape: (n_cells, 4)
        B_np = self.mesh.cells.B.to_numpy()
        W_np = self.mesh.cells.W.to_numpy()

        for c in range(self.n_cells):
            # Get vertex indices for this cell
            v_ids = cell_verts[c]

            # Get vertex positions
            x_cell = x_np[v_ids]  # 4x3

            # Compute deformation gradient
            Ds = np.zeros((3, 3))
            for i in range(3):
                Ds[:, i] = x_cell[i+1] - x_cell[0]

            B = B_np[c]
            F = Ds @ B
            W = W_np[c]
            para = W * self.dt ** 2

            # Compute element Hessian (12x12)
            He = self._compute_element_hessian(F, B, para)

            # Add to global matrix
            for i in range(4):
                for j in range(4):
                    for di in range(3):
                        for dj in range(3):
                            row_idx = v_ids[i] * 3 + di
                            col_idx = v_ids[j] * 3 + dj
                            val = He[i*3 + di, j*3 + dj]
                            if abs(val) > 1e-15:
                                rows.append(row_idx)
                                cols.append(col_idx)
                                data.append(val)

        # Build sparse matrix
        H = sparse.coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof))
        H = H.tocsr()

        # Ensure symmetry
        H = (H + H.T) / 2

        return H

    def _compute_element_hessian(self, F, B, para):
        """
        Compute 12x12 element Hessian for ARAP energy using numerical differentiation.
        """
        eps = 1e-6
        He = np.zeros((12, 12))

        # Get current gradient
        g0 = self._compute_element_gradient(F, B, para)

        # Finite difference
        for i in range(4):
            for d in range(3):
                # Perturb
                dF = np.zeros((3, 3))

                # dF/dx_i depends on B
                if i == 0:
                    # x0 affects all columns through -B
                    for k in range(3):
                        dF[d, k] = -B[d, k]
                else:
                    # xi (i>0) affects only column i-1
                    dF[d, i-1] = B[d, i-1]

                F_plus = F + eps * dF
                g_plus = self._compute_element_gradient(F_plus, B, para)

                # Hessian column
                col_idx = i * 3 + d
                He[:, col_idx] = (g_plus - g0) / eps

        # Symmetrize
        He = (He + He.T) / 2

        # Project to SPD (clip negative eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(He)
        eigvals = np.maximum(eigvals, 0.0)
        He = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return He

    def _compute_element_gradient(self, F, B, para):
        """Compute 12x1 element gradient for ARAP energy."""
        # ARAP: Psi = mu * ||F - R||^2
        # dPsi/dF = 2 * mu * (F - R)

        # SVD to get rotation
        U, S, Vt = np.linalg.svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Gradient w.r.t. F
        dPsidF = 2.0 * self.mu * (F - R)

        # Chain rule: dPsi/dx = dPsi/dF @ dF/dx
        g = np.zeros(12)

        # dF/dx for each vertex
        for i in range(4):
            for d in range(3):
                dFdx = np.zeros((3, 3))
                if i == 0:
                    for k in range(3):
                        dFdx[d, k] = -B[d, k]
                else:
                    dFdx[d, i-1] = B[d, i-1]

                g[i*3 + d] = para * np.sum(dPsidF * dFdx)

        return g

    def compute_ground_truth(self, g_np):
        """
        Compute ground truth preconditioned direction: z = H^{-1} @ g.

        Args:
            g_np: Gradient vector (ndof,)

        Returns:
            z_exact: Exact solution via sparse direct solve
        """
        print("Building sparse Hessian...")
        H = self.build_sparse_hessian()

        print("Solving H @ z = g via sparse direct solve...")
        try:
            z_exact = spsolve(H, g_np)

            # Verify solution
            residual = np.linalg.norm(H @ z_exact - g_np)
            print(f"  Solve residual: {residual:.2e}")

            return z_exact, H
        except Exception as e:
            print(f"  Direct solve failed: {e}")
            print("  Trying CG solve...")

            # Fallback to CG
            z_exact, info = cg(H, g_np, maxiter=1000, tol=1e-10)
            if info != 0:
                print(f"  CG did not converge (info={info})")
            return z_exact, H

    def run_test(self, initial_vy=-1.0, verbose=True, compute_exact=True):
        """
        Run the direction accuracy test at iter=0.

        Args:
            initial_vy: Initial downward velocity
            verbose: Print detailed output
            compute_exact: Whether to compute exact H^{-1} @ g (expensive for large meshes)

        Returns:
            dict with test results
        """
        print(f"\n{'='*70}")
        print(f"MAS-Small Preconditioner Direction Accuracy Test (iter=0)")
        print(f"{'='*70}")
        print(f"Demo: {self.demo}")
        print(f"Elastic: {self.elastic_type_str}")
        print(f"MAS Implementation: mas_preconditioner_small (IC(0) only)")
        print(f"{'='*70}\n")

        # Initialize
        self.init_v(initial_vy)
        self.assign_xn_xhat()

        # Compute gradient and diagonal Hessian (this is iter=0)
        print("Computing gradient and diagonal Hessian (iter=0)...")
        self.compute_grad_and_diagH()

        # Get gradient as numpy
        g_np = self.mesh.verts.grad.to_numpy().flatten()
        diagH_np = self.mesh.verts.diagH.to_numpy().flatten()

        print(f"  |g| = {np.linalg.norm(g_np):.6e}")
        print(f"  |g|_inf = {np.max(np.abs(g_np)):.6e}")

        # 1. Apply diagonal preconditioner
        print("\n[1] Applying diagonal preconditioner: z_diag = diag(H)^{-1} @ g")
        self.apply_diagonal_preconditioner()
        z_diag = self.mesh.verts.z.to_numpy().flatten()

        # 2. Apply MAS preconditioner (using simplified MAS-Small)
        print("\n[2] Applying MAS-Small preconditioner: z_mas = P_MAS @ g")
        # MAS-Small uses rebuild() which does: build_hierarchy + assemble + invert
        self.mas.rebuild(self)  # self has mu, la, dt attributes needed
        self.mas.apply()
        z_mas = self.mesh.verts.z.to_numpy().flatten()

        # Check for NaN
        if np.any(np.isnan(z_mas)):
            print("  [WARNING] MAS produced NaN!")
            z_mas = np.nan_to_num(z_mas, nan=0.0)

        # 3. Compute ground truth (for small/medium meshes)
        # Ground truth comparison is essential for validating preconditioner accuracy
        z_exact = None
        H = None
        MAX_VERTS_FOR_EXACT = 500  # Allow larger meshes for ground truth
        if compute_exact and self.n_verts <= MAX_VERTS_FOR_EXACT:
            print(f"\n[3] Computing ground truth: z_exact = H^{{-1}} @ g")
            z_exact, H = self.compute_ground_truth(g_np)
        elif compute_exact:
            print(f"\n[3] Skipping ground truth (mesh too large: {self.n_verts} verts > {MAX_VERTS_FOR_EXACT})")
            print(f"    Use --no-exact to explicitly disable, or use smaller mesh")
        else:
            print("\n[3] Ground truth computation disabled (--no-exact)")

        # ========================================================================
        # Compute metrics
        # ========================================================================
        print(f"\n{'='*70}")
        print("Direction Accuracy Metrics")
        print(f"{'='*70}")

        results = {
            'demo': self.demo,
            'n_verts': self.n_verts,
            'n_cells': self.n_cells,
            'g_norm': np.linalg.norm(g_np),
        }

        # --- Diagonal preconditioner metrics ---
        print("\n[Diagonal Preconditioner]")
        results['diag'] = self._compute_metrics(g_np, z_diag, 'diag', z_exact, verbose)

        # --- MAS preconditioner metrics ---
        print("\n[MAS Preconditioner]")
        results['mas'] = self._compute_metrics(g_np, z_mas, 'mas', z_exact, verbose)

        # --- Ground truth metrics (if available) ---
        if z_exact is not None:
            print("\n[Ground Truth (H^{-1} @ g)]")
            results['exact'] = self._compute_metrics(g_np, z_exact, 'exact', z_exact, verbose)

        # ========================================================================
        # Comparison summary
        # ========================================================================
        print(f"\n{'='*70}")
        print("Comparison Summary")
        print(f"{'='*70}")

        print(f"\n{'Metric':<30} {'Diagonal':<15} {'MAS':<15}")
        print(f"{'-'*60}")

        metrics_to_compare = [
            ('g^T z (descent)', 'gTz', '.6e'),
            ('cos(g, z)', 'cos_g_z', '.6f'),
            ('|z|', 'z_norm', '.6e'),
            ('|z| / |g|', 'z_g_ratio', '.6f'),
        ]

        if z_exact is not None:
            metrics_to_compare.extend([
                ('cos(z, z_exact)', 'cos_z_exact', '.6f'),
                ('|z - z_exact| / |z_exact|', 'rel_error', '.6e'),
            ])

        for name, key, fmt in metrics_to_compare:
            diag_val = results['diag'].get(key, float('nan'))
            mas_val = results['mas'].get(key, float('nan'))
            print(f"{name:<30} {diag_val:<15{fmt}} {mas_val:<15{fmt}}")

        # ========================================================================
        # Pass/Fail criteria
        # ========================================================================
        print(f"\n{'='*70}")
        print("Test Results")
        print(f"{'='*70}")

        # Check 1: Descent direction (g^T z > 0)
        diag_descent = results['diag']['gTz'] > 0
        mas_descent = results['mas']['gTz'] > 0

        print(f"\n1. Descent Direction Check (g^T z > 0):")
        print(f"   Diagonal: {'PASS' if diag_descent else 'FAIL'} (g^T z = {results['diag']['gTz']:.2e})")
        print(f"   MAS:      {'PASS' if mas_descent else 'FAIL'} (g^T z = {results['mas']['gTz']:.2e})")

        # Check 2: Direction quality (cos > 0.5)
        diag_quality = results['diag']['cos_g_z'] > 0.5
        mas_quality = results['mas']['cos_g_z'] > 0.5

        print(f"\n2. Direction Quality Check (cos(g, z) > 0.5):")
        print(f"   Diagonal: {'PASS' if diag_quality else 'FAIL'} (cos = {results['diag']['cos_g_z']:.4f})")
        print(f"   MAS:      {'PASS' if mas_quality else 'FAIL'} (cos = {results['mas']['cos_g_z']:.4f})")

        # Check 3: Accuracy vs exact (PRIMARY CHECK when available)
        diag_acc = None
        mas_acc = None
        if z_exact is not None:
            # This is the primary accuracy check - comparing against H^{-1} @ g
            diag_acc = results['diag']['cos_z_exact'] > 0.9
            mas_acc = results['mas']['cos_z_exact'] > 0.9

            print(f"\n3. GROUND TRUTH Accuracy Check (cos(z, z_exact) > 0.9):")
            print(f"   This is the PRIMARY metric - comparing P@g against H^{{-1}}@g")
            print(f"   Diagonal: {'PASS' if diag_acc else 'FAIL'} (cos = {results['diag']['cos_z_exact']:.4f})")
            print(f"   MAS:      {'PASS' if mas_acc else 'FAIL'} (cos = {results['mas']['cos_z_exact']:.4f})")

            # Also show relative error
            print(f"\n4. Ground Truth Relative Error Check:")
            diag_rel = results['diag']['rel_error']
            mas_rel = results['mas']['rel_error']
            print(f"   Diagonal: |z - z_exact| / |z_exact| = {diag_rel:.4e}")
            print(f"   MAS:      |z - z_exact| / |z_exact| = {mas_rel:.4e}")

        # Overall: Ground truth comparison takes priority when available
        if z_exact is not None:
            diag_pass = diag_descent and diag_acc
            mas_pass = mas_descent and mas_acc
            criterion = "descent + ground_truth"
        else:
            diag_pass = diag_descent and diag_quality
            mas_pass = mas_descent and mas_quality
            criterion = "descent + direction_quality"

        print(f"\n{'='*70}")
        print(f"Overall (criterion: {criterion}):")
        print(f"  Diagonal = {'PASS' if diag_pass else 'FAIL'}")
        print(f"  MAS      = {'PASS' if mas_pass else 'FAIL'}")
        print(f"{'='*70}\n")

        results['diag_pass'] = diag_pass
        results['mas_pass'] = mas_pass

        return results

    def _compute_metrics(self, g, z, name, z_exact=None, verbose=True):
        """Compute direction quality metrics."""
        metrics = {}

        # Basic norms
        g_norm = np.linalg.norm(g)
        z_norm = np.linalg.norm(z)
        metrics['g_norm'] = g_norm
        metrics['z_norm'] = z_norm

        if verbose:
            print(f"  |z| = {z_norm:.6e}")

        # Descent direction: g^T z
        gTz = np.dot(g, z)
        metrics['gTz'] = gTz

        if verbose:
            print(f"  g^T z = {gTz:.6e} ({'descent' if gTz > 0 else 'ASCENT!'})")

        # Cosine angle between g and z
        if g_norm > 1e-15 and z_norm > 1e-15:
            cos_g_z = gTz / (g_norm * z_norm)
        else:
            cos_g_z = 0.0
        metrics['cos_g_z'] = cos_g_z

        if verbose:
            angle_deg = np.arccos(np.clip(cos_g_z, -1, 1)) * 180 / np.pi
            print(f"  cos(g, z) = {cos_g_z:.6f} (angle = {angle_deg:.1f}deg)")

        # Scaling ratio
        if g_norm > 1e-15:
            z_g_ratio = z_norm / g_norm
        else:
            z_g_ratio = float('inf')
        metrics['z_g_ratio'] = z_g_ratio

        if verbose:
            print(f"  |z| / |g| = {z_g_ratio:.6f}")

        # Comparison with exact solution (if available)
        if z_exact is not None:
            z_exact_norm = np.linalg.norm(z_exact)

            # Cosine with exact
            if z_norm > 1e-15 and z_exact_norm > 1e-15:
                cos_z_exact = np.dot(z, z_exact) / (z_norm * z_exact_norm)
            else:
                cos_z_exact = 0.0
            metrics['cos_z_exact'] = cos_z_exact

            # Relative error
            if z_exact_norm > 1e-15:
                rel_error = np.linalg.norm(z - z_exact) / z_exact_norm
            else:
                rel_error = np.linalg.norm(z - z_exact)
            metrics['rel_error'] = rel_error

            if verbose:
                print(f"  cos(z, z_exact) = {cos_z_exact:.6f}")
                print(f"  |z - z_exact| / |z_exact| = {rel_error:.6e}")

        return metrics

    def test_hessian_matvec_accuracy(self, verbose=True):
        """
        Test MAS hessian_matvec accuracy against ground truth compute_zHz.

        This test verifies that z^T H z computed via MAS hessian_matvec is
        consistent with the ground truth computed via:
            z^T H z = sum_verts(z^T * m * z) + sum_cells(z^T * H_elastic * z)

        Note: MAS hessian_matvec is an APPROXIMATION because:
        - Level 0 stores only intra-block coupling
        - Cross-block coupling is stored in coarse levels with aggregation
        """
        print(f"\n{'='*70}")
        print("MAS hessian_matvec Accuracy Test")
        print(f"{'='*70}\n")

        # Initialize state
        self.init_v(-1.0)
        self.assign_xn_xhat()
        self.compute_grad_and_diagH()

        # Build hierarchy first (only once, avoid recompilation)
        if not self.mas.hierarchy_built:
            self.mas.build_hierarchy()

        # Assemble and invert block matrices
        self.mas.assemble_block_matrices(self)
        self.mas.invert_block_matrices()

        # Apply preconditioner
        self.mas.apply()

        z_np = self.mesh.verts.z.to_numpy().flatten()
        g_np = self.mesh.verts.grad.to_numpy().flatten()
        print(f"|z| = {np.linalg.norm(z_np):.6e}")
        print(f"|g| = {np.linalg.norm(g_np):.6e}")

        # Compute g^T z
        gTz = np.dot(g_np, z_np)
        print(f"g^T z = {gTz:.6e}")

        # Method 1: Ground truth z^T H z using sparse Hessian
        print("\nComputing ground truth z^T H z via sparse Hessian...")
        H = self.build_sparse_hessian()
        Hz_gt = H @ z_np
        zHz_gt = np.dot(z_np, Hz_gt)
        print(f"z^T H z (ground truth) = {zHz_gt:.6e}")
        print(f"|H @ z| (ground truth) = {np.linalg.norm(Hz_gt):.6e}")

        # Method 2: MAS hessian_matvec (approximate - uses coarse level approximation)
        print("\nComputing z^T H z via MAS hessian_matvec (APPROXIMATE)...")
        z_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)
        Hz_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)

        # Copy z to buffer
        z_reshaped = z_np.reshape(-1, 3)
        z_buffer.from_numpy(z_reshaped.astype(np.float64))

        # Apply hessian_matvec (approximate)
        self.mas.hessian_matvec(z_buffer, Hz_buffer)

        Hz_mas_approx = Hz_buffer.to_numpy().flatten()
        zHz_mas_approx = np.dot(z_np, Hz_mas_approx)
        print(f"z^T H z (MAS approx) = {zHz_mas_approx:.6e}")
        print(f"|H @ z| (MAS approx) = {np.linalg.norm(Hz_mas_approx):.6e}")

        # Method 3: MAS hessian_matvec_exact (uses triplet storage for cross-block)
        print("\nComputing z^T H z via MAS hessian_matvec_exact (EXACT)...")
        Hz_buffer_exact = ti.Vector.field(3, dtype=ti.f64, shape=self.n_verts)

        # Apply hessian_matvec_exact
        self.mas.hessian_matvec_exact(z_buffer, Hz_buffer_exact)

        Hz_mas = Hz_buffer_exact.to_numpy().flatten()
        zHz_mas = np.dot(z_np, Hz_mas)
        print(f"z^T H z (MAS exact) = {zHz_mas:.6e}")
        print(f"|H @ z| (MAS exact) = {np.linalg.norm(Hz_mas):.6e}")

        # Show cross-block statistics
        stats = self.mas.get_cross_block_stats()
        print(f"\nCross-block storage: {stats['n_triplets']} triplets ({stats['usage_percent']:.1f}% of max)")

        # Comparison
        print(f"\n{'='*70}")
        print("Comparison")
        print(f"{'='*70}")

        ratio = zHz_mas / zHz_gt if abs(zHz_gt) > 1e-15 else float('inf')
        print(f"Ratio (MAS / GT) = {ratio:.4f}")

        # Compare Hz vectors
        Hz_rel_error = np.linalg.norm(Hz_mas - Hz_gt) / np.linalg.norm(Hz_gt) if np.linalg.norm(Hz_gt) > 1e-15 else float('inf')
        print(f"|H@z_MAS - H@z_GT| / |H@z_GT| = {Hz_rel_error:.4e}")

        # Cosine similarity between Hz vectors
        if np.linalg.norm(Hz_mas) > 1e-15 and np.linalg.norm(Hz_gt) > 1e-15:
            cos_Hz = np.dot(Hz_mas, Hz_gt) / (np.linalg.norm(Hz_mas) * np.linalg.norm(Hz_gt))
        else:
            cos_Hz = 0.0
        print(f"cos(H@z_MAS, H@z_GT) = {cos_Hz:.6f}")

        # Compute alpha using both methods
        alpha_gt = gTz / zHz_gt if abs(zHz_gt) > 1e-15 else 0.0
        alpha_mas = gTz / zHz_mas if abs(zHz_mas) > 1e-15 else 0.0

        print(f"\nalpha = g^T z / z^T H z:")
        print(f"  alpha (GT):  {alpha_gt:.6f}")
        print(f"  alpha (MAS): {alpha_mas:.6f}")
        print(f"  Ratio:       {alpha_mas/alpha_gt if abs(alpha_gt) > 1e-15 else float('inf'):.4f}")

        # Step size comparison
        print(f"\nStep size |alpha * z|:")
        print(f"  GT:  {abs(alpha_gt) * np.linalg.norm(z_np):.6e}")
        print(f"  MAS: {abs(alpha_mas) * np.linalg.norm(z_np):.6e}")

        # Status assessment
        print(f"\n{'='*70}")
        print("Assessment")
        print(f"{'='*70}")

        if abs(ratio - 1.0) < 0.1:
            status = "EXCELLENT (<10% error)"
        elif abs(ratio - 1.0) < 0.5:
            status = "GOOD (<50% error)"
        elif 0.01 < ratio < 100.0:
            status = "APPROXIMATE (same order of magnitude)"
        else:
            status = "POOR (>2 orders of magnitude off)"

        print(f"z^T H z ratio status: {status}")
        print(f"Note: MAS stores block approximation, so some deviation is expected.")

        # Return results
        return {
            'zHz_gt': zHz_gt,
            'zHz_mas': zHz_mas,
            'ratio': ratio,
            'Hz_rel_error': Hz_rel_error,
            'cos_Hz': cos_Hz,
            'alpha_gt': alpha_gt,
            'alpha_mas': alpha_mas,
            'status': status,
        }


def main():
    parser = argparse.ArgumentParser(description='MAS-Small Direction Accuracy Test at iter=0')
    parser.add_argument('--demo', type=str, default='cube_freefall_10',
                        help='Demo configuration')
    parser.add_argument('--vy', type=float, default=-1.0,
                        help='Initial downward velocity')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--mas-verbose', action='store_true',
                        help='Enable MAS verbose output')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable Taichi offline cache')
    parser.add_argument('--no-exact', action='store_true',
                        help='Skip ground truth computation')
    parser.add_argument('--hessian-matvec', action='store_true',
                        help='Run hessian_matvec accuracy test (compare z^T H z)')
    args = parser.parse_args()

    # Enable MAS verbose if requested
    if args.mas_verbose:
        global _mas_verbose
        _mas_verbose = True
        builtins.print = _original_print

    # Initialize Taichi
    if args.no_cache:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    else:
        ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True)

    # Create tester and run
    tester = PrecondDirectionTester(demo=args.demo)

    # Run hessian_matvec test if requested
    if args.hessian_matvec:
        results = tester.test_hessian_matvec_accuracy(verbose=args.verbose or True)
        # Return 0 if ratio is reasonable (0.01 < ratio < 100)
        if 0.01 < results['ratio'] < 100.0:
            return 0
        else:
            return 1

    # Run standard direction accuracy test
    results = tester.run_test(
        initial_vy=args.vy,
        verbose=args.verbose or True,
        compute_exact=not args.no_exact
    )

    # Return exit code
    if results['mas_pass']:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
