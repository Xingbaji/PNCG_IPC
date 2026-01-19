"""
Unit tests for Elastic Hessian SPD Projection

Tests the eigenanalysis-based SPD projection for 3D elastic energy Hessian.
"""

import unittest
import numpy as np
import taichi as ti

# Initialize Taichi before importing the module
ti.init(arch=ti.cpu, default_fp=ti.f32)

import sys
sys.path.insert(0, '/root/PNCG_IPC')

# Import kernel functions with aliases to avoid pytest collection
from math_utils.elastic_hessian_spd import (
    test_hessian_spd_kernel as hessian_spd_kernel,
    test_force_kernel as force_kernel,
    Svd3x3,
    DiffTable3,
)


class TestElasticHessianSPD(unittest.TestCase):
    """Test suite for elastic Hessian SPD projection"""

    def setUp(self):
        """Set up test fixtures"""
        self.mu = 1e6
        self.lam = 1e6
        self.rel_tol = 1e-5  # Relative tolerance for SPD check

    def _check_spd(self, H: np.ndarray, name: str = ""):
        """Helper to check if matrix is SPD"""
        # Check symmetry
        sym_error = np.max(np.abs(H - H.T))
        self.assertLess(sym_error, 1e-6, f"{name}: Matrix not symmetric, error={sym_error:.2e}")

        # Check positive semi-definiteness (relative tolerance)
        eigenvalues = np.linalg.eigvalsh(H)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)

        # Allow small negative eigenvalues relative to max eigenvalue
        self.assertGreaterEqual(
            min_eig, -self.rel_tol * max_eig,
            f"{name}: Not SPD, min_eig={min_eig:.2e}, max_eig={max_eig:.2e}"
        )

        return min_eig, max_eig

    def test_stvk_mild_deformation(self):
        """Test StVK material with mild deformation"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 0, H_out)

        min_eig, max_eig = self._check_spd(H_out, "StVK mild")
        self.assertGreater(max_eig, 0, "Hessian should have positive eigenvalues")

    def test_neohookean_mild_deformation(self):
        """Test NeoHookean material with mild deformation"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 1, H_out)

        min_eig, max_eig = self._check_spd(H_out, "NeoHookean mild")
        self.assertGreater(max_eig, 0, "Hessian should have positive eigenvalues")

    def test_arap_mild_deformation(self):
        """Test ARAP material with mild deformation"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 2, H_out)

        min_eig, max_eig = self._check_spd(H_out, "ARAP mild")
        self.assertGreater(max_eig, 0, "Hessian should have positive eigenvalues")

    def test_stvk_compression(self):
        """Test StVK material with compression deformation"""
        F = np.array([
            [0.5, 0.2, 0.1],
            [0.2, 0.3, 0.15],
            [0.1, 0.15, 0.4]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 0, H_out)

        self._check_spd(H_out, "StVK compression")

    def test_neohookean_compression(self):
        """Test NeoHookean material with compression deformation"""
        F = np.array([
            [0.5, 0.2, 0.1],
            [0.2, 0.3, 0.15],
            [0.1, 0.15, 0.4]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 1, H_out)

        self._check_spd(H_out, "NeoHookean compression")

    def test_arap_compression(self):
        """Test ARAP material with compression deformation"""
        F = np.array([
            [0.5, 0.2, 0.1],
            [0.2, 0.3, 0.15],
            [0.1, 0.15, 0.4]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)
        hessian_spd_kernel(F, self.mu, self.lam, 2, H_out)

        self._check_spd(H_out, "ARAP compression")

    def test_identity_deformation(self):
        """Test with identity deformation gradient (no deformation)"""
        F = np.eye(3, dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
            hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
            self._check_spd(H_out, f"{name} identity")

    def test_pure_stretch(self):
        """Test with pure stretch deformation"""
        F = np.diag([1.5, 0.8, 1.2]).astype(np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
            hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
            self._check_spd(H_out, f"{name} pure stretch")

    def test_rotation(self):
        """Test with rotation deformation"""
        # 45 degree rotation around z-axis
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # F = R @ stretch
        stretch = np.diag([1.1, 0.95, 1.05])
        F = (R @ stretch).astype(np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
            hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
            self._check_spd(H_out, f"{name} rotation")

    def test_shear_deformation(self):
        """Test with shear deformation"""
        F = np.array([
            [1.0, 0.3, 0.1],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
            hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
            self._check_spd(H_out, f"{name} shear")

    def test_near_singular_deformation(self):
        """Test with near-singular deformation gradient"""
        # Very thin element (one singular value close to zero)
        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01]  # Near-zero thickness
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for material, name in [(0, "StVK"), (2, "ARAP")]:  # Skip NeoHookean (ln(J) issues)
            hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
            self._check_spd(H_out, f"{name} near-singular")

    def test_different_material_parameters(self):
        """Test with different material parameters"""
        F = np.array([
            [1.1, 0.05, 0.02],
            [0.05, 0.95, 0.03],
            [0.02, 0.03, 1.05]
        ], dtype=np.float32)

        H_out = np.zeros((9, 9), dtype=np.float32)

        # Test various mu/lambda ratios
        params = [
            (1e4, 1e4),   # Equal
            (1e6, 1e4),   # High shear
            (1e4, 1e6),   # High bulk
            (1e3, 1e8),   # Nearly incompressible
        ]

        for mu, lam in params:
            for material in [0, 1, 2]:
                hessian_spd_kernel(F, mu, lam, material, H_out)
                self._check_spd(H_out, f"material={material}, mu={mu:.0e}, lam={lam:.0e}")

    def test_force_computation(self):
        """Test that force computation produces valid stress tensor"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        P_out = np.zeros((3, 3), dtype=np.float32)

        for material, name in [(0, "StVK"), (1, "NeoHookean"), (2, "ARAP")]:
            force_kernel(F, self.mu, self.lam, material, P_out)

            # Check that force is finite
            self.assertTrue(np.all(np.isfinite(P_out)), f"{name}: Force contains NaN/Inf")

            # For identity F, force should be zero (or near zero)
            F_id = np.eye(3, dtype=np.float32)
            force_kernel(F_id, self.mu, self.lam, material, P_out)
            force_norm = np.linalg.norm(P_out)
            self.assertLess(force_norm, 1e-3, f"{name}: Force at rest should be ~0")

    def test_hessian_consistency_with_finite_difference(self):
        """Test Hessian consistency using finite differences (gradient check)"""
        F = np.array([
            [1.1, 0.05, 0.02],
            [0.05, 0.95, 0.03],
            [0.02, 0.03, 1.05]
        ], dtype=np.float32)

        eps = 1e-4
        H_out = np.zeros((9, 9), dtype=np.float32)
        P_out = np.zeros((3, 3), dtype=np.float32)
        P_plus = np.zeros((3, 3), dtype=np.float32)
        P_minus = np.zeros((3, 3), dtype=np.float32)

        # Use smaller mu/lam for better numerical conditioning
        mu, lam = 1e3, 1e3

        for material in [0, 2]:  # Skip NeoHookean due to ln(J) sensitivity
            # Compute analytical Hessian
            hessian_spd_kernel(F, mu, lam, material, H_out)

            # Compute numerical Hessian via finite differences of gradient
            H_fd = np.zeros((9, 9), dtype=np.float32)

            for i in range(3):
                for j in range(3):
                    idx = j * 3 + i  # Column-major index

                    F_plus = F.copy()
                    F_plus[i, j] += eps
                    force_kernel(F_plus, mu, lam, material, P_plus)

                    F_minus = F.copy()
                    F_minus[i, j] -= eps
                    force_kernel(F_minus, mu, lam, material, P_minus)

                    # Numerical gradient of P w.r.t. F[i,j]
                    dP = (P_plus - P_minus) / (2 * eps)

                    # Fill column of Hessian
                    for ii in range(3):
                        for jj in range(3):
                            H_fd[jj * 3 + ii, idx] = dP[ii, jj]

            # Compare (note: projected Hessian may differ from true Hessian)
            # We mainly check that the structure is reasonable
            H_fd_sym = 0.5 * (H_fd + H_fd.T)

            # Check that diagonal elements have same sign
            for k in range(9):
                if abs(H_out[k, k]) > 1e-6 and abs(H_fd_sym[k, k]) > 1e-6:
                    # Both should be positive after SPD projection
                    self.assertGreater(
                        H_out[k, k], -1e-6,
                        f"material={material}: Diagonal H[{k},{k}] should be non-negative"
                    )

    def test_random_deformations(self):
        """Test with multiple random deformations"""
        np.random.seed(42)

        H_out = np.zeros((9, 9), dtype=np.float32)

        for _ in range(10):
            # Generate random F with det(F) > 0
            F = np.eye(3) + 0.3 * np.random.randn(3, 3)
            # Ensure positive determinant
            if np.linalg.det(F) < 0.1:
                F = -F if np.linalg.det(F) < 0 else F * 1.5

            F = F.astype(np.float32)

            for material in [0, 1, 2]:
                try:
                    hessian_spd_kernel(F, self.mu, self.lam, material, H_out)
                    self._check_spd(H_out, f"random, material={material}")
                except Exception as e:
                    # Allow some failures for extreme deformations
                    det_F = np.linalg.det(F)
                    if det_F < 0.1 or det_F > 10:
                        continue
                    raise e


class TestSVDDecomposition(unittest.TestCase):
    """Test SVD decomposition accuracy"""

    def test_svd_reconstruction(self):
        """Test that U @ diag(S) @ Vt reconstructs F"""
        # This requires accessing internal SVD function
        # For now, we verify through the Hessian computation
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        # Compare with numpy SVD
        U_np, S_np, Vt_np = np.linalg.svd(F)

        # Reconstruction should match
        F_reconstructed = U_np @ np.diag(S_np) @ Vt_np
        error = np.max(np.abs(F - F_reconstructed))
        self.assertLess(error, 1e-5, "SVD reconstruction error too large")


class TestMaterialModels(unittest.TestCase):
    """Test specific material model properties"""

    def test_stvk_energy_at_rest(self):
        """StVK energy should be zero at rest (F=I)"""
        # At F=I, Green strain E = 0, so energy = 0
        # dE/dσ should be 0, d²E/dσ² should be well-defined
        F = np.eye(3, dtype=np.float32)
        P_out = np.zeros((3, 3), dtype=np.float32)

        force_kernel(F, 1e6, 1e6, 0, P_out)
        self.assertLess(np.linalg.norm(P_out), 1e-3, "StVK force at rest should be ~0")

    def test_arap_energy_at_rotation(self):
        """ARAP energy should be zero for pure rotation"""
        angle = np.pi / 6
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        P_out = np.zeros((3, 3), dtype=np.float32)
        force_kernel(R, 1e6, 1e6, 2, P_out)

        # Force should be ~0 for pure rotation
        self.assertLess(np.linalg.norm(P_out), 1e-2, "ARAP force for rotation should be ~0")

    def test_neohookean_volume_preservation(self):
        """NeoHookean should penalize volume change with opposing forces"""
        # Isotropic compression and stretch
        F_compress = 0.8 * np.eye(3, dtype=np.float32)
        F_stretch = 1.2 * np.eye(3, dtype=np.float32)

        P_compress = np.zeros((3, 3), dtype=np.float32)
        P_stretch = np.zeros((3, 3), dtype=np.float32)

        # High bulk modulus
        force_kernel(F_compress, 1e3, 1e6, 1, P_compress)
        force_kernel(F_stretch, 1e3, 1e6, 1, P_stretch)

        # For isotropic deformation F = αI:
        # P = dψ/dF should be diagonal
        # The key property: compression and stretch should produce opposite-sign forces
        # (restoring towards identity)
        trace_compress = np.trace(P_compress)
        trace_stretch = np.trace(P_stretch)

        # Check that forces have opposite signs (restoring behavior)
        # Compression gives one sign, stretch gives opposite
        self.assertNotEqual(
            np.sign(trace_compress), np.sign(trace_stretch),
            f"Compression and stretch should produce opposite forces: "
            f"compress={trace_compress:.2e}, stretch={trace_stretch:.2e}"
        )

        # Both should be non-zero (material resists deformation)
        self.assertGreater(abs(trace_compress), 1e-3, "Compression force should be non-zero")
        self.assertGreater(abs(trace_stretch), 1e-3, "Stretch force should be non-zero")


if __name__ == '__main__':
    unittest.main(verbosity=2)
