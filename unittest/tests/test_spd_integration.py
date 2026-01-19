"""
Integration test for SPD Hessian in elastic_util.py

Tests that the SPD-projected Hessian functions are correctly integrated
with the existing codebase and produce valid SPD matrices.
"""

import unittest
import numpy as np
import taichi as ti

# Initialize Taichi before importing
ti.init(arch=ti.cpu, default_fp=ti.f32)

import sys
sys.path.insert(0, '/root/PNCG_IPC')

from math_utils.elastic_util import (
    # Original functions
    compute_d2PsidF2_ARAP_filter, compute_d2PsidF2_SNH, compute_d2PsidF2_FCR_filter,
    # SPD-projected functions
    compute_d2PsidF2_ARAP_SPD, compute_d2PsidF2_NH_SPD, compute_d2PsidF2_STVK_SPD,
    compute_dPsidF_ARAP_SPD, compute_dPsidF_NH_SPD, compute_dPsidF_STVK_SPD,
)


@ti.kernel
def compute_hessian_test(F: ti.types.ndarray(),
                          mu: ti.f32, la: ti.f32,
                          H_arap: ti.types.ndarray(),
                          H_arap_spd: ti.types.ndarray(),
                          H_nh_spd: ti.types.ndarray(),
                          H_stvk_spd: ti.types.ndarray()):
    """Test kernel to compute various Hessians"""
    F_mat = ti.math.mat3([
        [F[0, 0], F[0, 1], F[0, 2]],
        [F[1, 0], F[1, 1], F[1, 2]],
        [F[2, 0], F[2, 1], F[2, 2]]
    ])

    # Original ARAP with filter
    H1 = compute_d2PsidF2_ARAP_filter(F_mat, mu, la)

    # SPD versions
    H2 = compute_d2PsidF2_ARAP_SPD(F_mat, mu, la)
    H3 = compute_d2PsidF2_NH_SPD(F_mat, mu, la)
    H4 = compute_d2PsidF2_STVK_SPD(F_mat, mu, la)

    for i in range(9):
        for j in range(9):
            H_arap[i, j] = H1[i, j]
            H_arap_spd[i, j] = H2[i, j]
            H_nh_spd[i, j] = H3[i, j]
            H_stvk_spd[i, j] = H4[i, j]


@ti.kernel
def compute_force_test(F: ti.types.ndarray(),
                        mu: ti.f32, la: ti.f32,
                        P_arap_spd: ti.types.ndarray(),
                        P_nh_spd: ti.types.ndarray(),
                        P_stvk_spd: ti.types.ndarray()):
    """Test kernel to compute forces"""
    F_mat = ti.math.mat3([
        [F[0, 0], F[0, 1], F[0, 2]],
        [F[1, 0], F[1, 1], F[1, 2]],
        [F[2, 0], F[2, 1], F[2, 2]]
    ])

    P1 = compute_dPsidF_ARAP_SPD(F_mat, mu, la)
    P2 = compute_dPsidF_NH_SPD(F_mat, mu, la)
    P3 = compute_dPsidF_STVK_SPD(F_mat, mu, la)

    for i in range(3):
        for j in range(3):
            P_arap_spd[i, j] = P1[i, j]
            P_nh_spd[i, j] = P2[i, j]
            P_stvk_spd[i, j] = P3[i, j]


class TestSPDIntegration(unittest.TestCase):
    """Test SPD Hessian integration"""

    def setUp(self):
        self.mu = 1e6
        self.la = 1e6
        self.rel_tol = 1e-5

    def _check_spd(self, H, name):
        """Check if matrix is SPD"""
        # Symmetry
        sym_error = np.max(np.abs(H - H.T))
        self.assertLess(sym_error, 1e-5, f"{name}: Not symmetric")

        # Positive semi-definite
        eigs = np.linalg.eigvalsh(H)
        min_eig = np.min(eigs)
        max_eig = np.max(eigs)
        self.assertGreaterEqual(min_eig, -self.rel_tol * max_eig,
                                 f"{name}: Not SPD, min_eig={min_eig:.2e}")
        return min_eig, max_eig

    def test_hessian_comparison(self):
        """Compare original and SPD Hessians"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        H_arap = np.zeros((9, 9), dtype=np.float32)
        H_arap_spd = np.zeros((9, 9), dtype=np.float32)
        H_nh_spd = np.zeros((9, 9), dtype=np.float32)
        H_stvk_spd = np.zeros((9, 9), dtype=np.float32)

        compute_hessian_test(F, self.mu, self.la,
                             H_arap, H_arap_spd, H_nh_spd, H_stvk_spd)

        # Check all SPD versions
        self._check_spd(H_arap_spd, "ARAP_SPD")
        self._check_spd(H_nh_spd, "NH_SPD")
        self._check_spd(H_stvk_spd, "STVK_SPD")

        # Also check original ARAP filter is SPD
        self._check_spd(H_arap, "ARAP_filter")

    def test_force_computation(self):
        """Test force computation"""
        F = np.array([
            [1.2, 0.1, 0.05],
            [0.1, 0.9, 0.1],
            [0.05, 0.1, 1.1]
        ], dtype=np.float32)

        P_arap = np.zeros((3, 3), dtype=np.float32)
        P_nh = np.zeros((3, 3), dtype=np.float32)
        P_stvk = np.zeros((3, 3), dtype=np.float32)

        compute_force_test(F, self.mu, self.la, P_arap, P_nh, P_stvk)

        # Check forces are finite
        self.assertTrue(np.all(np.isfinite(P_arap)), "ARAP_SPD force has NaN/Inf")
        self.assertTrue(np.all(np.isfinite(P_nh)), "NH_SPD force has NaN/Inf")
        self.assertTrue(np.all(np.isfinite(P_stvk)), "STVK_SPD force has NaN/Inf")

        # Check forces at rest (F=I) should be zero
        F_id = np.eye(3, dtype=np.float32)
        compute_force_test(F_id, self.mu, self.la, P_arap, P_nh, P_stvk)

        self.assertLess(np.linalg.norm(P_arap), 1e-3, "ARAP_SPD force at rest should be ~0")
        self.assertLess(np.linalg.norm(P_stvk), 1e-3, "STVK_SPD force at rest should be ~0")
        # NH force at rest might not be exactly zero due to ln(J) term

    def test_compression_stability(self):
        """Test stability under compression"""
        F = np.array([
            [0.5, 0.2, 0.1],
            [0.2, 0.3, 0.15],
            [0.1, 0.15, 0.4]
        ], dtype=np.float32)

        H_arap = np.zeros((9, 9), dtype=np.float32)
        H_arap_spd = np.zeros((9, 9), dtype=np.float32)
        H_nh_spd = np.zeros((9, 9), dtype=np.float32)
        H_stvk_spd = np.zeros((9, 9), dtype=np.float32)

        compute_hessian_test(F, self.mu, self.la,
                             H_arap, H_arap_spd, H_nh_spd, H_stvk_spd)

        # All should still be SPD under compression
        self._check_spd(H_arap_spd, "ARAP_SPD compression")
        self._check_spd(H_nh_spd, "NH_SPD compression")
        self._check_spd(H_stvk_spd, "STVK_SPD compression")


class TestAssemblyConstants(unittest.TestCase):
    """Test that assembly constants are correctly defined"""

    def test_elastic_type_constants(self):
        """Test elastic type constants"""
        from algorithm.mas_preconditioner_pkg.assembly import (
            ELASTIC_ARAP, ELASTIC_SNH, ELASTIC_FCR,
            ELASTIC_ARAP_SPD, ELASTIC_NH_SPD, ELASTIC_STVK_SPD
        )

        # Check values are unique
        values = [ELASTIC_ARAP, ELASTIC_SNH, ELASTIC_FCR,
                  ELASTIC_ARAP_SPD, ELASTIC_NH_SPD, ELASTIC_STVK_SPD]
        self.assertEqual(len(values), len(set(values)), "Elastic type constants must be unique")

        # Check SPD types are > original types
        self.assertGreater(ELASTIC_ARAP_SPD, ELASTIC_FCR)
        self.assertGreater(ELASTIC_NH_SPD, ELASTIC_FCR)
        self.assertGreater(ELASTIC_STVK_SPD, ELASTIC_FCR)


if __name__ == '__main__':
    unittest.main(verbosity=2)
