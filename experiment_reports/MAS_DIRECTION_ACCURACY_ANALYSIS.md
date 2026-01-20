# MAS Preconditioner Direction Accuracy Analysis

**Date**: 2026-01-20
**Status**: Completed
**Related Files**:
- `algorithm/mas_preconditioner_small/core.py`
- `unittest/tests/test_precond_direction.py`

---

## Summary

This report documents the analysis of MAS preconditioner direction accuracy, comparing the preconditioned direction `z_mas = P @ g` against the ground truth `z_exact = H^{-1} @ g`.

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Direction quality (cos) | 0.81 | Good alignment with exact solution |
| Amplitude ratio after alpha scaling | 0.40 | Conservative step size |
| Is descent direction | Yes | g^T (alpha*z) > 0 |

### Conclusion

The MAS-Small preconditioner produces a valid descent direction with good quality (cos = 0.81). The amplitude is controlled by line search parameter alpha, which is computed using the ground truth Hessian.

---

## 1. Problem Statement

### 1.1 Goal
Verify that the MAS preconditioner produces a valid descent direction for PNCG optimization:
- `z = P @ g` should satisfy `g^T z > 0` (descent direction)
- Direction should align well with exact solution `z_exact = H^{-1} @ g`

### 1.2 Test Configuration
- Demo: `cube_freefall_10` (491 vertices, 1736 cells)
- Elastic type: ARAP
- Initial velocity: vy = -1.0
- MAS implementation: `mas_preconditioner_small` (IC(0) inversion)

---

## 2. Experimental Results

### 2.1 Direction Quality Comparison

| Preconditioner | cos(z, z_exact) | Status |
|----------------|-----------------|--------|
| MAS-Small | 0.81 | Good |
| MAS-Pkg | 0.21 | Poor |
| Diagonal | 0.91 | Excellent |

**Observation**: `mas_preconditioner_small` achieves much better direction quality than `mas_preconditioner_pkg`.

### 2.2 Amplitude Analysis

Initial concern: `|z_mas| = 2.60` vs `|z_exact| = 0.24` (10x larger)

**Resolution**: In PNCG, amplitude is controlled by line search coefficient alpha:
```
alpha = (g^T z) / (z^T H z)
```

Using ground truth Hessian for H:
```
g^T z = 1.72e-01
z^T H z = 4.54e+00
alpha = 0.0378
|alpha * z| = 0.098
|z_exact| = 0.243
Ratio = 0.40
```

### 2.3 Descent Direction Verification
```
g^T (alpha * z) = 6.48e-03 > 0  ✓
```
The scaled direction IS a valid descent direction.

---

## 3. Technical Details

### 3.1 Inertia Bug Fix

**Bug**: Original code used `m/dt²` for inertia Hessian
**Fix**: Changed to `m` to match gradient scaling convention

```python
# Before (incorrect)
mass_val = m / (dt * dt)

# After (correct)
mass_val = m
```

**Reason**: The gradient is computed as `g = m * (x - x_hat)`, not `g = m/dt² * (x - x_hat)`.

### 3.2 MAS Block Matrices vs Ground Truth Hessian

**Important**: The MAS block matrices are NOT equal to the ground truth Hessian.

| Aspect | Ground Truth H | MAS Block Matrices |
|--------|----------------|-------------------|
| Structure | Full sparse matrix | Block-diagonal per level |
| Cross-block coupling | Direct entries | Stored in coarse levels |
| Matvec result | Exact H @ v | Approximate (100x different) |

**Implication**: The `hessian_matvec` function in MAS should NOT be used for computing PNCG alpha. Use the solver's actual Hessian-vector product instead.

### 3.3 Full Block vs Banded Solve

Tested using full 16x16 block inverse vs banded (bandwidth=6) approximation:
- Both produce similar direction quality (cos ≈ 0.81)
- IC(0) banded solve is confirmed correct

---

## 4. Comparison: MAS-Small vs MAS-Pkg

| Feature | MAS-Small | MAS-Pkg |
|---------|-----------|---------|
| Direction quality | cos = 0.81 | cos = 0.21 |
| Implementation | Single file | Modular (mixins) |
| Inversion | IC(0) banded | IC with regularization |
| Status | Correct | Has issues |

**Root cause of MAS-Pkg issues**: The modular implementation has additional complexity that may introduce bugs. Further investigation needed.

---

## 5. Recommendations

1. **Use MAS-Small for now**: Better direction quality and simpler implementation
2. **Don't use MAS hessian_matvec for alpha**: Use ground truth H instead
3. **Amplitude ratio of 0.4 is acceptable**: Conservative but will converge
4. **Investigate MAS-Pkg later**: Focus on getting simulation working first

---

## 6. Test Commands

```bash
# Run direction accuracy test
cd /root/PNCG_IPC/unittest/tests
python test_precond_direction.py --demo cube_freefall_10 --verbose

# Run all unit tests
cd /root/PNCG_IPC/unittest
python run_all_tests.py
```

---

## 7. Related Documentation

- [MAS Implementation](../docs/algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md)
- [Symmetry Bug Analysis](./MAS_SYMMETRY_BUG_ANALYSIS.md)
- [Test Report](../unittest/TEST_REPORT.md)
