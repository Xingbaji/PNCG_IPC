#!/usr/bin/env python
"""
Run all MAS unit tests and generate a report.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# Setup paths - go to the PNCG_IPC root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEMO_DIR = os.path.join(PROJECT_ROOT, 'demo')
N_E_DEMOS_DIR = os.path.dirname(SCRIPT_DIR)

# Test categories
TESTS = {
    "Core Test Suites": [
        ("test_mas_ground_truth.py", "Ground Truth Tests"),
        ("test_mas_multilevel.py", "Multi-Level Tests"),
    ],
    "Functional Validation": [
        ("test_mas_simple.py", "Simple MAS Test"),
        ("test_mas_freefall.py", "Freefall Validation"),
        ("test_mas_matrix_diagnostic.py", "Matrix Diagnostics"),
    ],
    "Assembly Tests": [
        ("test_assembly_detail.py", "Assembly Detail"),
        ("test_assembly_logic.py", "Assembly Logic"),
        ("test_assembly_precise.py", "Precise Assembly"),
    ],
    "Symmetry Tests": [
        ("test_He_symmetry.py", "Element Hessian Symmetry"),
        ("test_He_symmetry_simple.py", "Hessian Symmetry (NumPy)"),
    ],
    "Hierarchy Tests": [
        ("test_hierarchy_mapping.py", "Hierarchy Mapping"),
        ("test_level0_only.py", "Level 0 Isolation"),
        ("test_crosswarp_issue.py", "Cross-Warp Issues"),
    ],
    "Specific Issue Tests": [
        ("test_diagonal_contrib.py", "Diagonal Contributions"),
        ("test_diagonal_simple.py", "Diagonal Simple"),
        ("test_debug_simple.py", "Debug Simple"),
        ("test_nonopt_kernel.py", "Non-Optimized Kernel"),
        ("test_upper_triangle_bug.py", "Upper Triangle Bug"),
    ],
}

DEBUG_SCRIPTS = [
    ("debug_assembly_logic.py", "Assembly Logic Debug"),
    ("debug_block0_detailed.py", "Block0 Detailed Debug"),
    ("debug_mas_gTz.py", "gTz Debug"),
    ("debug_mas_nan.py", "NaN Debug"),
    ("debug_sym_expand.py", "Symmetric Expand Debug"),
]


def run_test(test_file, test_dir, timeout=120):
    """Run a single test file and return result."""
    test_path = os.path.join(test_dir, test_file)
    if not os.path.exists(test_path):
        return "NOT_FOUND", 0, f"File not found: {test_path}"

    start = time.time()
    try:
        # Run from demo directory for proper imports
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=DEMO_DIR,
            env={**os.environ, 'PYTHONPATH': f"{PROJECT_ROOT}:{DEMO_DIR}:{N_E_DEMOS_DIR}"}
        )
        elapsed = time.time() - start

        output = result.stdout + result.stderr

        # Parse result
        if result.returncode == 0:
            # Check for test summary
            if "OK" in output or "PASSED" in output.upper():
                # Extract test count if available
                import re
                match = re.search(r'Ran (\d+) test', output)
                if match:
                    return "PASS", elapsed, f"{match.group(1)} tests passed"
                return "PASS", elapsed, "All tests passed"
            return "PASS", elapsed, "Completed successfully"
        else:
            # Extract error message
            lines = output.strip().split('\n')
            error_lines = [l for l in lines[-10:] if l.strip()]
            error_msg = error_lines[-1] if error_lines else "Unknown error"
            if "FAILED" in output:
                match = re.search(r'FAILED.*\(failures?=(\d+)', output)
                if match:
                    return "FAIL", elapsed, f"{match.group(1)} tests failed"
            return "FAIL", elapsed, error_msg[:100]

    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, f"Timeout after {timeout}s"
    except Exception as e:
        return "ERROR", 0, str(e)[:100]


def main():
    print("=" * 70)
    print("MAS Preconditioner Unit Test Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    results = {
        "PASS": 0,
        "FAIL": 0,
        "ERROR": 0,
        "TIMEOUT": 0,
        "NOT_FOUND": 0,
        "SKIPPED": 0,
    }

    test_dir = os.path.join(SCRIPT_DIR, "tests")
    debug_dir = os.path.join(SCRIPT_DIR, "debug")

    all_results = []

    # Run tests by category
    for category, tests in TESTS.items():
        print(f"\n## {category}")
        print("-" * 50)

        for test_file, description in tests:
            status, elapsed, msg = run_test(test_file, test_dir)
            results[status] += 1

            status_icon = {
                "PASS": "[PASS]",
                "FAIL": "[FAIL]",
                "ERROR": "[ERR ]",
                "TIMEOUT": "[TIME]",
                "NOT_FOUND": "[N/A ]",
                "SKIPPED": "[SKIP]",
            }.get(status, "[????]")

            print(f"  {status_icon} {description:<35} ({elapsed:.1f}s)")
            if status != "PASS":
                print(f"         -> {msg}")

            all_results.append({
                "category": category,
                "file": test_file,
                "description": description,
                "status": status,
                "time": elapsed,
                "message": msg
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = sum(results.values())
    print(f"  Total:    {total}")
    print(f"  Passed:   {results['PASS']}")
    print(f"  Failed:   {results['FAIL']}")
    print(f"  Errors:   {results['ERROR']}")
    print(f"  Timeout:  {results['TIMEOUT']}")
    print(f"  Not Found: {results['NOT_FOUND']}")

    success_rate = results['PASS'] / total * 100 if total > 0 else 0
    print(f"\n  Success Rate: {success_rate:.1f}%")
    print("=" * 70)

    return 0 if results['FAIL'] == 0 and results['ERROR'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
