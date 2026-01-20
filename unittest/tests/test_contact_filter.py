"""
Contact Filter Unit Tests

Tests for the ContactFilter module that implements two-radius collision detection.
"""

import sys
import os
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
unittest_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(unittest_dir)
sys.path.insert(0, project_root)
demo_dir = os.path.join(project_root, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32)

from algorithm.contact_filter import ContactFilter


def test_configuration():
    """Test ContactFilter configuration."""
    print("\n=== Test: Configuration ===")

    filter = ContactFilter(max_contacts=1000)

    # Configure with dHat=0.01, multiplier=5.0
    dHat = 0.01
    multiplier = 5.0
    detection_dHat = filter.configure(dHat, multiplier)

    assert abs(detection_dHat - dHat * multiplier) < 1e-6, \
        f"Expected detection_dHat={dHat * multiplier}, got {detection_dHat}"
    assert abs(filter.get_detection_dHat() - detection_dHat) < 1e-6
    assert abs(filter.get_active_dHat() - dHat) < 1e-6

    print(f"  dHat={dHat}, multiplier={multiplier}")
    print(f"  detection_dHat={filter.get_detection_dHat()}")
    print(f"  active_dHat={filter.get_active_dHat()}")
    print("  PASSED")


def test_filter_basic():
    """Test basic filtering: contacts within active_dHat are kept."""
    print("\n=== Test: Basic Filtering ===")

    # Create filter
    max_contacts = 100
    filter = ContactFilter(max_contacts=max_contacts)

    # Configure: active_dHat=0.01, detection_dHat=0.05
    dHat = 0.01
    filter.configure(dHat, multiplier=5.0)

    # Create mock contact pairs with various distances
    pair_type = ti.types.struct(
        a=ti.types.vector(4, ti.u32),
        b=float,
        c=ti.types.vector(4, float),
        d=ti.types.vector(3, float)
    )
    cached_contacts = pair_type.field(shape=max_contacts)

    # Populate with test data
    # Distances: 0.005 (within), 0.008 (within), 0.015 (outside), 0.025 (outside), 0.04 (outside)
    test_distances = [0.005, 0.008, 0.015, 0.025, 0.04]
    n_cached = len(test_distances)

    @ti.kernel
    def populate_test_contacts(contacts: ti.template(), distances: ti.types.ndarray()):
        for i in range(distances.shape[0]):
            contacts[i] = pair_type(
                ti.Vector([ti.u32(i), ti.u32(i+1), ti.u32(i+2), ti.u32(i+3)]),
                distances[i],
                ti.Vector([1.0, 0.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0])
            )

    distances_np = np.array(test_distances, dtype=np.float32)
    populate_test_contacts(cached_contacts, distances_np)

    # Filter
    filter.filter_contacts(cached_contacts, n_cached)

    # Check results
    stats = filter.get_stats()
    print(f"  Test distances: {test_distances}")
    print(f"  active_dHat: {dHat}")
    print(f"  n_cached: {stats['n_cached']}")
    print(f"  n_filtered: {stats['n_filtered']}")

    # Expect 2 contacts (0.005 and 0.008 are < 0.01)
    expected_filtered = sum(1 for d in test_distances if d < dHat)
    assert stats['n_filtered'] == expected_filtered, \
        f"Expected {expected_filtered} filtered contacts, got {stats['n_filtered']}"

    print(f"  Expected {expected_filtered} contacts within dHat={dHat}")
    print("  PASSED")


def test_filter_empty():
    """Test filtering with no contacts within threshold."""
    print("\n=== Test: Empty Filter Result ===")

    filter = ContactFilter(max_contacts=100)
    filter.configure(0.01, multiplier=5.0)

    pair_type = ti.types.struct(
        a=ti.types.vector(4, ti.u32),
        b=float,
        c=ti.types.vector(4, float),
        d=ti.types.vector(3, float)
    )
    cached_contacts = pair_type.field(shape=100)

    # All distances above threshold
    test_distances = [0.02, 0.03, 0.04]

    @ti.kernel
    def populate(contacts: ti.template(), distances: ti.types.ndarray()):
        for i in range(distances.shape[0]):
            contacts[i] = pair_type(
                ti.Vector([ti.u32(i), ti.u32(i+1), ti.u32(i+2), ti.u32(i+3)]),
                distances[i],
                ti.Vector([1.0, 0.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0])
            )

    distances_np = np.array(test_distances, dtype=np.float32)
    populate(cached_contacts, distances_np)

    filter.filter_contacts(cached_contacts, len(test_distances))

    stats = filter.get_stats()
    assert stats['n_filtered'] == 0, f"Expected 0 filtered, got {stats['n_filtered']}"

    print(f"  All distances > dHat: {test_distances}")
    print(f"  n_filtered: {stats['n_filtered']}")
    print("  PASSED")


def test_filter_all_pass():
    """Test filtering when all contacts are within threshold."""
    print("\n=== Test: All Contacts Pass Filter ===")

    filter = ContactFilter(max_contacts=100)
    filter.configure(0.05, multiplier=5.0)  # Larger threshold

    pair_type = ti.types.struct(
        a=ti.types.vector(4, ti.u32),
        b=float,
        c=ti.types.vector(4, float),
        d=ti.types.vector(3, float)
    )
    cached_contacts = pair_type.field(shape=100)

    # All distances within threshold
    test_distances = [0.01, 0.02, 0.03, 0.04]

    @ti.kernel
    def populate(contacts: ti.template(), distances: ti.types.ndarray()):
        for i in range(distances.shape[0]):
            contacts[i] = pair_type(
                ti.Vector([ti.u32(i), ti.u32(i+1), ti.u32(i+2), ti.u32(i+3)]),
                distances[i],
                ti.Vector([1.0, 0.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0])
            )

    distances_np = np.array(test_distances, dtype=np.float32)
    populate(cached_contacts, distances_np)

    filter.filter_contacts(cached_contacts, len(test_distances))

    stats = filter.get_stats()
    assert stats['n_filtered'] == len(test_distances), \
        f"Expected {len(test_distances)} filtered, got {stats['n_filtered']}"

    print(f"  All distances < dHat: {test_distances}")
    print(f"  n_filtered: {stats['n_filtered']}")
    print("  PASSED")


def test_filter_preserves_data():
    """Test that filtered contacts preserve original data."""
    print("\n=== Test: Data Preservation ===")

    filter = ContactFilter(max_contacts=100)
    filter.configure(0.02, multiplier=5.0)

    pair_type = ti.types.struct(
        a=ti.types.vector(4, ti.u32),
        b=float,
        c=ti.types.vector(4, float),
        d=ti.types.vector(3, float)
    )
    cached_contacts = pair_type.field(shape=100)

    # Create specific test contact
    @ti.kernel
    def populate_specific(contacts: ti.template()):
        contacts[0] = pair_type(
            ti.Vector([ti.u32(10), ti.u32(20), ti.u32(30), ti.u32(40)]),
            0.015,  # Within threshold
            ti.Vector([0.5, -0.2, -0.2, -0.1]),
            ti.Vector([0.577, 0.577, 0.577])
        )
        contacts[1] = pair_type(
            ti.Vector([ti.u32(100), ti.u32(200), ti.u32(300), ti.u32(400)]),
            0.03,  # Outside threshold
            ti.Vector([1.0, 0.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0])
        )

    populate_specific(cached_contacts)

    filter.filter_contacts(cached_contacts, 2)

    # Verify filtered contact preserves data
    @ti.kernel
    def verify_data(filtered: ti.template()) -> ti.i32:
        passed = 1
        pair = filtered[0]
        if pair.a[0] != 10 or pair.a[1] != 20 or pair.a[2] != 30 or pair.a[3] != 40:
            passed = 0
        if ti.abs(pair.b - 0.015) > 1e-6:
            passed = 0
        if ti.abs(pair.c[0] - 0.5) > 1e-6:
            passed = 0
        return passed

    passed = verify_data(filter.filtered_contacts)
    assert passed == 1, "Filtered contact data mismatch"

    print("  Filtered contact preserves vertex IDs, distance, and coordinates")
    print("  PASSED")


def test_stats():
    """Test statistics reporting."""
    print("\n=== Test: Statistics ===")

    filter = ContactFilter(max_contacts=100)
    filter.configure(0.01, multiplier=5.0)

    pair_type = ti.types.struct(
        a=ti.types.vector(4, ti.u32),
        b=float,
        c=ti.types.vector(4, float),
        d=ti.types.vector(3, float)
    )
    cached_contacts = pair_type.field(shape=100)

    # 2 within, 3 outside
    test_distances = [0.005, 0.008, 0.015, 0.025, 0.04]

    @ti.kernel
    def populate(contacts: ti.template(), distances: ti.types.ndarray()):
        for i in range(distances.shape[0]):
            contacts[i] = pair_type(
                ti.Vector([ti.u32(i), ti.u32(i+1), ti.u32(i+2), ti.u32(i+3)]),
                distances[i],
                ti.Vector([1.0, 0.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0])
            )

    distances_np = np.array(test_distances, dtype=np.float32)
    populate(cached_contacts, distances_np)

    filter.filter_contacts(cached_contacts, len(test_distances))

    stats = filter.get_stats()
    expected_ratio = 2 / 5

    assert stats['n_cached'] == 5
    assert stats['n_filtered'] == 2
    assert abs(stats['filter_ratio'] - expected_ratio) < 1e-6

    print(f"  n_cached: {stats['n_cached']}")
    print(f"  n_filtered: {stats['n_filtered']}")
    print(f"  filter_ratio: {stats['filter_ratio']:.2%}")
    filter.print_stats()
    print("  PASSED")


def run_all_tests():
    """Run all contact filter tests."""
    print("=" * 60)
    print("Contact Filter Unit Tests")
    print("=" * 60)

    test_configuration()
    test_filter_basic()
    test_filter_empty()
    test_filter_all_pass()
    test_filter_preserves_data()
    test_stats()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
