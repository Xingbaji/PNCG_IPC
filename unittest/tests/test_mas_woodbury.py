"""
Unit tests for MAS Preconditioner Woodbury Update functionality.

Tests:
1. Woodbury structures initialization
2. Contact state save/restore
3. Update detection (new, changed, rotated contacts)
4. Capacitance matrix computation
5. Woodbury solve correctness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
import taichi as ti

# Initialize Taichi once for all tests
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True, offline_cache_file_path='.taichi_cache')


class MockSolver:
    """Mock solver for testing Woodbury updates."""

    def __init__(self, n_verts=128, n_contacts=10):
        self.n_verts = n_verts
        self.dHat = 0.01
        self.kappa = 1e5
        self.dt = 0.01

        # Contact pairs structure matching real solver
        self.contact_pairs = ti.Struct.field({
            'a': ti.types.vector(4, ti.u32),  # 4 vertex IDs
            'b': ti.f32,                       # distance
            'c': ti.types.vector(4, ti.f32),  # barycentric coords
            'd': ti.types.vector(3, ti.f32),  # direction
        }, shape=n_contacts * 2)

        self.n_contacts = ti.field(dtype=ti.i32, shape=())
        self.n_contacts[None] = 0

    def set_contacts(self, contacts_data):
        """Set contact data for testing.

        Args:
            contacts_data: List of dicts with keys 'ids', 'dist', 'cord', 'dir'
        """
        n = len(contacts_data)
        self.n_contacts[None] = n

        for i, c in enumerate(contacts_data):
            self.contact_pairs[i].a = ti.Vector(c['ids'], dt=ti.u32)
            self.contact_pairs[i].b = c['dist']
            self.contact_pairs[i].c = ti.Vector(c['cord'], dt=ti.f32)
            self.contact_pairs[i].d = ti.Vector(c['dir'], dt=ti.f32)


class TestWoodburyInit(unittest.TestCase):
    """Test Woodbury structures initialization."""

    def test_init_woodbury_structures(self):
        """Test that Woodbury structures are properly allocated."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact, BANKSIZE

        # Create a simple mesh for testing
        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)

        # Initialize Woodbury
        precond.init_woodbury()

        # Check that structures are created
        self.assertTrue(hasattr(precond, '_woodbury'))
        self.assertTrue(precond._woodbury.initialized)

        # Check field shapes
        n_blocks = (64 + BANKSIZE - 1) // BANKSIZE
        self.assertEqual(precond._woodbury.n_blocks, n_blocks)
        self.assertEqual(precond._woodbury.top_k, 8)

    def test_woodbury_stats(self):
        """Test Woodbury statistics reporting."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)

        # Before init
        stats = precond.get_woodbury_stats()
        self.assertFalse(stats['initialized'])

        # After init
        precond.init_woodbury()
        stats = precond.get_woodbury_stats()
        self.assertTrue(stats['initialized'])
        self.assertEqual(stats['n_base_contacts'], 0)
        self.assertEqual(stats['top_k'], 8)


class TestContactStateSave(unittest.TestCase):
    """Test contact state save/restore functionality."""

    def test_save_base_contact_state(self):
        """Test saving contact state as baseline."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)
        precond.init_woodbury()

        # Create mock solver with contacts
        solver = MockSolver(n_verts=64, n_contacts=10)
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.005, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
            {'ids': [4, 5, 6, 7], 'dist': 0.008, 'cord': [0.5, 0.5, 0.0, 0.0], 'dir': [1.0, 0.0, 0.0]},
        ])

        # Save base state
        precond.save_base_state(solver)

        # Check that contacts were saved
        stats = precond.get_woodbury_stats()
        self.assertEqual(stats['n_base_contacts'], 2)

    def test_barrier_stiffness_computation(self):
        """Test barrier stiffness computation for different distances."""
        from algorithm.mas_preconditioner_contact.woodbury import WoodburySupport

        # Create minimal preconditioner for testing
        mesh = create_test_mesh(n_verts=64)
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)

        woodbury = WoodburySupport(precond)

        dHat = 0.01
        kappa = 1e5

        # Test at various distances
        # At d >= dHat: stiffness should be 0
        k = woodbury._compute_barrier_stiffness(0.01, dHat, kappa)
        self.assertEqual(k, 0.0)

        k = woodbury._compute_barrier_stiffness(0.02, dHat, kappa)
        self.assertEqual(k, 0.0)

        # At d < dHat: stiffness should be positive
        k = woodbury._compute_barrier_stiffness(0.005, dHat, kappa)
        self.assertGreater(k, 0.0)

        # Closer distance = higher stiffness
        k1 = woodbury._compute_barrier_stiffness(0.005, dHat, kappa)
        k2 = woodbury._compute_barrier_stiffness(0.001, dHat, kappa)
        self.assertGreater(k2, k1)

        # At d <= 0: stiffness should be 0 (invalid)
        k = woodbury._compute_barrier_stiffness(0.0, dHat, kappa)
        self.assertEqual(k, 0.0)

        k = woodbury._compute_barrier_stiffness(-0.001, dHat, kappa)
        self.assertEqual(k, 0.0)


class TestUpdateDetection(unittest.TestCase):
    """Test contact change detection."""

    def test_detect_new_contact(self):
        """Test detection of new contacts."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)
        precond.init_woodbury()

        solver = MockSolver(n_verts=64, n_contacts=10)

        # Initial state: 1 contact
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.005, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
        ])
        precond.save_base_state(solver)

        # Add another contact
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.005, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
            {'ids': [4, 5, 6, 7], 'dist': 0.006, 'cord': [0.5, 0.5, 0.0, 0.0], 'dir': [1.0, 0.0, 0.0]},
        ])

        # Compute updates
        precond.woodbury_update(solver)

        # Should have updates for the new contact
        stats = precond.get_woodbury_stats()
        self.assertGreater(stats['n_updates_total'], 0)

    def test_detect_stiffness_increase(self):
        """Test detection of stiffness increase (distance decrease)."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)
        precond.init_woodbury()

        solver = MockSolver(n_verts=64, n_contacts=10)

        # Initial state: contact at distance 0.008
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.008, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
        ])
        precond.save_base_state(solver)

        # Same contact, closer distance (higher stiffness)
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.004, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
        ])

        # Compute updates
        precond.woodbury_update(solver)

        # Should have updates for the stiffness increase
        stats = precond.get_woodbury_stats()
        self.assertGreater(stats['n_updates_total'], 0)

    def test_should_use_woodbury(self):
        """Test the should_use_woodbury decision logic."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)

        solver = MockSolver(n_verts=64, n_contacts=100)

        # Before initialization
        self.assertFalse(precond.should_use_woodbury(solver))

        precond.init_woodbury()

        # No base contacts yet
        self.assertFalse(precond.should_use_woodbury(solver))

        # Set up base with 10 contacts
        contacts = []
        for i in range(10):
            contacts.append({
                'ids': [i*4, i*4+1, i*4+2, i*4+3],
                'dist': 0.005,
                'cord': [1.0, -0.3, -0.3, -0.4],
                'dir': [0.0, 1.0, 0.0]
            })
        solver.set_contacts(contacts)
        precond.save_base_state(solver)

        # Incremental change (< 50%)
        contacts_new = contacts[:8]  # 20% reduction
        solver.set_contacts(contacts_new)
        self.assertTrue(precond.should_use_woodbury(solver))

        # Large change (> 50%)
        contacts_new = contacts[:4]  # 60% reduction
        solver.set_contacts(contacts_new)
        self.assertFalse(precond.should_use_woodbury(solver))


class TestWoodburyApply(unittest.TestCase):
    """Test Woodbury preconditioner application."""

    def test_apply_with_woodbury_runs(self):
        """Test that apply_with_woodbury runs without errors."""
        from algorithm.mas_preconditioner_contact import MASPreconditionerContact

        mesh = create_test_mesh(n_verts=64)
        precond = MASPreconditionerContact(mesh, max_contacts=100, metis_reordered=True)

        # Need to build hierarchy and assemble first
        precond.build_hierarchy()

        solver = MockSolver(n_verts=64, n_contacts=10)
        solver.set_contacts([
            {'ids': [0, 1, 2, 3], 'dist': 0.005, 'cord': [1.0, -0.3, -0.3, -0.4], 'dir': [0.0, 1.0, 0.0]},
        ])

        # Full rebuild first
        precond.rebuild_with_contacts(solver)
        precond.save_base_state(solver)

        # Add gradient to mesh
        set_test_gradient(mesh)

        # Apply standard preconditioner
        precond.apply()
        z_standard = get_z_from_mesh(mesh)

        # Now apply with Woodbury (no changes, should be similar)
        set_test_gradient(mesh)
        precond.woodbury_update(solver)
        precond.apply_with_woodbury()
        z_woodbury = get_z_from_mesh(mesh)

        # Results should be identical when no actual updates
        np.testing.assert_allclose(z_standard, z_woodbury, rtol=1e-4, atol=1e-6)


def create_test_mesh(n_verts=64):
    """Create a simple test mesh with vertex and cell fields."""
    import meshtaichi_patcher as Patcher

    # Create a simple tet mesh
    n_cells = max(1, n_verts // 4)

    # Generate positions
    positions = np.random.randn(n_verts, 3).astype(np.float32) * 0.1

    # Generate cells (random tets using valid vertex indices)
    cells = []
    for i in range(n_cells):
        indices = np.random.choice(n_verts, 4, replace=False)
        cells.append(indices)
    cells = np.array(cells, dtype=np.int32)

    # Create mesh
    mesh = Patcher.load_mesh_rawdata(
        verts=positions,
        cells=cells.flatten(),
        mesh_type='tet'
    )

    # Place required fields
    mesh.verts.place({
        'x': ti.types.vector(3, float),
        'v': ti.types.vector(3, float),
        'm': float,
        'x_n': ti.types.vector(3, float),
        'x_hat': ti.types.vector(3, float),
        'x_init': ti.types.vector(3, float),
        'grad': ti.types.vector(3, float),
        'z': ti.types.vector(3, float),
    })
    mesh.cells.place({'B': ti.math.mat3, 'W': float})

    # Initialize positions
    mesh.verts.x.from_numpy(positions)

    # Initialize mass
    @ti.kernel
    def init_mass(mesh: ti.template()):
        for v in mesh.verts:
            v.m = 1.0

    init_mass(mesh)

    return mesh


def set_test_gradient(mesh):
    """Set a test gradient on mesh vertices."""
    @ti.kernel
    def set_grad(mesh: ti.template()):
        for v in mesh.verts:
            v.grad = ti.Vector([1.0, 0.5, 0.25])

    set_grad(mesh)


def get_z_from_mesh(mesh):
    """Get preconditioned gradient z from mesh."""
    n_verts = len(mesh.verts)
    z = np.zeros((n_verts, 3), dtype=np.float32)

    @ti.kernel
    def copy_z(mesh: ti.template(), z: ti.types.ndarray()):
        for v in mesh.verts:
            for i in ti.static(range(3)):
                z[v.id, i] = v.z[i]

    copy_z(mesh, z)
    return z


if __name__ == '__main__':
    unittest.main()
