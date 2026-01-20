"""
MAS Stiffness Test Demo

Tests the MAS preconditioner with 8 E-shaped objects having varying stiffness
from 1e4 (stiff) to 1e-7 (extremely soft). This stress tests the MAS
preconditioner's ability to handle mixed stiffness scenarios.

Based on the eight_E_drop_demo_contact configuration.
"""

import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

import taichi as ti
import numpy as np
from algorithm.mas_pncg_solver import MASPNCGSolver
from demo.demo_runner import DemoRunner


@ti.data_oriented
class MASStiffnessTestSolver(MASPNCGSolver):
    """
    MAS-PNCG solver with per-object stiffness support.

    Each of the 8 E-shaped objects has a different Young's modulus:
    - Object 0: E = 1e4 (soft)
    - Object 1: E = 1e4.43 (~2.7e4)
    - Object 2: E = 1e4.86 (~7.2e4)
    - Object 3: E = 1e5.29 (~1.9e5)
    - Object 4: E = 1e5.71 (~5.1e5)
    - Object 5: E = 1e6.14 (~1.4e6)
    - Object 6: E = 1e6.57 (~3.7e6)
    - Object 7: E = 1e7 (very stiff)
    """

    # Per-object Young's modulus values (from 1e4 to 1e7, logarithmically spaced)
    STIFFNESS_VALUES = [1e4, 2.7e4, 7.2e4, 1.9e5, 5.1e5, 1.4e6, 3.7e6, 1e7]

    def __init__(self, demo='eight_E_stiffness_test'):
        # First call parent init with base demo
        super().__init__(demo=demo)

        # Number of vertices per E object (from the mesh)
        self.object_size = 1046
        self.N_objects = len(self.STIFFNESS_VALUES)

        # Per-vertex and per-cell material properties
        self.mesh.verts.place({'object_id': ti.i32})
        self.mesh.cells.place({'mu': ti.f32, 'la': ti.f32})

        # Per-vertex color for visualization
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)

        # Initialize object IDs and material properties
        self._init_per_object_properties()

        print(f"MAS Stiffness Test: {self.N_objects} objects with stiffness range "
              f"[{min(self.STIFFNESS_VALUES):.1e}, {max(self.STIFFNESS_VALUES):.1e}]")

    def _init_per_object_properties(self):
        """Initialize per-object material properties and colors."""
        nu = 0.4  # Poisson's ratio (same for all)

        # Compute per-object Lame parameters
        mu_values = []
        la_values = []
        for E in self.STIFFNESS_VALUES:
            mu = E / (2.0 * (1.0 + nu))
            la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu_values.append(mu)
            la_values.append(la)

        self.mu_array = ti.field(dtype=ti.f32, shape=self.N_objects)
        self.la_array = ti.field(dtype=ti.f32, shape=self.N_objects)
        self.mu_array.from_numpy(np.array(mu_values, dtype=np.float32))
        self.la_array.from_numpy(np.array(la_values, dtype=np.float32))

        # Generate colors: blue (soft, 1e4) -> red (stiff, 1e7)
        colors = []
        for i in range(self.N_objects):
            t = i / (self.N_objects - 1)  # 0 to 1
            r = t          # More red = stiffer
            g = 0.2
            b = 1.0 - t    # More blue = softer
            colors.append([r, g, b])
        self.color_array = ti.Vector.field(3, dtype=ti.f32, shape=self.N_objects)
        self.color_array.from_numpy(np.array(colors, dtype=np.float32))

        # Assign object IDs and properties
        self._assign_object_ids()
        self._assign_cell_materials()
        self._assign_vertex_colors()

    @ti.kernel
    def _assign_object_ids(self):
        """Assign object ID to each vertex based on vertex index."""
        for vert in self.mesh.verts:
            obj_id = vert.id // self.object_size
            if obj_id >= self.N_objects:
                obj_id = self.N_objects - 1
            vert.object_id = obj_id

    @ti.kernel
    def _assign_cell_materials(self):
        """Assign Lame parameters to each cell based on first vertex's object ID."""
        for cell in self.mesh.cells:
            obj_id = cell.verts[0].object_id
            cell.mu = self.mu_array[obj_id]
            cell.la = self.la_array[obj_id]

    @ti.kernel
    def _assign_vertex_colors(self):
        """Assign per-vertex colors based on object ID."""
        for vert in self.mesh.verts:
            obj_id = vert.object_id
            self.per_vertex_color[vert.id] = self.color_array[obj_id]

    # Override elastic energy computation to use per-cell material properties
    @ti.kernel
    def compute_elastic_potential(self) -> ti.f32:
        """Compute total elastic potential energy using per-cell materials."""
        energy = 0.0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            # Use per-cell material properties
            Psi = self.compute_Psi(F, c.mu, c.la)
            energy += c.W * Psi
        return energy

    @ti.kernel
    def compute_grad_elastic(self):
        """Compute elastic gradient using per-cell materials."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = self.dt ** 2 * c.W
            # Use per-cell material properties
            dPsidx = para * self.compute_dPsidx(F, B, c.mu, c.la)
            for i in ti.static(range(4)):
                c.verts[i].grad += ti.Vector([dPsidx[i * 3], dPsidx[i * 3 + 1], dPsidx[i * 3 + 2]])

    @ti.kernel
    def compute_diagH_elastic(self):
        """Compute diagonal Hessian approximation using per-cell materials."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = self.dt ** 2 * c.W
            # Use per-cell material properties
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, c.mu, c.la)
            for i in ti.static(range(4)):
                c.verts[i].diagH += ti.Vector([diagH_d2Psidx2[i * 3],
                                               diagH_d2Psidx2[i * 3 + 1],
                                               diagH_d2Psidx2[i * 3 + 2]])

    @ti.kernel
    def compute_pHp_elastic(self) -> ti.f64:
        """Compute p^T H p for elastic term using per-cell materials."""
        pHp = 0.0
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            p = ti.Vector.zero(float, 12)
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    p[i * 3 + j] = c.verts[i].p[j]
            # Use per-cell material properties
            tmp = self.compute_p_d2Psidx2_p(F, B, p, c.mu, c.la)
            pHp += c.W * self.dt ** 2 * ti.max(tmp, 0.0)
        return pHp


class MASStiffnessTestDemo(DemoRunner):
    """Demo runner for MAS stiffness test."""

    def __init__(self, demo='eight_E_stiffness_test'):
        solver = MASStiffnessTestSolver(demo=demo)
        super().__init__(solver, demo_name=f"MAS Stiffness Test ({demo})")

    def get_per_vertex_color(self):
        """Return per-vertex colors for stiffness visualization."""
        return self.solver.per_vertex_color

    def setup(self):
        """Print stiffness information."""
        print("\nPer-object stiffness (Young's modulus E):")
        for i, E in enumerate(self.solver.STIFFNESS_VALUES):
            print(f"  Object {i}: E = {E:.1e}")
        print("\nColor legend: Blue = Soft (1e4), Red = Stiff (1e7)\n")


def add_stiffness_test_config():
    """Add the stiffness test demo configuration to model_loading."""
    # This demo configuration will be used by model_loading
    pass


def get_mas_test_demos():
    """Get list of available MAS test demos from YAML configs."""
    try:
        from demo_settings import list_demos
        demos = list_demos(by_category=True)
        return demos.get('mas_test', [])
    except ImportError:
        return ['eight_E_stiffness_test', 'eight_E_stiffness_mas']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MAS Stiffness Test Demo')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_test',
                        help='Demo name (default: eight_E_stiffness_test)')
    parser.add_argument('--list', action='store_true', help='List available MAS test demos')
    args, _ = parser.parse_known_args()

    if args.list:
        print("Available MAS test demos:")
        for demo in get_mas_test_demos():
            print(f"  - {demo}")
        sys.exit(0)

    ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

    demo = MASStiffnessTestDemo(demo=args.demo)
    demo.run()
