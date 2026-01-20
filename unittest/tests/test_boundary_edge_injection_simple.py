#!/usr/bin/env python3
"""
Simple test: Can we inject boundary edges into mesh loading?
"""

import sys
sys.path.insert(0, '/root/PNCG_IPC')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import numpy as np
import meshtaichi_patcher as Patcher


def main():
    print("Test: Boundary Edge Injection")
    print("="*50)

    # Load raw mesh data
    model_path = "/root/PNCG_IPC/model/cube"
    raw_data = Patcher.load_mesh_rawdata(model_path + ".node")

    vertices = raw_data[0]
    cells = raw_data[3]
    print(f"Vertices: {vertices.shape[0]}, Cells: {cells.shape[0]}")

    # Create some fake "boundary edges" for testing
    # Just take a small subset of edges manually
    # From a tetrahedron: edges are pairs of cell vertices
    boundary_edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        [2, 3],
    ], dtype=np.int32)

    print(f"Injecting {len(boundary_edges)} edges")

    # Create mesh dict with boundary edges
    mesh_dict = {
        0: vertices,
        1: boundary_edges,
        3: cells
    }

    print("Loading mesh with injected edges...")
    mesh = Patcher.load_mesh(mesh_dict, relations=['CV', 'EV'])

    n_loaded_edges = len(mesh.edges)
    print(f"Loaded edges: {n_loaded_edges}")

    # Did it work?
    if n_loaded_edges > 0:
        print("SUCCESS: Mesh has edges!")

        # Quick test of mesh-for
        mesh.verts.place({'x': ti.math.vec3})
        mesh.verts.x.from_numpy(vertices)

        @ti.kernel
        def test_meshfor() -> ti.f32:
            total = 0.0
            for e in mesh.edges:
                v0 = e.verts[0]
                v1 = e.verts[1]
                total += (v0.x - v1.x).norm()
            return total

        result = test_meshfor()
        print(f"Mesh-for result: {result:.4f}")
    else:
        print("FAILED: No edges loaded")


if __name__ == '__main__':
    main()
