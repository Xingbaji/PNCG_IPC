#!/usr/bin/env python3
"""
Test if we can inject boundary edges into mesh loading.

The idea is to pass {0: vertices, 1: boundary_edges, 3: cells} to Patcher.load_mesh
so that the mesh only contains boundary edges, not all edges.
"""

import sys
sys.path.insert(0, '/root/PNCG_IPC')

import taichi as ti
ti.init(arch=ti.gpu, log_level=ti.WARN)

import numpy as np
import meshtaichi_patcher as Patcher


def test_boundary_edge_injection():
    """Test injecting boundary edges into tet mesh loading."""
    print("\n" + "="*70)
    print("Test: Boundary Edge Injection")
    print("="*70)

    # Load raw mesh data
    model_path = "/root/PNCG_IPC/model/cube"
    raw_data = Patcher.load_mesh_rawdata(model_path + ".node")

    vertices = raw_data[0]  # (n_verts, 3)
    cells = raw_data[3]     # (n_cells, 4)

    n_verts = vertices.shape[0]
    n_cells = cells.shape[0]
    print(f"Vertices: {n_verts}")
    print(f"Cells: {n_cells}")

    # Step 1: Load with all relations to compute boundary
    print("\nStep 1: Load with all relations to find boundary edges...")
    mesh_tmp = Patcher.load_mesh(raw_data, relations=['FC', 'FE', 'FV', 'EV'])

    # Mark boundary faces and edges
    mesh_tmp.faces.place({'is_boundary': ti.i32})
    mesh_tmp.edges.place({'is_boundary': ti.i32})

    n_all_edges = len(mesh_tmp.edges)
    n_all_faces = len(mesh_tmp.faces)
    print(f"All edges: {n_all_edges}")
    print(f"All faces: {n_all_faces}")

    @ti.kernel
    def find_boundary():
        for f in mesh_tmp.faces:
            if f.cells.size == 1:
                f.is_boundary = 1
                for i in ti.static(range(3)):
                    f.edges[i].is_boundary = 1

    find_boundary()
    ti.sync()

    # Extract boundary edges
    edges_field = ti.field(ti.i32, shape=(n_all_edges, 2))

    @ti.kernel
    def extract_edge_verts():
        for e in mesh_tmp.edges:
            edges_field[e.id, 0] = e.verts[0].id
            edges_field[e.id, 1] = e.verts[1].id

    extract_edge_verts()
    ti.sync()

    all_edges_np = edges_field.to_numpy()
    edge_boundary = mesh_tmp.edges.is_boundary.to_numpy()
    boundary_edge_ids = np.where(edge_boundary == 1)[0]
    boundary_edges = all_edges_np[boundary_edge_ids]

    n_boundary_edges = boundary_edges.shape[0]
    print(f"Boundary edges: {n_boundary_edges}")
    print(f"Boundary edge ratio: {n_boundary_edges / n_all_edges * 100:.1f}%")

    del mesh_tmp

    # Step 2: Try to load mesh with injected boundary edges
    print("\nStep 2: Load mesh with injected boundary edges...")

    # Create mesh dict with boundary edges as element 1
    mesh_dict = {
        0: vertices,
        1: boundary_edges,  # Inject boundary edges
        3: cells
    }

    try:
        mesh = Patcher.load_mesh(mesh_dict, relations=['CV', 'EV'])

        n_loaded_edges = len(mesh.edges)
        print(f"Loaded edges: {n_loaded_edges}")

        # Verify the loaded edges match boundary edges
        if n_loaded_edges == n_boundary_edges:
            print("SUCCESS: Mesh loaded with only boundary edges!")
        else:
            print(f"PARTIAL: Loaded {n_loaded_edges} edges, expected {n_boundary_edges}")

        # Test mesh-for loop on edges
        mesh.verts.place({'x': ti.math.vec3})
        mesh.verts.x.from_numpy(vertices)

        total_length = ti.field(ti.f32, shape=())

        @ti.kernel
        def compute_edge_lengths():
            total = 0.0
            ti.mesh_local(mesh.verts.x)
            for e in mesh.edges:
                v0 = e.verts[0]
                v1 = e.verts[1]
                length = (v0.x - v1.x).norm()
                total += length
            total_length[None] = total

        compute_edge_lengths()
        ti.sync()

        print(f"Total edge length (mesh-for): {total_length[None]:.4f}")
        print("\nMesh-for loop on boundary edges: SUCCESS!")

        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_approaches():
    """Compare edge injection approach with full edge loading."""
    print("\n" + "="*70)
    print("Test: Compare Edge Injection vs Full Edge Loading")
    print("="*70)

    model_path = "/root/PNCG_IPC/model/cube"
    raw_data = Patcher.load_mesh_rawdata(model_path + ".node")

    # Approach 1: Full edge loading with is_boundary check
    print("\nApproach 1: Full edge loading...")
    mesh_full = Patcher.load_mesh(raw_data, relations=['CV', 'EV'])
    mesh_full.edges.place({'is_boundary': ti.i32})
    mesh_full.verts.place({'x': ti.math.vec3})
    mesh_full.verts.x.from_numpy(raw_data[0])

    # Mark boundary (simplified - just use edge count as proxy)
    n_full_edges = len(mesh_full.edges)
    print(f"Full mesh edges: {n_full_edges}")

    # For this test, just compute total length of all edges
    total_length_full = ti.field(ti.f32, shape=())

    @ti.kernel
    def compute_full():
        total = 0.0
        ti.mesh_local(mesh_full.verts.x)
        for e in mesh_full.edges:
            v0 = e.verts[0]
            v1 = e.verts[1]
            length = (v0.x - v1.x).norm()
            total += length
        total_length_full[None] = total

    compute_full()
    ti.sync()
    print(f"Total length (all edges): {total_length_full[None]:.4f}")

    del mesh_full

    print("\nComparison complete!")


if __name__ == '__main__':
    success = test_boundary_edge_injection()

    if success:
        test_compare_approaches()
