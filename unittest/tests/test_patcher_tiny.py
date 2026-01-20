#!/usr/bin/env python3
"""
Test patcher edge injection with tiny tet mesh.
"""

import sys
sys.path.insert(0, '/root/PNCG_IPC')

import numpy as np
import meshtaichi_patcher as Patcher
from meshtaichi_patcher.meshpatcher import MeshPatcher


def main():
    print("Test: Patcher Edge Injection (Tiny Tet)")
    print("="*60)

    # Load tiny tet mesh (4 vertices, 1 cell)
    model_path = "/root/PNCG_IPC/model/mesh/tet/tet"
    raw_data = Patcher.load_mesh_rawdata(model_path + ".node")

    vertices = raw_data[0]
    cells = raw_data[3]
    print(f"Vertices: {vertices.shape[0]}, Cells: {cells.shape[0]}")
    print(f"Vertices:\n{vertices}")
    print(f"Cells:\n{cells}")

    # Test 1: Normal loading
    print("\n--- Test 1: Load with CV only ---")
    m1 = MeshPatcher(raw_data)
    m1.patcher.patch_size = 256
    print("Patching...")
    m1.patch(-1, 'all')
    print("Getting meta...")
    meta1 = m1.get_meta(['CV'])
    print(f"Elements in meta: {[e['order'] for e in meta1['elements']]}")
    for e in meta1['elements']:
        print(f"  Order {e['order']}: {e['num']} elements")

    # Test 2: With EV relation
    print("\n--- Test 2: Load with CV + EV ---")
    m2 = MeshPatcher(raw_data)
    m2.patcher.patch_size = 256
    print("Patching...")
    m2.patch(-1, 'all')
    print("Getting meta...")
    meta2 = m2.get_meta(['CV', 'EV'])
    print(f"Elements in meta: {[e['order'] for e in meta2['elements']]}")
    for e in meta2['elements']:
        print(f"  Order {e['order']}: {e['num']} elements")

    # Test 3: Inject boundary edges
    print("\n--- Test 3: Inject boundary edges ---")
    # A tetrahedron has 6 edges, all are boundary
    boundary_edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ], dtype=np.int32)

    mesh_dict_with_edges = {
        0: vertices,
        1: boundary_edges,
        3: cells
    }

    m3 = MeshPatcher(mesh_dict_with_edges)
    m3.patcher.patch_size = 256
    print("Patching with injected edges...")
    m3.patch(-1, 'all')
    print("Getting meta...")
    meta3 = m3.get_meta(['CV', 'EV'])
    print(f"Elements in meta: {[e['order'] for e in meta3['elements']]}")
    for e in meta3['elements']:
        print(f"  Order {e['order']}: {e['num']} elements")

    # Check if injected edges are preserved
    n_injected = len(boundary_edges)
    n_loaded = meta3['elements'][1]['num']
    print(f"\nInjected edges: {n_injected}")
    print(f"Loaded edges: {n_loaded}")

    if n_loaded == n_injected:
        print("SUCCESS: Edge injection preserves count!")
    else:
        print(f"NOTE: Edge count changed from {n_injected} to {n_loaded}")

    # Examine the EV relation
    print("\n--- EV Relation Details ---")
    for rel in meta3['relations']:
        if rel['from_order'] == 1 and rel['to_order'] == 0:
            print(f"EV value: {rel['value']}")
            print(f"EV offset: {rel['offset']}")
            break


if __name__ == '__main__':
    main()
