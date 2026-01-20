#!/usr/bin/env python3
"""
Test patcher edge injection without Taichi kernel compilation.
"""

import sys
sys.path.insert(0, '/root/PNCG_IPC')

import numpy as np
import meshtaichi_patcher as Patcher
from meshtaichi_patcher.meshpatcher import MeshPatcher


def main():
    print("Test: Patcher Edge Injection (No Taichi Kernels)")
    print("="*60)

    # Load raw mesh data
    model_path = "/root/PNCG_IPC/model/cube"
    raw_data = Patcher.load_mesh_rawdata(model_path + ".node")

    vertices = raw_data[0]
    cells = raw_data[3]
    print(f"Vertices: {vertices.shape[0]}, Cells: {cells.shape[0]}")

    # Test 1: Normal loading without edges
    print("\n--- Test 1: Load without edges ---")
    m1 = MeshPatcher(raw_data)
    m1.patcher.patch_size = 256
    m1.patch(-1, 'all')
    meta1 = m1.get_meta(['CV'])
    print(f"Elements in meta: {[e['order'] for e in meta1['elements']]}")
    print(f"Vertex count: {meta1['elements'][0]['num']}")
    if len(meta1['elements']) > 1:
        print(f"Edge count: {meta1['elements'][1]['num']}")
    else:
        print("No edge element in metadata")

    # Test 2: Load with edges
    print("\n--- Test 2: Load with EV relation ---")
    m2 = MeshPatcher(raw_data)
    m2.patcher.patch_size = 256
    m2.patch(-1, 'all')
    meta2 = m2.get_meta(['CV', 'EV'])
    print(f"Elements in meta: {[e['order'] for e in meta2['elements']]}")
    for e in meta2['elements']:
        print(f"  Order {e['order']}: {e['num']} elements")

    # Test 3: Inject custom edges
    print("\n--- Test 3: Inject custom boundary edges ---")
    # Create a small set of fake boundary edges
    boundary_edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        [2, 3],
    ], dtype=np.int32)

    mesh_dict_with_edges = {
        0: vertices,
        1: boundary_edges,  # Inject edges as element order 1
        3: cells
    }

    m3 = MeshPatcher(mesh_dict_with_edges)
    m3.patcher.patch_size = 256
    m3.patch(-1, 'all')
    meta3 = m3.get_meta(['CV', 'EV'])
    print(f"Elements in meta: {[e['order'] for e in meta3['elements']]}")
    for e in meta3['elements']:
        print(f"  Order {e['order']}: {e['num']} elements")

    # Check if injected edges are preserved
    if len(meta3['elements']) > 1:
        n_edges_loaded = meta3['elements'][1]['num']
        print(f"\nInjected edges: {len(boundary_edges)}")
        print(f"Loaded edges: {n_edges_loaded}")
        if n_edges_loaded == len(boundary_edges):
            print("SUCCESS: Edge injection works!")
        else:
            print(f"NOTE: Edge count differs. Patcher may have deduced additional edges.")

    # Test 4: Verify EV relation
    print("\n--- Test 4: Check EV relation data ---")
    ev_relation = None
    for rel in meta3['relations']:
        if rel['from_order'] == 1 and rel['to_order'] == 0:
            ev_relation = rel
            break

    if ev_relation:
        print(f"EV relation value shape: {ev_relation['value'].shape}")
        print(f"EV relation offset shape: {ev_relation['offset'].shape}")
        print(f"First few values: {ev_relation['value'][:12]}")
        print(f"First few offsets: {ev_relation['offset'][:7]}")
    else:
        print("No EV relation found!")


if __name__ == '__main__':
    main()
