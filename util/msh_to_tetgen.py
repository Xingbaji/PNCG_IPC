"""
Mesh Converter: Gmsh 2.2 (.msh) to TetGen format (.node/.ele/.face/.edge)

Converts Stiff-GIPC mesh format to PNCG_IPC format.

Usage:
    python msh_to_tetgen.py input.msh output_dir/
    python msh_to_tetgen.py --batch --src /path/to/msh/ --dst /path/to/output/
"""

import os
import argparse
from collections import defaultdict
from pathlib import Path


def parse_msh_file(filepath):
    """
    Parse Gmsh 2.2 format file.

    Returns:
        nodes: list of (x, y, z) tuples, indexed from 0
        tets: list of (v0, v1, v2, v3) tuples with 0-indexed vertices
        triangles: list of (v0, v1, v2) surface triangles if present
        lines: list of (v0, v1) edges if present
    """
    nodes = []
    tets = []
    triangles = []
    lines = []

    # Map from 1-indexed MSH IDs to 0-indexed array indices
    node_id_map = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse nodes section
    # Handle both standard ($EndNodes) and non-standard (uses $Elements as terminator) formats
    if '$Nodes' in content:
        after_nodes = content.split('$Nodes')[1]
        if '$EndNodes' in after_nodes:
            nodes_section = after_nodes.split('$EndNodes')[0].strip()
        elif '$Elements' in after_nodes:
            nodes_section = after_nodes.split('$Elements')[0].strip()
        else:
            nodes_section = after_nodes.strip()

        lines_list = nodes_section.split('\n')
        num_nodes = int(lines_list[0])

        for i in range(1, num_nodes + 1):
            parts = lines_list[i].split()
            node_id = int(parts[0])  # 1-indexed in MSH
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            node_id_map[node_id] = len(nodes)  # Map to 0-indexed
            nodes.append((x, y, z))

    # Parse elements section
    # Note: Stiff-GIPC takes the LAST N elements as vertex indices,
    # ignoring the exact tag count. We do the same for compatibility.
    # Handle both standard ($EndElements) and non-standard (reads until EOF) formats
    if '$Elements' in content:
        after_elements = content.split('$Elements')[1]
        if '$EndElements' in after_elements:
            elements_section = after_elements.split('$EndElements')[0].strip()
        else:
            elements_section = after_elements.strip()

        lines_list = elements_section.split('\n')
        num_elements = int(lines_list[0])

        for i in range(1, min(num_elements + 1, len(lines_list))):
            parts = lines_list[i].split()
            if len(parts) < 5:  # Need at least id, type, num_tags, and some vertices
                continue
            elem_type = int(parts[1])

            # Take the last N elements as vertex indices (Stiff-GIPC style)
            if elem_type == 4:  # Tetrahedron (4 vertices)
                v0 = node_id_map[int(parts[-4])]
                v1 = node_id_map[int(parts[-3])]
                v2 = node_id_map[int(parts[-2])]
                v3 = node_id_map[int(parts[-1])]
                tets.append((v0, v1, v2, v3))
            elif elem_type == 2:  # Triangle (3 vertices)
                v0 = node_id_map[int(parts[-3])]
                v1 = node_id_map[int(parts[-2])]
                v2 = node_id_map[int(parts[-1])]
                triangles.append((v0, v1, v2))
            elif elem_type == 1:  # Line (2 vertices)
                v0 = node_id_map[int(parts[-2])]
                v1 = node_id_map[int(parts[-1])]
                lines.append((v0, v1))

    return nodes, tets, triangles, lines


def get_tet_faces(tet):
    """
    Get the 4 faces of a tetrahedron.
    Each face is a tuple of 3 vertex indices.

    For tet (v0, v1, v2, v3), faces are:
    - (v0, v1, v2)
    - (v0, v1, v3)
    - (v0, v2, v3)
    - (v1, v2, v3)
    """
    v0, v1, v2, v3 = tet
    return [
        (v0, v1, v2),
        (v0, v1, v3),
        (v0, v2, v3),
        (v1, v2, v3)
    ]


def extract_boundary_faces(tets):
    """
    Find boundary faces from tetrahedra.

    A face is on the boundary if it appears in exactly one tetrahedron.
    Uses sorted vertex tuples as keys for face counting.

    Returns:
        List of boundary face tuples (v0, v1, v2)
    """
    face_count = defaultdict(int)
    face_original = {}  # Map sorted face to original orientation

    for tet in tets:
        for face in get_tet_faces(tet):
            sorted_face = tuple(sorted(face))
            face_count[sorted_face] += 1
            face_original[sorted_face] = face

    # Boundary faces appear exactly once
    boundary_faces = [
        face_original[f] for f, count in face_count.items() if count == 1
    ]

    return boundary_faces


def extract_boundary_edges(boundary_faces):
    """
    Extract edges from boundary faces.

    Each triangle has 3 edges. Collect unique edges.

    Returns:
        List of edge tuples (v0, v1) sorted by vertex index
    """
    edge_set = set()

    for face in boundary_faces:
        v0, v1, v2 = face
        # Add edges as sorted tuples for uniqueness
        edge_set.add(tuple(sorted([v0, v1])))
        edge_set.add(tuple(sorted([v1, v2])))
        edge_set.add(tuple(sorted([v0, v2])))

    return list(edge_set)


def write_node_file(filepath, nodes):
    """Write .node file in TetGen format (0-indexed)."""
    with open(filepath, 'w') as f:
        f.write(f"{len(nodes)}\n")
        for i, (x, y, z) in enumerate(nodes):
            f.write(f"{i} {x} {y} {z}\n")


def write_ele_file(filepath, tets):
    """Write .ele file in TetGen format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(tets)}\n")
        for i, (v0, v1, v2, v3) in enumerate(tets):
            f.write(f"{i} {v0} {v1} {v2} {v3}\n")


def write_face_file(filepath, faces):
    """Write .face file in TetGen format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(faces)}\n")
        for i, (v0, v1, v2) in enumerate(faces):
            f.write(f"{i} {v0} {v1} {v2}\n")


def write_edge_file(filepath, edges):
    """Write .edge file in TetGen format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(edges)}\n")
        for i, (v0, v1) in enumerate(edges):
            f.write(f"{i} {v0} {v1}\n")


def convert_msh_to_tetgen(msh_path, output_dir, mesh_name=None):
    """
    Convert a single .msh file to TetGen format.

    Args:
        msh_path: Path to input .msh file
        output_dir: Directory to write output files
        mesh_name: Optional name for output files (default: derived from msh filename)
    """
    msh_path = Path(msh_path)
    output_dir = Path(output_dir)

    if mesh_name is None:
        # Remove .msh extension and any .1 suffix
        mesh_name = msh_path.stem
        if mesh_name.endswith('.1'):
            mesh_name = mesh_name[:-2]
        # Clean up common suffixes
        mesh_name = mesh_name.replace('_sorted', '').replace('_reset', '')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {msh_path.name}...")

    # Parse MSH file
    nodes, tets, triangles, lines = parse_msh_file(msh_path)

    print(f"  Nodes: {len(nodes)}")
    print(f"  Tetrahedra: {len(tets)}")

    # Extract boundary faces if not present in MSH
    if triangles:
        faces = triangles
        print(f"  Surface triangles from MSH: {len(faces)}")
    else:
        faces = extract_boundary_faces(tets)
        print(f"  Extracted boundary faces: {len(faces)}")

    # Extract boundary edges
    if lines:
        edges = lines
        print(f"  Edges from MSH: {len(edges)}")
    else:
        edges = extract_boundary_edges(faces)
        print(f"  Extracted boundary edges: {len(edges)}")

    # Write output files
    base_path = output_dir / mesh_name
    write_node_file(f"{base_path}.node", nodes)
    write_ele_file(f"{base_path}.ele", tets)
    write_face_file(f"{base_path}.face", faces)
    write_edge_file(f"{base_path}.edge", edges)

    print(f"  Written to {output_dir}/")

    return len(nodes), len(tets), len(faces), len(edges)


def batch_convert(src_dir, dst_dir):
    """
    Convert all .msh files from source directory.

    Creates a subdirectory for each mesh in the destination.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # List of meshes to convert with their clean names
    mesh_mappings = {
        'armadillo_sorted.msh': 'armadillo_stiff',
        'bunny2.msh': 'bunny_stiff',
        'bunny10.msh': 'bunny10',
        'cube.msh': 'cube_stiff',
        'dragon_50k.1.msh': 'dragon_50k',
        'dragon_70k.1.msh': 'dragon_70k',
        'dragon_high.1.msh': 'dragon_high',
        'letters_mas_pncg.1_sorted.msh': 'letters',
        'mat_new.1.msh': 'mat',
        'octopus_5KFaces_reset.1.msh': 'octopus',
        'teapot.1_sorted.msh': 'teapot',
        'box.1_sorted.msh': 'box',
    }

    results = []

    for msh_file, clean_name in mesh_mappings.items():
        msh_path = src_dir / msh_file
        if msh_path.exists():
            output_dir = dst_dir / clean_name
            stats = convert_msh_to_tetgen(msh_path, output_dir, clean_name)
            results.append((clean_name, stats))
        else:
            print(f"Warning: {msh_file} not found, skipping")

    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"{'Mesh':<20} {'Nodes':>10} {'Tets':>10} {'Faces':>10} {'Edges':>10}")
    print("-"*60)
    for name, (nodes, tets, faces, edges) in results:
        print(f"{name:<20} {nodes:>10} {tets:>10} {faces:>10} {edges:>10}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert Gmsh 2.2 (.msh) to TetGen format'
    )

    parser.add_argument('input', nargs='?', help='Input .msh file')
    parser.add_argument('output', nargs='?', help='Output directory')
    parser.add_argument('--batch', action='store_true',
                        help='Batch convert all meshes')
    parser.add_argument('--src', default='/root/Stiff-GIPC/Assets/tetMesh/',
                        help='Source directory for batch conversion')
    parser.add_argument('--dst', default='../model/mesh/',
                        help='Destination directory for batch conversion')

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.src, args.dst)
    elif args.input and args.output:
        convert_msh_to_tetgen(args.input, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
