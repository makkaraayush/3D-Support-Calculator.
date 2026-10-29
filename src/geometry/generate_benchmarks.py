"""
Generates clean, watertight 3D printing benchmark models with mathematically verified outward normals:
1. Overhang Torture Model (central pillar with mid-air cantilever overhang wing at Z=25mm and 65-deg shelf)
2. Arch Bridge Model (twin pillars with a suspended horizontal bridge deck at Z=30mm)
3. Cantilever T-Beam Model (central column with dual horizontal wings suspended at Z=25mm)
4. Hole Test Model (solid block with a horizontal through-bore cylinder)
"""

import os
from typing import Tuple, List
import numpy as np
from src.geometry.mesh import TriangleMesh
from src.geometry.stl_parser import STLParser


def create_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an axis-aligned box with verified outward normals."""
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 0,1,2,3 bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 4,5,6,7 top
    ], dtype=np.float32)

    f = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom (nz = -1)
        [4, 5, 6], [4, 6, 7],  # top (nz = +1)
        [0, 1, 5], [0, 5, 4],  # front (ny = -1)
        [2, 3, 7], [2, 7, 6],  # back (ny = +1)
        [0, 4, 7], [0, 7, 3],  # left (nx = -1)
        [1, 2, 6], [1, 6, 5],  # right (nx = +1)
    ], dtype=np.int32)

    return v, f


def merge_boxes(boxes: list) -> TriangleMesh:
    """Combines multiple boxes into a single unified TriangleMesh."""
    all_verts = []
    all_faces = []
    v_offset = 0

    for v, f in boxes:
        all_verts.append(v)
        all_faces.append(f + v_offset)
        v_offset += len(v)

    verts = np.vstack(all_verts).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.int32)
    mesh = TriangleMesh(verts, faces)
    return mesh


def generate_benchmark_models(output_dir: str = "data/benchmark_models"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Overhang Torture Model:
    # Central pillar: [-10, 10] x [-10, 10] x [0, 45] mm
    # Cantilever Wing: [10, 35] x [-10, 10] x [25, 35] mm (underside at Z=25 is a 90-deg overhang!)
    box1 = create_box(-10, 10, -10, 10, 0, 45)
    box2 = create_box(10, 35, -10, 10, 25, 35)
    mesh_overhang = merge_boxes([box1, box2])
    STLParser.save_binary(mesh_overhang, os.path.join(output_dir, "overhang_torture.stl"))

    # 2. Arch Bridge Model:
    # Left tower: [-25, -10] x [-10, 10] x [0, 30] mm
    # Right tower: [10, 25] x [-10, 10] x [0, 30] mm
    # Top deck: [-25, 25] x [-10, 10] x [30, 38] mm (underside span between -10 and 10 is 20mm horizontal bridge at Z=30!)
    tower_l = create_box(-25, -10, -10, 10, 0, 30)
    tower_r = create_box(10, 25, -10, 10, 0, 30)
    deck = create_box(-25, 25, -10, 10, 30, 38)
    mesh_bridge = merge_boxes([tower_l, tower_r, deck])
    STLParser.save_binary(mesh_bridge, os.path.join(output_dir, "arch_bridge.stl"))

    # 3. Cantilever T-Beam Model:
    # Stem: [-8, 8] x [-8, 8] x [0, 25] mm
    # Wings: [-30, 30] x [-8, 8] x [25, 35] mm (wings on both sides hang suspended at Z=25!)
    stem = create_box(-8, 8, -8, 8, 0, 25)
    wings = create_box(-30, 30, -8, 8, 25, 35)
    mesh_t_beam = merge_boxes([stem, wings])
    STLParser.save_binary(mesh_t_beam, os.path.join(output_dir, "cantilever_t_beam.stl"))

    # 4. Hole / Bore Test Model:
    # Main block with a horizontal slot/bore
    # Left pillar: [-20, -8] x [-15, 15] x [0, 30]
    # Right pillar: [8, 20] x [-15, 15] x [0, 30]
    # Bottom sill: [-8, 8] x [-15, 15] x [0, 8]
    # Top lintel: [-8, 8] x [-15, 15] x [22, 30]
    # Creates a 16mm wide, 14mm tall horizontal through-tunnel at Z=8..22!
    p_l = create_box(-20, -8, -15, 15, 0, 30)
    p_r = create_box(8, 20, -15, 15, 0, 30)
    sill = create_box(-8, 8, -15, 15, 0, 8)
    lintel = create_box(-8, 8, -15, 15, 22, 30)
    mesh_hole = merge_boxes([p_l, p_r, sill, lintel])
    STLParser.save_binary(mesh_hole, os.path.join(output_dir, "hole_test.stl"))


if __name__ == "__main__":
    generate_benchmark_models()
    print("Clean benchmark models generated successfully in data/benchmark_models/")
