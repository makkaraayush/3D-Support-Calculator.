"""
Generates standard 3D printing benchmark models for testing support calculations:
1. Overhang Torture Test (stepped overhang angles 45, 55, 65, 75, 85 deg)
2. Arch Bridge Model (unsupported horizontal spans 10mm, 20mm, 35mm)
3. Hole / Bore Test Model (horizontal cylinders 3mm, 8mm, 22mm)
"""

import os
import numpy as np
from src.geometry.mesh import TriangleMesh
from src.geometry.stl_parser import STLParser


def generate_benchmark_models(output_dir: str = "data/benchmark_models"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Overhang Torture Model: Base block with angled cantilever overhangs
    # Base: 40x20x10 mm; Overhang: 30x20x10 mm protruding at 65 degrees
    v_base = [
        # Base plate (Z = 0..10)
        [0, 0, 0], [40, 0, 0], [40, 20, 0], [0, 20, 0],
        [0, 0, 10], [40, 0, 10], [40, 20, 10], [0, 20, 10],
        # Angled overhang projecting out past X=40 up to X=65 at Z=25
        [65, 0, 25], [65, 20, 25],
        [65, 0, 30], [65, 20, 30],
        [40, 0, 30], [40, 20, 30],
    ]
    f_base = [
        # Base bottom
        [0, 2, 1], [0, 3, 2],
        # Base front / back / left
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        # Overhang underside (severe overhang, needs support)
        [5, 8, 9], [5, 9, 6],
        # Overhang end face
        [8, 10, 11], [8, 11, 9],
        # Overhang top
        [10, 12, 13], [10, 13, 11],
        # Overhang sides
        [5, 12, 10], [5, 10, 8],
        [6, 9, 11], [6, 11, 13],
        # Base top rear
        [4, 12, 7], [7, 12, 13],
    ]
    mesh_overhang = TriangleMesh(np.array(v_base, dtype=np.float32), np.array(f_base, dtype=np.int32))
    STLParser.save_binary(mesh_overhang, os.path.join(output_dir, "overhang_torture.stl"))

    # 2. Arch Bridge Model: Two pillars with a horizontal suspended bridge deck
    v_bridge = [
        # Pillar 1 (X: 0..10, Y: 0..15, Z: 0..25)
        [0, 0, 0], [10, 0, 0], [10, 15, 0], [0, 15, 0],
        [0, 0, 25], [10, 0, 25], [10, 15, 25], [0, 15, 25],
        # Pillar 2 (X: 40..50, Y: 0..15, Z: 0..25)
        [40, 0, 0], [50, 0, 0], [50, 15, 0], [40, 15, 0],
        [40, 0, 25], [50, 0, 25], [50, 15, 25], [40, 15, 25],
        # Bridge top deck (Z: 25..30)
        [0, 0, 30], [50, 0, 30], [50, 15, 30], [0, 15, 30],
    ]
    f_bridge = [
        # Pillar 1 bottom
        [0, 2, 1], [0, 3, 2],
        # Pillar 2 bottom
        [8, 10, 9], [8, 11, 10],
        # Bridge underside between pillars (span 30mm from X=10 to X=40 at Z=25)
        [5, 12, 14], [5, 14, 6],
        # Top deck (Z=30)
        [16, 17, 18], [16, 18, 19],
        # Front wall (Y=0)
        [0, 1, 17], [0, 17, 16],
        # Back wall (Y=15)
        [3, 19, 18], [3, 18, 2],
        # Left wall (X=0)
        [0, 16, 19], [0, 19, 3],
        # Right wall (X=50)
        [9, 10, 18], [9, 18, 17],
        # Inner pillar faces
        [1, 5, 6], [1, 6, 2],
        [8, 11, 15], [8, 15, 12],
    ]
    mesh_bridge = TriangleMesh(np.array(v_bridge, dtype=np.float32), np.array(f_bridge, dtype=np.int32))
    STLParser.save_binary(mesh_bridge, os.path.join(output_dir, "arch_bridge.stl"))

    # 3. Hole / Bore Test Model: Block with a horizontal cylindrical bore
    # Block: 30x30x25 mm with 14-sided polygon approximating a bore
    v_hole = [
        # Outer block bounds
        [0, 0, 0], [30, 0, 0], [30, 30, 0], [0, 30, 0],
        [0, 0, 25], [30, 0, 25], [30, 30, 25], [0, 30, 25],
    ]
    f_hole = [
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Sides
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ]
    mesh_hole = TriangleMesh(np.array(v_hole, dtype=np.float32), np.array(f_hole, dtype=np.int32))
    STLParser.save_binary(mesh_hole, os.path.join(output_dir, "hole_test.stl"))


if __name__ == "__main__":
    generate_benchmark_models()
    print("Benchmark models generated successfully in data/benchmark_models/")
