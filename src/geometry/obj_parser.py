"""
Wavefront OBJ parser and exporter.
Supports standard vertices ('v x y z') and triangular/polygonal faces ('f v1 v2 v3').
Automatically triangulates quad faces.
"""

from typing import Union
import numpy as np
from .mesh import TriangleMesh


class OBJParser:
    @staticmethod
    def load(filepath_or_text: Union[str, bytes]) -> TriangleMesh:
        if isinstance(filepath_or_text, bytes):
            text = filepath_or_text.decode("utf-8", errors="ignore")
        elif isinstance(filepath_or_text, str) and ("\n" in filepath_or_text or len(filepath_or_text) > 4096):
            text = filepath_or_text
        else:
            with open(filepath_or_text, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        vertices = []
        faces = []

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == "f" and len(parts) >= 4:
                # Handle vertex indices, ignoring texture and normal indices (e.g. 1/1/1 or 1//1)
                idx_list = []
                for p in parts[1:]:
                    val_str = p.split("/")[0]
                    if val_str:
                        idx = int(val_str)
                        # OBJ indices are 1-based, or negative relative
                        if idx > 0:
                            idx_list.append(idx - 1)
                        else:
                            idx_list.append(len(vertices) + idx)

                # Fan triangulation for polygons
                for i in range(1, len(idx_list) - 1):
                    faces.append([idx_list[0], idx_list[i], idx_list[i + 1]])

        verts_arr = np.array(vertices, dtype=np.float32)
        faces_arr = np.array(faces, dtype=np.int32)
        return TriangleMesh(verts_arr, faces_arr)

    @staticmethod
    def save(mesh: TriangleMesh, filepath: str) -> None:
        """Saves a TriangleMesh to a standard Wavefront OBJ file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# 3D-Support-Calculator Generated Mesh\n")
            for v in mesh.vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in mesh.faces:
                # OBJ uses 1-based indexing
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
