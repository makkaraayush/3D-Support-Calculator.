"""
High-speed Binary and ASCII STL parser and exporter.
Optimized with NumPy buffer operations to load large models in milliseconds
without consuming heavy memory on low-end CPUs.
"""

import struct
from typing import Union
import numpy as np
from .mesh import TriangleMesh


class STLParser:
    @staticmethod
    def load(filepath_or_bytes: Union[str, bytes]) -> TriangleMesh:
        """Loads an STL file from a file path or raw bytes."""
        if isinstance(filepath_or_bytes, str):
            with open(filepath_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = filepath_or_bytes

        if len(data) < 84:
            raise ValueError("Invalid STL: file is too small to contain header and face count.")

        # Check if binary or ASCII
        is_ascii = False
        try:
            # If startswith 'solid' and contains 'facet', could be ASCII
            header_sample = data[:512].decode("ascii", errors="ignore").strip().lower()
            if header_sample.startswith("solid") and "facet" in header_sample:
                is_ascii = True
        except Exception:
            pass

        if is_ascii:
            return STLParser._load_ascii(data.decode("utf-8", errors="ignore"))
        else:
            return STLParser._load_binary(data)

    @staticmethod
    def _load_binary(data: bytes) -> TriangleMesh:
        num_triangles = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + (num_triangles * 50)
        
        # In case the file size doesn't match perfectly, infer count from length
        if len(data) != expected_size:
            inferred_triangles = (len(data) - 84) // 50
            if inferred_triangles > 0:
                num_triangles = inferred_triangles

        # Binary STL layout per triangle (50 bytes):
        # normal: float32 * 3 (12 bytes)
        # vertex 0: float32 * 3 (12 bytes)
        # vertex 1: float32 * 3 (12 bytes)
        # vertex 2: float32 * 3 (12 bytes)
        # attribute byte count: uint16 (2 bytes)
        dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])

        record_array = np.frombuffer(data, dtype=dtype, offset=84, count=num_triangles)

        # Flatten vertices into (num_triangles * 3, 3)
        v0 = record_array["v0"]
        v1 = record_array["v1"]
        v2 = record_array["v2"]

        all_verts = np.empty((num_triangles * 3, 3), dtype=np.float32)
        all_verts[0::3] = v0
        all_verts[1::3] = v1
        all_verts[2::3] = v2

        # Faces index into vertices
        faces = np.arange(num_triangles * 3, dtype=np.int32).reshape((num_triangles, 3))
        normals = record_array["normal"].astype(np.float32)

        return TriangleMesh(all_verts, faces, normals)

    @staticmethod
    def _load_ascii(text: str) -> TriangleMesh:
        vertices = []
        faces = []
        normals = []

        curr_normal = [0.0, 0.0, 0.0]
        curr_face = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "facet" and len(parts) >= 5 and parts[1].lower() == "normal":
                curr_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            elif cmd == "vertex" and len(parts) >= 4:
                v_idx = len(vertices)
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                curr_face.append(v_idx)
            elif cmd == "endfacet":
                if len(curr_face) == 3:
                    faces.append(curr_face)
                    normals.append(curr_normal)
                curr_face = []

        verts_arr = np.array(vertices, dtype=np.float32)
        faces_arr = np.array(faces, dtype=np.int32)
        normals_arr = np.array(normals, dtype=np.float32)

        return TriangleMesh(verts_arr, faces_arr, normals_arr)

    @staticmethod
    def save_binary(mesh: TriangleMesh, filepath: str) -> None:
        """Exports a TriangleMesh to a standard binary STL file."""
        num_triangles = len(mesh.faces)
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        normals = mesh.face_normals

        dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])

        record_array = np.empty(num_triangles, dtype=dtype)
        record_array["normal"] = normals
        record_array["v0"] = v0
        record_array["v1"] = v1
        record_array["v2"] = v2
        record_array["attr"] = 0

        header = b"3D-Support-Calculator - Optimized Bambu Lab Slicing Mesh" + (b" " * 80)
        header = header[:80]

        with open(filepath, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", num_triangles))
            f.write(record_array.tobytes())
