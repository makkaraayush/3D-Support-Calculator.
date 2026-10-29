"""
High-performance 3D Triangle Mesh data structure implemented in pure NumPy.
Provides vectorized face normal computation, bounding box, surface area,
volume calculation via divergence theorem, and normal orientation verification.
Zero heavy external C++ dependencies: runs instantly on low-end CPUs (e.g. i5-2400s).
"""

from typing import Tuple, Optional
import numpy as np


class TriangleMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray] = None):
        """
        vertices: (N, 3) float32/float64 array
        faces: (M, 3) int32/int64 array of vertex indices
        normals: (M, 3) optional precomputed face normals
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        if normals is not None:
            self.face_normals = np.ascontiguousarray(normals, dtype=np.float32)
        else:
            self.face_normals = self._compute_face_normals()

        # Ensure consistent outward normals for closed solid meshes
        self.fix_normals()

    def _compute_face_normals(self) -> np.ndarray:
        if len(self.faces) == 0:
            return np.empty((0, 3), dtype=np.float32)

        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        # Avoid division by zero for degenerate triangles
        norm = np.where(norm < 1e-9, 1.0, norm)
        return (cross / norm).astype(np.float32)

    def fix_normals(self) -> None:
        """
        Ensures consistent outward normals.
        If signed divergence volume is negative, flips face winding so normals point outwards.
        """
        if len(self.faces) < 4:
            return

        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v0, v1)
        signed_volume = np.sum(np.sum(cross * v2, axis=1)) / 6.0

        if signed_volume < -1e-5:
            # Flip winding order [0, 1, 2] -> [0, 2, 1]
            self.faces = np.column_stack([self.faces[:, 0], self.faces[:, 2], self.faces[:, 1]])
            self.face_normals = self._compute_face_normals()

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (min_xyz, max_xyz) as (3,) arrays."""
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    @property
    def dimensions(self) -> np.ndarray:
        """Returns size in X, Y, Z (mm)."""
        b_min, b_max = self.bounds
        return b_max - b_min

    @property
    def center(self) -> np.ndarray:
        """Center of the bounding box."""
        b_min, b_max = self.bounds
        return (b_min + b_max) * 0.5

    def compute_surface_area(self) -> float:
        """Total mesh surface area in mm^2."""
        if len(self.faces) == 0:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return float(np.sum(areas))

    def compute_volume(self) -> float:
        """
        Mesh volume in mm^3 using the divergence theorem:
        V = 1/6 * sum( (v0 x v1) . v2 ) for closed manifold surfaces.
        """
        if len(self.faces) == 0:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v0, v1)
        dot = np.sum(cross * v2, axis=1)
        volume = np.abs(np.sum(dot)) / 6.0
        return float(volume)

    def center_on_bed(self, bed_center_xy: Tuple[float, float] = (128.0, 128.0)) -> "TriangleMesh":
        """Translates mesh so its bottom rests at Z=0 and center sits at bed_center_xy."""
        b_min, b_max = self.bounds
        current_center_xy = (b_min[:2] + b_max[:2]) * 0.5
        shift_x = bed_center_xy[0] - current_center_xy[0]
        shift_y = bed_center_xy[1] - current_center_xy[1]
        shift_z = -b_min[2]  # snap bottom to build plate Z=0

        new_verts = self.vertices.copy()
        new_verts[:, 0] += shift_x
        new_verts[:, 1] += shift_y
        new_verts[:, 2] += shift_z

        return TriangleMesh(new_verts, self.faces.copy(), self.face_normals.copy())

    def get_face_centers(self) -> np.ndarray:
        """Returns centroid (x,y,z) for each triangle face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return ((v0 + v1 + v2) / 3.0).astype(np.float32)

    def get_face_areas(self) -> np.ndarray:
        """Returns area (mm^2) for each face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32)
