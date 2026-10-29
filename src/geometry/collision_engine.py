import numpy as np
from typing import List, Tuple, Dict
from .mesh import TriangleMesh


class SliceCollisionEngine:
    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh
        self._cache: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = {}

        # Precompute face bounds for fast slice intersection filtering
        v0 = self.mesh.vertices[self.mesh.faces[:, 0]]
        v1 = self.mesh.vertices[self.mesh.faces[:, 1]]
        v2 = self.mesh.vertices[self.mesh.faces[:, 2]]
        self._f_zmin = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
        self._f_zmax = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])

    def get_slice_segments(self, z: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Snap z to 0.1mm resolution for exact slice clearance and cache reuse
        z_key = round(float(z) * 10.0) / 10.0
        if z_key in self._cache:
            return self._cache[z_key]

        intersecting = np.where((self._f_zmin <= z_key) & (self._f_zmax >= z_key))[0]
        segments: List[Tuple[np.ndarray, np.ndarray]] = []

        for fi in intersecting:
            verts = self.mesh.vertices[self.mesh.faces[fi]]
            pts: List[np.ndarray] = []
            for i in range(3):
                pA = verts[i]
                pB = verts[(i + 1) % 3]
                if (pA[2] <= z_key and pB[2] >= z_key) or (pB[2] <= z_key and pA[2] >= z_key):
                    dz = pB[2] - pA[2]
                    if abs(dz) > 1e-6:
                        t = (z_key - pA[2]) / dz
                        if 0.0 <= t <= 1.0:
                            pts.append(pA[:2] + t * (pB[:2] - pA[:2]))
            if len(pts) >= 2:
                segments.append((pts[0], pts[1]))

        self._cache[z_key] = segments
        return segments

    def query(
        self, point_3d: np.ndarray, radius: float, margin: float = 1.2
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Queries clearance between a 3D branch node and model surface.
        Returns:
            is_safe: True if point is outside the model and >= (radius + margin) away from walls
            signed_dist: Positive if outside, negative if inside model geometry
            push_vector: Direction and distance to move point into safe open air
        """
        x, y, z = float(point_3d[0]), float(point_3d[1]), float(point_3d[2])
        segs = self.get_slice_segments(z)
        if not segs:
            return True, 999.0, np.zeros(2, dtype=np.float32)

        p = np.array([x, y], dtype=np.float32)
        min_d = 999.0
        best_diff = np.zeros(2, dtype=np.float32)
        ray_count = 0

        for a, b in segs:
            # Raycast along +X for point-in-polygon
            if (a[1] > y) != (b[1] > y):
                denom = b[1] - a[1]
                if abs(denom) > 1e-9:
                    x_int = (y - a[1]) * (b[0] - a[0]) / denom + a[0]
                    if x_int > x:
                        ray_count += 1

            # Point to 2D line segment distance
            v = b - a
            w = p - a
            c1 = np.dot(w, v)
            if c1 <= 0:
                diff = p - a
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    diff = p - b
                else:
                    diff = p - (a + (c1 / max(1e-9, c2)) * v)

            d = float(np.linalg.norm(diff))
            if d < min_d:
                min_d = d
                best_diff = diff

        is_inside = (ray_count % 2) == 1
        needed = radius + margin
        norm = float(np.linalg.norm(best_diff))
        n_out = (best_diff / max(1e-6, norm)).astype(np.float32)

        if is_inside:
            push = n_out * (needed + min_d)
            return False, -min_d, push
        else:
            if min_d < needed:
                push = n_out * (needed - min_d)
                return False, min_d, push
            return True, min_d, np.zeros(2, dtype=np.float32)

    def get_bed_clearance(self, xy_pos: np.ndarray) -> float:
        """Returns distance from 2D point to nearest model footprint edge at bed level (Z=0)."""
        segs = self.get_slice_segments(0.0)
        if not segs:
            segs = self.get_slice_segments(0.4)
        if not segs:
            return 999.0

        p = xy_pos[:2].astype(np.float32)
        min_d = 999.0
        for a, b in segs:
            v = b - a
            w = p - a
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = float(np.linalg.norm(p - a))
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = float(np.linalg.norm(p - b))
                else:
                    d = float(np.linalg.norm(p - (a + (c1 / max(1e-9, c2)) * v)))
            if d < min_d:
                min_d = d
        return min_d
