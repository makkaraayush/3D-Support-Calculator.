"""
Vectorized Overhang and Feature Analyzer for 3D Triangle Meshes.
Detects:
- Overhang angles per face (0 deg = vertical wall, 90 deg = horizontal ceiling)
- Excludes bed-contact base faces (faces touching the build plate are self-supported by bed)
- Critical overhang zones exceeding material stability thresholds
- Horizontal bridging candidates eligible for bridge infill without supports
- Spatial clustering to generate sparse, high-efficiency contact anchor tips
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .mesh import TriangleMesh
from ..physics.precalculated import lookup_max_clean_bridge_span


@dataclass
class ContactPoint:
    position: np.ndarray  # (3,) x, y, z in mm
    normal: np.ndarray  # (3,) surface normal
    overhang_deg: float
    area_weight: float  # mm^2 of supported region
    requires_support: bool = True
    is_hole_ceiling: bool = False
    is_bridge: bool = False


class OverhangAnalyzer:
    def __init__(
        self,
        mesh: TriangleMesh,
        critical_angle_deg: float = 58.0,
        material: str = "PLA",
        layer_height: float = 0.20,
    ):
        self.mesh = mesh
        self.critical_angle_deg = critical_angle_deg
        self.material = material
        self.layer_height = layer_height
        self.max_bridge_span = lookup_max_clean_bridge_span(material, layer_height)

    def compute_face_overhang_angles(self) -> np.ndarray:
        """
        Computes overhang angle for every face in degrees [0, 90] from vertical.
        - Vertical wall (nz = 0): 0.0 deg
        - 45 deg overhang: 45.0 deg
        - Horizontal downward ceiling (nz = -1): 90.0 deg
        - Upward or horizontal faces (nz >= 0): 0.0 deg
        """
        nz = self.mesh.face_normals[:, 2]
        downward_mask = nz < -1e-4

        angles = np.zeros(len(self.mesh.faces), dtype=np.float32)
        if np.any(downward_mask):
            clamped_nz = np.clip(np.abs(nz[downward_mask]), 0.0, 1.0)
            angles[downward_mask] = np.degrees(np.arcsin(clamped_nz))

        return angles

    def get_critical_overhang_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (angles, critical_mask) where critical_mask includes ONLY faces that:
        1. Point downward with angle > critical_angle_deg (e.g. > 58 deg)
        2. Are elevated above the print bed (faces on the bed z <= min_z + 0.8mm are self-supported!)
        """
        angles = self.compute_face_overhang_angles()
        centers = self.mesh.get_face_centers()

        # Build plate threshold: faces touching bed do NOT need support
        min_z = float(self.mesh.bounds[0][2])
        bed_threshold = min_z + max(0.8, self.layer_height * 3.0)
        elevated_mask = centers[:, 2] > bed_threshold

        angle_mask = angles > self.critical_angle_deg
        critical_mask = angle_mask & elevated_mask

        return angles, critical_mask

    def analyze(self) -> Dict[str, Any]:
        """Performs full geometric overhang, bridge, and hole analysis."""
        angles, critical_mask = self.get_critical_overhang_mask()
        areas = self.mesh.get_face_areas()

        total_area = float(np.sum(areas))
        critical_area = float(np.sum(areas[critical_mask]))
        critical_ratio = (critical_area / max(1e-5, total_area)) * 100.0

        # Detect horizontal ceilings (bridges): elevated faces with angles >= 85 deg
        centers = self.mesh.get_face_centers()
        min_z = float(self.mesh.bounds[0][2])
        elevated = centers[:, 2] > (min_z + 0.8)
        ceiling_mask = (angles >= 85.0) & elevated
        bridge_count = int(np.sum(ceiling_mask))

        return {
            "total_faces": len(self.mesh.faces),
            "total_surface_area_mm2": total_area,
            "critical_overhang_area_mm2": critical_area,
            "critical_overhang_pct": critical_ratio,
            "critical_face_count": int(np.sum(critical_mask)),
            "bridge_face_count": bridge_count,
            "critical_angle_threshold_deg": self.critical_angle_deg,
            "angles": angles,
            "critical_mask": critical_mask,
        }

    @staticmethod
    def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """Determines whether 2D projection of point p lies within triangle (a, b, c) in XY."""
        v0 = b[:2] - a[:2]
        v1 = c[:2] - a[:2]
        v2 = p[:2] - a[:2]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-9:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return (u >= -1e-4) and (v >= -1e-4) and (u + v <= 1.0001)

    def generate_contact_points(
        self,
        grid_resolution_mm: float = 3.5,
        min_cluster_area_mm2: float = 0.20,
    ) -> List[ContactPoint]:
        """
        Samples and clusters critical elevated overhang surfaces into discrete,
        spatially distributed contact anchor tips across large overhang spans.
        Excludes bed-contact base faces and internally-supported surfaces (e.g. resting on pillars).
        """
        angles, critical_mask = self.get_critical_overhang_mask()
        if not np.any(critical_mask):
            return []

        min_z = float(self.mesh.bounds[0][2])
        bed_threshold = min_z + max(0.8, self.layer_height * 3.0)

        # Precompute upward-facing faces to check for internal support contact
        up_face_indices = [
            fi for fi in range(len(self.mesh.faces))
            if self.mesh.face_normals[fi, 2] > 0.4 and self.mesh.get_face_centers()[fi, 2] > bed_threshold
        ]

        def is_internally_supported(p: np.ndarray, z_tol: float = 0.3) -> bool:
            for ufi in up_face_indices:
                uv0 = self.mesh.vertices[self.mesh.faces[ufi, 0]]
                uv1 = self.mesh.vertices[self.mesh.faces[ufi, 1]]
                uv2 = self.mesh.vertices[self.mesh.faces[ufi, 2]]
                if abs(uv0[2] - p[2]) < z_tol and abs(uv1[2] - p[2]) < z_tol:
                    if self._point_in_triangle_2d(p, uv0, uv1, uv2):
                        return True
            return False

        spacing = max(1.5, grid_resolution_mm)
        crit_indices = np.where(critical_mask)[0]

        raw_points: List[np.ndarray] = []
        raw_normals: List[np.ndarray] = []
        raw_angles: List[float] = []
        raw_areas: List[float] = []

        for fi in crit_indices:
            v0 = self.mesh.vertices[self.mesh.faces[fi, 0]]
            v1 = self.mesh.vertices[self.mesh.faces[fi, 1]]
            v2 = self.mesh.vertices[self.mesh.faces[fi, 2]]
            f_norm = self.mesh.face_normals[fi]
            f_ang = float(angles[fi])
            f_area = float(self.mesh.get_face_areas()[fi])

            e1 = v1 - v0
            e2 = v2 - v0
            l1 = float(np.linalg.norm(e1))
            l2 = float(np.linalg.norm(e2))

            n1 = max(1, int(np.ceil(l1 / spacing)))
            n2 = max(1, int(np.ceil(l2 / spacing)))

            # If triangle is small, single centroid sample is sufficient
            if n1 <= 1 and n2 <= 1:
                centroid = (v0 + v1 + v2) / 3.0
                if centroid[2] > bed_threshold and not is_internally_supported(centroid):
                    raw_points.append(centroid)
                    raw_normals.append(f_norm)
                    raw_angles.append(f_ang)
                    raw_areas.append(f_area)
                continue

            # For large overhang faces, sample a uniform grid across the face
            sample_count = 0
            face_pts: List[np.ndarray] = []
            for i in range(n1):
                for j in range(n2):
                    s = (i + 0.5) / n1
                    t = (j + 0.5) / n2
                    if s + t <= 1.0:
                        pt = v0 + s * e1 + t * e2
                        if pt[2] > bed_threshold and not is_internally_supported(pt):
                            face_pts.append(pt)
                            sample_count += 1

            if face_pts:
                sub_area = f_area / max(1, sample_count)
                for pt in face_pts:
                    raw_points.append(pt)
                    raw_normals.append(f_norm)
                    raw_angles.append(f_ang)
                    raw_areas.append(sub_area)

        if not raw_points:
            return []

        pts_arr = np.array(raw_points, dtype=np.float32)
        norm_arr = np.array(raw_normals, dtype=np.float32)
        ang_arr = np.array(raw_angles, dtype=np.float32)
        area_arr = np.array(raw_areas, dtype=np.float32)

        # Voxel binning in (X, Y, Z) space to cluster into evenly spaced tips
        inv_res = 1.0 / spacing
        voxel_coords = np.floor(pts_arr * inv_res).astype(np.int64)

        voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, vc in enumerate(voxel_coords):
            key = (int(vc[0]), int(vc[1]), int(vc[2]))
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(idx)

        contact_points: List[ContactPoint] = []
        for key, indices in voxel_dict.items():
            cluster_area = float(np.sum(area_arr[indices]))
            if cluster_area < min_cluster_area_mm2:
                continue

            avg_pos = np.mean(pts_arr[indices], axis=0)
            avg_normal = np.mean(norm_arr[indices], axis=0)
            norm_mag = float(np.linalg.norm(avg_normal))
            if norm_mag > 1e-6:
                avg_normal = avg_normal / norm_mag
            else:
                avg_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            max_ang = float(np.max(ang_arr[indices]))
            is_flat_ceiling = max_ang >= 85.0

            contact_points.append(
                ContactPoint(
                    position=avg_pos.astype(np.float32),
                    normal=avg_normal.astype(np.float32),
                    overhang_deg=max_ang,
                    area_weight=cluster_area,
                    requires_support=True,
                    is_bridge=is_flat_ceiling,
                )
            )

        return contact_points
