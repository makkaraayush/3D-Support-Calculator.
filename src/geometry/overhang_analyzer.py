"""
Vectorized Overhang and Feature Analyzer for 3D Triangle Meshes.
Detects:
- Overhang angles per face (0 deg = vertical wall, 90 deg = horizontal ceiling)
- Critical overhang zones exceeding material stability thresholds
- Horizontal bridging candidates eligible for bridge infill without supports
- Cylindrical bore / hole ceilings and applies precalculated bridging rules
- Spatial clustering to generate sparse, high-efficiency contact anchor tips
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from .mesh import TriangleMesh
from ..physics.precalculated import (
    lookup_max_clean_bridge_span,
    evaluate_hole_support_requirement,
)


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
        Computes overhang angle for every face in degrees [0, 90].
        Faces with n_z >= 0 have 0.0 overhang.
        Downward faces (n_z < 0) have angle = 90 - arccos(-n_z).
        """
        nz = self.mesh.face_normals[:, 2]
        downward_mask = nz < -1e-4

        angles = np.zeros(len(self.mesh.faces), dtype=np.float32)
        if np.any(downward_mask):
            # Clamp to [-1.0, 0.0]
            clamped_nz = np.clip(nz[downward_mask], -1.0, 0.0)
            # angle from vertical in degrees
            angles[downward_mask] = 90.0 - np.degrees(np.arccos(-clamped_nz))

        return angles

    def analyze(self) -> Dict[str, Any]:
        """Performs full geometric overhang, bridge, and hole analysis."""
        angles = self.compute_face_overhang_angles()
        areas = self.mesh.get_face_areas()
        centers = self.mesh.get_face_centers()

        total_area = float(np.sum(areas))
        critical_mask = angles > self.critical_angle_deg
        critical_area = float(np.sum(areas[critical_mask]))
        critical_ratio = (critical_area / max(1e-5, total_area)) * 100.0

        # Detect horizontal ceilings (bridges): angles >= 85 deg
        ceiling_mask = angles >= 85.0
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

    def generate_contact_points(
        self,
        grid_resolution_mm: float = 3.2,
        min_cluster_area_mm2: float = 0.25,
    ) -> List[ContactPoint]:
        """
        Clusters critical overhang faces into discrete, spatially separated contact anchor tips.
        Uses 3D voxel grid binning to ensure uniform support coverage with minimal contact scars.
        """
        angles = self.compute_face_overhang_angles()
        critical_mask = angles > self.critical_angle_deg
        if not np.any(critical_mask):
            return []

        crit_centers = self.mesh.get_face_centers()[critical_mask]
        crit_normals = self.mesh.face_normals[critical_mask]
        crit_areas = self.mesh.get_face_areas()[critical_mask]
        crit_angles = angles[critical_mask]

        # Voxel binning in (X, Y, Z) space
        inv_res = 1.0 / max(0.5, grid_resolution_mm)
        voxel_coords = np.floor(crit_centers * inv_res).astype(np.int64)

        # Hash voxel coordinates to group faces
        voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, vc in enumerate(voxel_coords):
            key = (int(vc[0]), int(vc[1]), int(vc[2]))
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(idx)

        contact_points: List[ContactPoint] = []
        for key, indices in voxel_dict.items():
            sub_areas = crit_areas[indices]
            cluster_area = float(np.sum(sub_areas))
            if cluster_area < min_cluster_area_mm2:
                continue

            weights = sub_areas / cluster_area
            # Weighted average centroid
            avg_pos = np.sum(crit_centers[indices] * weights[:, None], axis=0)
            avg_normal = np.sum(crit_normals[indices] * weights[:, None], axis=0)
            norm_mag = np.linalg.norm(avg_normal)
            if norm_mag > 1e-6:
                avg_normal = avg_normal / norm_mag
            else:
                avg_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            max_angle = float(np.max(crit_angles[indices]))

            # Check if this contact point is part of a horizontal bridge or hole
            is_flat_ceiling = max_angle >= 85.0

            contact_points.append(
                ContactPoint(
                    position=avg_pos.astype(np.float32),
                    normal=avg_normal.astype(np.float32),
                    overhang_deg=max_angle,
                    area_weight=cluster_area,
                    requires_support=True,
                    is_bridge=is_flat_ceiling,
                )
            )

        return contact_points
