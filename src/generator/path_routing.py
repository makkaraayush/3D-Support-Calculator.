"""
Collision-aware 3D Branch Routing for Tree Supports.
Routes branches from elevated overhang contact tips downward to the print bed (Z=0).
Converges nearby branches organically to minimize material volume while avoiding model wall penetrations.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ..geometry.mesh import TriangleMesh
from ..geometry.overhang_analyzer import ContactPoint
from .tree_graph import TreeSupportGraph, TreeNode
from ..physics.precalculated import lookup_optimal_z_gap, get_recommended_branch_taper


class BranchRouter:
    def __init__(
        self,
        mesh: TriangleMesh,
        material: str = "PLA",
        layer_height: float = 0.20,
        branch_angle_deg: float = 28.0,
        merge_distance_mm: float = 9.0,
        tip_diameter_mm: float = 0.7,
        model_clearance_mm: float = 1.4,
    ):
        self.mesh = mesh
        self.material = material
        self.layer_height = layer_height
        self.branch_angle_deg = branch_angle_deg
        self.merge_distance_mm = merge_distance_mm
        self.tip_diameter_mm = tip_diameter_mm
        self.model_clearance_mm = model_clearance_mm
        self.z_gap_mm = lookup_optimal_z_gap(material, layer_height)

        # Precompute mesh bounds
        self.mesh_min, self.mesh_max = self.mesh.bounds
        self.mesh_center = self.mesh.center

    def check_point_clearance(self, point: np.ndarray, clearance_mm: float) -> bool:
        """
        Verifies that a support branch node doesn't collide with model geometry.
        Returns True if the point is safely outside the model with specified clearance.
        """
        # Fast AABB check with clearance margin
        if (
            point[0] < self.mesh_min[0] - clearance_mm
            or point[0] > self.mesh_max[0] + clearance_mm
            or point[1] < self.mesh_min[1] - clearance_mm
            or point[1] > self.mesh_max[1] + clearance_mm
            or point[2] < self.mesh_min[2] - clearance_mm
            or point[2] > self.mesh_max[2] + clearance_mm
        ):
            return True

        # Check distance against model vertices near this Z slice
        z_slice_mask = np.abs(self.mesh.vertices[:, 2] - point[2]) <= (clearance_mm * 2.0)
        if not np.any(z_slice_mask):
            return True

        slice_verts = self.mesh.vertices[z_slice_mask]
        dists_sq = np.sum((slice_verts - point) ** 2, axis=1)
        min_dist = np.sqrt(np.min(dists_sq))
        return min_dist >= clearance_mm

    def build_tree_graph(self, contact_points: List[ContactPoint]) -> TreeSupportGraph:
        """
        Constructs a complete TreeSupportGraph from contact points to the build bed.
        """
        graph = TreeSupportGraph()
        if not contact_points:
            return graph

        # Filter out any accidental bed contact points (Z <= 0.8)
        valid_contact_points = [
            cp for cp in contact_points if cp.position[2] > (self.mesh_min[2] + 0.8)
        ]
        if not valid_contact_points:
            return graph

        # 1. Create tip nodes offset by Z-air gap
        active_branch_heads: List[TreeNode] = []
        for cp in valid_contact_points:
            tip_pos = cp.position.copy()
            # Position tip exactly at interface layer
            tip_pos[2] = max(0.2, tip_pos[2] - self.z_gap_mm)

            tip_radius = self.tip_diameter_mm * 0.5
            tip_node = graph.add_node(
                position=tip_pos,
                radius=tip_radius,
                node_type="tip",
                parent_id=None,
            )
            active_branch_heads.append(tip_node)

        # 2. Descend layer by layer towards build plate (Z = 0)
        step_z = 3.0  # 3.0 mm vertical descent per segment
        current_heads = active_branch_heads

        while current_heads:
            next_heads: List[TreeNode] = []

            # Group heads into merge clusters
            clusters: List[List[TreeNode]] = []
            used_indices = set()

            for i in range(len(current_heads)):
                if i in used_indices:
                    continue
                cluster = [current_heads[i]]
                used_indices.add(i)

                p1 = current_heads[i].position
                for j in range(i + 1, len(current_heads)):
                    if j in used_indices:
                        continue
                    p2 = current_heads[j].position
                    xy_dist = np.linalg.norm(p1[:2] - p2[:2])
                    z_dist = abs(p1[2] - p2[2])
                    if xy_dist <= self.merge_distance_mm and z_dist <= step_z * 1.5:
                        cluster.append(current_heads[j])
                        used_indices.add(j)

                clusters.append(cluster)

            # Process each cluster
            for cluster in clusters:
                avg_pos = np.mean([h.position for h in cluster], axis=0)
                new_z = max(0.0, avg_pos[2] - step_z)

                taper_info = get_recommended_branch_taper(
                    total_height_mm=float(avg_pos[2]),
                    material=self.material,
                    target_tip_diameter_mm=self.tip_diameter_mm,
                )
                branch_radius = (taper_info["mid_diameter_mm"] * 0.5) * (1.0 + 0.12 * (len(cluster) - 1))
                branch_radius = min(4.0, max(1.2, branch_radius))

                new_pos = np.array([avg_pos[0], avg_pos[1], new_z], dtype=np.float32)

                # Collision clearance check and avoidance
                if not self.check_point_clearance(new_pos, self.model_clearance_mm):
                    outward_vec = new_pos[:2] - self.mesh_center[:2]
                    norm_out = np.linalg.norm(outward_vec)
                    if norm_out > 1e-4:
                        outward_dir = outward_vec / norm_out
                        new_pos[:2] += outward_dir * (self.model_clearance_mm * 1.8)

                is_at_bed = new_z <= 0.4
                if is_at_bed:
                    new_pos[2] = 0.0

                node_type = "root" if is_at_bed else ("trunk" if len(cluster) > 1 else "branch")
                new_node = graph.add_node(
                    position=new_pos,
                    radius=branch_radius if not is_at_bed else max(2.8, branch_radius * 1.4),
                    node_type=node_type,
                    is_bed_contact=is_at_bed,
                )

                # Connect clustered parents to this descending node
                for parent_node in cluster:
                    parent_node.parent_id = new_node.id
                    new_node.children_ids.append(parent_node.id)

                if not is_at_bed:
                    next_heads.append(new_node)

            current_heads = next_heads

        return graph
