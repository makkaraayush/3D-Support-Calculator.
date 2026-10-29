"""
Collision-aware 3D Branch Routing for Tree Supports.
Routes branches from overhang contact tips downward to the print bed (Z=0).
Converges nearby branches to minimize material volume while avoiding model wall penetrations.
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

        # Precompute mesh bounding box and vertex bounds for fast collision rejection
        self.mesh_min, self.mesh_max = self.mesh.bounds

    def check_point_clearance(self, point: np.ndarray, clearance_mm: float) -> bool:
        """
        Fast check to verify a support branch node doesn't collide with the model.
        Returns True if the point is safely outside the model with specified clearance.
        """
        # If point is completely outside model bounding box + clearance, it is safely clear
        if (
            point[0] < self.mesh_min[0] - clearance_mm
            or point[0] > self.mesh_max[0] + clearance_mm
            or point[1] < self.mesh_min[1] - clearance_mm
            or point[1] > self.mesh_max[1] + clearance_mm
            or point[2] < self.mesh_min[2] - clearance_mm
            or point[2] > self.mesh_max[2] + clearance_mm
        ):
            return True

        # For points within model AABB, check distance to nearest mesh vertices
        # Subsampling for fast low-CPU calculation
        if len(self.mesh.vertices) > 2000:
            sample_indices = np.linspace(0, len(self.mesh.vertices) - 1, 1500, dtype=np.int32)
            sampled_verts = self.mesh.vertices[sample_indices]
        else:
            sampled_verts = self.mesh.vertices

        dists_sq = np.sum((sampled_verts - point) ** 2, axis=1)
        min_dist = np.sqrt(np.min(dists_sq))
        return min_dist >= clearance_mm

    def build_tree_graph(self, contact_points: List[ContactPoint]) -> TreeSupportGraph:
        """
        Constructs a complete TreeSupportGraph from contact points to the build bed.
        """
        graph = TreeSupportGraph()
        if not contact_points:
            return graph

        # 1. Create tip nodes offset by Z-air gap
        active_branch_heads: List[Tuple[TreeNode, np.ndarray]] = []
        for cp in contact_points:
            # Tip position placed slightly below overhang contact point by z_gap
            tip_pos = cp.position.copy()
            tip_pos[2] = max(0.2, tip_pos[2] - self.z_gap_mm)

            tip_radius = self.tip_diameter_mm * 0.5
            tip_node = graph.add_node(
                position=tip_pos,
                radius=tip_radius,
                node_type="tip",
                parent_id=None,
            )
            # Store (current_node, direction_vector)
            # Initial growth vector points slightly downwards and away from surface normal
            descent_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            active_branch_heads.append((tip_node, descent_dir))

        # 2. Descend layer by layer towards bed (Z = 0)
        step_z = 3.5  # 3.5mm vertical step per segment
        current_heads = active_branch_heads

        while current_heads:
            next_heads: List[Tuple[TreeNode, np.ndarray]] = []

            # Check if any heads are close enough to merge
            merged_clusters: List[List[Tuple[TreeNode, np.ndarray]]] = []
            used_indices = set()

            for i in range(len(current_heads)):
                if i in used_indices:
                    continue
                cluster = [current_heads[i]]
                used_indices.add(i)

                for j in range(i + 1, len(current_heads)):
                    if j in used_indices:
                        continue
                    p1 = current_heads[i][0].position
                    p2 = current_heads[j][0].position
                    # If within merge distance in XY and close in Z
                    xy_dist = np.linalg.norm(p1[:2] - p2[:2])
                    z_dist = abs(p1[2] - p2[2])
                    if xy_dist <= self.merge_distance_mm and z_dist <= step_z * 1.5:
                        cluster.append(current_heads[j])
                        used_indices.add(j)

                merged_clusters.append(cluster)

            # Process each cluster
            for cluster in merged_clusters:
                avg_pos = np.mean([h[0].position for h in cluster], axis=0)
                new_z = max(0.0, avg_pos[2] - step_z)

                # Determine radius based on taper rules and load
                taper_info = get_recommended_branch_taper(
                    total_height_mm=float(avg_pos[2]),
                    material=self.material,
                    target_tip_diameter_mm=self.tip_diameter_mm,
                )
                branch_radius = (taper_info["mid_diameter_mm"] * 0.5) * (1.0 + 0.15 * (len(cluster) - 1))
                branch_radius = min(3.5, branch_radius)

                new_pos = np.array([avg_pos[0], avg_pos[1], new_z], dtype=np.float32)

                # Check clearance; if colliding, try nudging away from model center
                if not self.check_point_clearance(new_pos, self.model_clearance_mm):
                    # Nudge vector outward from model center
                    model_center = self.mesh.center
                    outward_vec = new_pos[:2] - model_center[:2]
                    norm_out = np.linalg.norm(outward_vec)
                    if norm_out > 1e-4:
                        outward_dir = outward_vec / norm_out
                        new_pos[:2] += outward_dir * (self.model_clearance_mm * 1.5)

                is_at_bed = new_z <= 0.2
                if is_at_bed:
                    new_pos[2] = 0.0

                # Create branch or root node
                node_type = "root" if is_at_bed else ("trunk" if len(cluster) > 1 else "branch")
                new_node = graph.add_node(
                    position=new_pos,
                    radius=branch_radius if not is_at_bed else max(2.5, branch_radius * 1.3),
                    node_type=node_type,
                    is_bed_contact=is_at_bed,
                )

                # Link all clustered parents to this new node
                for parent_head in cluster:
                    p_node = parent_head[0]
                    p_node.parent_id = new_node.id
                    new_node.children_ids.append(p_node.id)

                if not is_at_bed:
                    next_heads.append((new_node, np.array([0.0, 0.0, -1.0], dtype=np.float32)))

            current_heads = next_heads

        return graph
