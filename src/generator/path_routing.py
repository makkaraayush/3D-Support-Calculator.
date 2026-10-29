from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ..geometry.mesh import TriangleMesh
from ..geometry.overhang_analyzer import ContactPoint
from ..geometry.collision_engine import SliceCollisionEngine
from .tree_graph import TreeSupportGraph, TreeNode
from ..physics.precalculated import lookup_optimal_z_gap


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
        self.collision = SliceCollisionEngine(mesh)
        self.mesh_min, self.mesh_max = self.mesh.bounds

    def _partition_tips_into_families(
        self, contact_points: List[ContactPoint], max_cluster_radius: float = 9.0
    ) -> List[List[ContactPoint]]:
        """Groups contact points into spatial 3D clusters so branches do not cross across tiers."""
        if not contact_points:
            return []
        if len(contact_points) <= 3:
            return [contact_points]

        pts_3d = np.array([cp.position for cp in contact_points], dtype=np.float32)
        scale = np.array([1.0, 1.0, 3.0], dtype=np.float32)
        pts_scaled = pts_3d * scale
        centroids = [pts_scaled[0]]

        for _ in range(1, 24):
            dists = np.array([min(np.linalg.norm(p - c) for c in centroids) for p in pts_scaled])
            if np.max(dists) < max_cluster_radius:
                break
            centroids.append(pts_scaled[np.argmax(dists)])

        centroids_arr = np.array(centroids, dtype=np.float32)
        families: List[List[ContactPoint]] = [[] for _ in centroids]

        for i, p in enumerate(pts_scaled):
            c_idx = int(np.argmin([np.linalg.norm(p - c) for c in centroids_arr]))
            families[c_idx].append(contact_points[i])

        return [f for f in families if len(f) > 0]

    def build_tree_graph(self, contact_points: List[ContactPoint]) -> TreeSupportGraph:
        """Constructs non-colliding organic tree support graph with exact floor and bed clearance."""
        graph = TreeSupportGraph()
        if not contact_points:
            return graph

        min_z = float(self.mesh_min[2])
        valid_points = [cp for cp in contact_points if cp.position[2] > (min_z + 0.8)]
        if not valid_points:
            return graph

        families = self._partition_tips_into_families(valid_points, self.merge_distance_mm)
        step_z = 2.5
        max_xy_step = float(step_z * np.tan(np.radians(self.branch_angle_deg)))

        for fam in families:
            # Check for horizontal model floor beneath this family
            floor_z = max(self.collision.get_floor_beneath(cp.position[:2], cp.position[2]) for cp in fam)
            is_floor_supp = floor_z > 0.8
            z_target = (floor_z + self.z_gap_mm) if is_floor_supp else 0.0

            raw_centroid = np.mean([cp.position[:2] for cp in fam], axis=0)
            trunk_axis = raw_centroid.copy().astype(np.float32)
            min_tip_z = min(cp.position[2] for cp in fam)

            # Check trunk clearance down to z_target
            z_tests = np.linspace(z_target, min_tip_z, max(3, int((min_tip_z - z_target) / 10.0)))
            for _ in range(10):
                all_safe = True
                for tz in z_tests:
                    safe, _, push = self.collision.query(
                        np.array([trunk_axis[0], trunk_axis[1], tz]),
                        radius=2.2,
                        margin=self.model_clearance_mm,
                    )
                    if not safe:
                        trunk_axis += push
                        all_safe = False
                if all_safe:
                    break

            active_heads: List[TreeNode] = []
            for cp in fam:
                pos = cp.position.copy()
                pos[2] = max(z_target + 0.2, pos[2] - self.z_gap_mm)
                tip_radius = self.tip_diameter_mm * 0.5
                node = graph.add_node(
                    position=pos,
                    radius=tip_radius,
                    node_type="tip",
                )
                active_heads.append(node)

            while active_heads:
                avg_z = float(np.mean([h.position[2] for h in active_heads]))
                if avg_z <= (z_target + 0.4):
                    break
                new_z = max(z_target, avg_z - step_z)
                is_bottom = new_z <= (z_target + 0.4)

                dists_to_axis = [float(np.linalg.norm(h.position[:2] - trunk_axis)) for h in active_heads]
                max_d = max(dists_to_axis) if dists_to_axis else 0.0

                if len(active_heads) == 1 or max_d <= 2.0 or is_bottom:
                    trunk_r = min(3.8, 0.35 * np.sqrt(len(fam)) + 1.2)
                    if is_bottom and z_target == 0.0:
                        trunk_r = max(2.6, trunk_r * 1.3)

                    pos = np.array([trunk_axis[0], trunk_axis[1], z_target if is_bottom else new_z], dtype=np.float32)
                    safe, _, push = self.collision.query(pos, radius=trunk_r, margin=self.model_clearance_mm)
                    if not safe:
                        pos[:2] += push
                        trunk_axis = pos[:2].copy()

                    node_type = "root" if is_bottom else "trunk"
                    new_node = graph.add_node(
                        position=pos,
                        radius=trunk_r,
                        node_type=node_type,
                        is_bed_contact=(is_bottom and z_target == 0.0),
                    )
                    for h in active_heads:
                        h.parent_id = new_node.id
                        new_node.children_ids.append(h.id)

                    active_heads = [] if is_bottom else [new_node]
                else:
                    next_heads: List[TreeNode] = []
                    for h in active_heads:
                        d_vec = trunk_axis - h.position[:2]
                        dist = float(np.linalg.norm(d_vec))
                        new_xy = h.position[:2].copy()
                        if dist > 1e-4:
                            new_xy += (d_vec / dist) * min(dist, max_xy_step)

                        branch_r = min(2.4, h.radius + 0.08)
                        pos = np.array([new_xy[0], new_xy[1], new_z], dtype=np.float32)
                        safe, _, push = self.collision.query(pos, radius=branch_r, margin=self.model_clearance_mm)
                        if not safe:
                            pos[:2] += push

                        new_node = graph.add_node(
                            position=pos,
                            radius=branch_r,
                            node_type="branch",
                        )
                        h.parent_id = new_node.id
                        new_node.children_ids.append(h.id)
                        next_heads.append(new_node)
                    active_heads = next_heads

        return graph
