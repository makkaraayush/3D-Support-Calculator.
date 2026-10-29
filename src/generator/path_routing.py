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
        """Groups contact points into spatial tree clusters so branches do not cross."""
        if not contact_points:
            return []
        if len(contact_points) <= 3:
            return [contact_points]

        pts_xy = np.array([cp.position[:2] for cp in contact_points], dtype=np.float32)
        centroids = [pts_xy[0]]

        # Seed initial centroids spaced apart
        for _ in range(1, 12):
            dists = np.array([min(np.linalg.norm(p - c) for c in centroids) for p in pts_xy])
            if np.max(dists) < max_cluster_radius:
                break
            centroids.append(pts_xy[np.argmax(dists)])

        centroids_arr = np.array(centroids, dtype=np.float32)
        families: List[List[ContactPoint]] = [[] for _ in centroids]

        for i, p in enumerate(pts_xy):
            c_idx = int(np.argmin([np.linalg.norm(p - c) for c in centroids_arr]))
            families[c_idx].append(contact_points[i])

        return [f for f in families if len(f) > 0]

    def build_tree_graph(self, contact_points: List[ContactPoint]) -> TreeSupportGraph:
        """Constructs non-colliding organic tree support graph."""
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
            # Safe trunk target axis for this family
            raw_centroid = np.mean([cp.position[:2] for cp in fam], axis=0)
            trunk_axis = raw_centroid.copy().astype(np.float32)

            # Check that trunk axis maintains wall clearance down to the bed
            for test_z in [0.0, 5.0, 15.0]:
                safe, _, push = self.collision.query(
                    np.array([trunk_axis[0], trunk_axis[1], test_z]),
                    radius=2.2,
                    margin=self.model_clearance_mm,
                )
                if not safe:
                    trunk_axis += push

            # Initialize tip nodes offset by vertical air gap
            active_heads: List[TreeNode] = []
            for cp in fam:
                pos = cp.position.copy()
                pos[2] = max(0.2, pos[2] - self.z_gap_mm)
                tip_radius = self.tip_diameter_mm * 0.5
                node = graph.add_node(
                    position=pos,
                    radius=tip_radius,
                    node_type="tip",
                )
                active_heads.append(node)

            # Descend layer by layer towards bed
            while active_heads:
                avg_z = float(np.mean([h.position[2] for h in active_heads]))
                new_z = max(0.0, avg_z - step_z)
                is_bed = new_z <= 0.4

                # Check convergence towards trunk axis
                dists_to_axis = [float(np.linalg.norm(h.position[:2] - trunk_axis)) for h in active_heads]
                max_d = max(dists_to_axis) if dists_to_axis else 0.0

                if len(active_heads) == 1 or max_d <= 2.0 or is_bed:
                    # Merge branches into unified trunk
                    trunk_r = min(3.8, 0.35 * np.sqrt(len(fam)) + 1.2)
                    if is_bed:
                        trunk_r = max(2.6, trunk_r * 1.3)

                    pos = np.array([trunk_axis[0], trunk_axis[1], 0.0 if is_bed else new_z], dtype=np.float32)
                    safe, _, push = self.collision.query(pos, radius=trunk_r, margin=self.model_clearance_mm)
                    if not safe:
                        pos[:2] += push
                        trunk_axis = pos[:2].copy()

                    node_type = "root" if is_bed else "trunk"
                    new_node = graph.add_node(
                        position=pos,
                        radius=trunk_r,
                        node_type=node_type,
                        is_bed_contact=is_bed,
                    )
                    for h in active_heads:
                        h.parent_id = new_node.id
                        new_node.children_ids.append(h.id)

                    if not is_bed:
                        active_heads = [new_node]
                    else:
                        active_heads = []
                else:
                    # Steer branches inward towards trunk axis
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
