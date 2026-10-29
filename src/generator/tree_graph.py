"""
Hierarchical Tree Support Graph representation.
Encapsulates contact tips, branching nodes, trunks, and bed anchors.
Supports topological convergence (merging nearby branches into unified trunks)
to drastically reduce filament consumption and print time.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class TreeNode:
    id: int
    position: np.ndarray  # (3,) x, y, z
    radius: float  # mm
    node_type: str  # 'tip', 'branch', 'trunk', 'root'
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    is_bed_contact: bool = False
    load_capacity_factor: float = 1.0


class TreeSupportGraph:
    def __init__(self):
        self.nodes: Dict[int, TreeNode] = {}
        self._next_id: int = 1

    def add_node(
        self,
        position: np.ndarray,
        radius: float,
        node_type: str,
        parent_id: Optional[int] = None,
        is_bed_contact: bool = False,
    ) -> TreeNode:
        node_id = self._next_id
        self._next_id += 1

        node = TreeNode(
            id=node_id,
            position=np.ascontiguousarray(position, dtype=np.float32),
            radius=float(radius),
            node_type=node_type,
            parent_id=parent_id,
            is_bed_contact=is_bed_contact,
        )
        self.nodes[node_id] = node

        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)

        return node

    def get_roots(self) -> List[TreeNode]:
        """Returns all bed anchor root nodes."""
        return [n for n in self.nodes.values() if n.is_bed_contact or n.node_type == "root"]

    def get_tips(self) -> List[TreeNode]:
        """Returns all contact tip nodes."""
        return [n for n in self.nodes.values() if n.node_type == "tip"]

    def calculate_total_branch_length(self) -> float:
        """Total linear millimeters of tree support branches."""
        total_len = 0.0
        for node in self.nodes.values():
            if node.parent_id is not None and node.parent_id in self.nodes:
                p_node = self.nodes[node.parent_id]
                dist = float(np.linalg.norm(node.position - p_node.position))
                total_len += dist
        return total_len

    def estimate_volume_mm3(self) -> float:
        """
        Estimates total solid volume in mm^3 by integrating truncated cones (frustums)
        over each graph edge: V = pi/3 * h * (r1^2 + r1*r2 + r2^2)
        """
        total_vol = 0.0
        for node in self.nodes.values():
            if node.parent_id is not None and node.parent_id in self.nodes:
                p_node = self.nodes[node.parent_id]
                h = float(np.linalg.norm(node.position - p_node.position))
                r1 = node.radius
                r2 = p_node.radius
                frustum_vol = (np.pi / 3.0) * h * (r1**2 + r1 * r2 + r2**2)
                total_vol += frustum_vol

            # Add bed flange volume if root
            if node.is_bed_contact:
                flange_r = node.radius * 2.2
                flange_vol = np.pi * (flange_r**2) * 0.6  # 0.6mm (3 layers) adhesion skirt
                total_vol += flange_vol

        return float(total_vol)
