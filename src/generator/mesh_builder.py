"""
Translates abstract TreeSupportGraph structures into solid 3D TriangleMesh geometry.
Constructs lightweight, manifold frustum segments, contact tip interfaces, and bed adhesion discs
ready for Three.js 3D web preview and STL / Bambu 3MF file export.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from ..geometry.mesh import TriangleMesh
from .tree_graph import TreeSupportGraph, TreeNode


class SupportMeshBuilder:
    def __init__(self, radial_segments: int = 10):
        self.radial_segments = max(6, radial_segments)

    def build_mesh(
        self, graph: TreeSupportGraph, collision_engine: Optional[Any] = None
    ) -> TriangleMesh:
        """Constructs a combined 3D TriangleMesh from all branches and tips in the graph."""
        if not graph.nodes:
            return TriangleMesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32))

        all_vertices: List[np.ndarray] = []
        all_faces: List[List[int]] = []
        current_v_offset = 0

        for node in graph.nodes.values():
            # Connect branch edge to parent
            if node.parent_id is not None and node.parent_id in graph.nodes:
                p_node = graph.nodes[node.parent_id]
                verts, faces = self._create_frustum(
                    p1=node.position,
                    p2=p_node.position,
                    r1=max(0.3, node.radius),
                    r2=max(0.3, p_node.radius),
                )
                if len(verts) > 0:
                    all_vertices.append(verts)
                    all_faces.append(faces + current_v_offset)
                    current_v_offset += len(verts)

            # Build bed adhesion disc with collision clearance from model base
            if (node.is_bed_contact or (node.node_type == "root" and node.position[2] <= 0.4)) and node.position[2] <= 0.4:
                flange_r = max(2.4, node.radius * 1.5)
                if collision_engine is not None:
                    bed_dist = collision_engine.get_bed_clearance(node.position)
                    if bed_dist < 900.0:
                        flange_r = min(flange_r, max(node.radius + 0.1, bed_dist - 0.4))
                verts, faces = self._create_bed_disc(node.position, radius=flange_r, height=0.4)
                if len(verts) > 0:
                    all_vertices.append(verts)
                    all_faces.append(faces + current_v_offset)
                    current_v_offset += len(verts)

            # Cap bottom interface root nodes resting on model surfaces (Z > 0.4)
            if node.node_type == "root" and node.position[2] > 0.4:
                verts, faces = self._create_tip_cap(node.position, radius=node.radius)
                if len(verts) > 0:
                    all_vertices.append(verts)
                    all_faces.append(faces + current_v_offset)
                    current_v_offset += len(verts)

            # Cap tip nodes with closed circular cap at exact interface position
            if node.node_type == "tip":
                verts, faces = self._create_tip_cap(node.position, radius=node.radius)
                if len(verts) > 0:
                    all_vertices.append(verts)
                    all_faces.append(faces + current_v_offset)
                    current_v_offset += len(verts)

        if not all_vertices:
            return TriangleMesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32))

        merged_verts = np.vstack(all_vertices).astype(np.float32)
        merged_faces = np.vstack(all_faces).astype(np.int32)

        return TriangleMesh(merged_verts, merged_faces)

    def _create_frustum(
        self, p1: np.ndarray, p2: np.ndarray, r1: float, r2: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a truncated cone segment between 3D points p1 and p2."""
        axis = p2 - p1
        length = float(np.linalg.norm(axis))
        if length < 1e-4:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        z_axis = axis / length

        # Find orthogonal vectors: if z_axis is close to Z axis, choose X as arbitrary
        if abs(z_axis[2]) > 0.85:
            arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            arbitrary = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        x_axis = np.cross(arbitrary, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            x_axis = np.cross(arbitrary, z_axis)
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(1e-6, np.linalg.norm(y_axis))

        thetas = np.linspace(0, 2 * np.pi, self.radial_segments, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Top circle at p1
        top_circle = (
            p1[None, :]
            + (cos_t[:, None] * x_axis[None, :] * r1)
            + (sin_t[:, None] * y_axis[None, :] * r1)
        )
        # Bottom circle at p2
        bottom_circle = (
            p2[None, :]
            + (cos_t[:, None] * x_axis[None, :] * r2)
            + (sin_t[:, None] * y_axis[None, :] * r2)
        )

        verts = np.vstack([top_circle, bottom_circle]).astype(np.float32)
        verts[:, 2] = np.maximum(0.0, verts[:, 2])
        faces: List[List[int]] = []
        n = self.radial_segments

        for i in range(n):
            next_i = (i + 1) % n
            t1 = i
            t2 = next_i
            b1 = i + n
            b2 = next_i + n
            faces.append([t1, b1, t2])
            faces.append([t2, b1, b2])

        return verts, np.array(faces, dtype=np.int32)

    def _create_bed_disc(self, center: np.ndarray, radius: float, height: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a flat cylindrical adhesion brim disc on build plate Z=0."""
        thetas = np.linspace(0, 2 * np.pi, self.radial_segments, endpoint=False)
        x = radius * np.cos(thetas)
        y = radius * np.sin(thetas)

        c_base = np.array([center[0], center[1], 0.0], dtype=np.float32)
        c_top = np.array([center[0], center[1], height], dtype=np.float32)

        verts_bot = np.column_stack([center[0] + x, center[1] + y, np.zeros_like(x)])
        verts_top = np.column_stack([center[0] + x, center[1] + y, np.full_like(x, height)])

        verts = np.vstack([c_base, verts_bot, c_top, verts_top]).astype(np.float32)
        faces: List[List[int]] = []
        n = self.radial_segments
        top_c_idx = n + 1

        for i in range(n):
            next_i = (i + 1) % n
            faces.append([0, 1 + next_i, 1 + i])
            faces.append([top_c_idx, top_c_idx + 1 + i, top_c_idx + 1 + next_i])
            b1 = 1 + i
            b2 = 1 + next_i
            t1 = top_c_idx + 1 + i
            t2 = top_c_idx + 1 + next_i
            faces.append([b1, t1, b2])
            faces.append([b2, t1, t2])

        return verts, np.array(faces, dtype=np.int32)

    def _create_tip_cap(self, tip_pos: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a conical contact tip pointed upward toward the overhang interface."""
        thetas = np.linspace(0, 2 * np.pi, self.radial_segments, endpoint=False)
        x = radius * np.cos(thetas)
        y = radius * np.sin(thetas)

        apex = np.array([tip_pos[0], tip_pos[1], tip_pos[2] + 0.3], dtype=np.float32)
        ring_verts = np.column_stack([tip_pos[0] + x, tip_pos[1] + y, np.full_like(x, tip_pos[2])])

        verts = np.vstack([apex, ring_verts]).astype(np.float32)
        faces: List[List[int]] = []
        n = self.radial_segments

        for i in range(n):
            next_i = (i + 1) % n
            faces.append([0, 1 + i, 1 + next_i])

        return verts, np.array(faces, dtype=np.int32)
