"""
Unit tests for tree support generator, branch router, and physics evaluator.
"""

import unittest
import numpy as np
from src.geometry.mesh import TriangleMesh
from src.geometry.overhang_analyzer import OverhangAnalyzer, ContactPoint
from src.generator.path_routing import BranchRouter
from src.generator.mesh_builder import SupportMeshBuilder
from src.simulator.physics_evaluator import SupportPhysicsEvaluator


def create_t_shape_mesh() -> TriangleMesh:
    """Creates a 'T' shaped mesh with prominent overhang wings."""
    # Stem: 10x10x20 mm at bottom
    # Cap: 30x10x10 mm on top
    vertices = np.array([
        # Stem base (Z=0..20, X=-5..5, Y=-5..5)
        [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0],
        [-5, -5, 20], [5, -5, 20], [5, 5, 20], [-5, 5, 20],
        # Cap wings (Z=20..30, X=-15..15, Y=-5..5)
        [-15, -5, 20], [-15, 5, 20], [15, -5, 20], [15, 5, 20],
        [-15, -5, 30], [-15, 5, 30], [15, -5, 30], [15, 5, 30],
    ], dtype=np.float32)

    # Simplified box faces for the T-shape
    faces = np.array([
        # Stem base
        [0, 2, 1], [0, 3, 2],
        # Stem vertical walls (Z=0..20)
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
        # Overhang wings under surface (Z=20, facing down)
        [8, 9, 4], [9, 7, 4],
        [5, 6, 10], [6, 11, 10],
        # Top cap
        [12, 14, 13], [13, 14, 15],
    ], dtype=np.int32)

    return TriangleMesh(vertices, faces)


class TestTreeGenerator(unittest.TestCase):
    def setUp(self):
        self.mesh = create_t_shape_mesh()
        self.analyzer = OverhangAnalyzer(self.mesh, critical_angle_deg=50.0)
        self.contact_points = self.analyzer.generate_contact_points()

    def test_overhang_detection_on_t_shape(self):
        # The T-shape has downward facing wing faces
        self.assertGreaterEqual(len(self.contact_points), 1)

    def test_branch_routing(self):
        router = BranchRouter(
            mesh=self.mesh,
            material="PLA",
            layer_height=0.20,
            merge_distance_mm=10.0,
        )
        graph = router.build_tree_graph(self.contact_points)
        self.assertGreater(len(graph.nodes), 0)

        roots = graph.get_roots()
        self.assertGreaterEqual(len(roots), 1)
        # Verify roots are near bed (Z ~ 0)
        for r in roots:
            self.assertAlmostEqual(r.position[2], 0.0, places=1)

    def test_mesh_builder(self):
        router = BranchRouter(self.mesh, "PLA", 0.20)
        graph = router.build_tree_graph(self.contact_points)
        builder = SupportMeshBuilder(radial_segments=8)
        support_mesh = builder.build_mesh(graph)

        self.assertGreater(len(support_mesh.vertices), 0)
        self.assertGreater(len(support_mesh.faces), 0)

    def test_physics_evaluator(self):
        router = BranchRouter(self.mesh, "PLA", 0.20)
        graph = router.build_tree_graph(self.contact_points)
        evaluator = SupportPhysicsEvaluator(self.mesh, "PLA", 0.20)
        report = evaluator.evaluate(graph, self.contact_points, priority_profile="balanced")

        self.assertGreater(report.coverage_score_pct, 50.0)
        self.assertGreater(report.filament_saved_vs_standard_grid_pct, 0.0)
        self.assertGreater(report.removal_ease_score, 5.0)
        self.assertGreater(report.composite_fitness_score, 40.0)
        self.assertGreater(report.print_time_minutes, 1.0)
        self.assertGreater(report.grid_print_time_minutes, report.print_time_minutes)

    def test_slice_collision_clearance(self):
        from src.geometry.collision_engine import SliceCollisionEngine
        engine = SliceCollisionEngine(self.mesh)
        # Point inside stem (X=0, Y=0, Z=10)
        safe, dist, push = engine.query(np.array([0.0, 0.0, 10.0]), radius=1.0)
        self.assertFalse(safe)
        self.assertLess(dist, 0.0)
        self.assertGreater(np.linalg.norm(push), 0.0)

        # Point well outside in air (X=20, Y=0, Z=10)
        safe, dist, push = engine.query(np.array([20.0, 0.0, 10.0]), radius=1.0)
        self.assertTrue(safe)
        self.assertGreater(dist, 1.0)

    def test_bed_flange_clamped(self):
        router = BranchRouter(self.mesh, "PLA", 0.20)
        graph = router.build_tree_graph(self.contact_points)
        builder = SupportMeshBuilder(radial_segments=8)
        support_mesh = builder.build_mesh(graph, collision_engine=router.collision)
        self.assertGreaterEqual(float(support_mesh.bounds[0][2]), 0.0)


if __name__ == "__main__":
    unittest.main()
