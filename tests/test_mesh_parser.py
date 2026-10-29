"""
Unit tests for 3D mesh loaders, STL/OBJ/3MF parsers, and overhang analyzer.
"""

import os
import unittest
import numpy as np
from src.geometry.mesh import TriangleMesh
from src.geometry.stl_parser import STLParser
from src.geometry.obj_parser import OBJParser
from src.geometry.bambu_3mf import Bambu3MFParser
from src.geometry.overhang_analyzer import OverhangAnalyzer


def create_cube_mesh(size: float = 10.0) -> TriangleMesh:
    """Creates a 10x10x10 mm solid cube centered at origin."""
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # bottom (Z = -s)
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],      # top (Z = s)
    ], dtype=np.float32)

    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # front
        [0, 1, 5], [0, 5, 4],
        # back
        [2, 3, 7], [2, 7, 6],
        # left
        [0, 4, 7], [0, 7, 3],
        # right
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)

    return TriangleMesh(vertices, faces)


class TestMeshParsers(unittest.TestCase):
    def setUp(self):
        self.cube = create_cube_mesh(10.0)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        for f in os.listdir(self.tmp_dir):
            try:
                os.remove(os.path.join(self.tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.tmp_dir)
        except Exception:
            pass

    def test_cube_volume_and_bounds(self):
        vol = self.cube.compute_volume()
        self.assertAlmostEqual(vol, 1000.0, places=1)
        dim = self.cube.dimensions
        np.testing.assert_allclose(dim, [10.0, 10.0, 10.0], atol=1e-3)

    def test_stl_binary_roundtrip(self):
        stl_path = os.path.join(self.tmp_dir, "test_cube.stl")
        STLParser.save_binary(self.cube, stl_path)
        self.assertTrue(os.path.exists(stl_path))

        reloaded = STLParser.load(stl_path)
        self.assertEqual(len(reloaded.faces), 12)
        self.assertAlmostEqual(reloaded.compute_volume(), 1000.0, places=1)

    def test_obj_roundtrip(self):
        obj_path = os.path.join(self.tmp_dir, "test_cube.obj")
        OBJParser.save(self.cube, obj_path)
        self.assertTrue(os.path.exists(obj_path))

        reloaded = OBJParser.load(obj_path)
        self.assertEqual(len(reloaded.faces), 12)
        self.assertAlmostEqual(reloaded.compute_volume(), 1000.0, places=1)

    def test_bambu_3mf_roundtrip(self):
        threemf_path = os.path.join(self.tmp_dir, "test_project.3mf")
        Bambu3MFParser.export_bambu_3mf(
            self.cube,
            support_mesh=None,
            output_filepath=threemf_path,
            material="PLA",
            layer_height=0.20,
            z_gap_mm=0.22,
        )
        self.assertTrue(os.path.exists(threemf_path))

        reloaded_mesh, metadata = Bambu3MFParser.load(threemf_path)
        self.assertEqual(len(reloaded_mesh.faces), 12)
        self.assertEqual(metadata["material"], "PLA")
        self.assertEqual(metadata["layer_height"], 0.20)

    def test_overhang_analysis(self):
        # A cube sitting flat has:
        # 2 bottom faces pointing straight down (overhang angle = 90 deg)
        # 8 vertical side faces (overhang angle = 0 deg)
        # 2 top faces pointing straight up (overhang angle = 0 deg)
        analyzer = OverhangAnalyzer(self.cube, critical_angle_deg=50.0)
        report = analyzer.analyze()
        self.assertEqual(report["total_faces"], 12)
        self.assertEqual(report["critical_face_count"], 2)  # the 2 bottom faces

        contact_points = analyzer.generate_contact_points(grid_resolution_mm=5.0)
        self.assertGreaterEqual(len(contact_points), 1)


if __name__ == "__main__":
    unittest.main()
