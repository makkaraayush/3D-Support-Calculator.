"""
Unit tests for physics engine, beam mechanics, and precomputed lookups.
"""

import unittest
from src.physics.constants import BAMBU_A1_PROFILE, MATERIAL_DATABASE
from src.physics.beam_mechanics import SupportPhysicsEngine
from src.physics.precalculated import (
    lookup_optimal_z_gap,
    lookup_max_clean_bridge_span,
    evaluate_hole_support_requirement,
    get_recommended_branch_taper,
)


class TestPhysicsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SupportPhysicsEngine("PLA", nozzle_diameter=0.4)

    def test_bambu_profile(self):
        self.assertEqual(BAMBU_A1_PROFILE.bed_x, 256.0)
        self.assertEqual(BAMBU_A1_PROFILE.bed_y, 256.0)
        self.assertEqual(BAMBU_A1_PROFILE.bed_z, 256.0)
        self.assertEqual(BAMBU_A1_PROFILE.default_nozzle, 0.4)

    def test_euler_buckling_calculation(self):
        # A tall slender column (H=100mm, D=2mm) should have much lower critical load than stout column (D=5mm)
        p_slender = self.engine.calculate_critical_buckling_load(100.0, 2.0)
        p_stout = self.engine.calculate_critical_buckling_load(100.0, 5.0)
        self.assertGreater(p_stout, p_slender)
        self.assertGreater(p_slender, 0.0)

    def test_column_stability(self):
        # Slender tall column should flag instability
        stable_slender, sf_slender, def_slender = self.engine.evaluate_column_stability(120.0, 1.6)
        self.assertFalse(stable_slender)

        # Standard well-proportioned trunk (H=40mm, D=4.5mm) should be stable
        stable_stout, sf_stout, def_stout = self.engine.evaluate_column_stability(40.0, 4.5)
        self.assertTrue(stable_stout)
        self.assertGreater(sf_stout, 2.5)

    def test_breakaway_force(self):
        # Zero gap should give significantly higher peel force than 0.22mm air gap
        force_zero_gap = self.engine.calculate_breakaway_force(0.7, z_gap_mm=0.0)
        force_optimized_gap = self.engine.calculate_breakaway_force(0.7, z_gap_mm=0.22)
        self.assertGreater(force_zero_gap, force_optimized_gap)
        # Optimized gap should be in easy-breakaway human range (0.05N - 2.5N per tip)
        self.assertLess(force_optimized_gap, 2.0)

    def test_precomputed_lookups(self):
        pla_gap = lookup_optimal_z_gap("PLA", 0.20)
        self.assertAlmostEqual(pla_gap, 0.22, places=2)

        petg_gap = lookup_optimal_z_gap("PETG", 0.20)
        self.assertGreater(petg_gap, pla_gap)  # PETG needs more gap

        span = lookup_max_clean_bridge_span("PLA", 0.20)
        self.assertGreaterEqual(span, 25.0)

    def test_hole_decision_rules(self):
        micro_hole = evaluate_hole_support_requirement(3.0)
        self.assertFalse(micro_hole["needs_support"])
        self.assertEqual(micro_hole["category"], "micro")

        medium_hole = evaluate_hole_support_requirement(8.0)
        self.assertFalse(medium_hole["needs_support"])
        self.assertEqual(medium_hole["category"], "medium")

        large_hole = evaluate_hole_support_requirement(25.0)
        self.assertTrue(large_hole["needs_support"])
        self.assertEqual(large_hole["category"], "large")


if __name__ == "__main__":
    unittest.main()
