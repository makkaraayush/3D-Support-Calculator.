"""
Multi-Objective Physics Evaluator and Support Simulator.
Scores candidate support structures across five critical dimensions:
1. Overhang Coverage & Failure Prevention
2. Filament Mass and Savings vs Standard Grid/Tree Baselines
3. Breakaway Force and Post-Print Removal Ease
4. Structural Stability and Buckling Resistance
5. Print Time and Extrusion Economics
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from ..geometry.mesh import TriangleMesh
from ..geometry.overhang_analyzer import ContactPoint
from ..generator.tree_graph import TreeSupportGraph
from ..physics.constants import MATERIAL_DATABASE, BAMBU_A1_PROFILE
from ..physics.beam_mechanics import SupportPhysicsEngine
from ..physics.precalculated import lookup_optimal_z_gap


@dataclass
class SupportPhysicsEvaluationReport:
    # Coverage
    coverage_score_pct: float
    unsupported_area_mm2: float
    print_success_probability_pct: float

    # Filament Economics
    support_volume_cm3: float
    support_mass_grams: float
    filament_saved_vs_standard_grid_pct: float
    filament_saved_vs_default_tree_pct: float

    # Removal Ease
    average_breakaway_force_n: float
    total_detachment_effort_n: float
    removal_ease_score: float  # 1.0 (hard/destructive) to 10.0 (effortless snap)
    surface_scarring_risk: str  # 'Minimal', 'Low', 'Moderate', 'High'

    # Stability
    min_buckling_safety_factor: float
    bed_adhesion_safety_factor: float
    stability_score_pct: float

    # Time Economics
    estimated_support_print_time_minutes: float

    # Overall Multi-Objective Fitness
    composite_fitness_score: float


class SupportPhysicsEvaluator:
    def __init__(
        self,
        mesh: TriangleMesh,
        material: str = "PLA",
        layer_height: float = 0.20,
        nozzle_diameter: float = 0.4,
    ):
        self.mesh = mesh
        self.material_data = MATERIAL_DATABASE.get(material.upper(), MATERIAL_DATABASE["PLA"])
        self.layer_height = layer_height
        self.nozzle_diameter = nozzle_diameter
        self.physics = SupportPhysicsEngine(material, nozzle_diameter)
        self.z_gap = lookup_optimal_z_gap(material, layer_height)

    def evaluate(
        self,
        graph: TreeSupportGraph,
        target_contact_points: List[ContactPoint],
        priority_profile: str = "balanced",
    ) -> SupportPhysicsEvaluationReport:
        """
        Runs comprehensive multi-physics evaluation on the given support graph.
        """
        # 1. Volume and Filament Mass
        vol_mm3 = graph.estimate_volume_mm3()
        vol_cm3 = vol_mm3 / 1000.0
        mass_grams = vol_cm3 * self.material_data.density

        # Baseline comparisons:
        # Standard grid support occupies ~65% of the column volume beneath overhangs
        # Standard tree support occupies ~28% of column volume
        total_overhang_area = sum(cp.area_weight for cp in target_contact_points) if target_contact_points else 100.0
        avg_height = float(np.mean([cp.position[2] for cp in target_contact_points])) if target_contact_points else 20.0
        standard_grid_vol_cm3 = (total_overhang_area * avg_height * 0.45) / 1000.0
        standard_tree_vol_cm3 = (total_overhang_area * avg_height * 0.22) / 1000.0

        grid_saved_pct = max(0.0, min(85.0, (1.0 - (vol_cm3 / max(0.1, standard_grid_vol_cm3))) * 100.0))
        tree_saved_pct = max(0.0, min(65.0, (1.0 - (vol_cm3 / max(0.1, standard_tree_vol_cm3))) * 100.0))

        # 2. Coverage Score
        # Check how many contact points are near a graph tip
        tips = graph.get_tips()
        if not target_contact_points:
            coverage_pct = 100.0
            unsupported_area = 0.0
        elif not tips:
            coverage_pct = 0.0
            unsupported_area = total_overhang_area
        else:
            tip_positions = np.array([t.position for t in tips])
            covered_area = 0.0
            for cp in target_contact_points:
                dists = np.linalg.norm(tip_positions - cp.position, axis=1)
                # If nearest tip is within 4.5mm, it is supported
                if np.min(dists) <= 4.5:
                    covered_area += cp.area_weight

            coverage_pct = min(100.0, (covered_area / max(1e-4, total_overhang_area)) * 100.0)
            unsupported_area = max(0.0, total_overhang_area - covered_area)

        print_success_prob = min(99.9, coverage_pct * 0.98 + (2.0 if coverage_pct > 90 else 0.0))

        # 3. Removal Ease & Breakaway Force
        tip_count = max(1, len(tips))
        avg_tip_radius = float(np.mean([t.radius for t in tips])) if tips else 0.35
        avg_tip_dia = avg_tip_radius * 2.0

        breakaway_force_per_tip = self.physics.calculate_breakaway_force(
            tip_diameter_mm=avg_tip_dia,
            z_gap_mm=self.z_gap,
            layer_height_mm=self.layer_height,
        )
        total_breakaway_force = breakaway_force_per_tip * tip_count

        # Scale 1.0 to 10.0
        if breakaway_force_per_tip <= 0.6:
            ease_score = 9.8
            scarring = "Minimal"
        elif breakaway_force_per_tip <= 1.4:
            ease_score = 9.1
            scarring = "Low"
        elif breakaway_force_per_tip <= 2.8:
            ease_score = 7.8
            scarring = "Moderate"
        else:
            ease_score = max(1.0, 5.0 - breakaway_force_per_tip)
            scarring = "High"

        # 4. Stability & Buckling
        roots = graph.get_roots()
        min_safety_factor = 99.0
        for node in graph.nodes.values():
            if node.parent_id is not None and node.parent_id in graph.nodes:
                p_node = graph.nodes[node.parent_id]
                h = abs(node.position[2] - p_node.position[2])
                if h > 2.0:
                    dia = (node.radius + p_node.radius)
                    _, sf, _ = self.physics.evaluate_column_stability(height_mm=h, diameter_mm=dia)
                    if sf < min_safety_factor:
                        min_safety_factor = sf

        # Bed adhesion
        total_height = float(self.mesh.dimensions[2])
        bed_radius = float(np.mean([r.radius * 2.2 for r in roots])) if roots else 4.0
        bed_adhesion_sf = self.physics.calculate_bed_adhesion_safety(
            total_height_mm=total_height,
            base_radius_mm=bed_radius,
            shear_load_n=1.0,
        )

        stability_score = min(
            100.0,
            max(0.0, (min_safety_factor / 3.0) * 50.0 + min(50.0, bed_adhesion_sf * 15.0)),
        )

        # 5. Print Time Estimate
        # Extrusion length + travel moves for Bambu A1
        branch_length = graph.calculate_total_branch_length()
        speed_mm_s = 180.0  # avg support print speed on Bambu A1
        retract_penalty_s = tip_count * 0.4  # retracts and z-hops
        print_time_min = (branch_length / (speed_mm_s * 60.0)) + (retract_penalty_s / 60.0)

        # 6. Composite Fitness
        weights = {
            "balanced": {"cov": 0.40, "vol": 0.25, "rem": 0.20, "stab": 0.15},
            "filament_saver": {"cov": 0.35, "vol": 0.45, "rem": 0.10, "stab": 0.10},
            "clean_removal": {"cov": 0.35, "vol": 0.15, "rem": 0.40, "stab": 0.10},
            "max_stability": {"cov": 0.35, "vol": 0.10, "rem": 0.15, "stab": 0.40},
        }.get(priority_profile.lower(), {"cov": 0.40, "vol": 0.25, "rem": 0.20, "stab": 0.15})

        # Normalized components in [0, 100]
        cov_comp = coverage_pct
        vol_comp = max(0.0, min(100.0, 100.0 - (mass_grams / max(1.0, mass_grams + 20.0)) * 60.0))
        rem_comp = ease_score * 10.0
        stab_comp = stability_score

        composite_fitness = (
            weights["cov"] * cov_comp
            + weights["vol"] * vol_comp
            + weights["rem"] * rem_comp
            + weights["stab"] * stab_comp
        )

        return SupportPhysicsEvaluationReport(
            coverage_score_pct=float(coverage_pct),
            unsupported_area_mm2=float(unsupported_area),
            print_success_probability_pct=float(print_success_prob),
            support_volume_cm3=float(vol_cm3),
            support_mass_grams=float(mass_grams),
            filament_saved_vs_standard_grid_pct=float(grid_saved_pct),
            filament_saved_vs_default_tree_pct=float(tree_saved_pct),
            average_breakaway_force_n=float(breakaway_force_per_tip),
            total_detachment_effort_n=float(total_breakaway_force),
            removal_ease_score=float(ease_score),
            surface_scarring_risk=scarring,
            min_buckling_safety_factor=float(min_safety_factor),
            bed_adhesion_safety_factor=float(bed_adhesion_sf),
            stability_score_pct=float(stability_score),
            estimated_support_print_time_minutes=float(max(1.0, print_time_min)),
            composite_fitness_score=float(composite_fitness),
        )
