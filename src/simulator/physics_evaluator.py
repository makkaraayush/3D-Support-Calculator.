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
    coverage_score_pct: float
    unsupported_area_mm2: float
    print_success_probability_pct: float

    # Filament metrics
    support_volume_cm3: float
    support_mass_grams: float
    grid_mass_grams: float
    tree_mass_grams: float
    filament_saved_vs_standard_grid_pct: float
    filament_saved_vs_default_tree_pct: float

    # Removal metrics
    average_breakaway_force_n: float
    total_detachment_effort_n: float
    removal_ease_score: float
    surface_scarring_risk: str

    # Stability metrics
    min_buckling_safety_factor: float
    bed_adhesion_safety_factor: float
    stability_score_pct: float

    # Print time metrics (minutes)
    print_time_minutes: float
    grid_print_time_minutes: float
    standard_tree_print_time_minutes: float
    time_saved_minutes: float

    # Composite score
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
        # Base model volume and print time
        model_vol_mm3 = self.mesh.compute_volume()
        model_h = float(self.mesh.dimensions[2])
        num_layers = max(1, int(model_h / max(0.08, self.layer_height)))
        t_model_min = (model_vol_mm3 / (14.0 * 60.0)) + (num_layers * 2.0 / 60.0)

        # Support volume and mass
        vol_mm3 = graph.estimate_volume_mm3()
        vol_cm3 = vol_mm3 / 1000.0
        mass_grams = vol_cm3 * self.material_data.density

        # Baselines: grid and standard tree volumes
        total_overhang_area = sum(cp.area_weight for cp in target_contact_points) if target_contact_points else 100.0
        avg_height = float(np.mean([cp.position[2] for cp in target_contact_points])) if target_contact_points else 20.0

        v_grid_mm3 = total_overhang_area * avg_height * 0.40
        grid_mass = (v_grid_mm3 / 1000.0) * self.material_data.density
        t_grid_supp_min = (v_grid_mm3 / (9.0 * 60.0)) + (num_layers * 1.5 / 60.0)
        t_grid_total_min = t_model_min + t_grid_supp_min

        v_tree_mm3 = total_overhang_area * avg_height * 0.20
        tree_mass = (v_tree_mm3 / 1000.0) * self.material_data.density
        t_tree_supp_min = (v_tree_mm3 / (11.0 * 60.0)) + (num_layers * 1.0 / 60.0)
        t_tree_total_min = t_model_min + t_tree_supp_min

        # AI tree print time
        tips = graph.get_tips()
        tip_count = max(1, len(tips))
        t_ai_supp_min = (vol_mm3 / (14.0 * 60.0)) + (num_layers * 0.5 / 60.0) + (tip_count * 0.3 / 60.0)
        t_ai_total_min = t_model_min + t_ai_supp_min

        grid_saved_pct = max(0.0, min(90.0, (1.0 - (mass_grams / max(0.1, grid_mass))) * 100.0))
        tree_saved_pct = max(0.0, min(75.0, (1.0 - (mass_grams / max(0.1, tree_mass))) * 100.0))
        time_saved_min = max(0.0, t_grid_total_min - t_ai_total_min)

        # Coverage score
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
                if np.min(dists) <= 4.5:
                    covered_area += cp.area_weight

            coverage_pct = min(100.0, (covered_area / max(1e-4, total_overhang_area)) * 100.0)
            unsupported_area = max(0.0, total_overhang_area - covered_area)

        print_success_prob = min(99.9, coverage_pct * 0.98 + (2.0 if coverage_pct > 90 else 0.0))

        # Breakaway force and removal ease
        avg_tip_radius = float(np.mean([t.radius for t in tips])) if tips else 0.35
        avg_tip_dia = avg_tip_radius * 2.0

        breakaway_force_per_tip = self.physics.calculate_breakaway_force(
            tip_diameter_mm=avg_tip_dia,
            z_gap_mm=self.z_gap,
            layer_height_mm=self.layer_height,
        )
        total_breakaway_force = breakaway_force_per_tip * tip_count

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

        # Stability and buckling safety
        roots = graph.get_roots()
        min_safety_factor = 99.0
        for node in graph.nodes.values():
            if node.parent_id is not None and node.parent_id in graph.nodes:
                p_node = graph.nodes[node.parent_id]
                h = abs(node.position[2] - p_node.position[2])
                if h > 2.0:
                    dia = node.radius + p_node.radius
                    _, sf, _ = self.physics.evaluate_column_stability(height_mm=h, diameter_mm=dia)
                    if sf < min_safety_factor:
                        min_safety_factor = sf

        bed_radius = float(np.mean([r.radius * 1.5 for r in roots])) if roots else 4.0
        bed_adhesion_sf = self.physics.calculate_bed_adhesion_safety(
            total_height_mm=model_h,
            base_radius_mm=bed_radius,
            shear_load_n=1.0,
        )

        stability_score = min(
            100.0,
            max(0.0, (min_safety_factor / 3.0) * 50.0 + min(50.0, bed_adhesion_sf * 15.0)),
        )

        # Composite multi-objective score
        weights = {
            "balanced": {"cov": 0.40, "vol": 0.25, "rem": 0.20, "stab": 0.15},
            "filament_saver": {"cov": 0.35, "vol": 0.45, "rem": 0.10, "stab": 0.10},
            "clean_removal": {"cov": 0.35, "vol": 0.15, "rem": 0.40, "stab": 0.10},
            "max_stability": {"cov": 0.35, "vol": 0.10, "rem": 0.15, "stab": 0.40},
        }.get(priority_profile.lower(), {"cov": 0.40, "vol": 0.25, "rem": 0.20, "stab": 0.15})

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
            grid_mass_grams=float(grid_mass),
            tree_mass_grams=float(tree_mass),
            filament_saved_vs_standard_grid_pct=float(grid_saved_pct),
            filament_saved_vs_default_tree_pct=float(tree_saved_pct),
            average_breakaway_force_n=float(breakaway_force_per_tip),
            total_detachment_effort_n=float(total_breakaway_force),
            removal_ease_score=float(ease_score),
            surface_scarring_risk=scarring,
            min_buckling_safety_factor=float(min_safety_factor),
            bed_adhesion_safety_factor=float(bed_adhesion_sf),
            stability_score_pct=float(stability_score),
            print_time_minutes=float(t_ai_total_min),
            grid_print_time_minutes=float(t_grid_total_min),
            standard_tree_print_time_minutes=float(t_tree_total_min),
            time_saved_minutes=float(time_saved_min),
            composite_fitness_score=float(composite_fitness),
        )
