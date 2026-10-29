"""
Analytical physics models for 3D printing support columns and bridges:
- Euler buckling stability under nozzle dynamic drag
- Beam deflection and sagging of unsupported overhangs and bridges
- Interfacial shear stress and breakaway force for easy detachment
- Bed adhesion tipping moment safety factor
"""

import math
from typing import Dict, Tuple
from .constants import (
    MATERIAL_DATABASE,
    DEFAULT_LAYER_HEIGHT,
    DEFAULT_EXTRUSION_WIDTH,
    GRAVITY,
    MaterialPhysics,
)


class SupportPhysicsEngine:
    def __init__(self, material_name: str = "PLA", nozzle_diameter: float = 0.4):
        self.material: MaterialPhysics = MATERIAL_DATABASE.get(
            material_name.upper(), MATERIAL_DATABASE["PLA"]
        )
        self.nozzle_diameter = nozzle_diameter

    def calculate_second_moment_of_area(self, diameter_mm: float) -> float:
        """Area moment of inertia for circular cross-section: I = pi * d^4 / 64 (mm^4)"""
        radius = diameter_mm / 2.0
        return (math.pi * (radius ** 4)) / 4.0

    def calculate_critical_buckling_load(
        self,
        column_height_mm: float,
        diameter_mm: float,
        end_condition_k: float = 2.0,  # 2.0 = fixed-free cantilever column
    ) -> float:
        """
        Euler critical buckling load: P_cr = (pi^2 * E * I) / (K * L)^2
        Returns critical force in Newtons.
        """
        if column_height_mm <= 0.1:
            return 1e6

        # Convert E from GPa to N/mm^2 (MPa): 1 GPa = 1000 N/mm^2
        e_mpa = self.material.elastic_modulus * 1000.0
        i_mm4 = self.calculate_second_moment_of_area(diameter_mm)
        effective_length = end_condition_k * column_height_mm

        p_crit_newtons = (math.pi ** 2 * e_mpa * i_mm4) / (effective_length ** 2)
        return float(p_crit_newtons)

    def evaluate_column_stability(
        self,
        height_mm: float,
        diameter_mm: float,
        expected_nozzle_drag_n: float = 0.85,
    ) -> Tuple[bool, float, float]:
        """
        Evaluates whether a support branch or trunk will buckle or deflect under nozzle drag.
        Returns: (is_stable, buckling_safety_factor, lateral_deflection_mm)
        """
        p_crit = self.calculate_critical_buckling_load(height_mm, diameter_mm, end_condition_k=2.0)
        safety_factor = p_crit / max(0.01, expected_nozzle_drag_n)

        # Cantilever tip lateral deflection: delta = (F * L^3) / (3 * E * I)
        e_mpa = self.material.elastic_modulus * 1000.0
        i_mm4 = self.calculate_second_moment_of_area(diameter_mm)
        deflection_mm = (expected_nozzle_drag_n * (height_mm ** 3)) / (3.0 * e_mpa * max(1e-5, i_mm4))

        # Stable if safety factor >= 2.5 and deflection < 0.3mm (less than nozzle diameter)
        is_stable = (safety_factor >= 2.5) and (deflection_mm <= 0.35)
        return is_stable, safety_factor, deflection_mm

    def calculate_bridge_sag(
        self,
        span_length_mm: float,
        layer_height_mm: float = DEFAULT_LAYER_HEIGHT,
        extrusion_width_mm: float = DEFAULT_EXTRUSION_WIDTH,
    ) -> float:
        """
        Calculates maximum gravity sag deflection in center of unsupported horizontal bridge.
        Uses molten transition modulus near Tg to accurately capture filament droop.
        """
        if span_length_mm <= 0:
            return 0.0

        # Strand mass per unit length (kg/mm):
        # cross-section area in mm^2
        area_mm2 = layer_height_mm * extrusion_width_mm
        # density in g/cm^3 -> convert to kg/mm^3: 1 g/cm^3 = 1e-6 kg/mm^3
        density_kg_mm3 = self.material.density * 1e-6
        mass_per_length = area_mm2 * density_kg_mm3
        w_n_per_mm = mass_per_length * GRAVITY  # weight force per mm

        # Semi-molten modulus in MPa
        e_molten_mpa = self.material.molten_modulus * 1000.0
        # Rectangular strand moment of inertia: I = (w * h^3) / 12
        i_strand = (extrusion_width_mm * (layer_height_mm ** 3)) / 12.0

        # Beam simply supported with uniform load: delta = (5 * w * L^4) / (384 * E * I)
        # Filament tension stiffening reduces theoretical sag by ~40% under active cooling
        theoretical_sag = (5.0 * w_n_per_mm * (span_length_mm ** 4)) / (
            384.0 * e_molten_mpa * max(1e-9, i_strand)
        )
        effective_sag = theoretical_sag * 0.60
        return float(effective_sag)

    def calculate_breakaway_force(
        self,
        tip_diameter_mm: float,
        z_gap_mm: float,
        layer_height_mm: float = DEFAULT_LAYER_HEIGHT,
    ) -> float:
        """
        Calculates expected peel / shear force in Newtons required to detach a support interface tip.
        F = Area_contact * tau_adhesion * exp(-z_gap / (0.65 * layer_height))
        """
        tip_radius_mm = tip_diameter_mm / 2.0
        contact_area_mm2 = math.pi * (tip_radius_mm ** 2)

        # Interfacial adhesion decays exponentially with air gap
        decay_factor = math.exp(-z_gap_mm / max(0.04, layer_height_mm * 0.65))
        shear_force_newtons = contact_area_mm2 * self.material.interfacial_shear_strength * decay_factor
        return float(shear_force_newtons)

    def calculate_bed_adhesion_safety(
        self,
        total_height_mm: float,
        base_radius_mm: float,
        shear_load_n: float = 1.0,
        bed_adhesion_strength_kpa: float = 120.0,  # ~120 kPa for PEI sheet with PLA at 60C
    ) -> float:
        """
        Safety factor against base tipping/delamination from print bed.
        Overturning moment: M_overturn = F * H
        Resisting moment: M_resist = (pi * r^3 / 4) * sigma_bed
        """
        overturning_moment = shear_load_n * total_height_mm  # N*mm
        # sigma in MPa (120 kPa = 0.12 MPa = 0.12 N/mm^2)
        sigma_mpa = bed_adhesion_strength_kpa * 0.001
        resisting_moment = (math.pi * (base_radius_mm ** 3) / 4.0) * sigma_mpa
        return float(resisting_moment / max(0.01, overturning_moment))

    def evaluate_overhang_printability(
        self,
        overhang_angle_deg: float,
        layer_height_mm: float = DEFAULT_LAYER_HEIGHT,
        extrusion_width_mm: float = DEFAULT_EXTRUSION_WIDTH,
    ) -> Dict[str, float]:
        """
        Evaluates whether an overhang requires support based on overlap geometry.
        Overlap ratio R = 1 - (layer_height / extrusion_width) * cot(angle).
        When R >= 0.35, filament perimeter can freeze reliably without support.
        """
        angle_rad = math.radians(max(5.0, min(89.0, overhang_angle_deg)))
        cot_angle = 1.0 / math.tan(angle_rad)
        overlap_ratio = 1.0 - (layer_height_mm / extrusion_width_mm) * cot_angle

        # Calibrated threshold based on Bambu Lab A1 auxiliary cooling
        cooling_boost = 6.0 * (self.material.cooling_shrinkage_pct < 0.5)
        safe_angle = self.material.safe_overhang_base_deg + cooling_boost

        needs_support = overhang_angle_deg > safe_angle or overlap_ratio < 0.28
        droop_risk = max(0.0, min(1.0, (overhang_angle_deg - safe_angle) / 25.0))

        return {
            "overhang_angle_deg": overhang_angle_deg,
            "safe_threshold_deg": safe_angle,
            "overlap_ratio": float(overlap_ratio),
            "needs_support": float(1.0 if needs_support else 0.0),
            "droop_risk_index": float(droop_risk),
        }
