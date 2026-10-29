"""
Support interface layer generator for breakaway contact tips and detachment gaps.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from ..physics.beam_mechanics import SupportPhysicsEngine
from ..physics.precalculated import lookup_optimal_z_gap


@dataclass
class InterfaceSpecification:
    pattern: str  # 'conical_point', 'concentric_ring', 'micro_grid'
    tip_diameter_mm: float
    z_gap_mm: float
    num_interface_layers: int
    interface_extrusion_width_mm: float
    expected_breakaway_force_n: float
    removal_ease_rating: float  # 1.0 (very difficult) to 10.0 (effortless snap)


class InterfaceGenerator:
    def __init__(
        self,
        material: str = "PLA",
        layer_height: float = 0.20,
        nozzle_diameter: float = 0.4,
    ):
        self.material = material
        self.layer_height = layer_height
        self.nozzle_diameter = nozzle_diameter
        self.physics = SupportPhysicsEngine(material, nozzle_diameter)
        self.z_gap = lookup_optimal_z_gap(material, layer_height)

    def design_interface(
        self,
        target_tip_diameter_mm: float = 0.7,
        custom_z_gap_mm: float = None,
        pattern: str = "conical_point",
    ) -> InterfaceSpecification:
        """
        Designs the contact interface parameters and calculates breakaway physics.
        """
        z_gap = custom_z_gap_mm if custom_z_gap_mm is not None else self.z_gap

        # Calculate breakaway force using beam mechanics
        breakaway_force = self.physics.calculate_breakaway_force(
            tip_diameter_mm=target_tip_diameter_mm,
            z_gap_mm=z_gap,
            layer_height_mm=self.layer_height,
        )

        # Removal Ease Rating (1 to 10 scale):
        # < 0.5 N: 9.5 - 10.0 (finger flick removal)
        # 0.5 - 1.5 N: 8.5 - 9.4 (effortless 1-snap detachment)
        # 1.5 - 3.0 N: 7.0 - 8.4 (pliers with clean separation)
        # 3.0 - 6.0 N: 4.0 - 6.9 (firm grip required)
        # > 6.0 N: 1.0 - 3.9 (fused / risk of surface tearing)
        if breakaway_force <= 0.5:
            ease_rating = 9.8
        elif breakaway_force <= 1.5:
            ease_rating = 9.2 - ((breakaway_force - 0.5) / 1.0) * 0.8
        elif breakaway_force <= 3.0:
            ease_rating = 8.4 - ((breakaway_force - 1.5) / 1.5) * 1.4
        elif breakaway_force <= 6.0:
            ease_rating = 7.0 - ((breakaway_force - 3.0) / 3.0) * 3.0
        else:
            ease_rating = max(1.0, 4.0 - (breakaway_force - 6.0) * 0.5)

        return InterfaceSpecification(
            pattern=pattern,
            tip_diameter_mm=float(target_tip_diameter_mm),
            z_gap_mm=float(z_gap),
            num_interface_layers=3,
            interface_extrusion_width_mm=float(self.nozzle_diameter * 0.95),
            expected_breakaway_force_n=float(breakaway_force),
            removal_ease_rating=float(ease_rating),
        )
