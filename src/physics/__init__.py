"""
Physics package initialization.
"""

from .constants import (
    PrinterHardwareProfile,
    MaterialPhysics,
    BAMBU_A1_PROFILE,
    MATERIAL_DATABASE,
)
from .beam_mechanics import SupportPhysicsEngine
from .precalculated import (
    lookup_optimal_z_gap,
    lookup_max_clean_bridge_span,
    evaluate_hole_support_requirement,
    get_recommended_branch_taper,
)

__all__ = [
    "PrinterHardwareProfile",
    "MaterialPhysics",
    "BAMBU_A1_PROFILE",
    "MATERIAL_DATABASE",
    "SupportPhysicsEngine",
    "lookup_optimal_z_gap",
    "lookup_max_clean_bridge_span",
    "evaluate_hole_support_requirement",
    "get_recommended_branch_taper",
]
