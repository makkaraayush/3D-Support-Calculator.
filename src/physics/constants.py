"""
Physics constants, Bambu Lab A1 printer hardware limits, and material characteristics
for FDM 3D printing support optimization.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PrinterHardwareProfile:
    name: str
    bed_x: float  # mm
    bed_y: float  # mm
    bed_z: float  # mm
    default_nozzle: float  # mm
    max_print_speed: float  # mm/s
    max_acceleration: float  # mm/s^2
    max_nozzle_temp: float  # deg C
    max_bed_temp: float  # deg C
    cooling_power_index: float  # 0.0 - 1.0 (Bambu A1 dual-duct part fan is very high ~0.85)


@dataclass(frozen=True)
class MaterialPhysics:
    name: str
    density: float  # g/cm^3
    elastic_modulus: float  # GPa (room temp solid)
    molten_modulus: float  # GPa (cooling transition near Tg)
    tensile_strength: float  # MPa
    interfacial_shear_strength: float  # MPa (fusion bond strength at 0 gap)
    glass_transition_temp: float  # deg C
    print_temp: float  # deg C
    bed_temp: float  # deg C
    cooling_shrinkage_pct: float  # % linear shrinkage
    stringing_risk_factor: float  # 0.0 - 1.0
    safe_overhang_base_deg: float  # Base safe overhang angle at 0.2mm layer height


# Bambu Lab A1 default profile
BAMBU_A1_PROFILE = PrinterHardwareProfile(
    name="Bambu Lab A1",
    bed_x=256.0,
    bed_y=256.0,
    bed_z=256.0,
    default_nozzle=0.4,
    max_print_speed=500.0,
    max_acceleration=10000.0,
    max_nozzle_temp=300.0,
    max_bed_temp=100.0,
    cooling_power_index=0.88,
)

# Standard material database
MATERIAL_DATABASE: Dict[str, MaterialPhysics] = {
    "PLA": MaterialPhysics(
        name="PLA",
        density=1.24,
        elastic_modulus=3.5,
        molten_modulus=0.15,
        tensile_strength=50.0,
        interfacial_shear_strength=1.2,
        glass_transition_temp=60.0,
        print_temp=220.0,
        bed_temp=60.0,
        cooling_shrinkage_pct=0.3,
        stringing_risk_factor=0.2,
        safe_overhang_base_deg=58.0,
    ),
    "PETG": MaterialPhysics(
        name="PETG",
        density=1.27,
        elastic_modulus=2.1,
        molten_modulus=0.08,
        tensile_strength=53.0,
        interfacial_shear_strength=1.9,  # PETG sticks aggressively to interfaces
        glass_transition_temp=75.0,
        print_temp=255.0,
        bed_temp=70.0,
        cooling_shrinkage_pct=0.4,
        stringing_risk_factor=0.7,
        safe_overhang_base_deg=52.0,
    ),
    "ABS": MaterialPhysics(
        name="ABS",
        density=1.04,
        elastic_modulus=2.3,
        molten_modulus=0.10,
        tensile_strength=40.0,
        interfacial_shear_strength=1.0,
        glass_transition_temp=105.0,
        print_temp=260.0,
        bed_temp=95.0,
        cooling_shrinkage_pct=1.4,
        stringing_risk_factor=0.3,
        safe_overhang_base_deg=48.0,
    ),
    "TPU": MaterialPhysics(
        name="TPU 95A",
        density=1.21,
        elastic_modulus=0.08,  # Highly elastic
        molten_modulus=0.01,
        tensile_strength=35.0,
        interfacial_shear_strength=0.6,
        glass_transition_temp=-20.0,
        print_temp=230.0,
        bed_temp=45.0,
        cooling_shrinkage_pct=0.8,
        stringing_risk_factor=0.85,
        safe_overhang_base_deg=42.0,
    ),
}

# Empirical slicing guidelines
DEFAULT_LAYER_HEIGHT = 0.20  # mm
DEFAULT_EXTRUSION_WIDTH = 0.42  # mm
GRAVITY = 9.81  # m/s^2

# Support default geometry bounds
MIN_BRANCH_DIAMETER = 1.6  # mm
MAX_BRANCH_DIAMETER = 4.8  # mm
MIN_TRUNK_DIAMETER = 3.2  # mm
MAX_TRUNK_DIAMETER = 9.0  # mm
MIN_TIP_DIAMETER = 0.5  # mm
MAX_TIP_DIAMETER = 1.2  # mm
MIN_BRANCH_ANGLE_DEG = 15.0  # deg from vertical
MAX_BRANCH_ANGLE_DEG = 42.0  # deg from vertical
MODEL_CLEARANCE_DEFAULT = 1.5  # mm gap between support trunk and model walls
BASE_FLANGE_RADIUS_FACTOR = 2.2  # multiplier for bed adhesion pad
