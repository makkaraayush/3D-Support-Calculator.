"""
Evolutionary parameter optimizer for tree support topologies, branch angles,
merge distances, and interface geometries.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
from ..geometry.mesh import TriangleMesh
from ..geometry.overhang_analyzer import OverhangAnalyzer, ContactPoint
from ..generator.path_routing import BranchRouter
from ..simulator.physics_evaluator import SupportPhysicsEvaluator, SupportPhysicsEvaluationReport
from .surrogate_model import NeuralSurrogateModel


@dataclass
class SupportChromosome:
    branch_angle_deg: float
    merge_distance_mm: float
    tip_diameter_mm: float
    contact_grid_res_mm: float
    z_gap_offset_mm: float
    base_flange_factor: float

    @staticmethod
    def random() -> "SupportChromosome":
        return SupportChromosome(
            branch_angle_deg=float(np.random.uniform(18.0, 36.0)),
            merge_distance_mm=float(np.random.uniform(6.0, 14.0)),
            tip_diameter_mm=float(np.random.uniform(0.55, 0.95)),
            contact_grid_res_mm=float(np.random.uniform(2.4, 4.8)),
            z_gap_offset_mm=float(np.random.uniform(-0.02, 0.04)),
            base_flange_factor=float(np.random.uniform(1.8, 2.5)),
        )

    def mutate(self, mutation_rate: float = 0.25) -> "SupportChromosome":
        def perturb(val, low, high, scale):
            if np.random.rand() < mutation_rate:
                val += np.random.randn() * scale
            return float(np.clip(val, low, high))

        return SupportChromosome(
            branch_angle_deg=perturb(self.branch_angle_deg, 16.0, 38.0, 2.5),
            merge_distance_mm=perturb(self.merge_distance_mm, 5.0, 16.0, 1.2),
            tip_diameter_mm=perturb(self.tip_diameter_mm, 0.50, 1.10, 0.05),
            contact_grid_res_mm=perturb(self.contact_grid_res_mm, 2.0, 5.5, 0.3),
            z_gap_offset_mm=perturb(self.z_gap_offset_mm, -0.04, 0.06, 0.01),
            base_flange_factor=perturb(self.base_flange_factor, 1.6, 2.8, 0.15),
        )

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.branch_angle_deg,
            self.merge_distance_mm,
            self.tip_diameter_mm,
            self.contact_grid_res_mm,
            self.z_gap_offset_mm,
            self.base_flange_factor,
        ], dtype=np.float32)


class EvolutionarySupportOptimizer:
    def __init__(
        self,
        mesh: TriangleMesh,
        surrogate: NeuralSurrogateModel,
        material: str = "PLA",
        layer_height: float = 0.20,
        priority_profile: str = "balanced",
    ):
        self.mesh = mesh
        self.surrogate = surrogate
        self.material = material
        self.layer_height = layer_height
        self.priority_profile = priority_profile

        self.analyzer = OverhangAnalyzer(mesh, material=material, layer_height=layer_height)
        self.evaluator = SupportPhysicsEvaluator(mesh, material=material, layer_height=layer_height)

        # Extract mesh geometric feature vector for surrogate conditioning
        dims = mesh.dimensions
        surf_area = mesh.compute_surface_area()
        vol = mesh.compute_volume()
        overhang_report = self.analyzer.analyze()
        crit_area_pct = overhang_report["critical_overhang_pct"]

        self.mesh_features = np.array([
            dims[0], dims[1], dims[2],
            surf_area, vol,
            crit_area_pct,
        ], dtype=np.float32)

        # Best solution found so far
        self.best_chromosome = SupportChromosome.random()
        self.best_report: Optional[SupportPhysicsEvaluationReport] = None
        self.best_fitness: float = -1.0

    def get_feature_vector(self, chrom: SupportChromosome) -> np.ndarray:
        """Combines mesh geometry features with candidate support chromosome (12 features)."""
        return np.concatenate([self.mesh_features, chrom.to_vector()]).astype(np.float32)

    def evaluate_chromosome_physics(
        self, chrom: SupportChromosome
    ) -> Tuple[SupportPhysicsEvaluationReport, Any]:
        """Runs the true physical simulation on a chromosome."""
        contact_points = self.analyzer.generate_contact_points(
            grid_resolution_mm=chrom.contact_grid_res_mm
        )
        router = BranchRouter(
            mesh=self.mesh,
            material=self.material,
            layer_height=self.layer_height,
            branch_angle_deg=chrom.branch_angle_deg,
            merge_distance_mm=chrom.merge_distance_mm,
            tip_diameter_mm=chrom.tip_diameter_mm,
        )
        graph = router.build_tree_graph(contact_points)
        report = self.evaluator.evaluate(
            graph, contact_points, priority_profile=self.priority_profile
        )
        return report, graph

    def step_generation(self, population_size: int = 6) -> Dict[str, Any]:
        """
        Executes one evolutionary generation.
        Mutates candidates, ranks them via neural surrogate, evaluates top candidate in physics,
        and returns training experience.
        """
        # Create mutants from current best
        candidates = [self.best_chromosome]
        for _ in range(population_size - 1):
            candidates.append(self.best_chromosome.mutate())

        # Neural surrogate pre-screen
        feat_matrix = np.vstack([self.get_feature_vector(c) for c in candidates])
        predicted_outputs = self.surrogate.predict(feat_matrix)

        # Rank candidates using predicted fitness heuristic
        # predicted targets: [coverage, filament_mass, breakaway_force, stability]
        predicted_scores = []
        for p in predicted_outputs:
            pred_cov = p[0]
            pred_mass = p[1]
            pred_stab = p[3]
            score = pred_cov * 0.4 - pred_mass * 0.3 + pred_stab * 0.3
            predicted_scores.append(score)

        top_indices = np.argsort(predicted_scores)[::-1]
        chosen_idx = top_indices[0]
        chosen_chrom = candidates[chosen_idx]

        # Evaluate physical ground truth
        report, graph = self.evaluate_chromosome_physics(chosen_chrom)

        if report.composite_fitness_score > self.best_fitness:
            self.best_fitness = report.composite_fitness_score
            self.best_chromosome = chosen_chrom
            self.best_report = report

        # Ground truth training sample for surrogate:
        # [coverage, mass, breakaway_force, stability]
        y_true = np.array([
            report.coverage_score_pct,
            report.support_mass_grams,
            report.average_breakaway_force_n,
            report.stability_score_pct,
        ], dtype=np.float32)

        x_feat = self.get_feature_vector(chosen_chrom)

        return {
            "x_feature": x_feat,
            "y_target": y_true,
            "report": report,
            "chromosome": asdict(chosen_chrom),
            "best_fitness": self.best_fitness,
            "filament_saved_pct": report.filament_saved_vs_standard_grid_pct,
            "removal_ease": report.removal_ease_score,
            "graph": graph,
        }
