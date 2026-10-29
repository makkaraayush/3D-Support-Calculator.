"""
Unit tests for neural surrogate model, evolutionary search, and continuous learning engine.
"""

import os
import unittest
import numpy as np
from src.geometry.mesh import TriangleMesh
from src.ai.surrogate_model import NeuralSurrogateModel
from src.ai.evolutionary_search import EvolutionarySupportOptimizer, SupportChromosome
from src.ai.continuous_engine import ContinuousLearningEngine


def create_test_mesh() -> TriangleMesh:
    """Creates a basic overhang test mesh."""
    v = np.array([
        [-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0],
        [-10, -10, 15], [10, -10, 15], [10, 10, 15], [-10, 10, 15],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    return TriangleMesh(v, f)


class TestAIEngine(unittest.TestCase):
    def test_surrogate_forward_and_train(self):
        model = NeuralSurrogateModel(input_dim=12, hidden_dims=(32, 16), output_dim=4)
        x_dummy = np.random.randn(8, 12).astype(np.float32)
        y_dummy = np.random.randn(8, 4).astype(np.float32)

        # Initial prediction
        pred_before = model.predict(x_dummy)
        self.assertEqual(pred_before.shape, (8, 4))

        # Train multiple steps and verify loss is positive and decreases
        initial_loss = model.train_step(x_dummy, y_dummy)
        self.assertGreater(initial_loss, 0.0)

        for _ in range(15):
            last_loss = model.train_step(x_dummy, y_dummy)
        self.assertLess(last_loss, initial_loss)

    def test_chromosome_mutation(self):
        c1 = SupportChromosome.random()
        c2 = c1.mutate(mutation_rate=1.0)
        # Verify valid bounds
        self.assertGreaterEqual(c2.branch_angle_deg, 15.0)
        self.assertLessEqual(c2.branch_angle_deg, 40.0)
        self.assertGreaterEqual(c2.tip_diameter_mm, 0.4)

    def test_evolutionary_optimizer_step(self):
        mesh = create_test_mesh()
        surrogate = NeuralSurrogateModel(input_dim=12, hidden_dims=(16, 8), output_dim=4)
        opt = EvolutionarySupportOptimizer(mesh, surrogate, material="PLA", layer_height=0.20)

        step_res = opt.step_generation(population_size=4)
        self.assertIn("best_fitness", step_res)
        self.assertIn("x_feature", step_res)
        self.assertEqual(len(step_res["x_feature"]), 12)
        self.assertEqual(len(step_res["y_target"]), 4)

    def test_continuous_engine_lifecycle(self):
        mesh = create_test_mesh()
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_ai")
        engine = ContinuousLearningEngine(data_dir=tmp_dir)
        engine.set_active_mesh(mesh)

        status_init = engine.get_status()
        self.assertFalse(status_init["is_running"])

        started = engine.start()
        self.assertTrue(started)
        status_running = engine.get_status()
        self.assertTrue(status_running["is_running"])

        # Stop and clean up
        engine.stop()
        status_stopped = engine.get_status()
        self.assertFalse(status_stopped["is_running"])

        # Clean up files
        for f in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
