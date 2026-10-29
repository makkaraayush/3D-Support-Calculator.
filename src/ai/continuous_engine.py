"""
Continuous background learning loop for support topology exploration,
physics evaluation recording, and surrogate model fine-tuning.
"""

import json
import os
import threading
import time
from typing import Dict, Any, List, Optional
import numpy as np
from ..geometry.mesh import TriangleMesh
from .surrogate_model import NeuralSurrogateModel
from .evolutionary_search import EvolutionarySupportOptimizer, SupportChromosome


class ContinuousLearningEngine:
    def __init__(
        self,
        data_dir: str = "data",
        surrogate_weights_file: str = "surrogate_weights.npz",
        knowledge_bank_file: str = "knowledge_bank.json",
    ):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.weights_path = os.path.join(data_dir, surrogate_weights_file)
        self.knowledge_path = os.path.join(data_dir, knowledge_bank_file)

        # 1. Neural Surrogate
        self.surrogate = NeuralSurrogateModel(input_dim=12, hidden_dims=(64, 32), output_dim=4)
        if os.path.exists(self.weights_path):
            self.surrogate.load_weights(self.weights_path)

        # 2. Experience Replay Buffer
        self.replay_X: List[np.ndarray] = []
        self.replay_Y: List[np.ndarray] = []
        self.max_replay_size = 1000

        # Load persisted knowledge bank if available
        self.total_iterations = 0
        self.training_history: List[Dict[str, float]] = []
        self._load_knowledge_bank()

        # 3. Threading and State
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Active mesh and optimizer
        self.current_mesh: Optional[TriangleMesh] = None
        self.optimizer: Optional[EvolutionarySupportOptimizer] = None
        self.material = "PLA"
        self.layer_height = 0.20
        self.priority_profile = "balanced"

        # Latest state snapshot
        self.latest_status: Dict[str, Any] = {
            "is_running": False,
            "total_iterations": self.total_iterations,
            "best_fitness": 0.0,
            "filament_saved_pct": 0.0,
            "removal_ease_score": 0.0,
            "loss": 0.0,
            "best_parameters": None,
            "status_message": "Idle - ready to explore",
        }

    def _load_knowledge_bank(self) -> None:
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.total_iterations = data.get("total_iterations", 0)
                    self.training_history = data.get("history", [])
            except Exception:
                pass

    def _save_knowledge_bank(self) -> None:
        try:
            with open(self.knowledge_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_iterations": self.total_iterations,
                    "history": self.training_history[-200:],
                }, f, indent=2)
            self.surrogate.save_weights(self.weights_path)
        except Exception:
            pass

    def set_active_mesh(
        self,
        mesh: TriangleMesh,
        material: str = "PLA",
        layer_height: float = 0.20,
        priority_profile: str = "balanced",
    ) -> None:
        """Sets the active 3D model mesh for continuous background self-training."""
        with self._lock:
            self.current_mesh = mesh
            self.material = material
            self.layer_height = layer_height
            self.priority_profile = priority_profile

            self.optimizer = EvolutionarySupportOptimizer(
                mesh=mesh,
                surrogate=self.surrogate,
                material=material,
                layer_height=layer_height,
                priority_profile=priority_profile,
            )

    def start(self) -> bool:
        """Starts background continuous learning loop."""
        if self.is_running:
            return True
        if self.current_mesh is None or self.optimizer is None:
            return False

        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Pauses background continuous learning loop."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._save_knowledge_bank()

    def _loop(self) -> None:
        """Continuously tests new support configurations and trains the surrogate."""
        while self.is_running:
            try:
                with self._lock:
                    if self.optimizer is None:
                        break
                    step_result = self.optimizer.step_generation(population_size=6)

                # Store experience
                self.replay_X.append(step_result["x_feature"])
                self.replay_Y.append(step_result["y_target"])
                if len(self.replay_X) > self.max_replay_size:
                    self.replay_X.pop(0)
                    self.replay_Y.pop(0)

                # Retrain neural surrogate on mini-batch from experience buffer
                loss = 0.0
                if len(self.replay_X) >= 4:
                    batch_size = min(32, len(self.replay_X))
                    indices = np.random.choice(len(self.replay_X), batch_size, replace=False)
                    X_b = np.array([self.replay_X[i] for i in indices])
                    Y_b = np.array([self.replay_Y[i] for i in indices])
                    loss = self.surrogate.train_step(X_b, Y_b)

                self.total_iterations += 1

                # Update live status
                status_update = {
                    "is_running": True,
                    "total_iterations": self.total_iterations,
                    "best_fitness": round(float(step_result["best_fitness"]), 2),
                    "filament_saved_pct": round(float(step_result["filament_saved_pct"]), 1),
                    "removal_ease_score": round(float(step_result["removal_ease"]), 1),
                    "loss": round(float(loss), 5),
                    "best_parameters": step_result["chromosome"],
                    "status_message": f"Exploring generation {self.total_iterations}... Loss: {loss:.4f}",
                }

                with self._lock:
                    self.latest_status.update(status_update)

                # Append to history every 5 iterations
                if self.total_iterations % 5 == 0:
                    self.training_history.append({
                        "iteration": self.total_iterations,
                        "fitness": round(float(step_result["best_fitness"]), 2),
                        "saved_pct": round(float(step_result["filament_saved_pct"]), 1),
                        "loss": round(float(loss), 5),
                    })
                    self._save_knowledge_bank()

                # Gentle sleep to keep CPU usage low on low-end systems (e.g. i5 2400s)
                time.sleep(0.08)

            except Exception as e:
                with self._lock:
                    self.latest_status["status_message"] = f"Optimizer paused: {str(e)}"
                time.sleep(0.5)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            state = dict(self.latest_status)
            state["is_running"] = self.is_running
            return state

    def get_best_support_solution(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.optimizer and self.optimizer.best_report:
                report, graph = self.optimizer.evaluate_chromosome_physics(
                    self.optimizer.best_chromosome
                )
                return {
                    "chromosome": self.optimizer.best_chromosome,
                    "report": report,
                    "graph": graph,
                }
            return None
