"""
Neural surrogate model for predicting multi-physics support metrics
from geometric features and routing hyperparameters.
"""

import os
from typing import Tuple, Optional
import numpy as np


class NeuralSurrogateModel:
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Tuple[int, int] = (64, 32),
        output_dim: int = 4,
        learning_rate: float = 0.003,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = learning_rate

        # Initialize network weights (He initialization for LeakyReLU)
        np.random.seed(42)
        h1, h2 = hidden_dims

        self.W1 = np.random.randn(input_dim, h1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1, dtype=np.float32)

        self.W2 = np.random.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float32)

        self.W3 = np.random.randn(h2, output_dim).astype(np.float32) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

        # Adam optimizer state
        self.m = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.v = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # Feature normalization running statistics
        self.feat_mean = np.zeros(input_dim, dtype=np.float32)
        self.feat_std = np.ones(input_dim, dtype=np.float32)
        self.out_mean = np.zeros(output_dim, dtype=np.float32)
        self.out_std = np.ones(output_dim, dtype=np.float32)

    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, x * alpha)

    def _leaky_relu_grad(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        X: (N, input_dim) array of normalized features.
        Returns: (N, output_dim) predicted physics targets.
        """
        X = np.atleast_2d(X)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._leaky_relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._leaky_relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        """Denormalized prediction from unscaled input features."""
        raw_features = np.atleast_2d(raw_features)
        norm_X = (raw_features - self.feat_mean) / np.maximum(self.feat_std, 1e-5)
        norm_pred = self.forward(norm_X)
        pred = norm_pred * self.out_std + self.out_mean
        return pred

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray) -> float:
        """
        Single mini-batch training step using Adam optimizer with MSE loss.
        """
        N = len(X_batch)
        if N == 0:
            return 0.0

        # Update running normalization stats
        self.feat_mean = 0.95 * self.feat_mean + 0.05 * np.mean(X_batch, axis=0)
        self.feat_std = 0.95 * self.feat_std + 0.05 * np.std(X_batch, axis=0)
        self.out_mean = 0.95 * self.out_mean + 0.05 * np.mean(Y_batch, axis=0)
        self.out_std = 0.95 * self.out_std + 0.05 * np.std(Y_batch, axis=0)

        norm_X = (X_batch - self.feat_mean) / np.maximum(self.feat_std, 1e-5)
        norm_Y = (Y_batch - self.out_mean) / np.maximum(self.out_std, 1e-5)

        # Forward pass
        pred = self.forward(norm_X)
        diff = pred - norm_Y
        loss = float(np.mean(diff ** 2))

        # Backward pass
        d_z3 = (2.0 / N) * diff
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0)

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * self._leaky_relu_grad(self.z2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self._leaky_relu_grad(self.z1)
        d_W1 = np.dot(norm_X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        grads = [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        # Adam parameter update
        self.t += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grads[i] ** 2)
            params[i] -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.eps))

        return loss

    def save_weights(self, filepath: str) -> None:
        """Saves weights and normalization parameters to .npz file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            out_mean=self.out_mean,
            out_std=self.out_std,
        )

    def load_weights(self, filepath: str) -> bool:
        """Loads weights from .npz file if it exists."""
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]
            self.feat_mean = data["feat_mean"]
            self.feat_std = data["feat_std"]
            self.out_mean = data["out_mean"]
            self.out_std = data["out_std"]
            return True
        except Exception:
            return False
