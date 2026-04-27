"""
NumPy-only neural network implementation from scratch.

This module implements a Multi-Layer Perceptron (MLP) regressor using ONLY
NumPy for all mathematical operations.  No sklearn, PyTorch, TensorFlow,
Keras, or any other ML library is used for the learning algorithm.

Components implemented manually:
  - DenseLayer        : fully-connected layer  (y = X @ W + b)
  - ReLU activation   : max(0, x)
  - MSE loss          : mean((y_pred - y_true)^2)
  - Backpropagation   : chain-rule gradient computation
  - Adam optimiser    : adaptive moment estimation weight updates
  - Normaliser        : z-score normalisation (mean / std)
  - NumpyMLP          : training loop with mini-batches & early stopping

Author's note:
    NumPy is a numerical linear-algebra library — it provides matrix
    multiplication, element-wise operations, and random number generation.
    It does NOT contain any machine-learning models, training loops,
    optimisers, or loss functions.  Every learning component below is
    written by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normaliser – z-score standardisation implemented manually
# ---------------------------------------------------------------------------
class Normaliser:
    """
    z-score normalisation:  x_norm = (x - mean) / (std + eps)

    We compute mean and std from training data only, then apply the same
    transform to validation / test / inference data so there is no data
    leakage.
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._eps: float = 1e-8  # avoid division by zero

    def fit(self, X: np.ndarray) -> "Normaliser":
        """Compute mean and std from training data (axis=0 → per feature)."""
        self.mean = np.mean(X, axis=0)                       # shape (D,)
        self.std = np.std(X, axis=0)                         # shape (D,)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normaliser has not been fitted yet.")
        return (X - self.mean) / (self.std + self._eps)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Reverse the normalisation."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normaliser has not been fitted yet.")
        return X_norm * (self.std + self._eps) + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.std = np.asarray(d["std"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dense (Fully-Connected) Layer
# ---------------------------------------------------------------------------
class DenseLayer:
    """
    A single fully-connected layer:  output = input @ W + b

    Parameters
    ----------
    in_features  : number of input neurons
    out_features : number of output neurons

    Weight initialisation uses He (Kaiming) normal:
        W ~ N(0, sqrt(2 / in_features))
    which works well with ReLU activations.

    Attributes stored for backpropagation:
        _input_cache : the input that was fed during forward pass
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # He initialisation – good for ReLU networks
        scale = np.sqrt(2.0 / in_features)
        self.W: np.ndarray = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b: np.ndarray = np.zeros((1, out_features), dtype=np.float64)

        # Gradients (same shape as parameters)
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)

        # Cache for backward pass
        self._input_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass:  Z = X @ W + b

        X has shape (batch_size, in_features).
        Returns Z with shape (batch_size, out_features).
        """
        self._input_cache = X                    # save for backward
        return X @ self.W + self.b               # matrix multiply + bias

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Backward pass – compute gradients and propagate error back.

        dZ : gradient of the loss w.r.t. the output of this layer
             shape (batch_size, out_features)

        Math:
            dW = X^T @ dZ / batch_size
            db = mean(dZ, axis=0)
            dX = dZ @ W^T          (gradient to pass to the previous layer)

        Returns dX (shape = (batch_size, in_features)).
        """
        X = self._input_cache
        batch_size = X.shape[0]

        # Gradient of loss w.r.t. weights and biases
        self.dW = (X.T @ dZ) / batch_size
        self.db = np.mean(dZ, axis=0, keepdims=True)

        # Gradient to propagate backwards through the network
        dX = dZ @ self.W.T
        return dX


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------
class ReLU:
    """
    Rectified Linear Unit:  f(x) = max(0, x)

    This is the most common activation function for hidden layers.
    It introduces non-linearity so the network can learn complex patterns.

    Derivative:
        f'(x) = 1  if x > 0
        f'(x) = 0  if x <= 0
    """

    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self._mask = (Z > 0).astype(np.float64)   # 1 where positive, 0 elsewhere
        return Z * self._mask                       # element-wise max(0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        The gradient flows through only where the input was positive.
        """
        return dA * self._mask


# ---------------------------------------------------------------------------
# Loss Function – Mean Squared Error
# ---------------------------------------------------------------------------
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    MSE = (1/N) * Σ (y_pred_i - y_true_i)^2

    This measures the average squared difference between predictions
    and true values.  Lower is better.
    """
    diff = y_pred - y_true
    return float(np.mean(diff ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. y_pred:
        d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / N

    This is the starting point of backpropagation.
    """
    N = y_pred.shape[0]
    return 2.0 * (y_pred - y_true) / N


# ---------------------------------------------------------------------------
# Evaluation Metrics – implemented manually (no sklearn.metrics)
# ---------------------------------------------------------------------------
def manual_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error = (1/N) * Σ |y_pred_i - y_true_i|"""
    return float(np.mean(np.abs(y_pred - y_true)))


def manual_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error = sqrt( (1/N) * Σ (y_pred_i - y_true_i)^2 )"""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def manual_max_abs_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Maximum Absolute Error = max_i |y_pred_i - y_true_i|"""
    return float(np.max(np.abs(y_pred - y_true)))


# ---------------------------------------------------------------------------
# Adam Optimiser – implemented manually
# ---------------------------------------------------------------------------
class AdamOptimiser:
    """
    Adam (Adaptive Moment Estimation) optimiser.

    Adam maintains two running averages per parameter:
        m  –  first moment  (mean of gradients)           → "momentum"
        v  –  second moment (mean of squared gradients)    → "adaptive learning rate"

    Update rule for each parameter θ:
        m  = β₁ * m  + (1 - β₁) * grad
        v  = β₂ * v  + (1 - β₂) * grad²
        m̂  = m / (1 - β₁^t)          # bias-corrected
        v̂  = v / (1 - β₂^t)          # bias-corrected
        θ  = θ - lr * m̂ / (√v̂ + ε)

    Default hyper-parameters follow the original Adam paper (Kingma & Ba, 2015).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # State: one (m, v) pair per parameter array
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}
        self._t: int = 0   # global time step

    def step(self, layers: List[DenseLayer]) -> None:
        """
        Perform one Adam update for all parameters in all layers.
        Must be called AFTER backward() has populated each layer's dW, db.
        """
        self._t += 1

        for layer_idx, layer in enumerate(layers):
            for param_name in ("W", "b"):
                param: np.ndarray = getattr(layer, param_name)
                grad: np.ndarray = getattr(layer, f"d{param_name}")
                # Use a stable key (layer index + param name) instead of id(param)
                # because setattr creates new arrays whose ids may collide.
                key = (layer_idx, param_name)

                # Apply L2 weight decay (optional regularisation)
                if self.weight_decay > 0.0:
                    grad = grad + self.weight_decay * param

                # Initialise moment buffers on first call
                if key not in self._m:
                    self._m[key] = np.zeros_like(param)
                    self._v[key] = np.zeros_like(param)

                # Update biased first and second moment estimates
                self._m[key] = self.beta1 * self._m[key] + (1.0 - self.beta1) * grad
                self._v[key] = self.beta2 * self._v[key] + (1.0 - self.beta2) * (grad ** 2)

                # Bias correction
                m_hat = self._m[key] / (1.0 - self.beta1 ** self._t)
                v_hat = self._v[key] / (1.0 - self.beta2 ** self._t)

                # Parameter update
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                setattr(layer, param_name, param - update)


# ---------------------------------------------------------------------------
# MLP Network – the full neural network built from the components above
# ---------------------------------------------------------------------------
@dataclass
class TrainingHistory:
    """Stores per-epoch loss values for plotting training curves."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


class NumpyMLP:
    """
    Multi-Layer Perceptron built from scratch with NumPy.

    Architecture (configurable):
        Input → [Dense → ReLU] × (n_hidden - 1) → Dense → Output

    Default architecture per the project spec:
        Input(D) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Dense(1)
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, ...] = (128, 128, 64, 1),
        n_input: int = 1,
        seed: int = 42,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.n_input = n_input
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Build layers: input → hidden1 → hidden2 → ... → output
        self.layers: List[DenseLayer] = []
        self.activations: List[Optional[ReLU]] = []

        prev_size = n_input
        for i, size in enumerate(layer_sizes):
            self.layers.append(DenseLayer(prev_size, size, rng))
            # ReLU after every layer except the last (output) layer
            if i < len(layer_sizes) - 1:
                self.activations.append(ReLU())
            else:
                self.activations.append(None)  # linear output
            prev_size = size

        self.optimiser = AdamOptimiser(lr=lr, weight_decay=weight_decay)
        self.history = TrainingHistory()

    # ---- Forward Pass ----
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation:
            Pass the input through each layer and activation sequentially.
            Each layer computes Z = X @ W + b, then applies ReLU (except output).

        Returns the final prediction (batch_size, output_dim).
        """
        out = X
        for layer, activation in zip(self.layers, self.activations):
            out = layer.forward(out)           # linear transform
            if activation is not None:
                out = activation.forward(out)  # non-linearity
        return out

    # ---- Backward Pass ----
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Backpropagation:
            1. Compute the loss gradient at the output layer.
            2. Walk backwards through each layer, computing gradients
               of the loss w.r.t. weights and biases using the chain rule.

        The chain rule says:
            ∂L/∂W_i = ∂L/∂Z_out · ∂Z_out/∂A_(i+1) · ... · ∂Z_i/∂W_i

        In practice we propagate dZ backwards through activations and layers.
        """
        loss = mse_loss(y_pred, y_true)

        # Starting gradient: d(MSE)/d(y_pred)
        dout = mse_grad(y_pred, y_true)

        # Walk backwards through layers
        for i in reversed(range(len(self.layers))):
            activation = self.activations[i]
            if activation is not None:
                dout = activation.backward(dout)   # gradient through ReLU
            dout = self.layers[i].backward(dout)   # gradient through Dense

        return loss

    # ---- Optimiser Step ----
    def step(self) -> None:
        """Apply one Adam update to all layer weights and biases."""
        self.optimiser.step(self.layers)

    # ---- Predict (no gradient tracking) ----
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass for inference (same as forward, just a semantic alias)."""
        return self.forward(X)

    # ---- Save / Load ----
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Export all weights and biases as a flat dict for saving."""
        d: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            d[f"layer_{i}_W"] = layer.W
            d[f"layer_{i}_b"] = layer.b
        return d

    def set_weights(self, d: Dict[str, np.ndarray]) -> None:
        """Load weights and biases from a dict (produced by get_weights)."""
        for i, layer in enumerate(self.layers):
            layer.W = np.asarray(d[f"layer_{i}_W"], dtype=np.float64)
            layer.b = np.asarray(d[f"layer_{i}_b"], dtype=np.float64)
            # Update internal sizes
            layer.in_features = layer.W.shape[0]
            layer.out_features = layer.W.shape[1]


# ---------------------------------------------------------------------------
# Training Loop – mini-batch SGD with early stopping
# ---------------------------------------------------------------------------
def train_mlp(
    model: NumpyMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 300,
    batch_size: int = 512,
    patience: int = 30,
    min_delta: float = 1e-6,
    verbose: bool = True,
    target_name: str = "",
) -> NumpyMLP:
    """
    Mini-batch training loop with early stopping.

    Parameters
    ----------
    model        : NumpyMLP instance to train
    X_train      : normalised training features   (N_train, D)
    y_train      : normalised training targets     (N_train, 1)
    X_val        : normalised validation features  (N_val, D)
    y_val        : normalised validation targets    (N_val, 1)
    epochs       : maximum number of full passes through the training data
    batch_size   : number of samples per mini-batch
    patience     : stop if validation loss hasn't improved for this many epochs
    min_delta    : minimum improvement to count as "better"
    verbose      : print progress every 10 epochs
    target_name  : label for logging

    Returns the trained model (with best weights restored).
    """
    n_train = X_train.shape[0]
    rng = np.random.default_rng(model.seed + 999)

    best_val_loss = np.inf
    best_weights = model.get_weights()
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        # --- Mini-batch training ---
        epoch_losses: List[float] = []
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Backward pass – compute gradients
            batch_loss = model.backward(y_pred, y_batch)
            epoch_losses.append(batch_loss)

            # Update weights using Adam
            model.step()

        train_loss = float(np.mean(epoch_losses))

        # --- Validation loss ---
        y_val_pred = model.predict(X_val)
        val_loss = mse_loss(y_val_pred, y_val)

        model.history.train_loss.append(train_loss)
        model.history.val_loss.append(val_loss)

        # --- Early stopping check ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or epochs_no_improve == patience):
            print(
                f"  [{target_name}] epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"best_val={best_val_loss:.6f}  no_improve={epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  [{target_name}] Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    model.set_weights(best_weights)
    return model
