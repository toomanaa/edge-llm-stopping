#!/usr/bin/env python3
"""Pure-numpy inference for the trained sufficiency probe. Runs on the
Pi with no scikit-learn or PyTorch dependency -- just the exported
probe_weights.npz and numpy, which the Pi already has.

Reimplements exactly what scikit-learn's StandardScaler.transform() and
MLPClassifier.predict_proba() do internally, for a single hidden-layer
MLP with ReLU activation and a logistic (sigmoid) output for binary
classification -- the architecture used throughout this project
(hidden_layer_sizes=(64,)).

Usage:
    from probe_inference import Probe
    probe = Probe("probe_weights.npz")
    prob_safe = probe.predict_proba(hidden_state_vector)  # float in [0, 1]
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class Probe:
    def __init__(self, weights_path):
        d = np.load(weights_path)
        self.scaler_mean = d["scaler_mean"]
        self.scaler_scale = d["scaler_scale"]
        self.activation = str(d["activation"])
        self.out_activation = str(d["out_activation"])
        self.n_layers = int(d["n_layers"])
        self.weights = [d[f"W{i}"] for i in range(self.n_layers)]
        self.biases = [d[f"b{i}"] for i in range(self.n_layers)]

    def _scale(self, x):
        # matches sklearn.preprocessing.StandardScaler.transform exactly:
        # (x - mean) / scale
        return (x - self.scaler_mean) / self.scaler_scale

    def _forward(self, x):
        # matches sklearn.neural_network.MLPClassifier's forward pass:
        # hidden layers use `self.activation` (relu here), the final
        # layer uses `self.out_activation_` (logistic for binary,
        # softmax for multi-class) -- no activation function is applied
        # a second time to the output of the last matmul beyond that.
        h = x
        for i in range(self.n_layers):
            h = h @ self.weights[i] + self.biases[i]
            is_last = (i == self.n_layers - 1)
            if not is_last:
                if self.activation == "relu":
                    h = relu(h)
                else:
                    raise NotImplementedError(
                        f"activation '{self.activation}' not implemented "
                        f"in this lightweight reimplementation")
        if self.out_activation == "logistic":
            return sigmoid(h)
        elif self.out_activation == "softmax":
            return softmax(h)
        else:
            raise NotImplementedError(
                f"out_activation '{self.out_activation}' not implemented")

    def predict_proba(self, hidden_state):
        """hidden_state: 1D numpy array, the model's raw hidden state
        vector (before scaling). Returns P(safe to stop) as a float."""
        x = self._scale(np.asarray(hidden_state, dtype=np.float64))
        out = self._forward(x)
        # sklearn's binary MLPClassifier.predict_proba returns
        # [P(class 0), P(class 1)] built from a single sigmoid output
        # `p`: P(class 1) = p, P(class 0) = 1 - p. We return P(class 1)
        # (the positive / "safe to stop" class) directly, since that is
        # the only thing the controller needs.
        if self.out_activation == "logistic":
            return float(out[0]) if out.ndim > 0 else float(out)
        else:  # softmax, multi-class -- return P(class 1) by convention
            return float(out[1])
