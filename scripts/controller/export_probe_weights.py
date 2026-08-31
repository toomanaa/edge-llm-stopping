#!/usr/bin/env python3
"""Export cell for probe_training_v2.ipynb (or a standalone follow-up).

Extracts the trained probe's weights and the scaler's parameters as
plain numpy arrays, saved to a single .npz file. This is what lets the
probe run on the Pi with nothing but numpy -- no scikit-learn, no
PyTorch, no pickle version compatibility risk between the Colab
environment that trained it and the Pi that runs it.

Run this in the same Colab session where `probe_final` and
`scaler_final` already exist (Section 10 of probe_training_v2.ipynb),
or after loading probe_v2_and_scaler.pkl fresh.

Usage (paste as a new cell after Section 10, or run standalone with the
pickle uploaded):
"""
import pickle
import numpy as np

# If probe_final / scaler_final are already in memory from Section 10,
# this block is skipped. Otherwise, load from the saved pickle.
try:
    probe_final, scaler_final
    print("using probe_final / scaler_final already in memory")
except NameError:
    from google.colab import files
    print("Upload probe_v2_and_scaler.pkl:")
    uploaded = files.upload()
    saved = pickle.load(open(list(uploaded.keys())[0], "rb"))
    probe_final = saved["probe"]
    scaler_final = saved["scaler"]
    print("loaded from pickle")

# --- sanity checks before export ---
assert hasattr(probe_final, "coefs_"), "probe does not look like a fitted MLPClassifier"
n_layers = len(probe_final.coefs_)
print(f"probe: {n_layers} weight matrices, "
     f"shapes {[w.shape for w in probe_final.coefs_]}")
print(f"activation: {probe_final.activation}, "
     f"output activation: {probe_final.out_activation_}")
print(f"classes: {probe_final.classes_}")

# --- extract everything needed for a from-scratch forward pass ---
export = {
    "scaler_mean": scaler_final.mean_,
    "scaler_scale": scaler_final.scale_,
    "activation": probe_final.activation,          # e.g. "relu"
    "out_activation": probe_final.out_activation_,  # e.g. "logistic"
    "n_layers": n_layers,
}
for i, (W, b) in enumerate(zip(probe_final.coefs_, probe_final.intercepts_)):
    export[f"W{i}"] = W
    export[f"b{i}"] = b

np.savez("probe_weights.npz", **export)

# report total parameter count and file size, for the paper's overhead
# accounting (Section 6.7 / E4)
n_params = sum(W.size + b.size for W, b in
               zip(probe_final.coefs_, probe_final.intercepts_))
n_params += scaler_final.mean_.size + scaler_final.scale_.size
print(f"\ntotal exported parameters: {n_params:,}")

import os
print(f"file size: {os.path.getsize('probe_weights.npz') / 1024:.1f} KB")

from google.colab import files
files.download("probe_weights.npz")
print("\ndownloaded probe_weights.npz -- copy this to the Pi.")
