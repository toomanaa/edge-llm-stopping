# ============================================================
# Probe-on-Q4 evaluation (the direct deployment-gap test)
# Loads the ALREADY-TRAINED probe (from probe_and_scaler.pkl) and runs
# it directly on real Q4 hidden states collected from the Pi, using the
# SAME train/test split logic (grouped by response id) as the original
# training run, so this is an honest held-out evaluation, not just a
# re-run on the training data.
# ============================================================
import json, pickle, random
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- Step 1: load the trained probe + scaler ---
# Upload probe_and_scaler.pkl when prompted (the file downloaded at the
# end of the original probe-training run)
from google.colab import files
print("Upload probe_and_scaler.pkl:")
uploaded = files.upload()
probe_path = list(uploaded.keys())[0]
saved = pickle.load(open(probe_path, "rb"))
probe = saved["probe"]
scaler = saved["scaler"]
print(f"loaded probe (hidden_size={saved['hidden_size']})")

# --- Step 2: load the Q4 embeddings collected from the Pi ---
print("\nUpload q4_embeddings_full.json:")
uploaded2 = files.upload()
q4_path = list(uploaded2.keys())[0]
q4_data = json.load(open(q4_path))
print(f"loaded {len(q4_data)} Q4 embeddings")

n_safe = sum(1 for e in q4_data if e["safe_stop"])
print(f"{n_safe} safe / {len(q4_data)-n_safe} unsafe")

# --- Step 3: same grouped train/test split as the original training run ---
# (test_size=0.25, random_state=42 -- matching probe_training.ipynb so the
# test set here corresponds to genuinely held-out responses)
unique_ids = list(set(e["id"] for e in q4_data))
random.seed(42)
random.shuffle(unique_ids)
n_test_ids = int(len(unique_ids) * 0.25)
test_ids = set(unique_ids[:n_test_ids])

test_data = [e for e in q4_data if e["id"] in test_ids]
X_test_q4 = np.array([e["q4_embedding"] for e in test_data])
y_test_q4 = np.array([1 if e["safe_stop"] else 0 for e in test_data])

print(f"\nheld-out test set (Q4): {len(X_test_q4)} boundaries")
print(f"test set positives: {sum(y_test_q4)}/{len(y_test_q4)}")

# --- Step 4: run the probe (trained on full-precision states) on Q4 states ---
X_test_q4_scaled = scaler.transform(X_test_q4)
probs_q4 = probe.predict_proba(X_test_q4_scaled)[:, 1]

precisions_q4, recalls_q4, thresholds_q4 = precision_recall_curve(y_test_q4, probs_q4)
ap_q4 = average_precision_score(y_test_q4, probs_q4)
print(f"\nprobe average precision ON Q4 DATA: {ap_q4:.3f}")

print("\nProbe operating points on Q4 hidden states (subsample):")
print(f"{'threshold':>10} {'precision':>10} {'recall':>8}")
step = max(1, len(thresholds_q4) // 10)
for i in range(0, len(thresholds_q4), step):
    print(f"{thresholds_q4[i]:>10.3f} {precisions_q4[i]:>10.3f} {recalls_q4[i]:>8.3f}")

# --- Step 5: plot against the original (full-precision) curve and entropy ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

entropy_recall =    [0.308, 0.373, 0.451, 0.535, 0.655, 0.794, 0.961, 1.000, 1.000]
entropy_precision = [0.957, 0.964, 0.961, 0.963, 0.963, 0.963, 0.965, 0.966, 0.966]

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(recalls_q4, precisions_q4, color="#4f81c4", linewidth=1.8,
       label="Probe on Q4 hidden states")
ax.plot(entropy_recall, entropy_precision, color="#c46b4f", linewidth=1.5,
       marker=".", linestyle="--", label="Entropy (S3)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Probe (trained on FP32, tested on Q4) vs. entropy")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("fig_probe_on_q4.pdf")
fig.savefig("fig_probe_on_q4.png", dpi=150)
plt.show()

files.download("fig_probe_on_q4.pdf")
files.download("fig_probe_on_q4.png")

print(f"""
--- Summary ---
Test set: {len(X_test_q4)} boundaries, {sum(y_test_q4)} safe / {len(y_test_q4)-sum(y_test_q4)} unsafe
Probe average precision on Q4 hidden states: {ap_q4:.3f}

This is the direct test: the probe was trained on full-precision
hidden states, and is now evaluated on real Q4/llama.cpp hidden states
from the actual deployment platform. If precision/recall here is
close to what was seen on full-precision data, the deployment gap is
small in practice, not just in raw vector similarity. If it degrades
substantially, that's real evidence the probe needs to be retrained
directly on Q4 hidden states rather than full-precision ones.
""")
