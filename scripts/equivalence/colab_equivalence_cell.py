# ============================================================
# Quantization equivalence check (Step B)
# Compares Q4/llama.cpp hidden states (collected on the Pi) against
# full-precision HuggingFace hidden states for the SAME 20 text snippets.
# Run this AFTER the model-loading cell in probe_training.ipynb.
# ============================================================
import json, numpy as np

# Upload q4_embeddings.json when prompted
from google.colab import files
uploaded = files.upload()
q4_path = list(uploaded.keys())[0]
q4_data = json.load(open(q4_path))
print(f"loaded {len(q4_data)} Q4 embeddings from the Pi")

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def l2_dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

results = []
for i, entry in enumerate(q4_data):
    full_text = chat_prompt(entry["question"]) + entry["prefix_text"]
    fp_vec = get_hidden_state_for(entry["question"], entry["prefix_text"])
    q4_vec = np.array(entry["q4_embedding"])

    if fp_vec.shape[0] != q4_vec.shape[0]:
        print(f"[{i+1}] {entry['id']} boundary {entry['boundary']}: "
              f"DIMENSION MISMATCH fp={fp_vec.shape[0]} q4={q4_vec.shape[0]} -- skipping")
        continue

    cos = cosine_sim(fp_vec, q4_vec)
    l2 = l2_dist(fp_vec, q4_vec)
    # also compare against fp vector's own norm, for scale context
    fp_norm = float(np.linalg.norm(fp_vec))
    results.append({"id": entry["id"], "boundary": entry["boundary"],
                    "cosine": cos, "l2": l2, "fp_norm": fp_norm,
                    "chars": len(entry["prefix_text"])})
    print(f"[{i+1:>2}/{len(q4_data)}] {entry['id']:>10} b{entry['boundary']:<3} "
          f"cos={cos:.4f}  L2={l2:.2f}  (fp norm={fp_norm:.2f})")

cos_vals = [r["cosine"] for r in results]
l2_vals = [r["l2"] for r in results]
print(f"\n--- summary over {len(results)} pairs ---")
print(f"cosine similarity: mean={np.mean(cos_vals):.4f}  min={np.min(cos_vals):.4f}  max={np.max(cos_vals):.4f}")
print(f"L2 distance:        mean={np.mean(l2_vals):.2f}  min={np.min(l2_vals):.2f}  max={np.max(l2_vals):.2f}")

# rough interpretation guide -- not a formal statistical test, just a sanity read
low_cos = [r for r in results if r["cosine"] < 0.90]
if low_cos:
    print(f"\n{len(low_cos)} pair(s) with cosine < 0.90 (possible meaningful divergence):")
    for r in low_cos:
        print(f"  {r['id']} boundary {r['boundary']} ({r['chars']} chars): cos={r['cosine']:.4f}")
else:
    print("\nAll pairs above cosine 0.90 -- no obvious large divergence in this sample.")

print("""
How to read this:
- cosine close to 1.0 means the two hidden-state vectors point in
  almost the same direction -- the Q4 model's internal representation
  at this point is structurally very similar to the full-precision one.
- cosine noticeably below 1.0 (especially < 0.9) means real divergence --
  worth checking whether it's a specific kind of input causing it (long
  vs short, easy vs hard boundary) rather than assuming it's noise.
- L2 distance depends on the vectors' scale (see fp_norm for context);
  a large L2 relative to fp_norm matters more than the raw number alone.
This is a first, coarse check on 20 examples -- a signal to decide
whether deeper work is needed, not a final validation.
""")
