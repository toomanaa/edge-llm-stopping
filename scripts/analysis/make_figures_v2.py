#!/usr/bin/env python3
"""Generate figures for the v2 (1,074-question) results.

Reads out/qa_v2_gen_merged.jsonl, out/qa_v2_labels.jsonl, and
out/qa_v2_report.txt, producing:

    figures/fig_v2_waste_by_type.pdf     waste comparison: factual vs explain
    figures/fig_v2_length_hist.pdf       response length distribution, by type
    figures/fig_v2_entropy_roc.pdf       entropy precision/recall on v2 (all boundaries)
    figures/fig_v1_vs_v2_entropy.pdf     v1's inflated curve vs v2's real curve
    figures/fig_v2_summary.txt           plain-text numbers for the paper's prose

Usage (run from ~/e1 on the Pi, or after copying out/ to a laptop):
    pip install matplotlib --break-system-packages   # if needed
    python3 make_figures_v2.py
"""
import json, os, sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Install with:")
    print("  pip install matplotlib --break-system-packages")
    sys.exit(1)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT = "out"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


# =====================================================================
# Figure 1: waste comparison, factual vs explain (v2 internal comparison)
# =====================================================================
def fig_waste_by_type():
    gen = load_jsonl(os.path.join(OUT, "qa_v2_gen_merged.jsonl"))
    labels = load_jsonl(os.path.join(OUT, "qa_v2_labels.jsonl"))
    if not gen or not labels:
        print("skip fig_waste_by_type: missing data")
        return

    gen_by_id = {r["id"]: r for r in gen}
    by_id = defaultdict(list)
    for r in labels:
        by_id[r["id"]].append(r)

    waste_by_type = defaultdict(list)
    for rid, rows in by_id.items():
        safe = [w for w in rows if w["safe_stop"]]
        if not safe:
            continue
        earliest = min(safe, key=lambda w: w["boundary"])
        n_tok = max(w["tok_idx"] for w in rows)
        wasted = n_tok - earliest["tok_idx"]
        qtype = gen_by_id.get(rid, {}).get("type", "unknown")
        waste_by_type[qtype].append(wasted)

    if not waste_by_type:
        print("skip fig_waste_by_type: no waste data")
        return

    types = sorted(waste_by_type.keys())
    avgs = [sum(waste_by_type[t]) / len(waste_by_type[t]) for t in types]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    colors = ["#8fa8c4", "#c46b4f"]
    bars = ax.bar(types, avgs, color=colors[:len(types)], width=0.5)
    ax.set_ylabel("Avg. tokens wasted\nafter earliest safe stop")
    ax.set_title("Post-answer waste by question type (v2)")
    for b, v in zip(bars, avgs):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_v2_waste_by_type.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_v2_waste_by_type.png"))
    plt.close(fig)
    print(f"fig_v2_waste_by_type: " + ", ".join(f"{t}={a:.1f}" for t, a in zip(types, avgs)))


# =====================================================================
# Figure 2: response length distribution by type
# =====================================================================
def fig_length_hist():
    gen = load_jsonl(os.path.join(OUT, "qa_v2_gen_merged.jsonl"))
    if not gen:
        print("skip fig_length_hist: no data")
        return

    factual = [r["n_tokens"] for r in gen if r.get("type") == "factual" and r.get("n_tokens")]
    explain = [r["n_tokens"] for r in gen if r.get("type") == "explain" and r.get("n_tokens")]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bins = range(0, 800, 25)
    ax.hist(factual, bins=bins, alpha=0.7, label=f"Factual (n={len(factual)})", color="#8fa8c4")
    ax.hist(explain, bins=bins, alpha=0.7, label=f"Explain (n={len(explain)})", color="#c46b4f")
    ax.set_xlabel("Response length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Response length by question type (v2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_v2_length_hist.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_v2_length_hist.png"))
    plt.close(fig)
    print(f"fig_v2_length_hist: factual n={len(factual)}, explain n={len(explain)}")


# =====================================================================
# Figure 3: entropy precision/recall on v2 (recomputed directly, not
# just parsed from the report text, so the figure is self-verifying)
# =====================================================================
def compute_entropy_curve(labels):
    pairs = [(r["entropy_topk"], r["safe_stop"]) for r in labels
            if r.get("entropy_topk") is not None]
    if not pairs:
        return None, None
    thresholds = sorted(set(round(h, 3) for h, _ in pairs))
    if len(thresholds) > 60:
        step = len(thresholds) // 60
        thresholds = thresholds[::step]
    precisions, recalls = [], []
    for T in thresholds:
        tp = sum(1 for h, ok in pairs if h < T and ok)
        fp = sum(1 for h, ok in pairs if h < T and not ok)
        fn = sum(1 for h, ok in pairs if h >= T and ok)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
    return recalls, precisions


def fig_entropy_roc():
    labels = load_jsonl(os.path.join(OUT, "qa_v2_labels.jsonl"))
    if not labels:
        print("skip fig_entropy_roc: no labels")
        return
    recalls, precisions = compute_entropy_curve(labels)
    if recalls is None:
        print("skip fig_entropy_roc: no entropy data")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(recalls, precisions, color="#c46b4f", linewidth=1.5, marker=".")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Entropy-threshold stopping on v2\n(n={len(labels)} boundaries)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_v2_entropy_roc.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_v2_entropy_roc.png"))
    plt.close(fig)
    print(f"fig_v2_entropy_roc: {len(labels)} boundaries, "
         f"precision range [{min(precisions):.3f}, {max(precisions):.3f}]")
    return recalls, precisions


# =====================================================================
# Figure 4: v1 vs v2 entropy comparison -- the key methodological finding
# =====================================================================
def fig_v1_vs_v2():
    labels = load_jsonl(os.path.join(OUT, "qa_v2_labels.jsonl"))
    if not labels:
        print("skip fig_v1_vs_v2: no v2 labels")
        return
    v2_recall, v2_precision = compute_entropy_curve(labels)
    if v2_recall is None:
        print("skip fig_v1_vs_v2: no entropy data")
        return

    # v1 numbers, hardcoded from the earlier (304-question) run's report
    v1_recall =    [0.308, 0.373, 0.451, 0.535, 0.655, 0.794, 0.961, 1.000, 1.000]
    v1_precision = [0.957, 0.964, 0.961, 0.963, 0.963, 0.963, 0.965, 0.966, 0.966]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(v1_recall, v1_precision, color="#8fa8c4", linewidth=1.5,
           marker="o", markersize=4, label="v1 (304 questions, mostly factual)")
    ax.plot(v2_recall, v2_precision, color="#c46b4f", linewidth=1.5,
           marker=".", label="v2 (1,074 questions, 60% explain)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Entropy baseline: small vs. large dataset")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_v1_vs_v2_entropy.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_v1_vs_v2_entropy.png"))
    plt.close(fig)
    print(f"fig_v1_vs_v2_entropy: v1 precision range "
         f"[{min(v1_precision):.3f},{max(v1_precision):.3f}], "
         f"v2 precision range [{min(v2_precision):.3f},{max(v2_precision):.3f}]")


# =====================================================================
def summary():
    lines = ["=== v2 summary numbers for the paper's prose ===\n"]
    report_path = os.path.join(OUT, "qa_v2_report.txt")
    if os.path.exists(report_path):
        lines.append(open(report_path).read())
    txt = "\n".join(lines)
    with open(os.path.join(FIGDIR, "fig_v2_summary.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    fig_waste_by_type()
    fig_length_hist()
    fig_entropy_roc()
    fig_v1_vs_v2()
    summary()
    print(f"\nAll figures written to ./{FIGDIR}/")
