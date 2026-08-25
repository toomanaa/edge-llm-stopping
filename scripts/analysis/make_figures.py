#!/usr/bin/env python3
"""Generate all figures for the FGCS paper's E0/E1 sections.

Reads the raw experiment outputs (gen.jsonl, qa_gen_run3.jsonl, labels, power
CSVs) and produces publication-ready figures in ./figures/.

Usage (run from the e1/ project directory, after copying this script in):
    pip install matplotlib --break-system-packages   # if not already present
    python3 make_figures.py

Or copy the whole out/ folder to a laptop and run it there -- either works,
this script has no dependency on the Pi specifically.

Produces:
    figures/fig_waste_comparison.pdf   E0 vs E1 wasted-tokens bar chart
    figures/fig_entropy_trace.pdf      per-token entropy trajectory, 2 examples
    figures/fig_entropy_roc.pdf        entropy-threshold precision/recall curve
    figures/fig_length_hist.pdf        response length distribution (E1)
    figures/fig_summary_table.txt      plain-text numbers for the paper's prose
"""
import json, math, os, sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Install with:")
    print("  pip install matplotlib --break-system-packages")
    sys.exit(1)

# ---- style: plain, print-friendly, no seaborn dependency ----
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT = "out"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def find_first(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# =====================================================================
# Figure 1: E0 vs E1 wasted-token comparison
# =====================================================================
def fig_waste_comparison():
    e0 = load_jsonl(os.path.join(OUT, "gen.jsonl"))
    e1_path = find_first(os.path.join(OUT, "qa_gen_run3.jsonl"),
                         os.path.join(OUT, "qa_gen_run2.jsonl"),
                         os.path.join(OUT, "qa_gen_run1.jsonl"))
    e1 = load_jsonl(e1_path) if e1_path else []
    if not e0 and not e1:
        print("skip fig_waste_comparison: no data found")
        return

    # E0: waste is documented as exactly 0 (see qa_e1_report / e1_report)
    e0_waste = 0.0
    # E1: recompute average wasted tokens from labels if available
    labels_path = find_first(os.path.join(OUT, "qa_labels.jsonl"))
    e1_waste = 0.0
    if labels_path:
        rows = load_jsonl(labels_path)
        by_id = defaultdict(list)
        for r in rows:
            by_id[r["id"]].append(r)
        wasted = []
        for rid, rws in by_id.items():
            safe = [w for w in rws if w["safe_stop"]]
            if not safe:
                continue
            earliest = min(safe, key=lambda w: w["boundary"])
            n_tok = max(w["tok_idx"] for w in rws)
            wasted.append(n_tok - earliest["tok_idx"])
        if wasted:
            e1_waste = sum(wasted) / len(wasted)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["E0: Commands\n(format-anchored)", "E1: QA\n(open-ended)"],
                  [e0_waste, e1_waste], color=["#8fa8c4", "#c46b4f"], width=0.55)
    ax.set_ylabel("Avg. tokens generated\nafter earliest safe stop")
    ax.set_title("Post-answer waste by task type")
    for b, v in zip(bars, [e0_waste, e1_waste]):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_waste_comparison.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_waste_comparison.eps"))
    fig.savefig(os.path.join(FIGDIR, "fig_waste_comparison.png"))
    plt.close(fig)
    print(f"fig_waste_comparison: E0={e0_waste:.2f}, E1={e1_waste:.2f}")


# =====================================================================
# Figure 2: per-token entropy trajectory for 2 example responses
# =====================================================================
def fig_entropy_trace():
    path = find_first(os.path.join(OUT, "qa_gen_run3.jsonl"),
                      os.path.join(OUT, "qa_gen_run2.jsonl"))
    if not path:
        print("skip fig_entropy_trace: no run with entropy data found")
        return
    rows = load_jsonl(path)
    # pick two examples: one short/terse, one with visible tail
    rows_with_tokens = [r for r in rows if r.get("tokens") and
                        any(t.get("entropy_topk", 0) > 0 for t in r["tokens"])]
    if not rows_with_tokens:
        print("skip fig_entropy_trace: entropy values are all zero "
             "(re-run with the logprob fix before plotting this figure)")
        return
    rows_with_tokens.sort(key=lambda r: r["n_tokens"], reverse=True)
    examples = [rows_with_tokens[len(rows_with_tokens)//2],  # a mid-length one
               rows_with_tokens[0]]                          # the longest

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, r in zip(axes, examples):
        ents = [t["entropy_topk"] for t in r["tokens"]]
        ax.plot(range(len(ents)), ents, marker="o", markersize=3,
               color="#c46b4f", linewidth=1.2)
        ax.set_xlabel("Token position")
        ax.set_title(r["question"][:40] + ("..." if len(r["question"]) > 40 else ""),
                    fontsize=8)
    axes[0].set_ylabel("Next-token entropy (nats)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_trace.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_trace.eps"))
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_trace.png"))
    plt.close(fig)
    print("fig_entropy_trace: saved 2 example trajectories")


# =====================================================================
# Figure 3: entropy-threshold precision/recall curve
# =====================================================================
def fig_entropy_roc():
    labels_path = find_first(os.path.join(OUT, "qa_labels.jsonl"))
    if not labels_path:
        print("skip fig_entropy_roc: no labels file found")
        return
    rows = load_jsonl(labels_path)
    pairs = [(r["entropy_topk"], r["safe_stop"]) for r in rows
            if r.get("entropy_topk") is not None]
    if not pairs or all(h == 0 for h, _ in pairs):
        print("skip fig_entropy_roc: entropy values are all zero "
             "(re-run with the logprob fix before plotting this figure)")
        return

    thresholds = sorted(set(round(h, 3) for h, _ in pairs))
    if len(thresholds) > 40:  # subsample for a clean curve
        step = len(thresholds) // 40
        thresholds = thresholds[::step]

    precisions, recalls = [], []
    for T in thresholds:
        tp = sum(1 for h, ok in pairs if h < T and ok)
        fp = sum(1 for h, ok in pairs if h < T and not ok)
        fn = sum(1 for h, ok in pairs if h >= T and ok)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(recalls, precisions, color="#c46b4f", linewidth=1.5, marker=".")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Entropy-threshold stopping:\nprecision vs. recall")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_roc.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_roc.eps"))
    fig.savefig(os.path.join(FIGDIR, "fig_entropy_roc.png"))
    plt.close(fig)
    print(f"fig_entropy_roc: {len(thresholds)} threshold points plotted")


# =====================================================================
# Figure 4: response length distribution (E1)
# =====================================================================
def fig_length_hist():
    path = find_first(os.path.join(OUT, "qa_gen_run3.jsonl"),
                      os.path.join(OUT, "qa_gen_run2.jsonl"),
                      os.path.join(OUT, "qa_gen_run1.jsonl"))
    if not path:
        print("skip fig_length_hist: no QA run found")
        return
    rows = load_jsonl(path)
    lens = [r["n_tokens"] for r in rows if r.get("n_tokens")]
    if not lens:
        print("skip fig_length_hist: no token counts found")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(lens, bins=20, color="#8fa8c4", edgecolor="white")
    ax.set_xlabel("Response length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"E1 response length distribution (n={len(lens)})")
    ax.axvline(sum(lens) / len(lens), color="#c46b4f", linestyle="--",
              linewidth=1.2, label=f"mean = {sum(lens)/len(lens):.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_length_hist.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig_length_hist.eps"))
    fig.savefig(os.path.join(FIGDIR, "fig_length_hist.png"))
    plt.close(fig)
    print(f"fig_length_hist: n={len(lens)}, mean={sum(lens)/len(lens):.1f}")


# =====================================================================
# Summary numbers (for prose, not a figure)
# =====================================================================
def summary_table():
    lines = ["=== Summary numbers for the paper's prose ===\n"]
    e0 = load_jsonl(os.path.join(OUT, "gen.jsonl"))
    if e0:
        lines.append(f"E0: {len(e0)} commands generated")
    e1_report = find_first(os.path.join(OUT, "qa_e1_report.txt"))
    if e1_report:
        lines.append("\n--- E1 report (qa_e1_report.txt) ---")
        lines.append(open(e1_report).read())
    txt = "\n".join(lines)
    with open(os.path.join(FIGDIR, "fig_summary_table.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    fig_waste_comparison()
    fig_entropy_trace()
    fig_entropy_roc()
    fig_length_hist()
    summary_table()
    print(f"\nAll figures written to ./{FIGDIR}/")
