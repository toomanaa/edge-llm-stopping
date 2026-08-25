#!/usr/bin/env python3
"""E1 hindsight labeler: answer-containment scoring at each sentence
boundary, plus joules attached from a power CSV and an entropy-threshold
precision/recall sweep.

Usage:
  python3 label_qa_boundaries.py --gen out/qa_gen.jsonl --power out/qa_power.csv --outdir out
"""
import argparse, json, re, bisect, pathlib
from collections import defaultdict

BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

def contains_answer(text, answers):
    t = text.lower()
    return any(a.lower() in t for a in answers)

def boundaries(text):
    offs = [m.start() for m in BOUNDARY.finditer(text)]
    if not offs or offs[-1] < len(text):
        offs.append(len(text))
    return offs

def load_power(csv_path):
    ts, ws = [], []
    for line in open(csv_path):
        if line.startswith("t,"):
            continue
        t, w = line.strip().split(",")
        ts.append(float(t)); ws.append(float(w))
    return ts, ws

def joules(ts, ws, t0, t1):
    i = bisect.bisect_left(ts, t0); j = bisect.bisect_right(ts, t1)
    total = 0.0
    for k in range(max(i, 1), j):
        dt = ts[k] - ts[k - 1]
        total += 0.5 * (ws[k] + ws[k - 1]) * dt
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--power", default=None)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir); outdir.mkdir(exist_ok=True)
    pw = load_power(args.power) if args.power else None

    rows, stats = [], defaultdict(int)
    tokens_after_safe, entropy_pairs = [], []

    for line in open(args.gen):
        r = json.loads(line)
        text, answers = r["text"], r["answers"]
        offs = boundaries(text)
        n_tok = max(len(r.get("tokens", [])), 1)
        full_ok = contains_answer(text, answers)
        stats["total"] += 1
        stats["full_correct"] += int(full_ok)
        if not full_ok:
            continue
        earliest = None
        for b_idx, off in enumerate(offs):
            trunc = text[:off]
            ok = contains_answer(trunc, answers)
            tok_idx = round(off / max(len(text), 1) * n_tok)
            ent = (r["tokens"][min(tok_idx, n_tok - 1)]["entropy_topk"]
                   if r.get("tokens") else None)
            row = {"id": r["id"], "boundary": b_idx, "tok_idx": tok_idx,
                   "safe_stop": ok, "delta_q": 0.0 if ok else 1.0,
                   "entropy_topk": ent}
            if pw:
                frac = off / max(len(text), 1)
                t_b = r["t_start"] + (r["t_end"] - r["t_start"]) * frac
                row["joules_so_far"] = joules(*pw, r["t_start"], t_b)
                row["joules_total"] = joules(*pw, r["t_start"], r["t_end"])
            rows.append(row)
            if ok and earliest is None:
                earliest = (b_idx, tok_idx)
            if ent is not None:
                entropy_pairs.append((ent, ok))
        if earliest:
            stats["has_safe_stop"] += 1
            tokens_after_safe.append(n_tok - earliest[1])

    with open(outdir / "qa_labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = [f"responses: {stats['total']}, contain correct answer: {stats['full_correct']}",
              f"with a safe early stop boundary: {stats['has_safe_stop']}"]
    if tokens_after_safe:
        avg = sum(tokens_after_safe) / len(tokens_after_safe)
        mx = max(tokens_after_safe)
        report.append(f"avg tokens generated AFTER earliest safe stop: {avg:.1f} (max {mx})")
        report.append(f"total wasted tokens across {len(tokens_after_safe)} correct responses: "
                      f"{sum(tokens_after_safe)}")
    if pw and rows:
        by_id = defaultdict(list)
        for row in rows:
            by_id[row["id"]].append(row)
        wasted_j = 0.0
        for rid, rws in by_id.items():
            safe = [w for w in rws if w["safe_stop"]]
            if not safe or "joules_total" not in rws[0]:
                continue
            earliest_row = min(safe, key=lambda w: w["boundary"])
            wasted_j += rws[0]["joules_total"] - earliest_row.get("joules_so_far", 0.0)
        report.append(f"estimated joules wasted after earliest safe stop: {wasted_j:.2f} J")
    if entropy_pairs:
        report.append("\nentropy-threshold baseline (stop if H<T):")
        report.append(f"{'T':>6} {'precision':>10} {'recall':>8}")
        for T in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]:
            pred = [(h < T, ok) for h, ok in entropy_pairs]
            tp = sum(1 for p, ok in pred if p and ok)
            fp = sum(1 for p, ok in pred if p and not ok)
            fn = sum(1 for p, ok in pred if not p and ok)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            report.append(f"{T:>6} {prec:>10.3f} {rec:>8.3f}")
    txt = "\n".join(report)
    (outdir / "qa_e1_report.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {len(rows)} boundary labels to {outdir/'qa_labels.jsonl'}")

if __name__ == "__main__":
    main()
