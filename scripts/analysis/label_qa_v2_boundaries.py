#!/usr/bin/env python3
"""Hindsight labeler for the v2 QA set (factual + explain/why questions).

Extends v1's label_qa_boundaries.py with one change: correctness checking
respects match_mode.
  "any" (factual questions): correct if text contains at least one of
        `answers` -- same as v1.
  "all" (explain/why questions): correct if text contains every one of
        `answers` (the required concept keywords) -- new in v2, since an
        explain answer needs to cover multiple ideas, not match one
        string.

Usage:
  python3 label_qa_v2_boundaries.py --gen out/qa_v2_gen.jsonl --power out/qa_v2_power.csv --outdir out
"""
import argparse, json, re, bisect, pathlib
from collections import defaultdict
from correctness import is_correct

BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

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

def in_power_range(t0, t1, ts):
    """True only if [t0, t1] falls inside the power log's actual recorded
    span. Without this check, bisect-based joules() silently returns a
    near-zero or wrong value for timestamps outside the logged range,
    rather than correctly signaling "no data" -- caught by testing this
    script against a synthetic partial-coverage case before running it
    on real (partially-covered) data."""
    if not ts:
        return False
    return ts[0] <= t0 and t1 <= ts[-1]

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
    by_type = defaultdict(lambda: defaultdict(int))

    for line in open(args.gen):
        r = json.loads(line)
        text, answers = r["text"], r["answers"]
        match_mode = r.get("match_mode", "any")
        qtype = r.get("type", "unknown")
        offs = boundaries(text)
        n_tok = max(len(r.get("tokens", [])), 1)
        full_ok = is_correct(text, answers, match_mode)
        stats["total"] += 1
        stats["full_correct"] += int(full_ok)
        by_type[qtype]["total"] += 1
        by_type[qtype]["correct"] += int(full_ok)
        if not full_ok:
            continue
        earliest = None
        for b_idx, off in enumerate(offs):
            trunc = text[:off]
            ok = is_correct(trunc, answers, match_mode)
            tok_idx = round(off / max(len(text), 1) * n_tok)
            ent = (r["tokens"][min(tok_idx, n_tok - 1)]["entropy_topk"]
                   if r.get("tokens") else None)
            row = {"id": r["id"], "boundary": b_idx, "tok_idx": tok_idx,
                   "safe_stop": ok, "delta_q": 0.0 if ok else 1.0,
                   "entropy_topk": ent, "type": qtype}
            if pw and in_power_range(r["t_start"], r["t_end"], pw[0]):
                frac = off / max(len(text), 1)
                t_b = r["t_start"] + (r["t_end"] - r["t_start"]) * frac
                row["joules_so_far"] = joules(*pw, r["t_start"], t_b)
                row["joules_total"] = joules(*pw, r["t_start"], r["t_end"])
                row["has_power"] = True
            elif pw:
                row["has_power"] = False
            rows.append(row)
            if ok and earliest is None:
                earliest = (b_idx, tok_idx)
            if ent is not None:
                entropy_pairs.append((ent, ok))
        if earliest:
            stats["has_safe_stop"] += 1
            tokens_after_safe.append(n_tok - earliest[1])

    with open(outdir / "qa_v2_labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = [f"responses: {stats['total']}, correct: {stats['full_correct']}"]
    for qtype, d in by_type.items():
        report.append(f"  {qtype}: {d['correct']}/{d['total']} correct "
                      f"({d['correct']/max(d['total'],1)*100:.1f}%)")
    report.append(f"with a safe early stop boundary: {stats['has_safe_stop']}")
    if tokens_after_safe:
        avg = sum(tokens_after_safe) / len(tokens_after_safe)
        mx = max(tokens_after_safe)
        report.append(f"avg tokens generated AFTER earliest safe stop: {avg:.1f} (max {mx})")
        report.append(f"total wasted tokens: {sum(tokens_after_safe)}")

    # hard-boundary (multi-boundary response) stats -- the ones that
    # actually matter for probe training/eval
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    hard_rows = [r for rid, rws in by_id.items() if len(rws) >= 2 for r in rws]
    n_hard_safe = sum(1 for r in hard_rows if r["safe_stop"])
    report.append(f"\nhard boundaries (multi-boundary responses): {len(hard_rows)}")
    report.append(f"  safe: {n_hard_safe}  unsafe: {len(hard_rows)-n_hard_safe}")

    if pw and rows:
        wasted_j = 0.0
        n_energy_responses = 0
        for rid, rws in by_id.items():
            safe = [w for w in rws if w["safe_stop"]]
            if not safe or "joules_total" not in rws[0]:
                continue
            earliest_row = min(safe, key=lambda w: w["boundary"])
            wasted_j += rws[0]["joules_total"] - earliest_row.get("joules_so_far", 0.0)
            n_energy_responses += 1
        report.append(f"\nestimated joules wasted after earliest safe stop: {wasted_j:.2f} J")
        report.append(f"  (computed over {n_energy_responses} responses with power "
                      f"coverage, out of {stats['full_correct']} correct responses total --")
        report.append(f"   power logger stopped partway through this run; joules figures "
                      f"cover only the {n_energy_responses}-response subset with real "
                      f"power data, while token/entropy/accuracy figures above cover all "
                      f"{stats['total']} responses)")

    if entropy_pairs:
        report.append("\nentropy-threshold baseline (stop if H<T), all boundaries:")
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
    (outdir / "qa_v2_report.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {len(rows)} boundary labels to {outdir/'qa_v2_labels.jsonl'}")

if __name__ == "__main__":
    main()
