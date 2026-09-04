#!/usr/bin/env python3
"""Simulates two baselines retroactively from data already collected for
the v2 dataset -- no new live Pi runs needed, since every boundary's
token position, entropy, and (via the power trace) energy cost is
already on disk from the offline labeling pass.

Both baselines operate at the SAME granularity as every other method in
this paper: they decide at sentence boundaries, not mid-word, exactly
like the entropy baseline, the probe, and the live controller. This
keeps the comparison apples-to-apples rather than giving a token-level
policy a resolution advantage no other method has.

  fixed max_tokens(T): stop at the first sentence boundary whose token
      index is >= T (i.e., stop at the first natural sentence break
      once at least T tokens have been generated); if no boundary
      reaches T, the response runs to its natural end.

  entropy-threshold(H): stop at the first sentence boundary where the
      model's next-token entropy is below H; same fallback to natural
      end if none qualifies. This reproduces exactly the same policy
      already reported as the "entropy baseline" throughout this paper,
      just computing per-response outcomes instead of only aggregate
      precision/recall.

For the ~194 of 1,074 questions that were wrong even at full length
(and therefore never appear in the boundary labels, since the offline
labeler only labels correct responses), no policy has finer-grained
data to work with; all policies are charged that response's full
energy and a wrong verdict identically, so this does not bias the
comparison toward or against any one method.

Usage:
    python3 simulate_baselines.py \\
        --gen out/qa_v2_gen_merged.jsonl \\
        --labels out/qa_v2_labels.jsonl \\
        --power out/qa_v2_power_merged.csv \\
        --out out/baseline_sweep_results.json
"""
import argparse, json, bisect
from collections import defaultdict


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


def in_power_range(t0, t1, ts):
    if not ts:
        return False
    return ts[0] <= t0 and t1 <= ts[-1]


def simulate_policy(gen_records, boundaries_by_id, power, select_boundary_fn):
    """select_boundary_fn(sorted_boundaries) -> chosen boundary dict, or
    None to mean 'run to natural end' (either because no boundary
    qualified, or because this response has no boundary data at all)."""
    ts, ws = power
    n_correct = 0
    n_total = 0
    total_joules = 0.0
    n_energy_covered = 0

    for r in gen_records:
        n_total += 1
        bounds = boundaries_by_id.get(r["id"])

        if not bounds:
            # wrong at full length (or no boundary data available) --
            # every policy is charged the same full-response outcome
            t0, t1 = r["t_start"], r["t_end"]
            if in_power_range(t0, t1, ts):
                total_joules += joules(ts, ws, t0, t1)
                n_energy_covered += 1
            continue

        chosen = select_boundary_fn(bounds)
        if chosen is None:
            # policy never triggered -- run to the last labeled boundary
            # (the full, correct response)
            chosen = bounds[-1]

        n_correct += int(chosen["safe_stop"])
        n_tokens = r.get("n_tokens") or 1
        frac = min(1.0, chosen["tok_idx"] / max(n_tokens, 1))
        t0 = r["t_start"]
        t1 = r["t_start"] + (r["t_end"] - r["t_start"]) * frac
        if in_power_range(t0, t1, ts):
            total_joules += joules(ts, ws, t0, t1)
            n_energy_covered += 1

    return {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "n_correct": n_correct, "n_total": n_total,
        "total_joules": total_joules,
        "n_energy_covered": n_energy_covered,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen_records = [json.loads(l) for l in open(args.gen)]
    print(f"loaded {len(gen_records)} responses")

    boundaries_by_id = defaultdict(list)
    for l in open(args.labels):
        row = json.loads(l)
        boundaries_by_id[row["id"]].append(row)
    for k in boundaries_by_id:
        boundaries_by_id[k].sort(key=lambda b: b["boundary"])
    print(f"loaded boundary labels for {len(boundaries_by_id)} responses "
         f"(the rest were wrong even at full length)")

    power = load_power(args.power)
    print(f"loaded {len(power[0])} power samples")

    results = {}

    # --- fixed max_tokens sweep ---
    print("\n=== fixed max_tokens sweep ===")
    for T in [30, 50, 75, 100, 150, 200, 300, 500]:
        def select(bounds, T=T):
            for b in bounds:
                if b["tok_idx"] >= T:
                    return b
            return None  # never reached T -> natural end
        r = simulate_policy(gen_records, boundaries_by_id, power, select)
        key = f"max_tokens_{T}"
        results[key] = r
        print(f"  T={T:>4}: accuracy={r['accuracy']*100:.1f}%  "
             f"total_J={r['total_joules']:.1f}  "
             f"(energy coverage {r['n_energy_covered']}/{r['n_total']})")

    # --- entropy threshold sweep (same thresholds used throughout the paper) ---
    print("\n=== entropy-threshold sweep ===")
    for H in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]:
        def select(bounds, H=H):
            for b in bounds:
                if b.get("entropy_topk") is not None and b["entropy_topk"] < H:
                    return b
            return None
        r = simulate_policy(gen_records, boundaries_by_id, power, select)
        key = f"entropy_{H}"
        results[key] = r
        print(f"  H={H:>4}: accuracy={r['accuracy']*100:.1f}%  "
             f"total_J={r['total_joules']:.1f}  "
             f"(energy coverage {r['n_energy_covered']}/{r['n_total']})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote full sweep results to {args.out}")


if __name__ == "__main__":
    main()
