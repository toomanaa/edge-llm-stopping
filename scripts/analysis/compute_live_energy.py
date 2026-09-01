#!/usr/bin/env python3
"""Compute real measured energy (joules) for the live controller's full
1,074-question run, using its actual power trace and per-response
timestamps -- not the word-count approximation used for the quick
sanity check. Compares against the offline (no-stopping) full-generation
energy for the SAME questions, using the already-merged v2 offline data,
so the two numbers are directly comparable.

Usage:
    python3 compute_live_energy.py \\
        --live-gen out/live_full_t99.jsonl \\
        --live-power out/live_full_t99_power.csv \\
        --offline-gen out/qa_v2_gen_merged.jsonl \\
        --offline-power out/qa_v2_power_merged.csv
"""
import argparse, json, bisect


def load_power(csv_path):
    ts, ws = [], []
    for line in open(csv_path):
        if line.startswith("t,"):
            continue
        t, w = line.strip().split(",")
        ts.append(float(t)); ws.append(float(w))
    return ts, ws


def joules(ts, ws, t0, t1):
    """Integrate watts over [t0, t1] using the trapezoidal rule between
    logged power samples -- identical logic to every other energy
    computation in this project (label_qa_v2_boundaries.py,
    merge_v2_results.py), reused here rather than reimplemented, so
    numbers stay comparable across the whole pipeline."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-gen", required=True)
    ap.add_argument("--live-power", required=True)
    ap.add_argument("--offline-gen", required=True)
    ap.add_argument("--offline-power", required=True)
    args = ap.parse_args()

    live_results = [json.loads(l) for l in open(args.live_gen)]
    live_ts, live_ws = load_power(args.live_power)
    print(f"loaded {len(live_results)} live results, "
         f"{len(live_ts)} power samples "
         f"({(live_ts[-1]-live_ts[0])/3600:.2f} hours logged)")

    offline_by_id = {json.loads(l)["id"]: json.loads(l)
                     for l in open(args.offline_gen)}
    offline_ts, offline_ws = load_power(args.offline_power)
    print(f"loaded {len(offline_by_id)} offline results, "
         f"{len(offline_ts)} power samples")

    # --- live controller: real measured joules per response, summed ---
    live_total_j = 0.0
    live_covered = 0
    live_uncovered = 0
    for r in live_results:
        t0, t1 = r["t_start"], r["t_end"]
        if in_power_range(t0, t1, live_ts):
            live_total_j += joules(live_ts, live_ws, t0, t1)
            live_covered += 1
        else:
            live_uncovered += 1
    print(f"\nlive controller: {live_covered}/{len(live_results)} responses "
         f"with power coverage ({live_uncovered} outside the logged range)")
    print(f"live controller total measured energy: {live_total_j:.2f} J")

    # --- offline (no stopping): same computation, same question ids ---
    offline_total_j = 0.0
    offline_covered = 0
    offline_uncovered = 0
    matched_ids = 0
    for r in live_results:
        off = offline_by_id.get(r["id"])
        if off is None:
            continue
        matched_ids += 1
        t0, t1 = off["t_start"], off["t_end"]
        if in_power_range(t0, t1, offline_ts):
            offline_total_j += joules(offline_ts, offline_ws, t0, t1)
            offline_covered += 1
        else:
            offline_uncovered += 1
    print(f"\noffline (no stopping): {matched_ids} matching question ids found")
    print(f"offline: {offline_covered}/{matched_ids} responses with power "
         f"coverage ({offline_uncovered} outside the logged range)")
    print(f"offline total measured energy (same questions): {offline_total_j:.2f} J")

    # --- comparison ---
    if offline_total_j > 0:
        savings_pct = (1 - live_total_j / offline_total_j) * 100
        print(f"\n=== REAL MEASURED SAVINGS ===")
        print(f"live controller:  {live_total_j:.2f} J")
        print(f"offline baseline: {offline_total_j:.2f} J")
        print(f"measured energy savings: {savings_pct:.1f}%")
        print(f"\n(for reference, the earlier word-count approximation gave ~69.1%)")

    # accuracy, for the same report
    n_correct = sum(1 for r in live_results if r["correct"])
    print(f"\nlive controller accuracy: {n_correct}/{len(live_results)} "
         f"= {n_correct/len(live_results)*100:.1f}%")


if __name__ == "__main__":
    main()
