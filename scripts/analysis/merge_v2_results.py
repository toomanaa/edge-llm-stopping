#!/usr/bin/env python3
"""Merge the original qa_v2_gen.jsonl (1,074 responses) with the refill
qa_v2_gen_refill.jsonl (599 responses re-collected specifically to get
power coverage), preferring the refill version for any id that appears
in both -- the refill version is guaranteed to have power data within
one of the merged CSV files, the original may not.

Also merges the (possibly overlapping-in-time, but that's fine since we
look up by timestamp range per response) power CSVs into one file.

Usage:
    python3 merge_v2_results.py \\
        --gen-original out/qa_v2_gen.jsonl \\
        --gen-refill out/qa_v2_gen_refill.jsonl \\
        --power out/qa_v2_power.csv out/qa_v2_power_refill.csv out/qa_v2_power_refill2.csv \\
        --out-gen out/qa_v2_gen_merged.jsonl \\
        --out-power out/qa_v2_power_merged.csv
"""
import argparse, json, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-original", required=True)
    ap.add_argument("--gen-refill", required=True)
    ap.add_argument("--power", nargs="+", required=True)
    ap.add_argument("--out-gen", required=True)
    ap.add_argument("--out-power", required=True)
    args = ap.parse_args()

    # --- merge generation data, refill wins on id collision ---
    original = {json.loads(l)["id"]: json.loads(l) for l in open(args.gen_original)}
    refill = {json.loads(l)["id"]: json.loads(l) for l in open(args.gen_refill)}

    print(f"original: {len(original)} responses")
    print(f"refill:   {len(refill)} responses")

    merged = dict(original)
    overlap = 0
    for rid, rec in refill.items():
        if rid in merged:
            overlap += 1
        merged[rid] = rec  # refill wins

    print(f"overlap (refill replaced original): {overlap}")
    print(f"merged total: {len(merged)} responses")

    outp = pathlib.Path(args.out_gen)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for rid in sorted(merged.keys()):
            f.write(json.dumps(merged[rid]) + "\n")
    print(f"wrote merged generation data to {outp}")

    # --- merge power CSVs: concatenate, sort by timestamp, dedupe ---
    all_rows = []
    for csv_path in args.power:
        n = 0
        for line in open(csv_path):
            if line.startswith("t,"):
                continue
            t, w = line.strip().split(",")
            all_rows.append((float(t), float(w)))
            n += 1
        print(f"  {csv_path}: {n} rows")

    all_rows = sorted(set(all_rows))  # dedupe exact (t, w) pairs, sort by time
    print(f"merged power rows (deduped, sorted): {len(all_rows)}")

    outp2 = pathlib.Path(args.out_power)
    with open(outp2, "w") as f:
        f.write("t,watts\n")
        for t, w in all_rows:
            f.write(f"{t:.4f},{w:.4f}\n")
    print(f"wrote merged power data to {outp2}")

    # --- report final coverage estimate ---
    ts = [t for t, w in all_rows]
    covered = 0
    for rid, rec in merged.items():
        # crude check: is there SOME power sample within this response's
        # time window, not necessarily continuous coverage
        import bisect
        i = bisect.bisect_left(ts, rec["t_start"])
        j = bisect.bisect_right(ts, rec["t_end"])
        if j > i:
            covered += 1
    print(f"\nresponses with at least some power data in their time window: "
         f"{covered}/{len(merged)}")

if __name__ == "__main__":
    main()
