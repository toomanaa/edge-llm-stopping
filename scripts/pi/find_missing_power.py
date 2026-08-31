#!/usr/bin/env python3
"""Find which responses from qa_v2_gen.jsonl fall OUTSIDE the power log's
recorded time range, and write those questions out as a smaller dataset
to re-run through the generation script -- this time with robust,
restart-safe power logging, so we don't have to redo the whole 1,074-
question run.

Usage:
    python3 find_missing_power.py \\
        --gen out/qa_v2_gen.jsonl --power out/qa_v2_power.csv \\
        --data data/qa_v2.jsonl \\
        --out data/qa_v2_missing_power.jsonl
"""
import argparse, json, pathlib

def load_power_range(csv_path):
    ts = []
    for line in open(csv_path):
        if line.startswith("t,"):
            continue
        t, w = line.strip().split(",")
        ts.append(float(t))
    if not ts:
        return None, None
    return min(ts), max(ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--data", required=True,
                    help="original qa_v2.jsonl, to pull question/answers/"
                         "match_mode/type for the missing subset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_min, t_max = load_power_range(args.power)
    print(f"power log covers: {t_min:.1f} to {t_max:.1f} "
         f"({(t_max - t_min)/60:.1f} minutes)")

    gen_records = {json.loads(l)["id"]: json.loads(l) for l in open(args.gen)}
    orig_records = {json.loads(l)["id"]: json.loads(l) for l in open(args.data)}

    missing_ids = []
    for rid, r in gen_records.items():
        # covered only if BOTH t_start and t_end fall inside the logged range
        if not (t_min <= r["t_start"] and r["t_end"] <= t_max):
            missing_ids.append(rid)

    print(f"total responses: {len(gen_records)}")
    print(f"missing power coverage: {len(missing_ids)}")

    # Pull original question data (not the generated text) for these ids,
    # so the refill run re-asks the same questions fresh.
    missing_questions = [orig_records[rid] for rid in missing_ids
                         if rid in orig_records]
    print(f"matched back to original dataset: {len(missing_questions)}")

    n_factual = sum(1 for q in missing_questions if q["type"] == "factual")
    n_explain = sum(1 for q in missing_questions if q["type"] == "explain")
    print(f"  factual: {n_factual}  explain: {n_explain}")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for q in missing_questions:
            f.write(json.dumps(q) + "\n")
    print(f"\nwrote {len(missing_questions)} questions to {outp}")
    print("Next: run the restart-safe generation script on this file.")

if __name__ == "__main__":
    main()
