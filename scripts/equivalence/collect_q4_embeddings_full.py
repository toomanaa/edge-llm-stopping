#!/usr/bin/env python3
"""Collect Q4 hidden states for the FULL labeled boundary set, matched
against the true safe/unsafe labels already computed by
label_qa_boundaries.py. This is the direct follow-up to the 20-sample
equivalence check: instead of just comparing vectors for similarity, we
collect enough real Q4 data to test the trained probe's actual
precision/recall on the deployment platform.

Run this ON THE PI, with llama-server already running with --embeddings
enabled, in the same folder as qa_gen_run3.jsonl and qa_labels.jsonl.

Usage:
    python3 collect_q4_embeddings_full.py \\
        --gen out/qa_gen_run3.jsonl --labels out/qa_labels.jsonl \\
        --out out/q4_embeddings_full.json

Note: this makes one HTTP request per boundary (up to a few hundred),
so it takes longer than the 20-sample check -- expect several minutes,
not seconds. It prints progress every 20 requests.
"""
import argparse, json, re, urllib.request, pathlib
from collections import defaultdict

BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

def boundaries(text):
    offs = [m.start() for m in BOUNDARY.finditer(text)]
    if not offs or offs[-1] < len(text):
        offs.append(len(text))
    return offs

SYSTEM = ("You are a helpful assistant. Answer the user's question clearly "
          "and correctly.")

def chat_prompt(question):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + question + "<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n")

def get_embedding(text, url):
    body = json.dumps({"content": text, "pooling": "none"}).encode()
    req = urllib.request.Request(url + "/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d[0]["embedding"][-1]  # last-token vector, matching the probe

def contains_answer(text, answers):
    t = text.lower()
    return any(a.lower() in t for a in answers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--labels", required=True,
                    help="qa_labels.jsonl -- used only to confirm which "
                         "(id, boundary) pairs were part of the E1 analysis; "
                         "labels are recomputed here directly for consistency")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--hard-only", action="store_true", default=True,
                    help="only collect boundaries from multi-boundary "
                         "responses (default: on, matching the probe's "
                         "evaluation set)")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.gen)]

    candidates = []
    for r in records:
        text, answers = r["text"], r["answers"]
        if not contains_answer(text, answers):
            continue
        offs = boundaries(text)
        for b_idx, off in enumerate(offs):
            trunc = text[:off]
            safe = contains_answer(trunc, answers)
            candidates.append({
                "id": r["id"], "question": r["question"],
                "prefix_text": trunc, "boundary": b_idx,
                "n_boundaries_in_response": len(offs), "safe_stop": safe,
            })

    if args.hard_only:
        by_id = defaultdict(list)
        for c in candidates:
            by_id[c["id"]].append(c)
        candidates = [c for c in candidates
                     if len(by_id[c["id"]]) >= 2]

    print(f"collecting Q4 embeddings for {len(candidates)} boundaries "
         f"({'hard-only' if args.hard_only else 'all'})")
    n_safe = sum(1 for c in candidates if c["safe_stop"])
    print(f"{n_safe} safe / {len(candidates)-n_safe} unsafe")

    results = []
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if outp.exists():
        for l in open(outp):
            pass  # will overwrite fresh below; kept simple, no resume logic
    for i, c in enumerate(candidates):
        full_text = chat_prompt(c["question"]) + c["prefix_text"]
        try:
            vec = get_embedding(full_text, args.url)
        except Exception as e:
            print(f"  ERROR on {c['id']} boundary {c['boundary']}: {e} -- skipping")
            continue
        results.append({
            "id": c["id"], "boundary": c["boundary"],
            "safe_stop": c["safe_stop"], "q4_embedding": vec,
        })
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(candidates)}] collected")

    with open(outp, "w") as f:
        json.dump(results, f)
    print(f"\nwrote {len(results)} Q4 embeddings (with labels) to {outp}")
    n_safe_out = sum(1 for r in results if r["safe_stop"])
    print(f"final set: {n_safe_out} safe / {len(results)-n_safe_out} unsafe")

if __name__ == "__main__":
    main()
