#!/usr/bin/env python3
"""Step A of the quantization equivalence check.

Picks ~20 real text snippets from qa_gen_run3.jsonl -- a mix of full
responses and truncated (mid-answer) prefixes, matching exactly what the
probe notebook feeds to the model -- and queries the Pi's llama.cpp
/embeddings endpoint (with pooling=none) for the last-token hidden state
of each one. This is the Q4 side of the comparison.

Run this ON THE PI, in the same folder as qa_gen_run3.jsonl, with the
llama-server already running with --embeddings enabled.

Usage:
    python3 collect_q4_embeddings.py --gen out/qa_gen_run3.jsonl --out out/q4_embeddings.json
"""
import argparse, json, re, random, urllib.request, pathlib

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.gen)]

    # Build the same (question, prefix_text) pairs the probe notebook uses,
    # only for responses the model got right end-to-end.
    def contains_answer(text, answers):
        t = text.lower()
        return any(a.lower() in t for a in answers)

    candidates = []
    for r in records:
        text, answers = r["text"], r["answers"]
        if not contains_answer(text, answers):
            continue
        offs = boundaries(text)
        for b_idx, off in enumerate(offs):
            candidates.append({
                "id": r["id"], "question": r["question"],
                "prefix_text": text[:off], "boundary": b_idx,
            })

    random.seed(7)
    random.shuffle(candidates)
    sample = candidates[: args.n]
    print(f"selected {len(sample)} (question, boundary) pairs to check")

    results = []
    for i, c in enumerate(sample):
        full_text = chat_prompt(c["question"]) + c["prefix_text"]
        vec = get_embedding(full_text, args.url)
        results.append({
            "id": c["id"], "boundary": c["boundary"],
            "question": c["question"], "prefix_text": c["prefix_text"],
            "q4_embedding": vec,
        })
        print(f"[{i+1}/{len(sample)}] {c['id']} boundary {c['boundary']} "
              f"({len(c['prefix_text'])} chars)")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"\nwrote {len(results)} Q4 embeddings to {args.out}")
    print("Next: bring this file to Colab and run the matching "
          "collect_fp_embeddings step there for comparison.")

if __name__ == "__main__":
    main()
