#!/usr/bin/env python3
"""Restart-safe generation runner. Same generation logic as
run_qa_v2_generation.py, but with the fix for last night's failure mode
built in: llama-server's memory grew from normal to 89% of the Pi's RAM
over a 9-hour unattended run, which corrupted the Python installation
via a severe memory crunch and killed the power logger.

This script does NOT start/stop llama-server itself (server lifecycle
is simpler to manage by hand, in a separate terminal, restarted between
batches) -- instead it processes questions in small batches and, after
each batch, checks available memory and PAUSES with a clear instruction
if memory looks dangerous, rather than plowing ahead into another crash.

Usage:
    python3 run_qa_v2_generation_safe.py \\
        --data data/qa_v2_missing_power.jsonl --out out/qa_v2_gen_refill.jsonl \\
        --batch-size 50

Recommended operating pattern:
    1. Start llama-server fresh.
    2. Start power_logger.py fresh (new output file).
    3. Run this script.
    4. When it reports "batch complete, restart the server", Ctrl-C the
       server, restart it, and re-run this exact same command -- it
       resumes automatically from where it left off (same resumability
       as the original script).
"""
import argparse, json, time, math, pathlib, urllib.request, shutil

SYSTEM = ("You are a helpful assistant. Answer the user's question clearly "
          "and correctly.")

def call_server(prompt, url, n_predict=800):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.0, "n_probs": 5, "stream": False,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def entropy_from_probs(probs):
    ps = []
    for p in probs:
        if "prob" in p and p["prob"] > 0:
            ps.append(p["prob"])
        elif "logprob" in p:
            ps.append(math.exp(p["logprob"]))
    return -sum(p * math.log(p) for p in ps if p > 0) if ps else 0.0

def chat_prompt(question):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + question + "<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n")

def available_memory_mb():
    """Read available memory from /proc/meminfo -- no extra dependencies."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / 1024
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="pause for a server restart after this many "
                         "questions in a single invocation")
    ap.add_argument("--min-free-mb", type=float, default=1500,
                    help="if available memory drops below this, stop and "
                         "recommend a server restart before continuing")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.data)]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    outp = pathlib.Path(args.out)
    if outp.exists():
        done = {json.loads(l)["id"] for l in open(outp)}
        print(f"resuming: {len(done)} already done")

    processed_this_batch = 0
    with open(outp, "a") as f:
        for i, ex in enumerate(prompts):
            if ex["id"] in done:
                continue

            mem = available_memory_mb()
            if mem is not None and mem < args.min_free_mb:
                print(f"\n*** STOPPING: available memory is {mem:.0f} MB, "
                     f"below the {args.min_free_mb:.0f} MB safety threshold. ***")
                print("*** Restart llama-server now (Ctrl-C it, then start "
                     "it again), then re-run this exact command to resume. ***")
                return

            t_start = time.time()
            resp = call_server(chat_prompt(ex["question"]), args.url)
            t_end = time.time()
            toks = []
            for tp in resp.get("completion_probabilities", []):
                toks.append({"tok": tp.get("token", ""),
                             "entropy_topk": entropy_from_probs(
                                 tp.get("top_probs", tp.get("top_logprobs", [])))})
            rec = {"id": ex["id"], "question": ex["question"],
                   "answers": ex["answers"], "match_mode": ex["match_mode"],
                   "type": ex["type"], "text": resp.get("content", ""),
                   "tokens": toks, "t_start": t_start, "t_end": t_end,
                   "n_tokens": resp.get("tokens_predicted")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            processed_this_batch += 1

            if (i + 1) % 10 == 0:
                mem_str = f", mem_avail={mem:.0f}MB" if mem is not None else ""
                print(f"[{i+1}/{len(prompts)}] {ex['id']} ({ex['type']}) "
                     f"{rec['n_tokens']} tok in {t_end-t_start:.1f}s{mem_str}")

            if processed_this_batch >= args.batch_size:
                print(f"\n*** Batch of {args.batch_size} complete. "
                     f"Restart llama-server now (for memory hygiene), "
                     f"then re-run this exact command to continue. ***")
                return

    print("done")

if __name__ == "__main__":
    main()
