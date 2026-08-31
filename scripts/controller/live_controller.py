#!/usr/bin/env python3
"""Live budget-aware stopping controller. Unlike every previous script
in this project, this one makes the stop-or-continue decision WHILE
generation is happening, using the trained probe -- not by generating a
full response and grading it afterward.

At each sentence boundary:
  1. Ask llama.cpp's /embeddings endpoint for the hidden state of the
     text generated so far (same endpoint used in the deployment-gap
     check).
  2. Run the probe on that hidden state.
  3. If the probe's confidence that stopping here is safe exceeds a
     threshold, stop generation (with a short grammatical landing).
     Otherwise, ask the server for the next chunk and repeat.

This is a genuinely different code path from run_qa_v2_generation.py:
that script asks for one full response in a single request. This one
asks for short chunks in a loop, checking the probe after each one.

Usage:
    python3 live_controller.py --data data/qa_v2.jsonl \\
        --out out/live_controller_results.jsonl \\
        --probe probe_weights.npz --threshold 0.7 --limit 20
"""
import argparse, json, re, time, pathlib, urllib.request
from correctness import is_correct
from probe_inference import Probe

BOUNDARY = re.compile(r"(?<!\d\.)(?<=[.!?])\s+")
# Excludes both bare newlines (see above) AND numbered-list markers like
# "1." "2." "3." -- a lone digit followed by a period looks identical to
# a real sentence-ending period to a plain [.!?] regex, but it is
# markdown list structure, not a completed thought. A second live
# debugging pass found the probe stopping confidently and wrongly right
# after "...as follows:\n\n1." -- essentially no content, just a list
# marker -- the same failure mode as the header/newline bug, just from
# a different piece of formatting. The negative lookbehind (?<![0-9])
# rejects any period immediately preceded by a digit, so "1." "12." etc.
# no longer register as checkable boundaries, while genuine sentence
# endings (which are never preceded by a bare digit) are unaffected.
SYSTEM = ("You are a helpful assistant. Answer the user's question clearly "
          "and correctly.")


def chat_prompt(question):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + question + "<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n")


def call_completion(prompt, url, n_predict):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.0, "stream": False,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def call_embedding(text, url):
    body = json.dumps({"content": text, "pooling": "none"}).encode()
    req = urllib.request.Request(url + "/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return d[0]["embedding"][-1]  # last-token vector, matching the probe


def latest_boundary_offset(text):
    """Find the character offset of the LAST sentence boundary anywhere
    in the text (not just whether the text happens to end at one right
    now). This is the fix for a real bug: with fixed-size generation
    chunks, the raw chunk end almost never coincides with a sentence's
    final punctuation, so checking only "does the text end here" meant
    the controller went hundreds of words between checks. Searching the
    whole accumulated text for the most recent real boundary, the same
    way the offline labeler always did, catches every sentence as soon
    as it completes, regardless of where the chunk happened to cut off."""
    offsets = [m.start() for m in BOUNDARY.finditer(text)]
    return offsets[-1] if offsets else None


def generate_with_live_stopping(question, answers, match_mode, url, probe,
                                threshold, chunk_tokens=20,
                                max_total_tokens=800, landing_tokens=15,
                                debug=False):
    """The actual controller loop. Returns a dict with the final text,
    whether it stopped early (via the probe) or ran to natural
    completion, how many boundary checks were made, timing, AND whether
    the final text is actually correct against the ground truth --
    savings numbers alone do not tell us if the probe is stopping at
    genuinely safe points or just early ones.

    Checks EVERY new sentence boundary as it appears in the accumulated
    text, not just whether the raw chunk happened to end at one -- see
    latest_boundary_offset() for why this matters. If the probe says
    stop, any text generated past that boundary (the "overshoot" from
    finishing the current chunk) is discarded, so the final output ends
    cleanly at the sentence the decision was actually made at."""
    prompt = chat_prompt(question)
    accumulated = ""
    last_checked_offset = 0
    boundary_checks = 0
    stopped_early = False
    approx_tokens_used = 0
    t_start = time.time()

    while True:
        if approx_tokens_used >= max_total_tokens:
            break

        resp = call_completion(prompt + accumulated, url,
                               n_predict=chunk_tokens)
        chunk = resp.get("content", "")
        if not chunk:
            break  # model produced nothing more; natural end
        accumulated += chunk
        approx_tokens_used += chunk_tokens

        # Check every boundary that appeared since the last check, not
        # just the very end of the current chunk.
        offs = [m.start() for m in BOUNDARY.finditer(accumulated)]
        new_offs = [o for o in offs if o > last_checked_offset]
        for off in new_offs:
            boundary_checks += 1
            prefix = accumulated[:off]
            try:
                hidden_state = call_embedding(prompt + prefix, url)
                p_safe = probe.predict_proba(hidden_state)
            except Exception as e:
                # Fail open: if the embedding call errors (observed cause
                # in practice: HTTP 500 on long accumulated text, likely
                # from prompt + prefix approaching the server's context
                # limit, -c 2048 in our setup), don't stop and don't
                # crash -- just keep generating. This matches the
                # paper's fail-open design: an unreliable signal should
                # never cause premature (and here, mid-response) failure.
                p_safe = 0.0
                print(f"    [probe call failed: {e}, continuing generation]")

            if p_safe >= threshold:
                accumulated = prefix  # discard any overshoot past this point
                stopped_early = True
                if debug:
                    print(f"    [boundary {boundary_checks}: p_safe={p_safe:.4f} "
                         f">= {threshold} -> STOP here: {prefix[-60:]!r}]")
                break
            elif debug:
                print(f"    [boundary {boundary_checks}: p_safe={p_safe:.4f} "
                     f"< {threshold} -> continue]")
            last_checked_offset = off

        if stopped_early:
            break

    t_end = time.time()
    correct = is_correct(accumulated, answers, match_mode)
    return {
        "question": question, "text": accumulated,
        "stopped_early_by_probe": stopped_early,
        "boundary_checks": boundary_checks,
        "n_words_approx": len(accumulated.split()),
        "t_start": t_start, "t_end": t_end,
        "wall_seconds": t_end - t_start,
        "correct": correct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default="probe_weights.npz")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="stop when P(safe to stop) >= this value")
    ap.add_argument("--chunk-tokens", type=int, default=20,
                    help="tokens requested per generation step")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="print the probe's raw confidence at every "
                         "boundary check, to diagnose threshold behavior")
    args = ap.parse_args()

    probe = Probe(args.probe)
    print(f"loaded probe from {args.probe}")

    prompts = [json.loads(l) for l in open(args.data)]
    if args.limit:
        prompts = prompts[:args.limit]

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        done = {json.loads(l)["id"] for l in open(outp)
                if "id" in json.loads(l)}

    n_correct = 0
    n_total = 0
    with open(outp, "a") as f:
        for i, ex in enumerate(prompts):
            qid = ex.get("id", f"q{i}")
            if qid in done:
                continue
            result = generate_with_live_stopping(
                ex["question"], ex["answers"], ex["match_mode"],
                args.url, probe, args.threshold,
                chunk_tokens=args.chunk_tokens, debug=args.debug)
            result["id"] = qid
            f.write(json.dumps(result) + "\n"); f.flush()
            n_total += 1
            n_correct += int(result["correct"])
            tag = "STOPPED-EARLY" if result["stopped_early_by_probe"] else "natural-end"
            correct_tag = "CORRECT" if result["correct"] else "WRONG"
            print(f"[{i+1}/{len(prompts)}] {qid} [{tag}] [{correct_tag}] "
                 f"{result['n_words_approx']} words, "
                 f"{result['boundary_checks']} boundary checks, "
                 f"{result['wall_seconds']:.1f}s  "
                 f"(running accuracy: {n_correct}/{n_total} = "
                 f"{n_correct/n_total*100:.1f}%)")

    print("done")


if __name__ == "__main__":
    main()
