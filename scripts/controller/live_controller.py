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
import argparse, json, re, sys, time, pathlib, urllib.request
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
                       "cache_prompt": True}).encode()
    # cache_prompt=True is the fix for a real, measured problem: with it
    # False, every completion call re-processes the ENTIRE accumulated
    # prompt from scratch rather than reusing the already-computed
    # context and only processing what's new. A full 1,074-question run
    # measured an effective throughput of 2.67 words/sec against a real
    # generation speed of ~11 words/sec -- a 4.1x overhead factor, with
    # 76% of total wall-clock time spent on repeated re-processing
    # rather than actual generation. That run's energy numbers are not
    # usable as a result: it measured this overhead, not the underlying
    # stopping decision's real cost.
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def call_embedding(text, url):
    body = json.dumps({"content": text, "pooling": "none",
                       "cache_prompt": True}).encode()
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
                                check_every_n=2, debug=False):
    """The actual controller loop. Returns a dict with the final text,
    whether it stopped early (via the probe) or ran to natural
    completion, how many boundary checks were made, timing, AND whether
    the final text is actually correct against the ground truth --
    savings numbers alone do not tell us if the probe is stopping at
    genuinely safe points or just early ones.

    Checks every Nth new sentence boundary (check_every_n), not every
    single one. This is a direct response to a measured problem: a
    diagnostic comparing /completion and /embeddings call latency on
    identical growing text found /completion caching works correctly
    (flat latency, ~0.18s regardless of length) but /embeddings latency
    grows roughly linearly with text length (0.26s to 1.58s over 8
    steps) -- the cache_prompt flag has no effect on that endpoint in
    this server version. Since a boundary check requires one embeddings
    call per check, and later checks in a long response reprocess an
    increasingly long prefix, checking every boundary made the
    embeddings overhead the dominant cost of the whole controller,
    enough to erase the energy savings from stopping early at all (a
    full-scale run measured negative net savings before this fix).
    Checking every Nth boundary instead directly cuts the number of
    these expensive calls.

    If the probe says stop, any text generated past the triggering
    boundary (the "overshoot" from finishing the current chunk, plus
    any skipped intervening boundaries) is discarded, so the final
    output still ends cleanly at a real sentence."""
    prompt = chat_prompt(question)
    accumulated = ""
    last_checked_offset = 0
    boundary_checks = 0
    boundaries_since_check = 0
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

        # Check every Nth boundary that appeared since the last check,
        # not every single one -- see the docstring above for why.
        offs = [m.start() for m in BOUNDARY.finditer(accumulated)]
        new_offs = [o for o in offs if o > last_checked_offset]
        for off in new_offs:
            boundaries_since_check += 1
            if boundaries_since_check < check_every_n:
                last_checked_offset = off  # advance past this boundary,
                continue                    # but skip the expensive check

            boundaries_since_check = 0
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


def available_memory_mb():
    """Read available memory from /proc/meminfo -- same approach used in
    run_qa_v2_generation_safe.py, which caught the memory buildup that
    corrupted the Pi's Python installation during the unappwatched
    overnight v2 generation run. The live controller makes MORE HTTP
    requests per question than that script did (multiple /embeddings
    calls per response, not just one /completion call), so the same
    risk applies here, arguably more so."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


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
    ap.add_argument("--check-every-n", type=int, default=2,
                    help="only call the probe every Nth sentence boundary, "
                         "not every one -- reduces expensive /embeddings "
                         "calls, which were found to dominate overhead")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="print the probe's raw confidence at every "
                         "boundary check, to diagnose threshold behavior")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="pause for a server restart after this many "
                         "questions in a single invocation")
    ap.add_argument("--min-free-mb", type=float, default=1500,
                    help="if available memory drops below this, stop and "
                         "recommend a server restart before continuing")
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
    processed_this_batch = 0
    with open(outp, "a") as f:
        for i, ex in enumerate(prompts):
            qid = ex.get("id", f"q{i}")
            if qid in done:
                continue

            mem = available_memory_mb()
            if mem is not None and mem < args.min_free_mb:
                print(f"\n*** STOPPING: available memory is {mem:.0f} MB, "
                     f"below the {args.min_free_mb:.0f} MB safety threshold. ***")
                print("*** Restart llama-server now (Ctrl-C it, then start "
                     "it again), then re-run this exact command to resume. ***")
                sys.exit(2)  # signals "paused, needs a restart" to a wrapper script

            result = generate_with_live_stopping(
                ex["question"], ex["answers"], ex["match_mode"],
                args.url, probe, args.threshold,
                chunk_tokens=args.chunk_tokens,
                check_every_n=args.check_every_n, debug=args.debug)
            result["id"] = qid
            f.write(json.dumps(result) + "\n"); f.flush()
            n_total += 1
            n_correct += int(result["correct"])
            processed_this_batch += 1
            tag = "STOPPED-EARLY" if result["stopped_early_by_probe"] else "natural-end"
            correct_tag = "CORRECT" if result["correct"] else "WRONG"
            mem_str = f", mem_avail={mem:.0f}MB" if mem is not None else ""
            print(f"[{i+1}/{len(prompts)}] {qid} [{tag}] [{correct_tag}] "
                 f"{result['n_words_approx']} words, "
                 f"{result['boundary_checks']} boundary checks, "
                 f"{result['wall_seconds']:.1f}s{mem_str}  "
                 f"(running accuracy: {n_correct}/{n_total} = "
                 f"{n_correct/n_total*100:.1f}%)")

            if processed_this_batch >= args.batch_size:
                print(f"\n*** Batch of {args.batch_size} complete. "
                     f"Restart llama-server now (for memory hygiene), "
                     f"then re-run this exact command to continue. ***")
                sys.exit(2)  # same "paused, needs a restart" signal

    print("done")
    sys.exit(0)  # signals "genuinely finished" to a wrapper script


if __name__ == "__main__":
    main()
