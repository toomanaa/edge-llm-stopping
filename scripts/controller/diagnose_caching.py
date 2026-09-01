#!/usr/bin/env python3
"""Diagnostic: isolates whether /completion and /embeddings calls both
benefit from cache_prompt as the accumulated text grows, or whether one
of them is silently doing a full re-prefill every time regardless of
the cache_prompt setting.

Sends a sequence of growing prefixes of the SAME base text to both
endpoints and times each call. If caching is working for an endpoint,
its per-call latency should stay roughly flat (or grow very slowly)
as the prefix grows, since only the new tail needs processing. If
caching is NOT effectively used, latency should grow roughly linearly
with prefix length, since the whole thing is reprocessed every time.

Usage:
    python3 diagnose_caching.py --url http://127.0.0.1:8080
"""
import argparse, json, time, urllib.request

BASE_TEXT = (
    "The history of computing spans many decades of innovation. "
    "Early computers filled entire rooms and required teams of engineers. "
    "Transistors replaced vacuum tubes and made devices smaller and faster. "
    "Integrated circuits then allowed thousands of transistors on one chip. "
    "Personal computers brought computing power into ordinary homes. "
    "The internet connected these computers into a global network. "
    "Smartphones later combined computing and communication into one device. "
    "Cloud computing moved storage and processing to remote data centers. "
    "Artificial intelligence models now run on specialized hardware accelerators. "
    "Edge devices increasingly run smaller models locally for privacy and speed. "
)

def call_completion(prompt, url, cache_prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                       "stream": False, "cache_prompt": cache_prompt}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return time.time() - t0

def call_embedding(text, url, cache_prompt):
    body = json.dumps({"content": text, "pooling": "none",
                       "cache_prompt": cache_prompt}).encode()
    req = urllib.request.Request(url + "/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    sentences = BASE_TEXT.split(". ")
    print(f"{'step':>4} {'chars':>7} {'completion_s':>13} {'embedding_s':>13}")

    for i in range(1, min(args.steps, len(sentences)) + 1):
        prefix = ". ".join(sentences[:i]) + "."
        t_completion = call_completion(prefix, args.url, cache_prompt=True)
        t_embedding = call_embedding(prefix, args.url, cache_prompt=True)
        print(f"{i:>4} {len(prefix):>7} {t_completion:>13.3f} {t_embedding:>13.3f}")

    print("\nInterpretation:")
    print("- If completion_s stays roughly flat across steps: completion caching works.")
    print("- If embedding_s grows roughly linearly with chars: embeddings are")
    print("  NOT effectively cached, and are doing a full re-prefill every call --")
    print("  this would confirm embeddings calls (one per boundary check) are the")
    print("  dominant remaining overhead in the live controller.")

if __name__ == "__main__":
    main()
