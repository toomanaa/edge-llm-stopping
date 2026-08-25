# Lab notebook

Chronological record of what was done, what broke, and how it was fixed.
Kept in plain language on purpose — this is a working log, not the paper.

## Setup

Hardware acquired: Raspberry Pi 5 (8 GB, CanaKit Essentials kit — includes
the official active cooler, a 45 W USB-C PD supply, and a 32 GB SD card),
an FNIRSI FNB48S USB-C power meter, and an Anker 20,000 mAh / 30 W power
bank.

Network problems dominated the first setup session: the model download and
the `git clone` of llama.cpp both crawled at single-digit KB/s and
repeatedly failed with TLS errors over Wi-Fi, traced to the connection
routing through a NAT64 gateway (`64:ff9b::...`). Switching the Pi to a
wired Ethernet connection resolved it immediately — the same 770 MB model
download that had an ETA of 20+ hours over Wi-Fi completed in 40 seconds
over Ethernet at 17 MB/s, and the `git clone` that had failed twice
completed in 6 seconds. Lesson: prefer Ethernet for any Pi that will do
sustained downloads, and don't assume a slow transfer is the remote
server's fault before checking the routing path.

llama.cpp was built from source (`cmake` + `cmake --build`, ~15 minutes on
the Pi's four cores) rather than installed as a package, to get the
`llama-server` binary with its HTTP completion API.

The Pi 5's power management IC (PMIC) was confirmed working via
`vcgencmd pmic_read_adc`, which gives free, no-extra-hardware power
sensing at roughly 20 Hz.

## E0: command parsing

**First attempt (no few-shot examples in the prompt).** The model matched
0 of 5 test commands. It copied parameter *names* from the tool schema
into the call itself (e.g. `dim(room, percent, 25)` instead of
`dim(bedroom, 25)`), and sometimes answered in plain English with no tool
call at all.

**Fix:** added three worked examples to the system prompt, showing bare,
correctly-formatted tool calls. Re-tested on 5 prompts: 4/5 exact matches
(the one miss was the model hallucinating an extra argument on a
temperature command). Judged good enough to run the full set.

**Full run (333 commands):** 198/333 (59.5%) exactly correct. Of the
correct responses, **all 198** stopped immediately after the tool call —
zero tokens of trailing text. The few-shot examples used to fix the format
had, as a side effect, also removed the verbosity a stopping mechanism was
meant to catch. Interpreted as a genuine, useful negative result rather
than a failure: for rigidly formatted tasks, prompt design alone can solve
the problem this project is trying to solve at the mechanism level.

## E1: open-ended QA

Motivated by E0's null result — needed a task where the model would
naturally over-answer, with no format anchor suppressing it.

Built a 304-question dataset of short factual questions (capitals,
science, history, culture) with accepted-answer lists, and a matching
generation script using an open system prompt ("answer clearly and
correctly," no worked examples).

**Smoke test (5 questions):** 5/5 correct, with 4 terse one-sentence
answers and one (`Who wrote The Old Man and the Sea?`) that added an
unprompted extra fact — exactly the phenomenon being tested for. Judged
good enough to run the full set.

**Full run (304 questions):** 291/304 (95.7%) correct. Response lengths
ranged from 8 to 76 tokens on the smoke test, later up to 155 on the full
run — real variance, unlike E0.

**First labeler pass** on this run showed an entropy-threshold table that
was flat and perfect (precision/recall = 1.000 across every threshold
tested). Investigated and found this was an artifact of too little data
at the boundary level (447 boundary points across 291 responses, but only
61 responses had more than one boundary at all — most answers were single
sentences with nothing to threshold). Restricting to the 61
multi-boundary responses gave 217 boundary-level test points, but the
entropy values themselves were found to be uniformly near zero regardless
of position.

## Debugging the entropy signal (the long part)

Traced the flat/near-zero entropy to a chain of two separate bugs, found
by working backward from a raw `curl` request to the llama.cpp server and
comparing it, field by field, against what the generation script was
parsing.

1. **Field name mismatch.** The generation script read `tp.get("content")`
   and `tp.get("probs")` from each token's completion-probability entry.
   The running llama.cpp server actually returns `"token"` and
   `"top_probs"` (or, depending on a request flag, `"logprob"` and
   `"top_logprobs"` — see point 2). The script's use of `.get(key,
   default)` meant this mismatch failed silently: every token entry came
   back with an empty string and zero entropy instead of raising an
   error. Fixed by reading the correct keys.

2. **`post_sampling_probs` and greedy decoding.** With
   `post_sampling_probs: true` in the request (needed to get the `"prob"` /
   `"top_probs"` fields at all), and `temperature: 0` (greedy decoding),
   the server reports the probability *after* the sampler has already
   collapsed to a single deterministic choice. That probability is always
   exactly `1.0` for the chosen token, and `top_probs` contains only that
   one entry. Entropy of a certain outcome is exactly zero, so this is not
   a bug in the strict sense — it is a correct computation of an
   uninformative quantity. Confirmed directly via a manual `curl` request
   with `post_sampling_probs` both on and off: turning it off returns the
   raw pre-selection distribution instead (field names `"logprob"` /
   `"top_logprobs"`), which does vary meaningfully across positions and
   tokens (e.g. 0.60 nats on a sentence-initial "The", dropping to 0.0003
   through a confidently-generated phrase, rising again to 0.27–0.42 near
   the end of the response).

**Fix applied:** stopped requesting `post_sampling_probs`, and updated the
entropy function to compute from `logprob` values (`exp(logprob)`) when
`prob` is not present. Verified on a 3-question test that entropy values
now vary sensibly across a response before committing to a full 304-run.

**Re-run (full 304, with the fix):** results were quantitatively identical
in every measure that didn't depend on entropy — same 291/304 correct,
same 8.2 avg wasted tokens (max 155), same ~1,348 J — confirming the fix
didn't change generation behavior, only what we could observe about it.
The entropy-threshold table this time showed a real, non-degenerate
precision-recall tradeoff: precision roughly flat at 0.96–0.97 across the
sweep, recall climbing from 0.31 at a conservative threshold to 1.00 at a
permissive one.

**Reproducibility check:** the full 304-question run was repeated three
times at temperature 0. Token counts matched exactly (304/304) between
the first two runs; the third (post-entropy-fix) run reproduced the same
correctness count, waste, and total energy to within about 1 J. Concluded
the underlying phenomenon (waste on open QA) is real and repeatable, and
the entropy finding, once measured correctly, is a genuine
precision-recall tradeoff rather than either a perfect or a useless
signal.

## Open questions carried forward

- Whether a learned probe (reading the model's hidden state directly,
  rather than a scalar entropy statistic) can sit above the entropy
  precision-recall curve established here, not just replace a broken
  baseline with a working one.
- Whether the smart-home command task should be revisited with a more
  realistic, less rigidly-anchored prompt (natural voice-assistant
  confirmations rather than bare tool calls), since the current E0 result
  shows the *rigid* version of that task doesn't need this system, but
  says nothing about a more natural version of the same task.

## Verification pass (re-checking E0 and E1 before building on them)

Before starting probe work, re-ran both labelers fresh against the
existing result files on the Pi (`out/gen.jsonl` for E0,
`out/qa_gen_run3.jsonl` for E1) and diffed the output against the numbers
already written into the paper. Both matched exactly, digit for digit:
E0 — 198/333 correct, 0.0 avg tokens wasted; E1 — 291/304 correct, 8.2 avg
wasted tokens (max 155), 2,388 total, 1,348.24 J, and the full entropy
threshold sweep. No drift, no stale numbers. This was a useful check to
do before trusting either result as a foundation for new work.

Also found and fixed a real gap in the paper draft during this pass:
Figures 1 and 2 (the response-length histogram and the E0-vs-E1 waste
comparison) had `\label{}`s and appeared in the PDF, but were never
referenced by name anywhere in the prose -- only Figures 3 and 4
(entropy ROC and entropy trace) had inline `Figure~\ref{}` mentions.
Fixed by adding one sentence near each figure's natural place in the
text. Worth remembering as a general check before submission: every
figure needs at least one explicit textual pointer, not just a caption.

## Probe training: first pass (Colab)

Built a Colab notebook (`probe_training.ipynb`, not yet added to this
repo) to test whether a learned probe on the model's hidden states can
predict safe-stop boundaries better than the entropy baseline from
Section 6.3. Loads Llama-3.2-1B-Instruct in full precision (needed for
hidden-state access, which the Pi's quantized GGUF build doesn't expose),
replays the 304 QA responses from `qa_gen_run3.jsonl`, extracts the
final-layer hidden state at each sentence boundary, and trains a small
MLP (one 64-unit hidden layer, ~131K parameters) to predict the same
safe-stop label used throughout E1.

**First run: precision/recall came back suspiciously perfect** (near 1.0
precision across nearly the full recall range). Traced this to **data
leakage in the train/test split**: the split was done at the boundary
level (`train_test_split` on all ~450 boundaries), but a single response
can contribute several boundaries that share nearly identical text and
hidden states. When boundaries from the same response landed on both
sides of the split, the probe could partly recognize "I have seen this
response before" rather than learning a general pattern. Fixed by
grouping boundaries by response id first, then splitting on the group,
so every boundary from a given response stays entirely in train or
entirely in test.

**Second run, same near-perfect curve.** Investigated and found the fix
had not actually taken effect -- the notebook's cells had been edited
and re-run out of order in the browser, so an old cell was still
supplying a stale, ungrouped split. This surfaced a second, real problem
once the fix was confirmed applied: even with grouping correct, the test
set was 92/96 (96%) positive. Most responses in this dataset are a
single sentence, which is trivially "safe to stop" by definition -- there
is no real decision being tested at those boundaries. Added a second
filter, keeping only boundaries from responses with two or more sentence
boundaries (the same "genuine decision point" filter used earlier when
characterizing the entropy baseline on the Pi).

**Third run hit a `ValueError: Found input variables with inconsistent
numbers of samples: [37, 96]`.** `X_test_s` and `y_test` had been
computed in different, out-of-order cell executions and no longer
matched. Root cause was purely a Jupyter/Colab hazard, not a logic bug:
editing cells in place and re-running them individually, rather than
top-to-bottom, leaves stale variables from earlier runs sitting in
memory. Fixed at the process level by using Runtime > Restart runtime
followed by Runtime > Run all, and fixed at the notebook level by
merging the split, hard-boundary filter, and training into a single
cell with an explicit `assert X_test.shape[0] == y_test.shape[0]` guard,
plus a matching assert before evaluation, so a stale-variable bug now
fails loudly and immediately instead of producing a silently wrong
number three cells later.

**Fourth run, clean.** train (hard only): 168 boundaries; test (hard
only): 49 boundaries; test set positives: 45/49. The probe's
precision-recall curve sat at or slightly above the entropy baseline
across most of the range, with a dip to about 0.93 precision near
recall 1.0 -- a believable, imperfect shape, unlike the earlier flat 1.0
lines that turned out to be leakage artifacts.

**Honest reading of this result.** With only 4 negative examples in the
test set, this is not strong evidence the probe beats entropy -- it is
evidence the probe is *not worse*, on a sample too small to say more.
The dataset's structure is the limiting factor: genuinely unsafe
boundaries (where stopping mid-answer would be wrong) are rare in this
304-question set, because most answers are single, clean sentences. The
pipeline itself (hidden-state extraction, grouped splitting,
hard-boundary filtering, training, evaluation, and comparison against a
real baseline) is now working correctly end to end, which was the actual
goal of this pass. The number it currently produces is a pilot, not a
result to report as a system-level claim.

**Next step identified, not yet started:** expand the QA dataset,
roughly 3-4x, with a deliberate mix of short factual questions (more of
the existing style) and "explain" / "why" questions that naturally
provoke longer, multi-sentence answers -- since that is specifically
where genuine stop-or-continue decisions occur, and where the current
dataset is thin.
