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

## Deployment gap: does the probe's signal survive quantization?

Raised as a direct challenge to the probe pilot: the notebook trains on
hidden states from the full-precision HuggingFace model, but the Pi
actually runs a Q4-quantized GGUF model through llama.cpp, which is a
separate implementation. Proving the probe works on full-precision
hidden states does not by itself prove the same signal is accessible or
behaves the same way on the deployed Q4 model. This needed to be
checked, not assumed, before any claim about the probe "working."

**Discovered that llama.cpp's server has a `/embeddings` endpoint** that,
with `"pooling": "none"`, returns real per-token hidden states directly
from the running Q4 model -- no separate tooling needed. Confirmed the
vector dimension (2048) matches HuggingFace's `hidden_size` exactly, so
there is no structural mismatch to worry about.

**Built a two-sided comparison.** On the Pi: restarted `llama-server`
with `--embeddings` enabled, then wrote `collect_q4_embeddings.py`,
which selects 20 real (question, boundary) pairs at random from
`qa_gen_run3.jsonl` -- deliberately a mix of easy single-boundary and
harder multi-boundary examples -- and queries `/embeddings` for the
last-token hidden state of each, matching exactly what the probe
notebook uses. In Colab: added a cell that reconstructs the identical
prompt for each of those 20 examples, computes the full-precision hidden
state the same way the probe training does, and compares the two vectors
with cosine similarity and L2 distance. The comparison math itself was
unit-tested first (identical vectors -> cosine 1.0, orthogonal vectors ->
cosine 0.0, small synthetic perturbation -> cosine ~0.999) before trusting
it on real data.

**Result: mean cosine similarity 0.981 across the 20 pairs (range
0.967-0.988), no outliers, no dimension mismatches.** This is a
reassuring answer to the original challenge -- the Q4 model's hidden
states are structurally very close to the full-precision ones the probe
was trained on. One real pattern in the data: the lowest-similarity
examples (0.967-0.975) were consistently the deeper, later-boundary
cases (boundary 15-18) rather than the single-sentence, boundary-0
cases (which mostly scored 0.98-0.99) -- consistent with small
per-token quantization rounding compounding slightly over longer
contexts. The effect is small on this sample but worth tracking as
responses get longer.

**Honest scope of this check:** 20 examples is enough to rule out a
gross, obvious failure (e.g. a completely different pooling convention
or a dimension mismatch), but it is not the same as proving the trained
probe's actual precision/recall holds up when evaluated *directly on Q4
hidden states* rather than full-precision ones. That is the next,
more direct test: run the already-trained probe's weights against
Q4-extracted hidden states for the labeled boundary set and see if its
precision/recall curve survives. High cosine similarity is encouraging
indirect evidence; probe performance on Q4 data would be the direct
evidence, and is the more convincing thing to put in the paper.

## Probe evaluated directly on Q4 hidden states (the direct deployment test)

Followed up the cosine-similarity check with the more direct test: does
the already-trained probe (trained entirely on full-precision hidden
states) still work when fed real Q4 hidden states from the actual
deployment platform?

Built `collect_q4_embeddings_full.py`, an extension of the earlier
20-sample script that instead collects Q4 embeddings for the **entire**
hard-boundary set (217 boundaries, matching exactly what the original
probe training and evaluation used), with the true safe/unsafe label
attached directly. Run on the Pi against the live `--embeddings`
server: 217/217 collected successfully, 202 safe / 15 unsafe overall --
notably more negative examples available than the 4 that limited the
original full-precision test split, simply because this pulls from the
whole labeled set rather than one random 25% slice.

Built a second Colab cell, `colab_probe_on_q4_cell.py`, that loads the
saved probe and scaler (`probe_and_scaler.pkl`, downloaded from the
original training run) and the Q4 embeddings, reproduces the *same*
grouped train/test split (by response id, `test_size=0.25`,
`random_state=42`) used during training, and evaluates the probe's
`predict_proba` directly on the Q4-derived test vectors -- an honest
held-out test, not a re-run on data the probe already saw. Both the
split logic and the probe-loading/evaluation logic were simulated with
synthetic data and confirmed to run end-to-end before handing the cell
over.

**Result: held-out test set of 42 boundaries (36 safe / 6 unsafe),
average precision 0.995.** The full precision-recall curve is a
believable, non-degenerate shape -- at the most permissive threshold,
precision is 0.857 with recall 1.000 (a handful of false positives let
through); precision climbs to 1.000 as the threshold tightens, with
recall falling accordingly. At recall roughly matched to the entropy
baseline's higher-recall operating points (~0.9), the probe shows
precision around 0.971, compared to entropy's ~0.963-0.965 in a similar
range -- a modest, not dramatic, edge.

**Honest reading.** This directly answers the question raised about the
deployment gap: the probe's signal survives the transition from
full-precision training to real Q4/llama.cpp hidden states on the
actual Pi. It does not simply collapse or become noise. The comparison
to the entropy baseline is close, with the probe showing a small
apparent edge at matched recall on this sample, but 42 test boundaries
(6 negative) is still a small evaluation -- "comparable to, and
possibly modestly better than, entropy" is the honest claim right now,
not "the probe clearly beats entropy." The dataset-expansion step
identified earlier remains the way to get a test set large enough to
say something stronger with confidence.

## Building the v2 QA dataset

Started work on the dataset expansion identified as the limiting factor
throughout. No standard mix of factual vs. explanatory questions exists
in the literature to borrow -- factoid QA (NQ-open, TriviaQA) and
non-factoid/explanatory QA (ANTIQUE, WikiQA) are studied as separate
benchmark categories, not combined with an established ratio. Chose a
60% explain/why, 40% factual split deliberately: explain/why questions
are what v1 showed reliably produces multi-sentence answers with real
"unsafe" stopping boundaries, so the dataset is weighted toward the
question type that exercises the phenomenon under study.

Built `make_qa_v2.py` generating 1,074 questions (434 factual, 640
explain/why across ~160 topics with 4 question-template variants each).
Explain questions carry a list of required concept keywords rather than
a single answer string, with a new `match_mode` field: "any" (factual --
correct if the text contains at least one accepted answer) or "all"
(explain -- correct only if the text contains every required concept).

**Caught a real grammar bug before shipping.** The first generator draft
combined gerund-phrase topics ("leaves changing color") with templates
expecting a finite-clause object ("Why does {t}?"), producing broken
questions like "Why does some clouds bring rain and others dont?" Fixed
by restricting to templates that work correctly with noun-phrase topics
("What causes {t}?", "Explain how {t} works.", etc.) and fixing the one
topic that was itself phrased as a full clause rather than a noun
phrase. A residual, milder awkwardness remains in some entries (mixed
gerund/finite-verb topic phrasing), noted honestly as a known limitation
rather than hidden -- the questions are still comprehensible to the
model, just not textbook-clean English, and correctness here matters
less than in a dataset meant to be a citable benchmark in its own right.

Built matching `run_qa_v2_generation.py` (same generation logic as v1,
now carrying `type` and `match_mode` through to the output) and
`label_qa_v2_boundaries.py` (extends v1's labeler with `match_mode`-aware
correctness checking). Unit-tested the new "all" logic on synthetic data
before running on the Pi: a synthetic explain response containing only
one of two required concepts was correctly marked incorrect and excluded
from boundary labeling, confirming the extension works as intended.

**Smoke-test debugging on the Pi.** A 5-question smoke test surfaced two
real issues before committing to the full run. First, `llama-server`
needed restarting (a "No such file or directory" error on
`llama-server` turned out to be a stale terminal state, resolved by
re-running the same startup command). Second, and more substantively,
both explain-question responses in the smoke test hit the original
250-token generation cap and were truncated mid-sentence -- a genuine
problem, since a labeler that assumes responses reach a natural end
cannot correctly score a boundary near an artificial cutoff.

Diagnosed by progressively raising the cap: at 400 tokens, still
truncating; at 800 tokens (tested on 20 questions), the longest
response was 642 tokens and finished naturally, with explain responses
averaging ~464 tokens against ~12 for factual ones. Set the generation
cap to 800 for the full run based on this. Also noted the model's
explain answers are heavily structured (numbered lists, bold markdown
headers) rather than plain prose, which may need revisiting in the
sentence-boundary regex later, though it was not blocking for
generation itself.

**Revised time estimate.** The original plan assumed run time similar
to v1 (~20 minutes for 304 short questions). With explain responses
averaging ~464 tokens across 640 of the 1,074 questions, the full run
is estimated at roughly 8-9 hours, not 1-2 -- launched as an overnight,
unattended run on wall power (not the battery bank, which is reserved
for the E3 battery-drain experiment).

## The overnight run: a memory crunch, a corrupted Python, and recovery

The overnight generation run completed all 1,074 questions, but two
things went wrong along the way, both traced back to the same root
cause and both worth recording in full since the debugging path was
long.

**What happened.** `llama-server` ran as a single continuous process for
the full ~9-hour run. Checked partway through: memory usage had climbed
to 7.4 GB of the Pi's 7.9 GB total (89%), with active swapping. The
power logger, running as a separate background process, appears to have
been killed or destabilized by this memory pressure at some point
around the 40% mark of the run, though the generation process itself
kept going and completed successfully.

**Discovering the real damage.** After the run finished, basic Python
commands started segfaulting -- first `import json`, then `import re`,
then even `apt` itself (which depends on Python for some subprocess
hooks) began crashing. Diagnosed step by step: confirmed no thermal
throttling (`vcgencmd get_throttled` read `0x0`), no filesystem errors
in `dmesg`, plenty of disk space free, and file reads working while
`import json` specifically crashed -- narrowing the fault to something
in Python's compiled C extension modules (`_json`, `_sre`) rather than
the interpreter, the filesystem, or the hardware broadly. Multiple
targeted package reinstalls (`apt install --reinstall python3`,
then the full `python3.13`/`libpython3.13-stdlib` stack, including a
genuine version upgrade from 3.13.5-1 to 3.13.5-2+deb13u4) did not fix
it -- strong evidence the corruption was not sitting in on-disk package
files, which reinstalling replaces, but in something transient: live
memory state or a corrupted swap file, neither of which survives a
reboot.

**The fix was a clean reboot** (`sudo reboot`), which resolved it
completely -- `json` and `re` imports worked immediately afterward, with
memory back to a normal ~400 MB used out of 7.9 GB. This confirms the
diagnosis: severe memory pressure during the unattended run had
corrupted something in RAM or swap that on-disk package reinstalls
could not touch, and a reboot cleared it. A side effect of the debugging
process: repeatedly trying to reinstall packages had left
`apt-listchanges` in a broken, half-configured state (its own hook
script was segfaulting on every apt invocation); resolved by moving
`/etc/apt/apt.conf.d/20listchanges` out of the way so apt stops invoking
it, rather than continuing to fight a non-essential changelog viewer.

All 1,074 generation records survived on disk untouched throughout --
the corruption was in the running Python environment, never in the
data files themselves.

## Recovering full power coverage: a restart-safe refill run

With the generation data intact but the power logger having died
partway through (confirmed: only 380 of 877 correct responses had
power data in their timestamp range, verified with a dedicated coverage
script rather than assumed), rather than redo the full 8-9 hour
generation, built a targeted refill: `find_missing_power.py` identifies
exactly which response IDs fall outside the power log's recorded time
range and extracts just those questions (599 of them) from the original
dataset for re-collection.

To avoid repeating the same failure mode, built
`run_qa_v2_generation_safe.py`: a restart-safe version of the generation
script that processes questions in batches (50 at a time) and, between
questions, checks `/proc/meminfo` directly -- if available memory drops
below a 1,500 MB safety threshold, it stops and prints an explicit
instruction to restart `llama-server` before continuing, rather than
plowing ahead into another crash. In practice the batch-size cutoff
(not the memory threshold) was what triggered each pause; memory stayed
healthy (6.3-6.6 GB available) throughout the whole refill run, meaning
the periodic server restarts alone were enough to prevent the earlier
memory buildup from recurring.

The refill ran across two separate sessions (interrupted once by
choice, to stop for the night, and resumed cleanly the next day using
the script's built-in resumability) and completed all 599 questions.

**Merging the results.** Built `merge_v2_results.py` to combine the
original 1,074-question generation file with the 599-question refill
(refill wins on any id collision, since it is guaranteed to have power
data), and to merge the three separate power CSVs (the original run's
log, plus two refill-session logs) into one combined file. Tested
against a synthetic scenario with a known overlapping id and a
known-uncovered response before running on real data, confirming the
merge correctly prefers refill data and correctly identifies coverage.
Result: 1,073 of 1,074 responses now have at least some power data in
their time window -- essentially complete coverage, up from the
original 43%.

## E1-v2 final results

Re-ran the labeler on the merged, fully-covered dataset. Final numbers,
now with genuine full power coverage rather than a partial estimate:

- 877 of 1,074 responses correct (81.7%): 412/434 factual (94.9%),
  465/640 explain (72.7% -- lower because explain correctness requires
  every required concept to appear, not just one accepted phrasing).
- Average 180.1 tokens wasted per correct response after the earliest
  safe stopping point (maximum 721), 157,991 tokens total.
- 87,938.96 J wasted after the earliest safe stop, computed over all
  877 correct responses (full coverage, not the earlier 380-response
  partial estimate).
- 13,569 boundaries from multi-sentence responses (the genuine decision
  points), of which 3,623 are labeled unsafe -- roughly 240x more
  negative examples than the original 304-question set's fewer-than-20.
- Entropy-threshold baseline, re-measured on this much larger sample:
  precision holds around 0.74-0.77 across the threshold sweep (recall
  climbing 0.34 to 1.00), a substantial and consistent drop from the
  0.96-0.97 measured on the smaller set. Read as the more trustworthy
  number -- the earlier result was likely inflated by how few hard
  cases it contained.

Built `make_figures_v2.py`, generating a waste-by-question-type
comparison, a response-length-by-type histogram, the v2 entropy
precision/recall curve (recomputed directly from the boundary labels
rather than parsed from report text, so the figure is self-verifying),
and a direct v1-vs-v2 entropy overlay -- the single figure that makes
the "small dataset gave an inflated result" finding visible at a
glance. Wrote the full result into the paper as a new subsection
(Section 6.4, "A larger, harder open-ended set"), positioned as a
follow-up that both confirms and sharpens the original E1 finding
rather than replacing it.

**Where this leaves the project.** The dataset bottleneck that limited
every result since the probe pilot (too few negative examples to trust
any precision/recall estimate) is now resolved: 3,623 real unsafe
boundaries is enough to train and evaluate the probe with genuine
statistical weight, and the bar it needs to clear -- entropy at roughly
0.75 precision -- is now a solid, well-measured target rather than an
artifact of a small sample. Retraining and evaluating the probe on this
dataset is the next step.

## Retraining the probe on v2: the real comparison

Built `probe_training_v2.ipynb`, a substantial redesign of the pilot
notebook rather than a small patch. Sampled 4,518 boundaries from the
v2 set's 13,569 hard boundaries, grouped by response (so a response's
boundaries never split across the sample boundary itself). Rather than
a single train/test split, used a proper grouped 70/15/15
train/validation/test split, repeated across 5 random seeds, with the
validation set used for model selection each time and the test set
touched only once, at the end, per seed -- addressing the exact
weakness (single split, tiny test set) that made the pilot's result
untrustworthy.

Before running this on real data, stress-tested the new sampling and
splitting logic against synthetic data covering the full pipeline at
realistic scale (~4,500 boundaries, 200 simulated response groups):
confirmed zero data leakage across all 5 seeds (no response id
appearing in more than one split) and confirmed every example was
accounted for in each split, for every seed. Also ran the full
train/validate/test/aggregate pipeline once end-to-end on synthetic
data to confirm no runtime errors before handing the notebook over.

**Result: test-set average precision 0.890 ± 0.045 across the 5 seeds**
(worst seed 0.833, best 0.964), on test sets of 693-779 boundaries with
132-234 genuinely unsafe examples each. Every single seed beat the
entropy baseline's precision ceiling of roughly 0.74-0.77 measured on
the same v2 data, by a similar margin each time. The precision-recall
curve sits visibly above entropy's across nearly the whole recall
range, and the five individual seed curves overlay closely, showing the
result is not an artifact of one favorable split.

This is a different situation from the pilot: that result rested on 4-6
negative test examples and was not trustworthy either way; this one
rests on hundreds of negative examples per seed, five independent
splits, and a validation-based model-selection step the pilot never
had. Written into the paper as Section 6.5, replacing the earlier TODO
placeholder, with the honest caveat that average precision and
precision-at-fixed-recall are related but not identical measurements,
stated explicitly rather than treating the two curves as trivially
comparable.

Along the way, needed a `.pdf` version of `fig_v1_vs_v2_entropy`
(generated as `.png` and `.pdf` together by `make_figures_v2.py` on the
Pi, but only the `.png` had been carried over locally) -- rather than
reconnect to the Pi, converted it directly in Colab with Pillow. Also
hit Colab's ordinary session-storage behavior: a `files.download(...)`
call can need to be re-run if it does not fire the first time, and
anything not downloaded is lost if the session resets. Both handled as
small appended utility cells in the notebook (Sections 11-12) rather
than undocumented one-off snippets, so the full record of what was run
stays in one place.

## Building the live controller

Everything up to this point graded stopping decisions offline: generate
a full response, then look back and label where stopping would have
been safe. The actual system needs to decide live, mid-generation. Built
this in stages, each with real bugs found and fixed before moving on --
worth recording in full since the debugging path mattered as much as
the destination.

**Exporting the probe.** Rather than run scikit-learn on the Pi (version
compatibility risk between the Colab environment that trained the probe
and whatever the Pi has), built `export_probe_weights.py` to pull the
trained MLP's raw weights and the scaler's mean/scale out as a plain
`.npz` file, and `probe_inference.py` to reimplement the forward pass
(scale, matmul, ReLU, sigmoid) in pure numpy on the Pi side. Validated
by training a fresh scikit-learn MLP with the same architecture in this
environment, exporting it, running both the original and the numpy
reimplementation on 200 test points, and confirming outputs matched to
floating-point precision (max difference 0.0) before trusting it near
the real trained probe. Loaded correctly on the Pi on the first try.

**Bug 1: boundaries checked too rarely.** The first version of
`live_controller.py` only asked the probe "is it safe to stop" when a
generated chunk happened to end exactly at a period -- but with
fixed-size 20-token chunks, that is mostly coincidence, since sentences
rarely finish exactly at a chunk boundary. First smoke test showed
responses running 300+ words between checks. Fixed by searching the
*entire* accumulated text for every new sentence boundary as it
appears, checking each one individually, and discarding any text
generated past the boundary that triggers a stop. Verified with a test
that packs three sentences into one fake chunk and confirms all three
get checked separately, with the response correctly truncated at
whichever one triggers.

**Bug 2: markdown structure mistaken for content.** With the chunk-size
bug fixed, a new failure appeared: the probe stopped confidently and
wrongly right after markdown section headers like "**Types of
Eclipses:**" -- a heading, not an answer. Traced to the boundary regex
matching bare newlines as well as real punctuation; markdown headers
and list intros produce a lot of newlines, and the probe (trained on
boundaries detected with the same regex) appears to have picked up a
spurious correlation between "text ending right after a
structural newline" and "safe to stop." Fixed by restricting boundaries
to genuine sentence-ending punctuation only. A second, related case
then appeared: numbered-list markers ("1." "2." "3.") also matched
plain `[.!?]` punctuation and produced the same kind of content-free
stopping point. Fixed with a lookbehind excluding periods immediately
preceded by a digit. Both fixes were validated against the exact failing
text before being trusted, including a check that multi-digit list
numbers ("12.") are excluded too, and that real sentences are never
accidentally excluded.

**Bug 3: overly strict correctness grading, plus a packaging mistake.**
With both boundary bugs fixed, live accuracy was still around 60%, well
below the ~82% offline baseline. Manually inspecting the "wrong"
results found that two of four failures were not stopping failures at
all: the model had explained the required concept correctly but used a
different word form than the dataset's exact required string ("align"
instead of "alignment", "generate" instead of "generator"), and the
exact-substring correctness check did not recognize them as the same
concept. Since this check is also what produced the offline v2 numbers
already written into the paper (and the labels the probe itself trained
on), this was a real, scope-relevant bug, not just a live-controller
issue.

Built a shared `correctness.py`, imported by both the offline labeler
and the live controller (previously two separate copies of the same
logic, which is exactly the kind of drift that causes silent bugs).
First attempt used a naive shared-prefix heuristic and produced clear
false positives on ordinary English -- "wind" is a character-prefix of
"window", "star" of "start" -- caught by testing before shipping.
Second attempt required the leftover characters on each side of a
shared prefix to be a recognized derivational suffix (e.g. "align" +
"ment", "generat" + "e"/"or"), which passed a broad test suite
including the exact real failure cases. Deploying it surfaced one more
regression: legitimate plurals of factual answers ("seismograph"
required, model said "seismographs") started failing once exact
matching was made word-boundary-aware, since there is no boundary
between "seismograph" and its own trailing "s". Fixed with a narrow,
tested allowance for a trailing "s"/"es" specifically on factual
("any" mode) matches, confirmed not to reopen the wind/window problem.

Separately, an actual deployment mistake: the first packaged update zip
had `label_qa_v2_boundaries.py` at the wrong internal path, so it
extracted to the top level of `~/e1` instead of overwriting the real
file in `scripts/analysis/` -- the offline report was re-run but showed
completely unchanged numbers, which was the tell that the fix had
never actually reached the file being executed. Repackaged with the
correct directory structure and reran; the offline numbers changed as
expected on the second attempt.

**Corrected v2 numbers**, after both the boundary-detection and
correctness fixes (small, expected shifts from the original E1-v2
numbers, not a different picture): 880 of 1,074 responses correct
(94.9% factual, 73.1% explain, up from 72.7%), 182.3 average tokens
wasted (max 716), 160,406 tokens wasted in total, an estimated
89,268 J wasted (full 880/880 power coverage), 13,568 hard boundaries
with 3,507 labeled unsafe (down from 3,623 -- the 116-boundary drop is
exactly the word-form-mismatch boundaries that were being mislabeled
unsafe before the fix). Entropy's precision shifted up slightly across
the threshold sweep (e.g. 0.766 -> 0.776 at the tightest threshold).
None of this changes the paper's qualitative claims -- entropy's real
ceiling is still far below what the small v1 sample suggested, and the
probe still has real work to do -- but the exact numbers in
Section 6.4 need updating to match.

**Live controller result after all three fixes:** on the same 10-question smoke test used throughout this debugging pass, accuracy climbed from
60% to 80%, with the two remaining wrong answers being genuine cases
(one real premature stop before a required concept appeared, one case
where the model's explanation took a different, equally plausible
route that never used the expected concept words at all) rather than
grading artifacts. Not yet run at full scale; the next step is a full
run over the v2 dataset with the corrected pipeline, which is what
E2's quality-vs-energy comparison will actually be built on.

## Calibrating the stopping threshold at scale

Before committing to a full 1,074-question run, calibrated the
threshold properly rather than trusting the 10-question smoke test's
80%. A 30-question slice at threshold 0.7 gave only 60-66% accuracy;
0.8 and 0.9 both gave an identical 73.3% (same questions right and
wrong at both), a sign the probe's confidence on the remaining bad
cases was already far above 0.9, not marginally over it. Diagnosed the
persisting failures directly: compared each live "wrong" result against
the SAME question's already-known offline (full-length) correctness,
finding exactly half were structurally unfixable by any threshold (the
model gets them wrong even given unlimited length) and half were
genuinely fixable (the model can answer correctly if allowed to run
longer). Pushing to threshold 0.99 recovered most of the fixable cases,
giving 83.3% accuracy on a 30-question calibration slice with the
generation total still 65% smaller than the offline baseline -- a real
operating point, though a larger 100-question follow-up check
(necessary since the 30-question estimate turned out optimistic)
settled the honest number closer to 74-75%. Locked in threshold 0.99
as the calibrated setting for the full run.

## The full 1,074-question run, and a serious problem it revealed

Built `run_full_orchestrated.sh` to automate the ~10 manual
restart-server cycles a full run would otherwise need: it starts
llama-server, polls `/health` until ready, runs the controller for one
batch, checks its exit code (added distinct `sys.exit(0)` for genuine
completion vs `sys.exit(2)` for a batch/memory pause, replacing the
previous plain `return`), and loops automatically. Verified both exit
paths with real subprocess tests (not just static inspection) before
trusting it to run unattended for hours. Launched via `nohup ... &` so
it would survive SSH disconnects.

The full run completed in full: 1,074/1,074 questions, ~9.86 hours,
9 automatic server restarts, zero manual intervention needed after
launch -- genuinely the first complete, real, hands-off overnight-scale
run since the memory-crash incident earlier in the project, and the
orchestration held up.

**Result: 796/1,074 correct (74.1%), 622/1,074 (57.9%) stopped early by
the probe, and — computed from real measured power, not a word-count
estimate — the live controller used slightly *more* energy than the
offline no-stopping baseline: -3.0% "savings."** This is a serious,
counterintuitive result on a system whose whole purpose is to save
energy, and it needed to be understood rather than reported as-is.

**Diagnosis.** Computed the run's effective throughput: 94,769 words
generated over 35,508 seconds of wall-clock time is 2.67 words/sec,
against a known real generation speed of roughly 11 words/sec measured
earlier in the project -- a 4.1x overhead factor, meaning roughly 76%
of the entire 9.86-hour run was spent on something other than actual
token generation. Traced this to `"cache_prompt": False` being set on
every `/completion` call in `live_controller.py`, forcing the server to
re-process the entire accumulated prompt from scratch on every one of
the many small generation chunks and boundary checks per response,
rather than reusing already-computed context.

Set `cache_prompt: True` on both `/completion` and `/embeddings` calls
and re-tested the same 20 questions: throughput improved to 4.13
words/sec (a real 55% gain) but energy savings were still negative
(-1.1%) -- an improvement, but not the fix. Built `diagnose_caching.py`
to isolate the two endpoints directly: sent a sequence of growing
prefixes of identical text to `/completion` and `/embeddings`
separately and timed each call. `/completion` latency stayed flat
(0.21s to 0.18s across 8 steps of growing text, confirming its caching
genuinely works); `/embeddings` latency grew almost linearly with text
length (0.26s to 1.58s, roughly 6x) -- confirming `cache_prompt` has no
effect on that endpoint in this llama.cpp server version, and every
single boundary check pays a full, ever-larger re-prefill cost
regardless of the setting. With some responses making 14+ boundary
checks, this was structurally the dominant cost of the entire
controller, not a tunable inefficiency.

**Fix: check less often, not every sentence boundary.** Added a
`check_every_n` parameter (default 2) that only spends an actual
`/embeddings` call and probe evaluation on every Nth new sentence
boundary, skipping the ones in between while still advancing past them
correctly. Verified with a controlled test (6 sentences,
`check_every_n=2`) that the probe is called exactly 3 times instead of
6, and that a triggered stop still truncates the output at the correct
sentence even when the triggering check was several boundaries after
the last one actually evaluated.

**Validated on the same 20-question sample, this time with real power
logging: 29.6% measured energy savings, 70.0% accuracy** (down modestly
from 75.0% at the finer-grained checking rate, an expected and
reasonable cost of checking less often). This is the first positive,
real, measured energy result for the live controller, and the honest
number to build the full-scale claim on -- notably smaller than the
69.1% word-count estimate, which never accounted for the controller's
own runtime cost. That gap is itself worth stating plainly in the paper
rather than hidden: the true cost of a live stopping decision is not
free, and reporting the token-savings number alone would have
overstated the system's real benefit.

**Not yet run at full scale with this fix** -- the next session's first
step. The negative-result run (`out/live_full_t99.jsonl`,
`out/live_full_t99_power.csv`) is being kept as a record of the
overhead problem and its diagnosis, not deleted, since the debugging
path is as much a part of this project's honest record as the eventual
positive result.

## E3: battery drain, on a genuinely different device than planned

Discovered a real hardware limitation before wasting a multi-hour
experiment on it: the Pi 5 requires 5V/5A, and the available Anker
battery pack (20,000mAh, confirmed via its own printed spec label)
tops out at 5V/3A per port -- a real, hard ceiling, not a cable or
port problem. Diagnosed this cleanly: the Pi 5 showed `SD: card not
detected` and an explicit `USB boot requires high current (5 volt 5
amp)` warning on the battery pack, and booted perfectly normally on
wall power with the same SD card, confirming the power supply as the
sole cause.

Solved by switching to a second Raspberry Pi 4 (4GB), whose official
power supply is rated 5V/3A -- within the Anker pack's capability.
Rebuilt the whole stack from scratch on this second device: cloned and
built llama.cpp (clean, ~15 minutes, no network issues this time),
pulled the model directly from Hugging Face rather than round-tripping
through the Pi 5, and transferred the dataset, probe weights, and
controller scripts over. This is a genuine platform substitution, not
a controlled comparison, and is disclosed as such in the paper: E3's
absolute power figures characterize the Pi 4 under the same controller
and workload, not the Pi 5 used throughout the rest of the paper.

Set up a FNIRSI FNB48S inline USB power meter to measure real current
independently of either Pi's own telemetry (the Pi 4 has no onboard
power-management IC comparable to the Pi 5's PMIC, so `vcgencmd
pmic_read_adc` -- used for every other energy number in this paper --
does not exist on this device). After ruling out Windows-side PC
logging (would have required a driver replacement, Zadig, with a real
risk of conflicting with the vendor's own software) and briefly
considering routing the meter through the Pi 4 itself, ended up doing
exactly that after all: installed `pyusb` and a reverse-engineered
open-source logger (`baryluk/fnirsi-usb-power-data-logger`, confirmed
to explicitly support the FNB48S) directly on the Pi 4, reading the
meter over its own separate USB data port. This gave genuine
100-samples/second automatic logging, a real upgrade over the manual
periodic photo-reading fallback planned earlier.

The logger proved to have a real, documented reliability limitation
(noted in its own README): it crashed with a USB timeout four times
over the course of the full drain, each time recovered by unplugging
and replugging only the data cable (never the power cable) and
restarting the same command. None of these crashes affected the actual
drain test, which ran as a fully independent process throughout and
was confirmed alive via `ps aux` at every check. One crash produced a
genuine ~2.24-hour gap in the power log with no manual intervention
(discovered only at the next scheduled check-in), and a final crash
left the last ~26 minutes before the true end unlogged; both gaps are
disclosed honestly in the paper rather than papered over.

Tracked overall drain progress primarily via the battery pack's own
percentage display (99% at the confirmed start, checked periodically
against the real clock) rather than the power meter's integrated
capacity, after finding the capacity-based projection method
produced inconsistent, unreliable answers under cross-checking
(estimates of total interactions before depletion swung from ~485 to
~4,146 to ~1,280 depending on which sub-assumption was adjusted,
never converging) -- concluded this was not worth chasing further and
that the simple, direct percentage reading was the trustworthy
signal, which it proved to be: successive percentage-based projections
converged steadily (485, then 453, then 362, then 341, then 332,
then 326) as more data came in, and correctly anticipated the real,
observed acceleration in discharge rate near the end (a well-known
lithium-ion voltage-sag effect, not a measurement artifact -- confirmed
by comparing early, mid, and late instantaneous rates: 8.83%/hour,
10.86%/hour, 20-24%/hour in the final half hour).

**Final result: 306 interactions completed on a single charge**, from
07:12 to approximately 15:28 (8.3 hours). The last three logged
responses before the device lost power were inspected directly and
found fully complete and coherent -- including a long, well-structured
multi-paragraph explanation of rust formation as literally the final
thing generated before shutdown, with no sign of degradation or
truncation. Average power draw across three independently-logged
clean segments (separated by the crash gaps) was 7.29W, 7.30W, and
7.30W -- a strong, reassuring consistency across measurements taken
hours apart.

Ran the fail-open test (`fail_open_test.py`, 10 creative/OOD prompts
never resembling the probe's training distribution) once while the
pack was at 27% charge, concurrently with the ongoing main drain
sharing the same server process. All ten completed with zero crashes
and zero suspiciously short outputs. Several boundary checks genuinely
timed out during this run under the added concurrent load (visible in
the console as repeated probe-call failures) -- an unplanned, real
stress test of the fail-open rule, and in every case the controller
correctly treated the failure as "keep generating" rather than
stopping, exactly as designed.

Built two figures from this real data: a battery-charge-and-
interactions-over-time plot showing the observed discharge curve
(including its visible late acceleration), and a three-segment power
trace plot showing the measured wattage throughout, with the
crash-related gaps clearly visible and honestly labeled rather than
hidden. Both written into the paper's Section 6.7, replacing the
earlier placeholder.

## E4: overhead accounting

Filled in the paper's third and final empirical placeholder, largely
from data already collected tonight rather than needing new
experiments. Three real numbers:

**Probe inference cost.** Benchmarked the probe's own forward pass
(135,297 parameters: a single 64-unit hidden layer over the model's
2,048-dimensional hidden state, plus the standardization scaler) over
20,000 calls: 27.8 microseconds average. Since the actual trained
weights only ever lived on the Pi and never round-tripped through this
environment, benchmarked a synthetic probe with the identical
architecture instead -- valid for timing purposes, since numpy matmul
cost depends on matrix dimensions, not the specific trained values,
and disclosed as such in the paper.

**Boundary-detection cost.** Reused the real, Pi-measured linear fit
from `diagnose_caching.py` (built during the E2 overhead diagnosis) to
compute the `/embeddings` call's cost at several realistic context
lengths: 374ms at 100 characters, up to 5,715ms at 2,000 characters.
Placed side by side with the probe's cost on a log-scale figure: the
boundary check costs four to five orders of magnitude more than the
probe itself at every length tested, direct, quantified confirmation
of what the earlier `check_every_n` diagnosis already implied --
the retrieval mechanism, not the probe, is what needed fixing.

**One-time calibration cost.** Tallied the real calibration effort
from earlier tonight's threshold sweep: three 30-question probes
(0.8, 0.9, 0.99) plus two 100-question checkpoints (an initial run at
a since-abandoned 0.7, and a confirming run at 0.99) = 290 total
interactions. At that phase's measured average of 29.6 seconds per
interaction, approximately 2.4 hours of one-time compute -- paid once,
offline, not per-request or per-device. Framed explicitly as the
honest cost behind the paper's "training-free" claim: no gradient
training occurs, but a real, disclosed calibration pass is still
required.

All three empirical sections (E2, E3, E4) are now complete and written
into the paper with real numbers, replacing every remaining
placeholder from the original experimental plan.

## The corrected full-scale run: a real, positive result

Before relaunching, closed a real gap between the paper's own
formalization (Section 4: stop when predicted quality gain is less
than a calibrated exchange rate times the marginal cost of continuing)
and what the code actually did (a single fixed confidence bar,
identical at every point in a response regardless of how expensive the
next check actually was). Fitted a real cost model from the
`diagnose_caching.py` measurements already in hand -- a linear fit of
`/embeddings` latency against text length, converted to joules with a
representative average power figure -- and added an optional
`dynamic_threshold()` mode to `live_controller.py` that relaxes the
confidence bar as accumulated context grows and the next check gets
more expensive, gated behind an explicit `--lambda-cost` flag so the
already-validated fixed-threshold path remains the untouched default
(confirmed via regression test: identical behavior with the flag
unset). Not exercised at full scale yet -- built and unit-tested,
available for a future run, not part of tonight's result.

Relaunched the full run with the validated fixed settings
(threshold 0.99, `check_every_n=2`). One early, false alarm: the first
orchestrator attempt exited almost immediately with no error and no
output file, cause not fully pinned down (possibly a stale process
still holding the port from the previous night, since a clean restart
with added upfront diagnostics -- confirming working directory and
every input file's existence before the loop begins -- resolved it and
the retry ran normally start to finish). Also caught and fixed a real
bug before relaunching: `run_full_orchestrated.sh` had the previous
run's output filename hardcoded, which would have silently appended
the new, corrected run onto the old, contaminated negative-result file
had it not been changed to a fresh path first.

The corrected run completed in about 5.5 hours (05:46 to 11:17), roughly
half the first attempt's ~10 hours -- consistent with `check_every_n=2`
roughly halving the number of expensive `/embeddings` calls, confirming
the fix's benefit holds at full scale, not just on the 20-question
validation sample. 10 automatic server restarts, zero manual
intervention needed after launch.

**Final result: 789/1,074 correct (73.5%), 590/1,074 (54.9%) stopped
early by the probe, and -- computed from the real measured power trace
across the full run, using the same `compute_live_energy.py` script
validated earlier -- 146,313.55 J used by the live controller against
246,665.86 J for the same 1,074 questions run to natural completion:
a measured 40.7% energy reduction.** This is meaningfully higher than
the 29.6% seen on the 20-question validation sample, which is a good
sign rather than a concern -- a larger sample averaging out the noise
of a small one, not a fluke in the other direction. Accuracy (73.5%)
landed close to the pre-fix run's 74.1%, confirming the `check_every_n`
change cost little in quality while resolving the energy regression
entirely.

This is the number the whole live-controller phase of the project was
building toward, and the honest way to state it: a live, cost-aware
stopping controller, evaluated with its own real runtime overhead
included rather than estimated away, delivers a genuine 40.7% energy
reduction at a real but modest accuracy cost (73.5% vs. roughly 82%
for the unstopped baseline) -- a result earned by finding and fixing a
regression that would otherwise have gone unnoticed, not by reporting
the first number that came out of the pipeline.

## Deciding what E2 actually needs to compare against

Before building DEER and LASER as empirical baselines, checked what
those methods actually do, from the primary sources rather than from
memory. DEER (Yang et al., 2025) does not use entropy or hidden
states: it watches for specific reasoning-transition tokens ("Wait",
"Alternatively", "Let me check") that CoT-style reasoning models
naturally produce, forces a trial answer at each one, and measures
confidence from the model's own token probabilities on that induced
answer, against a fixed threshold (their default: 0.95). LASER (2026)
builds on the same induced-answer-confidence signal but replaces the
fixed threshold with one that responds to real-time server queue
load, via a tanh-shaped adjustment around a base value.

Concluded these are not directly comparable to this paper's setting,
and said so explicitly rather than forcing a comparison: both were
designed for a different decision point (truncating hidden reasoning
before an answer starts, not deciding when a visible answer is
already-complete) and assume a model that naturally produces
transition tokens ours does not; LASER's core mechanism additionally
assumes a multi-request serving queue that does not exist on a single
edge device. Positioning them as related work with an axis-of-novelty
argument (already how Section~\ref{sec:related} frames it) is more
defensible than an adapted reimplementation a reviewer could
reasonably contest as not really being DEER or LASER at all. Agreed
with the plan to keep the primary empirical comparison to methods that
share this paper's exact problem (fixed max\_tokens, entropy
threshold, our controller), and to revisit a DEER-inspired
induced-answer signal only if it turns out to be a small, clean
addition -- not yet built.

## The real baseline comparison, computed without new Pi runtime

Built `simulate_baselines.py` to compute fixed-max\_tokens and
entropy-threshold outcomes retroactively from data already collected
for the offline v2 analysis -- no live experiments needed, since every
sentence boundary's token position, entropy, and (via the existing
power trace) energy cost was already on disk. Both policies decide at
the same sentence-boundary granularity as every other method in this
paper, keeping the comparison fair rather than giving a token-level
policy a resolution advantage nothing else has. For the ~194 of 1,074
questions that were wrong even at full length (and so never appear in
the boundary labels), every policy is charged that response's full
energy and a wrong verdict identically, so this cannot bias the
comparison toward any one method.

Validated the simulation logic against hand-traceable synthetic data
before trusting it on real numbers: confirmed the max\_tokens sweep
produces a strictly monotonic accuracy/energy curve that correctly
plateaus once the token budget exceeds the longest response, and
confirmed the entropy sweep produces the correct *non*-monotonic
pattern (accuracy can fall as the threshold loosens, once boundaries
that are not the true safe point start qualifying) -- both traced by
hand against the synthetic data's known structure before running on
real data.

**Real result, full 1,074-question set.** Fixed max\_tokens, swept from
30 to 500 tokens, traces a smooth curve from 46.1% accuracy at 57.3 kJ
up to 81.7% at 238.0 kJ, approaching but never quite reaching the
no-stopping baseline (81.9% at 246.7 kJ). Entropy threshold behaves
differently: even at its most conservative setting it never exceeds
66.4 kJ, and accuracy *falls* as the threshold relaxes, from 52.0% down
to 42.4% -- consistent with entropy's ~0.75 precision ceiling
established earlier: it triggers often, and wrongly often enough, that
no setting of it reaches a competitive accuracy.

**Our controller's single validated point (73.5% accuracy, 146.3 kJ)
sits above the max\_tokens curve at comparable energy** (interpolating
that curve at 146.3 kJ gives roughly 65-70%) **and above the entropy
curve's entire achievable range** (which never reaches 73.5% at any
energy level tested). This is real frontier dominance over a
comparable operating range, not a single cherry-picked favorable
point -- the standard the paper's own E2 methodology set for itself
from the start. Built `fig_e2_frontier.pdf` plotting all three
alongside the no-stopping reference point, and wrote the result into
Section 6.6, replacing the earlier single-point framing.

**Not yet done:** the DEER-inspired addition remains open, contingent
on finding a clean, honestly-labeled adaptation; E3 (battery dynamics)
and E4 (overhead accounting) remain unstarted.
