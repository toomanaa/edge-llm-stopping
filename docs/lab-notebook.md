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

**Live controller result after all three fixes:** on the same 10-question
smoke test used throughout this debugging pass, accuracy climbed from
60% to 80%, with the two remaining wrong answers being genuine cases
(one real premature stop before a required concept appeared, one case
where the model's explanation took a different, equally plausible
route that never used the expected concept words at all) rather than
grading artifacts. Not yet run at full scale; the next step is a full
run over the v2 dataset with the corrected pipeline, which is what
E2's quality-vs-energy comparison will actually be built on.
