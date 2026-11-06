# Lab notebook

A running record of what we tried, what broke, and how we fixed it.

## Setup

Hardware: a Raspberry Pi 5 (8 GB) from a CanaKit kit (official active cooler, 45 W USB-C supply, 32 GB SD card), a FNIRSI FNB48S USB power meter, and an Anker 20,000 mAh / 30 W power bank.

The first setup session went badly because of the network. The model download and the llama.cpp download both crawled at a few KB/s and kept failing with TLS errors over Wi-Fi. The cause was the connection routing through a NAT64 gateway. Switching the Pi to wired Ethernet fixed it at once: the 770 MB model that had a 20+ hour estimate on Wi-Fi finished in 40 seconds. Lesson: use Ethernet for big downloads, and check the network path before blaming the remote server.

We built llama.cpp from source on the Pi (about 15 minutes on four cores) to get the `llama-server` program and its HTTP API.

We confirmed that the Pi 5's built-in power sensor (the PMIC) works. It gives free power readings at about 20 Hz with no extra hardware.

## E0: command parsing

**First attempt, no examples in the prompt.** 0 of 5 test commands matched. The model copied the parameter names from the tool description into the call (for example `dim(room, percent, 25)` instead of `dim(bedroom, 25)`), and sometimes answered in plain English with no tool call.

**Fix.** We added three worked examples to the system prompt. On 5 test prompts, 4 matched exactly. The one miss was an extra made-up argument on a temperature command. That was good enough to run the full set.

**Full run, 333 commands.** 198 were exactly correct (59.5%). In all 198, generation stopped right after the tool call, with no trailing text. The examples that fixed the format also removed the extra text we wanted to catch. We treat this as a useful negative result: for tasks with a rigid output format, good prompting can already solve the problem this project targets.

## E1: open-ended questions

E0 found no waste, so we needed a task where the model naturally says more than it has to, with no format example holding it back.

We built 304 short factual questions (capitals, science, history, culture) with lists of accepted answers, and a generation script with an open prompt ("answer clearly and correctly", no examples).

**Smoke test, 5 questions.** All 5 correct. Four answers were short. One (who wrote The Old Man and the Sea) added an extra fact nobody asked for, which is exactly the behavior we were looking for.

**Full run, 304 questions.** 291 correct (95.7%). Response lengths varied a lot, up to 155 extra tokens in one case, unlike E0.

**First labeling pass.** The entropy table came out flat and perfect (precision and recall both 1.000 at every threshold). That was suspicious. There were 447 boundary points across 291 responses, but only 61 responses had more than one boundary, because most answers were a single sentence. The entropy values were also close to zero everywhere.

## Debugging the entropy signal

We traced this to two separate bugs. We found them by sending a raw request to the llama.cpp server by hand and comparing the reply, field by field, with what our script read.

1. **Wrong field names.** The script read fields called `content` and `probs`. The server actually returns `token` and `top_probs`, or `logprob` and `top_logprobs`, depending on a request option. Our code used a default value when a field was missing, so the mismatch failed silently: every token had an empty string and zero entropy instead of raising an error. We fixed the field names.

2. **Greedy decoding hides the distribution.** With the option for post-sampling probabilities turned on and temperature 0, the server reports the probability after the sampler has already picked one token. That value is always exactly 1.0, so entropy is exactly zero. This isn't a bug, it is a correct calculation of something uninformative. We checked it by hand, with the option on and off. With it off, the server returns the raw distribution, which varies a lot: 0.60 nats on a sentence-initial "The", 0.0003 inside a confident phrase, and 0.27 to 0.42 near the end of a response.

**Fix.** We stopped asking for post-sampling probabilities and computed entropy from the log probabilities. We checked on 3 questions that entropy now varied sensibly before rerunning everything.

**Rerun, 304 questions.** Everything that did not depend on entropy came out identical: 291 correct, 8.2 average wasted tokens (maximum 155), about 1,348 J. So the fix changed only what we could observe, not how the model generated. The entropy table now showed a real tradeoff: precision about 0.96 to 0.97 across thresholds, recall rising from 0.31 to 1.00.

**Repeatability.** We ran the full 304-question set three times at temperature 0. Token counts matched exactly between the first two runs. The third run (after the entropy fix) matched on correctness, waste, and energy to within about 1 J. The waste on open questions is real and repeatable.

## Open questions at this point

- Can a learned probe that reads the model's hidden state beat the entropy curve, instead of only replacing a broken baseline with a working one?
- Should the smart-home task be revisited with a more natural prompt (spoken confirmations instead of bare tool calls)? The E0 result only covers the rigid version.

## Verification pass

Before building on E0 and E1, we reran both labelers on the saved result files and compared the output with the numbers in the paper. Everything matched exactly: E0 at 198/333 correct with 0.0 wasted tokens, E1 at 291/304 correct, 8.2 average wasted tokens (maximum 155), 2,388 total, 1,348.24 J, and the full entropy sweep.

We also found a gap in the paper draft. The response-length histogram and the E0 versus E1 comparison had labels but were never mentioned in the text. We added a sentence for each. Check before submission that every figure is referred to in the text.

## Probe training, first pass (Colab)

We built a Colab notebook to test whether a probe on the model's hidden states can predict safe stopping points better than entropy. It loads Llama-3.2-1B-Instruct at full precision (needed to read hidden states, which the Pi's quantized model does not give), replays the 304 saved responses, takes the last-layer hidden state at each sentence boundary, and trains a small neural network (one 64-unit hidden layer, about 131K parameters) on the same safe-to-stop label.

**First run: suspiciously perfect.** Precision was near 1.0 almost everywhere. The cause was leakage in the split. We split at the boundary level, but one response can give several boundaries with nearly identical text and hidden states. When they landed on both sides of the split, the probe could partly recognize responses it had seen. We fixed it by splitting by response, so all boundaries from one response stay together.

**Second run: same perfect curve.** The fix hadn't taken effect. We had edited and rerun notebook cells out of order, so an old cell still supplied the old split. Once that was fixed, a second problem showed up: the test set was 96% positive (92 of 96). Most answers were one sentence, which is trivially safe to stop at, so no real decision was being tested. We kept only boundaries from responses with two or more sentence boundaries.

**Third run: a ValueError** about mismatched sample counts (37 versus 96). The test features and labels came from different cell runs. This was a notebook hazard, not a logic bug. We now restart the runtime and run all cells from the top, and we merged the split, filter, and training steps into one cell with explicit shape checks, so a stale variable fails immediately instead of giving a wrong number later.

**Fourth run: clean.** Train (hard boundaries only): 168. Test: 49, of which 45 were positive. The probe's curve sat at or slightly above entropy for most of the range, with a dip to about 0.93 precision near recall 1.0. That is a believable shape, unlike the earlier flat lines.

**Honest reading.** With only 4 negative test examples, this shows the probe is not worse than entropy, and nothing more. The limit was the dataset: truly unsafe boundaries were rare, because most answers were single clean sentences. The pipeline itself (extract, grouped split, filter, train, evaluate, compare) now worked end to end, which was the goal of this pass. The number is a pilot, not a result to report.

**Next step:** expand the question set by roughly 3 to 4 times, mixing short factual questions with "explain" and "why" questions that produce longer, multi-sentence answers, since that is where real stop-or-continue decisions occur.

## Deployment gap: does the probe survive quantization?

The notebook trained on full-precision hidden states, but the Pi runs a 4-bit quantized model in llama.cpp, which is a different implementation. A probe that works on full-precision states does not automatically work on the deployed model, so we checked.

llama.cpp's server has an embeddings endpoint that, with pooling turned off, returns per-token hidden states from the running 4-bit model. The vector size (2048) matches the original model's hidden size exactly.

**Comparison.** On the Pi, we collected the last-token hidden state for 20 randomly chosen (question, boundary) pairs, a mix of easy single-boundary and harder multi-boundary cases. In Colab, we rebuilt the same prompts, computed the full-precision hidden states, and compared the two vectors with cosine similarity and L2 distance. We first tested the comparison code on known cases (identical vectors gave cosine 1.0, orthogonal vectors gave 0.0).

**Result.** Mean cosine similarity 0.981 (range 0.967 to 0.988), with no outliers and no size mismatches. The lowest values (0.967 to 0.975) were the later, deeper boundaries (15 to 18); the single-sentence cases scored 0.98 to 0.99. That is consistent with small rounding differences adding up over longer contexts. The effect is small here but worth watching with longer responses.

**Scope.** Twenty examples rule out an obvious failure, such as a different pooling convention or a size mismatch. They do not prove the trained probe keeps its precision and recall on 4-bit states. That needed a direct test.

## Probe on 4-bit hidden states (the direct test)

We collected 4-bit embeddings for the whole hard-boundary set (217 boundaries, matching the original probe training), with the true safe or unsafe label attached. All 217 were collected: 202 safe and 15 unsafe. That is more negative examples than the 4 in the original test split, because it covers the whole labeled set.

In Colab, we loaded the saved probe and scaler, rebuilt the same grouped split (by response, 25% test, fixed seed), and evaluated the probe on the 4-bit vectors of the held-out part. We first checked the split and loading code on synthetic data.

**Result.** Test set of 42 boundaries (36 safe, 6 unsafe), average precision 0.995. At the most permissive threshold, precision was 0.857 at recall 1.000, rising to 1.000 as the threshold tightened. At recall near 0.9, the probe had precision about 0.971 against entropy's roughly 0.963 to 0.965.

**Reading.** The probe's signal survives the move from full-precision training to real 4-bit hidden states on the Pi. The comparison with entropy is close, with a small edge for the probe, but 42 boundaries (6 negative) is a small test. The honest claim then was "comparable to, and possibly slightly better than, entropy". A larger dataset was needed to say more.

## Building the v2 question set

No standard mix of factual and explanatory questions exists in the literature. Factoid QA (NQ-open, TriviaQA) and non-factoid QA (ANTIQUE, WikiQA) are studied as separate categories. We chose 60% explain/why and 40% factual, on purpose: explain/why questions are what produced multi-sentence answers with real unsafe stopping points in v1.

`make_qa_v2.py` generates 1,074 questions: 434 factual and 640 explain/why across about 160 topics, with 4 question templates each. Explain questions carry a list of required concept keywords, and a new `match_mode` field: "any" for factual (one accepted answer is enough) and "all" for explain (every required concept must appear).

**A grammar bug, caught before shipping.** The first draft combined topics written as verb phrases ("leaves changing color") with templates that expected a full clause ("Why does {t}?"), which gave broken questions such as "Why does some clouds bring rain and others dont?". We switched to templates that work with noun-phrase topics ("What causes {t}?", "Explain how {t} works") and fixed the one topic written as a full clause. Some mildly awkward phrasing remains and is a known limitation. The model understands the questions, and this dataset is not meant to be a citable benchmark.

We wrote a matching generation script and labeler. We tested the new "all" logic on synthetic data: a response with only one of two required concepts was correctly marked incorrect and left out of boundary labeling.

**Smoke test problems on the Pi.** Two issues came up in a 5-question test. First, the server needed a restart (a "No such file or directory" error was a stale terminal state). Second, both explain answers hit the original 250-token limit and were cut mid-sentence, which breaks a labeler that assumes responses finish naturally. We raised the limit step by step: 400 tokens still truncated. At 800 tokens (tested on 20 questions) the longest response was 642 tokens and ended naturally. Explain answers averaged about 464 tokens and factual ones about 12. We set the limit to 800. We also noticed that explain answers are heavily formatted (numbered lists, bold headers), which may matter for the sentence-boundary pattern later.

**Time estimate.** With explain answers near 464 tokens for 640 of the 1,074 questions, we estimated 8 to 9 hours, and ran it overnight on wall power. The battery pack was kept for the battery experiment.

## The overnight run: a memory crunch and a damaged Python

The run finished all 1,074 questions, but two things went wrong, both from the same cause.

**What happened.** The server ran as one process for about 9 hours. At a mid-run check, memory use was 7.4 GB of 7.9 GB (89%) with heavy swapping. The power logger, a separate background process, seems to have been killed or disturbed by the memory pressure at about the 40% mark. Generation itself kept going and finished.

**The damage.** Afterward, basic Python imports started crashing with segmentation faults (first `json`, then `re`, and then `apt`, which uses Python). We checked step by step: no overheating (the throttle flag read zero), no filesystem errors, plenty of disk space, file reads working. That narrowed it down to Python's compiled extension modules. Several package reinstalls, including a real upgrade of the Python packages, did not help. That suggested the damage was not in the files on disk but in memory or swap, which a reboot clears.

**Fix.** A normal reboot fixed everything. Memory use went back to about 400 MB and the imports worked. The reinstall attempts had also left a changelog tool in a broken state, so we disabled its apt hook instead of fixing it.

All 1,074 generation records on disk were untouched. The damage was only in the running Python environment.

## Recovering power coverage: a restart-safe refill

The generation data was fine, but the power logger had died partway through. A coverage check (not an assumption) showed that only 380 of 877 correct responses had power data. Instead of repeating the 8 to 9 hour run, we built a targeted refill. A script found which response ids fell outside the power log's time range (599 questions) and extracted just those for re-collection.

To avoid the same failure, we wrote `run_qa_v2_generation_safe.py`. It processes questions in batches of 50 and checks available memory between questions. If available memory drops below 1,500 MB, it stops and tells us to restart the server. In practice, the batch limit triggered every pause. Memory stayed healthy (6.3 to 6.6 GB available) the whole time, so the regular restarts alone prevented the earlier buildup.

The refill ran over two sessions (we stopped once for the night and resumed from where it left off) and finished all 599 questions.

**Merging.** `merge_v2_results.py` combines the original generation file with the refill (the refill wins if an id appears in both, since it has power data) and merges the three power logs (original plus two refill sessions) into one file. We tested it on synthetic data with a known overlapping id and a known uncovered response. After merging, 1,073 of 1,074 responses had power data, up from 43%.

## E1-v2 results

We reran the labeler on the merged data. These first-pass numbers were later corrected (see below):

- 877 of 1,074 correct (81.7%): 412 of 434 factual (94.9%), 465 of 640 explain (72.7%).
- 180.1 average wasted tokens per correct response (maximum 721), 157,991 in total.
- 87,938.96 J wasted, over all 877 correct responses.
- 13,569 boundaries from multi-sentence responses, 3,623 of them unsafe, about 240 times more unsafe examples than the 304-question set's fewer than 20.
- Entropy baseline on the larger set: precision about 0.74 to 0.77 across thresholds (recall from 0.34 to 1.00), well below the 0.96 to 0.97 on the small set. The earlier number was probably inflated by how few hard cases it had.

We built `make_figures_v2.py` for the waste-by-question-type chart, the response-length histogram, the entropy curve (recomputed from the boundary labels), and a direct v1 versus v2 entropy overlay, the figure that shows the small-set result was inflated.

With this dataset, the earlier bottleneck (too few negative examples to trust any precision estimate) was gone. The bar for the probe was now entropy at about 0.75 precision.

## Retraining the probe on v2

We redesigned the probe notebook instead of patching it. We sampled 4,518 boundaries from the 13,569 hard boundaries, grouped by response. Instead of one split, we used a grouped 70/15/15 train/validation/test split repeated over 5 random seeds, with the validation set used to pick the model and the test set used once per seed at the end. This fixes the pilot's weakness (one split, a tiny test set).

Before using real data, we stress-tested the sampling and splitting on synthetic data at realistic size (about 4,500 boundaries, 200 response groups). There was no leakage across any of the 5 seeds, and every example was accounted for. We also ran the full pipeline once on synthetic data to catch runtime errors.

**Result: test average precision 0.890 ± 0.045 over 5 seeds** (worst 0.833, best 0.964), on test sets of 693 to 779 boundaries with 132 to 234 unsafe ones. Every seed beat entropy's ceiling of about 0.74 to 0.77, by a similar margin. The curve sits above entropy across almost the whole recall range, and the five seed curves overlap closely, so the result isn't from one lucky split.

This is different from the pilot, which rested on 4 to 6 negative test examples. This one has hundreds per seed, five independent splits, and a validation step. In the paper we state that average precision and precision at a fixed recall are related but not the same measurement.

We also needed a PDF of the v1 versus v2 figure (only the PNG had been copied locally), so we converted it in Colab. Colab's download call sometimes needs to be run twice, and anything not downloaded is lost when the session resets.

## Building the live controller

Everything so far graded stopping decisions after the fact: generate the full response, then look back to see where stopping would have been safe. The real system has to decide while generating. We built it in stages, and found bugs at each one.

**Exporting the probe.** To avoid scikit-learn version problems between Colab and the Pi, we exported the network's weights and the scaler's mean and scale to a plain `.npz` file, and rewrote the forward pass (scale, matrix multiply, ReLU, sigmoid) in plain numpy. We checked it by training a fresh network of the same shape, exporting it, and comparing both versions on 200 test points. The outputs matched exactly (maximum difference 0.0). It loaded correctly on the Pi on the first try.

**Bug 1: boundaries checked too rarely.** The first version only asked the probe when a 20-token chunk happened to end exactly on a period. That is mostly luck, and responses ran over 300 words between checks. We changed it to search all the accumulated text for every new sentence boundary and check each one, throwing away any text generated past the boundary where a stop is triggered. We tested this by packing three sentences into one fake chunk and confirming that all three were checked separately and the response was cut at the right one.

**Bug 2: markdown mistaken for content.** The probe then stopped, confidently and wrongly, right after markdown headers such as "**Types of Eclipses:**". The boundary pattern also matched bare newlines, and markdown makes many of them. The probe, trained with the same pattern, seems to have learned a false link between "text ends after a structural newline" and "safe to stop". We limited boundaries to real sentence-ending punctuation. Numbered-list markers ("1.", "2.") caused the same problem, so we excluded periods that follow a digit. We tested both fixes on the exact failing text, including multi-digit numbers like "12.", and confirmed that real sentences were not excluded.

**Bug 3: grading too strict, plus a packaging mistake.** Live accuracy was still about 60%, far below the offline 82%. When we checked the "wrong" results by hand, two of four were not stopping failures. The model had explained the concept correctly but used a different word form ("align" for "alignment", "generate" for "generator"), and the exact-text check missed it. This check also produced the offline v2 numbers and the probe's training labels, so it was a real bug.

We wrote one shared `correctness.py`, used by both the offline labeler and the live controller (before, they were two copies of the same logic, which is how silent drift happens). The first version, which matched words by shared prefix, gave false positives on ordinary English ("wind" and "window", "star" and "start"). The second version required the leftover letters on each side to be a known word ending (for example "align" plus "ment", "generat" plus "e" or "or"), and passed a broad test suite, including the real failures. Deploying it caused one more problem: once exact matching respected word boundaries, plurals of factual answers ("seismograph" required, model said "seismographs") started failing. We added a small, tested allowance for a trailing "s" or "es" on factual matches only, and confirmed it does not bring back the wind/window problem.

We also made a packaging mistake. The first update zip had `label_qa_v2_boundaries.py` at the wrong path, so it landed in the top folder instead of replacing the real file. The rerun showed unchanged numbers, which told us the fix had never reached the file being run. We repackaged it with the right folders, and the numbers changed as expected.

**Corrected v2 numbers**, after both fixes (small, expected shifts, not a different picture):
- 880 of 1,074 correct (94.9% factual, 73.1% explain, up from 72.7%).
- 182.3 average wasted tokens (maximum 716), 160,406 wasted tokens in total.
- About 89,268 J wasted, with power data for all 880 responses.
- 13,568 hard boundaries, 3,507 of them unsafe (down from 3,623). The drop of 116 is exactly the boundaries that had been wrongly labeled unsafe because of the word-form mismatch.
- Entropy precision rose slightly (for example from 0.766 to 0.776 at the tightest threshold).

None of this changes the qualitative picture. The numbers in the paper's v2 section had to be updated to match.

**Live result after the three fixes.** On the same 10-question smoke test, accuracy went from 60% to 80%. The two remaining wrong answers were real cases: one genuine early stop before a required concept appeared, and one where the model explained the topic correctly but never used the expected words.

## Calibrating the threshold

Before a full run, we calibrated the threshold instead of trusting the 10-question test. On a 30-question slice, threshold 0.7 gave only 60 to 66% accuracy. Thresholds 0.8 and 0.9 both gave exactly 73.3%, with the same questions right and wrong, which means the probe was already far above 0.9 on the remaining bad cases.

We compared each live "wrong" result with the same question's offline result at full length. Exactly half were impossible to fix with any threshold (the model gets them wrong even with unlimited length). The other half were fixable: the model answers correctly if allowed to run longer. A threshold of 0.99 recovered most of these, giving 83.3% on the 30-question slice, with total generation still 65% smaller than the offline baseline. A larger 100-question check showed that the 30-question figure was optimistic, and put the honest number at about 74 to 75%. We locked in 0.99.

## First full run: a serious problem

`run_full_orchestrated.sh` automates the 10 or so server restarts a full run needs. It starts the server, waits until it is ready, runs the controller for one batch, checks the exit code (0 for finished, 2 for a batch or memory pause), and repeats. We tested both exit paths with real subprocesses before trusting it for hours, and started it so that it would survive a lost connection.

It completed all 1,074 questions in about 9.86 hours, with 9 automatic restarts and no manual steps.

**Result: 796 of 1,074 correct (74.1%), 622 (57.9%) stopped early, and, computed from the real power trace, the controller used slightly more energy than the no-stopping baseline (−3.0% savings).** This was a serious, surprising result for a system meant to save energy, and we needed to understand it.

**Diagnosis.** The run generated 94,769 words in 35,508 seconds, which is 2.67 words per second, against about 11 words per second measured earlier. That is a 4.1 times overhead, so about 76% of the run was spent on something other than generating tokens. The cause was the prompt-cache option being turned off on every generation call, which made the server reprocess the whole growing prompt for every small chunk and boundary check.

Turning the cache on for both endpoints and retesting 20 questions raised throughput to 4.13 words per second (a 55% gain), but savings were still negative (−1.1%). We wrote `diagnose_caching.py` to test the two endpoints separately, sending growing prefixes of the same text. The generation endpoint stayed flat at about 0.2 s per call, so its caching works. The embeddings endpoint grew almost linearly with text length, from 0.26 s to 1.58 s (about 6 times), so the cache option has no effect there in this server version, and every boundary check pays an ever larger cost. With 14 or more checks in some responses, this was the main cost of the whole controller.

**Fix: check less often.** We added a `check_every_n` setting (default 2) so that only every Nth new sentence boundary gets an embeddings call and a probe check. The skipped boundaries are still passed correctly. We tested it with 6 sentences and `check_every_n=2`: the probe was called 3 times instead of 6, and a stop still cut the output at the right sentence.

**20-question check with real power logging: 29.6% measured energy savings at 70.0% accuracy** (down a little from 75.0% with finer checking, an expected cost). This is the first real, positive energy result for the live controller. It is smaller than the 69.1% word-count estimate, which never counted the controller's own running cost, and we state that gap openly in the paper. Counting only saved tokens would have overstated the benefit.

We kept the first full run's output files as a record of the overhead problem and how we diagnosed it.

## E3: battery drain, on a different device than planned

We found a hardware limit before spending hours on the experiment: the Pi 5 needs 5 V / 5 A, and the Anker pack (checked on its printed label) tops out at 5 V / 3 A per port. On the battery pack, the Pi 5 showed "SD: card not detected" and a warning that USB boot needs high current (5 V, 5 A). With the same SD card on wall power it booted normally, so the power supply was the only cause.

We switched to a Raspberry Pi 4 (4 GB), whose official supply is rated 5 V / 3 A, which the pack can deliver. We rebuilt the stack there: llama.cpp built cleanly (about 15 minutes, no network problems this time), the model was pulled directly from Hugging Face, and the dataset, probe weights, and controller scripts were copied over. This is a real platform change, not a controlled comparison, and the paper says so. E3's power figures describe the Pi 4 with the same controller and workload, not the Pi 5 used elsewhere.

The Pi 4 has no built-in power sensor like the Pi 5's, so we measured current with the FNIRSI FNB48S meter. We ruled out logging on the Windows PC (it would have needed a driver replacement with a risk of conflicting with the vendor's software), and ended up reading the meter on the Pi 4 itself through its separate USB data port, with an open-source logger that supports the FNB48S. That gave automatic logging at 100 samples per second, better than the manual photo-reading we had planned.

The logger has a known reliability problem (noted in its own README). It crashed four times with a USB timeout over the drain. Each time we unplugged and replugged only the data cable (never the power cable) and restarted it. None of the crashes affected the drain test, which ran as a separate process and was confirmed alive at every check. One crash left a gap of about 2.24 hours in the power log, which we only noticed at the next scheduled check, and a final crash left the last 26 minutes before shutdown unlogged. The paper discloses both gaps.

We tracked drain progress mainly with the pack's own percentage display (99% at the start), checked against the clock, instead of the meter's integrated capacity. The capacity-based projections were inconsistent when we cross-checked them (estimates of total interactions swung from about 485 to 4,146 to 1,280 depending on one assumption), so we dropped that method. The percentage-based projections converged steadily (485, 453, 362, 341, 332, 326), and they correctly anticipated the faster discharge near the end, a known lithium-ion voltage-sag effect (8.83% per hour early, 10.86% per hour mid-run, 20 to 24% per hour in the last half hour).

**Result: 306 interactions on one charge**, from 07:12 to about 15:28 (8.3 hours). We read the last three responses before shutdown and found them complete and coherent, including a long, well-structured explanation of rust formation as the very last output. Average power over three clean segments (separated by the crash gaps) was 7.29 W, 7.30 W, and 7.30 W, very consistent across measurements taken hours apart.

**Fail-open test.** We ran `fail_open_test.py` (10 creative, out-of-distribution prompts unlike the probe's training data) once when the pack was at 27%, while the main drain was running on the same server. All ten finished with no crashes and no suspiciously short outputs. Several boundary checks timed out under the extra load. Each time, the controller treated the failure as "keep generating", as designed. That was an unplanned stress test of the fail-open rule.

We built two figures from this data: battery charge and interactions over time (showing the faster discharge at the end) and a three-segment power trace with the crash gaps clearly marked. Both are in the paper.

## E4: overhead accounting

Most of this came from data we already had.

**Probe cost.** We timed the probe's forward pass (135,297 parameters: one 64-unit hidden layer over the 2,048-dimensional hidden state, plus the scaler) over 20,000 calls: 27.8 microseconds on average. The trained weights only ever lived on the Pi, so we timed a synthetic probe of the same shape. That is valid for timing, since matrix-multiply cost depends on the sizes, not on the values. The paper says so.

**Boundary-detection cost.** We used the linear fit measured on the Pi in `diagnose_caching.py` to compute the embeddings call's cost at several context lengths: 374 ms at 100 characters, up to 5,715 ms at 2,000 characters. On a log-scale figure next to the probe's cost, the boundary check is four to five orders of magnitude more expensive at every length. This confirms what the `check_every_n` diagnosis already suggested: getting the representation is the expensive part, not the probe.

**One-time calibration cost.** The threshold sweep used three 30-question probes (0.8, 0.9, 0.99) plus two 100-question checks (an early run at 0.7, which we dropped, and a confirming run at 0.99): 290 interactions. At about 29.6 seconds per interaction, that is roughly 2.4 hours, paid once, offline, not per request or per device. This is the honest cost behind the "training-free" claim: no gradient training happens, but a calibration pass is still needed.

## The corrected full run: a positive result

Before relaunching, we closed a gap between the paper's formal rule (stop when the predicted quality gain is below a calibrated exchange rate times the marginal cost of continuing) and what the code did (one fixed confidence bar everywhere, whatever the next check cost). We fitted a cost model from the `diagnose_caching.py` data, a linear fit of embeddings latency against text length, converted to joules with a typical average power. We added an optional `dynamic_threshold()` mode to the controller that relaxes the bar as context grows and checks get more expensive. It is switched on only by the `--lambda-cost` flag, so the validated fixed-threshold path stays the default (a regression test showed identical behavior with the flag off). It has been built and unit-tested but not run at full scale.

We relaunched the full run with the validated settings (threshold 0.99, `check_every_n=2`). The first attempt exited almost at once with no error and no output file. We don't fully know why (possibly an old process still holding the port from the night before). A clean restart, with upfront checks of the working directory and every input file, ran normally. We also caught a real bug before relaunching: the orchestrator script had the previous run's output filename hardcoded, which would have silently added the new run onto the old, contaminated file. We changed it to a fresh path first.

The run took about 5.5 hours (05:46 to 11:17), about half the first attempt, consistent with `check_every_n=2` roughly halving the expensive embeddings calls. It had 10 automatic restarts and no manual steps.

**Final result: 789 of 1,074 correct (73.5%), 590 (54.9%) stopped early by the probe, and, computed from the real power trace across the whole run with the validated `compute_live_energy.py`, 146,313.55 J used by the live controller against 246,665.86 J for the same questions run to completion: a measured 40.7% energy reduction.** This is higher than the 29.6% on the 20-question sample, which is what we'd expect from a larger sample averaging out noise. Accuracy (73.5%) is close to the first run's 74.1%, so `check_every_n` cost little quality and fully fixed the energy regression.

This is the number the whole live-controller phase was building toward. A live stopping controller, measured with its own running cost included, gives a 40.7% energy reduction at a real but modest accuracy cost (73.5% against about 82% with no stopping). We got there by finding and fixing a regression that would otherwise have gone unnoticed.

## Deciding what E2 should compare against

Before building DEER and LASER as baselines, we checked what they actually do, from the original papers. DEER (Yang et al., 2025) doesn't use entropy or hidden states. It watches for reasoning-transition words ("Wait", "Alternatively", "Let me check") that step-by-step reasoning models produce, forces a trial answer at each one, and measures confidence from the model's own token probabilities on that answer, against a fixed threshold (default 0.95). LASER (2026) builds on the same signal but replaces the fixed threshold with one that responds to server queue load.

We decided they are not directly comparable to this paper's setting, and said so openly instead of forcing it. Both were designed for a different decision (cutting hidden reasoning before an answer starts, not deciding when a visible answer is already complete) and assume a model that produces transition words, which ours does not. LASER also assumes a multi-request server queue, which a single edge device doesn't have. Treating them as related work with a clear difference is easier to defend than an adapted reimplementation that a reviewer could say isn't really DEER or LASER. The main comparison stays with methods that share our exact problem: a fixed max_tokens limit, an entropy threshold, and our controller. A DEER-inspired signal could be added later if it turns out to be small and clean. It has not been built.

## Baseline comparison, without new Pi runs

`simulate_baselines.py` computes fixed max_tokens and entropy-threshold results from data already collected, since every boundary's token position, entropy, and (through the power trace) energy cost was already on disk. Both policies decide at sentence boundaries, like every other method here, so no policy gets a finer resolution than the others. For the roughly 194 of 1,074 questions that were wrong even at full length (and so don't appear in the boundary labels), every policy is charged that response's full energy and a wrong verdict, so this can't favor any method.

We tested the simulation on hand-traceable synthetic data first. The max_tokens sweep gave a strictly increasing accuracy and energy curve that levels off once the budget exceeds the longest response. The entropy sweep gave the expected non-monotonic pattern (accuracy can drop as the threshold loosens, once boundaries that aren't the real safe point qualify). We checked both by hand against the synthetic data.

**Result, full 1,074 questions.** Fixed max_tokens, swept from 30 to 500 tokens, goes from 46.1% accuracy at 57.3 kJ up to 81.7% at 238.0 kJ, approaching but never reaching the no-stopping point (81.9% at 246.7 kJ). The entropy threshold behaves differently: even at its most cautious setting it never uses more than 66.4 kJ, and accuracy falls as the threshold relaxes, from 52.0% to 42.4%. That fits entropy's precision ceiling of about 0.75: it triggers often, and wrongly often enough that no setting reaches competitive accuracy.

**Our controller's validated point (73.5% accuracy, 146.3 kJ)** sits above the max_tokens curve at comparable energy (interpolating at 146.3 kJ gives roughly 65 to 70%) and above the whole achievable range of the entropy curve (which never reaches 73.5% at any energy tested). That is dominance over a comparable range, not a single favorable point. We built `fig_e2_frontier.pdf` showing all three plus the no-stopping point, and put the result in the paper.
