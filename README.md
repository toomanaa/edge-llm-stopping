# Budget-Aware Optimal Stopping for On-Device LLM Generation

Research project and lab notebook for a paper submitted to *Future
Generation Computer Systems* (FGCS). We treat the "should the model keep
generating or stop now" decision as a budget-aware optimal stopping
problem, priced in **measured joules** rather than token counts, on a
Raspberry Pi 5.

## Why this exists

Small language models on battery-powered edge devices often keep
generating text after they've already produced a complete, correct
answer. Every extra token costs real energy. Existing early-stopping
methods (DEER, Dynasor, LASER) decide when to stop using answer
confidence or server load — never using the physical energy state of the
device that is paying for each token. This project builds and measures a
stopping controller that does.

## Status

| Milestone | Status |
|---|---|
| Hardware setup (Pi 5, PMIC power sensing, cooler, USB power meter, battery bank) | Done |
| E0 — command parsing under format-anchored prompting | Done — zero waste found (a negative but informative result) |
| E1 preliminary — open-ended factual QA | Done — real waste found: 2,388 wasted tokens / ~1,348 J across 291 correct responses, reproduced across 3 runs |
| Entropy baseline (S3), v1 | Done — informative once read from pre-selection logprobs; precision ~0.96–0.97 on the small v1 set (later found to be inflated by too few hard cases — see v2 below) |
| Learned sufficiency probe (S1), pilot | Pipeline working end-to-end (`scripts/probe_training.ipynb`); pilot result comparable to entropy but on a test set too small to be conclusive (49 boundaries, 4 negative) — superseded by the v2 retrain below |
| Quantization deployment gap check | Done — mean cosine similarity 0.981 (range 0.967–0.988) between full-precision and Q4 hidden states on 20 real examples; probe evaluated directly on real Q4 hidden states scored 0.995 average precision on a 42-boundary held-out set. The probe's signal survives the full-precision-to-Q4 transition — see `docs/lab-notebook.md` |
| Expanded QA dataset (v2) — built and run | Done, with a real incident along the way — an overnight memory crunch corrupted the Pi's Python installation (fixed by reboot) and killed the power logger partway through. Recovered with a restart-safe refill run and a merge step to reach full (880/880) power coverage — full story in `docs/lab-notebook.md` |
| E1-v2 final results | Done — 880/1,074 correct (94.9% factual, 73.1% explain), 182.3 avg tokens wasted (max 716), 89,268 J wasted (880/880 power coverage), 13,568 hard boundaries with 3,507 genuinely unsafe (vs. under 20 in v1). Entropy re-measured: precision ~0.75–0.78 (down substantially from v1's misleading ~0.96). Written into the paper as Section 6.4 |
| Learned probe retrained/evaluated on v2's boundaries | Done — trained on 4,518 boundaries sampled from the v2 set, grouped 70/15/15 train/val/test split, 5 random seeds. Test-set average precision 0.890 ± 0.045 across seeds (worst 0.833, best 0.964), every seed beating the entropy baseline's ceiling. Written into the paper as Section 6.5. `scripts/probe_training_v2.ipynb` |
| Shared correctness-grading module | Done — `scripts/analysis/correctness.py` / `scripts/controller/correctness.py`, one tested definition of "correct" used by both the offline labeler and the live controller. Handles word-form variants for explain concepts (align/alignment) and plurals for factual answers, while rejecting accidental substring matches (wind/window) — see `docs/lab-notebook.md` |
| Live controller (probe export + real-time stopping) | **Done — final validated result.** Full-scale run (1,074/1,074 questions) with the overhead fix: **73.5% accuracy, 40.7% real measured energy reduction** (146,314 J vs. 246,666 J for the same questions run to completion, computed from real power traces). Completed in ~5.5 hours (half the broken first attempt) via `run_full_orchestrated.sh`. A cost-aware dynamic threshold matching the paper's own formalization (`--lambda-cost`) was also built and unit-tested as an optional mode, not yet run at scale. See `docs/lab-notebook.md` for the full diagnosis-and-fix story |
| E2 — quality vs. measured energy | **Frontier comparison done.** Our controller (73.5%/146.3 kJ) dominates a swept fixed-max_tokens baseline (interpolated ~65-70% at the same energy) and the entire achievable range of an entropy-threshold baseline (max 52.0% at any energy level, computed retroactively with `scripts/analysis/simulate_baselines.py` — no new Pi runtime needed). DEER/LASER checked against their real published mechanisms and kept as related work rather than empirical baselines — see `docs/lab-notebook.md` for why. Written into paper Section 6.6 with `figures/fig_e2_frontier.pdf` |
| E3 — battery drain runs | **Done.** Real hardware constraint discovered and honestly worked around: the Pi 5 needs 5V/5A, the available battery pack tops out at 5V/3A — solved by running this specific experiment on a Pi 4 instead (disclosed as a genuine platform difference in the paper, not a controlled comparison). **306 interactions completed on a single charge** (8.3 hours), average power 7.29-7.30W across three independently-measured segments, last outputs before shutdown fully complete and coherent, fail-open test passed cleanly (0/10 crashes) including several real probe-call timeouts handled correctly under load. Two figures built from real data (`figures/fig_e3_battery_drain.pdf`, `figures/fig_e3_power_trace.pdf`). See `docs/lab-notebook.md` for the full device-setup and power-logging story |
| E4 — overhead accounting | Not started — next up |

See `docs/lab-notebook.md` for the detailed, chronological account of
what was tried, what broke, and how it was fixed — including a full
walkthrough of three data-leakage and stale-variable bugs found and
fixed during the first probe-training pass, and the quantization
equivalence check that followed.

## Repository layout

```
paper/                LaTeX source (elsarticle / FGCS format)
scripts/pi/            scripts that run ON the Raspberry Pi (generation, power logging, setup)
scripts/analysis/      scripts that run on any machine (labeling, figure generation)
scripts/generators/    dataset generation scripts (seeded, reproducible)
scripts/equivalence/   Q4-vs-full-precision hidden state comparison (deployment gap check)
scripts/controller/    the live controller (probe export, pure-numpy inference, real-time stopping)
scripts/probe_training.ipynb      Colab notebook: probe pilot training (v1, superseded)
scripts/probe_training_v2.ipynb   Colab notebook: probe training on v2 data (current)
data/                   datasets (smart-home commands, QA questions) with ground truth
figures/                generated plots (populated by scripts/analysis/make_figures.py)
docs/                   lab notebook and design notes
```

## Reproducing the measurements

Hardware: a Raspberry Pi 5 (8 GB), any active cooler, and (optionally,
for validating the built-in sensor) an inline USB-C power meter. No GPU
required for the experiments in this repo; a free Colab GPU is used
later for probe training.

```bash
# On the Pi, once:
bash scripts/pi/setup_pi.sh
# follow the printed instructions to download a GGUF model

# Three terminals on the Pi:
# 1) the model server
./llama.cpp/build/bin/llama-server -m models/<model>.gguf -c 2048 --port 8080
# 2) the power logger
python3 scripts/pi/power_logger.py --out out/power.csv --hz 20
# 3) the experiment
python3 scripts/pi/run_command_generation.py --data data/commands.jsonl --out out/gen.jsonl
python3 scripts/pi/run_qa_generation.py --data data/qa.jsonl --out out/qa_gen.jsonl

# On any machine with the out/ folder copied over:
python3 scripts/analysis/label_command_boundaries.py --gen out/gen.jsonl --power out/power.csv --outdir out
python3 scripts/analysis/label_qa_boundaries.py --gen out/qa_gen.jsonl --power out/qa_power.csv --outdir out
python3 scripts/analysis/make_figures.py
```

## Measuring power without extra hardware

The Raspberry Pi 5 has a power management IC (PMIC) on the board that
reports per-rail voltage and current through `vcgencmd pmic_read_adc`.
`scripts/pi/power_logger.py` polls this at 20 Hz and integrates to get
joules for any time window. This is what every measurement in this repo
is based on; an inline USB-C power meter (e.g. an FNIRSI FNB48S) can be
run alongside it as an independent check, but is not required to
reproduce the numbers reported so far.

## Headline results so far

- **Command parsing (E0):** 333 smart-home voice commands, answered under
  a few-shot, format-anchored prompt. 198/333 (59.5%) exactly correct;
  **zero** tokens generated after the correct tool call in any of them.
  Good prompting alone solved the stopping problem for this task.
- **Open QA (E1):** 304 short factual questions, answered under an open
  prompt. 291/304 (95.7%) correct; an average of 8.2 tokens (max 155)
  generated after the answer was already stated, totaling 2,388 wasted
  tokens and ~1,348 J. Reproduced 3× at temperature 0 with token counts
  matching exactly and energy within ~1 J each time.
- **Entropy as a stopping signal:** useless when read from post-sampling
  probabilities under greedy decoding (always exactly 1.0 for the chosen
  token, hence zero entropy everywhere, by construction). Informative
  when read from the model's pre-selection logprobs instead: precision
  stays around 0.96–0.97 across a threshold sweep, recall climbs from
  0.31 to 1.00.

## Citation

If this repository or the accompanying paper is useful to you, please
cite (details to be finalized on publication):

```bibtex
@article{TODO2026budgetaware,
  title={Budget-Aware Optimal Stopping for On-Device {LLM} Generation:
         Pricing the Continue-or-Stop Decision in Measured Energy},
  author={TODO},
  journal={Future Generation Computer Systems},
  year={2026},
  note={Under review}
}
```

## License

Code: MIT (see `LICENSE`). Data: released for research use; see
`data/README.md` for provenance.
