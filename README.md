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
| Entropy baseline (S3) | Done — informative once read from pre-selection logprobs; precision ~0.96–0.97, recall 0.31→1.00 across threshold sweep |
| Learned sufficiency probe (S1), pilot | Pipeline working end-to-end (`scripts/probe_training.ipynb`); pilot result is comparable to entropy but on a test set too small to be conclusive (49 boundaries, 4 negative) — see `docs/lab-notebook.md` |
| Expanded QA dataset for a trustworthy probe evaluation | Not started — identified as the next step |
| Draft-agreement signal (S2) | Not started |
| Live controller on the Pi | Not started |
| E2 — quality vs. measured energy (baseline comparison) | Not started |
| E3 — battery drain runs | Not started |
| E4 — overhead accounting | Not started |

See `docs/lab-notebook.md` for the detailed, chronological account of
what was tried, what broke, and how it was fixed — including a full
walkthrough of three data-leakage and stale-variable bugs found and
fixed during the first probe-training pass.

## Repository layout

```
paper/                LaTeX source (elsarticle / FGCS format)
scripts/pi/            scripts that run ON the Raspberry Pi (generation, power logging, setup)
scripts/analysis/      scripts that run on any machine (labeling, figure generation)
scripts/generators/    dataset generation scripts (seeded, reproducible)
scripts/probe_training.ipynb   Colab notebook: hidden-state extraction + probe training (Phase 4)
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
