# Budget-Aware Optimal Stopping for On-Device LLM Generation

This project looks at one question: when should a language model
running on an edge device stop writing text? We treat this as a
budget-aware stopping problem, priced in **measured joules** (real
energy), not just token counts, on a Raspberry Pi.

## Why this exists

Small language models on battery-powered edge devices often keep
writing text even after they've already given a complete, correct
answer. Every extra word costs real energy. Existing early-stopping
methods (DEER, Dynasor, LASER) decide when to stop based on how
confident the model is, or how busy a server is — not on the real
energy state of the device paying for each word. This project builds
and tests a stopping controller that uses real, measured energy
instead.

## Headline results

- **Command parsing:** 333 smart-home voice commands. 198 out of 333
  (59.5%) were exactly correct, with zero extra words generated after
  the correct command in any of them.
- **Open-ended QA:** 1,074 questions (short factual questions plus
  explain/why questions). 880 out of 1,074 (81.9%) were correct.
  Correct answers had, on average, 182.3 extra words generated after
  the answer was already complete.
- **Learned probe vs. entropy:** our trained probe is better at
  spotting safe stopping points than a simple entropy-based baseline
  (0.890 vs. about 0.75-0.78 average precision, on the same data).
- **Live controller, on real hardware:** running live on a Raspberry
  Pi with a real power meter, our controller cuts measured energy use
  by 40.7%, while still getting 73.5% of answers right (compared to
  81.9% with no stopping at all).
- **Battery test:** the system ran on battery power until it died,
  completing 306 real interactions on one charge, with steady power
  draw the whole time.
- **Overhead:** the probe itself is very cheap to run (microseconds).
  The real cost is in getting the model's internal representation for
  the probe to read.

See `docs/lab-notebook.md` for the full, step-by-step story of what we
tried, what broke, and how we fixed it.

## Repository layout

    scripts/pi/            scripts that run ON the Raspberry Pi (generation, power logging, setup)
    scripts/analysis/      scripts that run on any machine (labeling, figure generation)
    scripts/generators/    dataset generation scripts (seeded, reproducible)
    scripts/equivalence/   compares the model's outputs before and after quantization
    scripts/controller/    the live controller (probe export, real-time stopping)
    data/                  datasets (smart-home commands, QA questions) with ground truth
    figures/               generated plots
    docs/                  lab notebook and design notes

## Reproducing the measurements

You will need: a Raspberry Pi (8 GB), any active cooler, and, if you
want to double-check the Pi's own power sensor, an inline USB power
meter (this is optional). No GPU is needed for the experiments in this
repo; a free Colab GPU is used later, just for training the probe.

On the Pi, once:

    bash scripts/pi/setup_pi.sh
    # follow the printed instructions to download a GGUF model

Then open three terminals on the Pi:

    # 1) the model server
    ./llama.cpp/build/bin/llama-server -m models/<model>.gguf -c 2048 --port 8080
    # 2) the power logger
    python3 scripts/pi/power_logger.py --out out/power.csv --hz 20
    # 3) the experiment
    python3 scripts/pi/run_command_generation.py --data data/commands.jsonl --out out/gen.jsonl
    python3 scripts/pi/run_qa_generation.py --data data/qa.jsonl --out out/qa_gen.jsonl

Then, on any machine, once you've copied the out/ folder over:

    python3 scripts/analysis/label_command_boundaries.py --gen out/gen.jsonl --power out/power.csv --outdir out
    python3 scripts/analysis/label_qa_boundaries.py --gen out/qa_gen.jsonl --power out/qa_power.csv --outdir out
    python3 scripts/analysis/make_figures.py

## The battery test (a second device)

The battery test was run on a **Raspberry Pi 4**, not the Pi 5 used
everywhere else in this project. The Pi 5 needs more electrical current
than our battery pack could reliably supply, so we used a Pi 4 for
this one test instead. The steps are the same as above, just run on
the Pi 4, with the controller left running on its own, on battery
power, until the device ran out of power.

## Measuring power without extra hardware

The Raspberry Pi 5 has a power sensor built in (called a PMIC), which
reports voltage and current through the command `vcgencmd
pmic_read_adc`. `scripts/pi/power_logger.py` reads this 20 times a
second and adds it up to get joules for any time period. This is what
every measurement in this project is based on. You can also plug in an
inline USB power meter (like an FNIRSI FNB48S) to double-check the
built-in sensor, but this is optional.

Note: the Raspberry Pi 4 does not have this built-in sensor. For the
battery test, power was measured only with the inline USB power meter.

## License

Code: MIT (see `LICENSE`). Data: released for research use; see
`data/README.md` for where it came from.
