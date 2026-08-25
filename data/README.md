# Data

## commands.jsonl

333 synthetic smart-home voice commands with ground-truth tool calls,
covering lights (dim/on/off), thermostat, locks, blinds, media, timers,
and fans, across 6 rooms. Includes ~100 "correction" examples of the form
"turn off the lights... actually just dim them to X%" where the ground
truth is the corrected command, not the first one spoken. Generated
programmatically (not scraped or drawn from an existing benchmark); see
the generator logic described in the project history if regenerating is
needed.

Each line:
```json
{"id": "cmd_0000", "utterance": "...", "ground_truth": "dim(bedroom, 25)", "family": "lights_dim", "has_correction": false}
```

## qa.jsonl

304 short-form factual questions (world capitals, basic science, history,
culture) with lists of acceptable answer strings (to handle aliases like
"Beijing" appearing in different phrasings). Curated by hand, not drawn
from an existing benchmark such as TriviaQA or NQ-open, so there are no
licensing constraints on redistribution.

Each line:
```json
{"id": "qa_0000", "question": "What is the largest city in Japan?", "answers": ["tokyo"]}
```

## Regenerating

Both datasets were built by short, seeded Python scripts for
reproducibility. If the generator scripts are not present in this
checkout, they can be reconstructed from the specifications above; the
`id` field ordering is deterministic given a fixed random seed.
