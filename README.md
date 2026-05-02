# Persuasion Arguments Generation

Generates paired persuasive essays that vary in tone (strong vs. weak) while holding factual content constant. The essays are stimuli for an experiment studying how the confidence/strength of an argument's tone affects persuasion of readers about contested political claims (gun background checks; abortion regret).

This repository is the **stimulus pipeline** for the study; participant-facing data collection happens in a separate Qualtrics survey, which loads stimuli from this repo at runtime via the loader script in `generation/qualtrics_essay_loader.js`.

## Research project

Standalone — no companion analysis or research-project repo is currently referenced from this codebase.

## Deployment

The generated stimulus JSON is served as static files via **GitHub Pages** at `https://Zachary-Wojtowicz.github.io/persuasion_experiments/generation/all_essays.json` (the `.nojekyll` file at the repo root keeps Pages from filtering the directory). The Qualtrics survey fetches that URL at survey load time and assigns one essay per participant. There is no separate webapp deployment — the participant interface is the Qualtrics survey itself.

## What gets logged

This repo does not log participant data directly. The Qualtrics loader (`qualtrics_essay_loader.js`) reads three embedded fields that Qualtrics randomizes upstream — `topic` (`guns` or `abortion`), `politics` (`liberal` or `conservative`), and `tone` (`strong` or `weak`) — and writes the following back as embedded data on each participant's response:

- `essay` — the full assigned essay text (with `\n` converted to `<br>` for HTML display)
- `essay_issue`, `essay_politics`, `essay_tone` — confirmed assignment values
- `essay_pair_index` — index of the chosen essay pair within the stance run
- `essay_stance`, `essay_contrary_stance` — the position the essay argues and the position it argues against

These embedded fields, plus whatever rating/attitude items the Qualtrics survey itself asks, are the per-participant record. Outcome measures and exclusion/attention-check logic live in the Qualtrics survey, not in this repo.

## Randomization scheme

Within this repo:

- **Stimulus generation** (offline, run once per stimulus set): for each (issue, politics) cell, a pool of 15 argument points is generated and 5 are randomly sampled per essay pair (C(15,5) = 3,003 combinations). Each essay pair shares the same 5 points; the strong and weak essays differ only in tone instruction. Each essay is assigned a random rhetorical approach (8 options, e.g. "open with a question," "lead with strongest evidence") and a random tone-instruction phrasing.
- **Participant assignment** (at survey time, in the loader): given Qualtrics-assigned `topic` × `politics` × `tone`, the loader collects all matching essays and picks one uniformly at random. If `essay` is already set on the participant's record (e.g., refresh/back-button), it is not reassigned.

The primary experimental factor is **tone** (`strong` vs. `weak`), crossed with `topic` and `politics`. Tone-relevant content (point set, stance, contrary stance) is held constant within each pair; only confidence/hedging language differs. The exact tone variant, rhetorical approach, and full LLM prompt are saved alongside each essay in the output for traceability.

## Stimulus generation pipeline

The generator follows a two-step process:

1. **Point pool generation** — For each issue/stance, the script generates a pool of argument points (default 15) using the point-generation prompt.
2. **Essay generation** — For each essay pair, 5 points are randomly sampled from the pool and turned into two essays: one strong-toned and one weak-toned, sharing a randomly assigned rhetorical approach.

Variation mechanisms baked in to reduce style/lexical leakage between essays:

- **Point pool sampling** (3,003 combinations per stance).
- **Tone instruction variants** for each tone level, focused on linguistic markers (boosters vs. hedges) rather than formulaic structure.
- **Rhetorical approach rotation** (8 structural options) — assigned independently of tone so structure does not confound confidence.
- **Anti-formula nudge** in every essay prompt to vary transitions and avoid template-y output.

## Setup

```bash
cd generation
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key(s):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Usage

By default, the script uses **Claude Opus 4.6** and generates essays for **all issues**, **both political orientations**, and **both tones** (strong + weak).

### Generate essays (default settings)

```bash
python generation/generate_arguments.py --out generation/essays.json
```

This produces 3 essay pairs per issue/politics combination (= 3 pairs × 2 issues × 2 politics = 24 essays total).

### Generate more essays per combination

```bash
python generation/generate_arguments.py --num-essays 25 --out generation/essays.json
```

### Restrict to one issue or one political orientation

```bash
python generation/generate_arguments.py --issue guns --out generation/essays.json
python generation/generate_arguments.py --politics liberal --out generation/essays.json
```

### Use OpenAI instead

```bash
python generation/generate_arguments.py --provider openai --model gpt-4o --out generation/essays.json
```

### Single essay mode

Pass `--tone` to generate just one essay (useful for testing):

```bash
python generation/generate_arguments.py --tone weak --issue guns --politics liberal
```

### Reproducible runs

```bash
python generation/generate_arguments.py --seed 42 --out generation/essays.json
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--issue` | all | Restrict to one issue: `guns` or `abortion` |
| `--politics` | both | Restrict to one orientation: `liberal` or `conservative` |
| `--tone` | — | Single-essay mode: generate one essay with this tone (`weak` or `strong`) |
| `--num-essays` | `3` | Number of essay pairs per issue/politics combination |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `--model` | `claude-opus-4-6` | Model name |
| `--temperature` | `1.0` | Sampling temperature |
| `--seed` | — | Random seed for reproducible point sampling and variant selection |
| `--pool-size` | `15` | Number of argument points to generate in the pool before sampling |
| `--out` | — | Output JSON file path |

## Output format

Batch mode produces a JSON array. Each element has:

```json
[
  {
    "issue": "guns",
    "stance_runs": [
      {
        "politics": "liberal",
        "stance": "Many guns that are sold in ...",
        "contrary_stance": "Few guns that are sold in ...",
        "point_pool": ["point 1", "point 2", "..."],
        "point_pool_prompt": "the prompt used to generate points",
        "pairs": [
          {
            "pair_index": 0,
            "points": ["5 sampled points for this pair"],
            "essays": [
              {
                "tone": "strong",
                "tone_variant": "the specific tone instruction used",
                "rhetorical_approach": "the structural approach used",
                "essay": "the full essay text",
                "prompt": "the exact prompt sent to the LLM"
              },
              { "tone": "weak", "..." : "..." }
            ]
          }
        ]
      }
    ]
  }
]
```

The file consumed in production by the Qualtrics loader is `generation/all_essays.json`, served via GitHub Pages.

## Issues and stances

| Issue | Liberal stance | Conservative stance |
|-------|---------------|-------------------|
| Guns | Many guns sold in the US are sold without a background check | Few guns sold in the US are sold without a background check |
| Abortion | Few women regret having had an elective abortion | Many women regret having had an elective abortion |
