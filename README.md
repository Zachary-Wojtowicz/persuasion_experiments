# Persuasion Arguments Generation

Generates matched persuasive essays that vary on two orthogonal stylistic axes — how *confident* the writer is that the claims are true, and how *brash* the writer is toward people who disagree — while holding factual content constant, providing stimuli for an experiment on how an argument's confidence and interpersonal tone affect persuasion on contested political claims (gun background checks; abortion regret).

![Status](https://img.shields.io/badge/status-data_collection-yellow)
![Preregistration](https://img.shields.io/badge/preregistered-no-lightgrey)

## Methods
- **Design:** Current set is a 2 (confidence: low vs. high) × 2 (brashness: low vs. high) within-*quad* stimulus design, crossed with 2 issues (guns vs. abortion) × 2 political orientations of the stance (liberal vs. conservative). Within each quad, all 4 essays share the same 5 argument points sampled from a per-stance pool of 15 (C(15,5) = 3,003 combinations) and a single rhetorical approach drawn from 8 structural options; only confidence-signaling language (boosters vs. hedges) and interpersonal brashness (dismissive vs. warm/charitable) vary across the cells. Each variant prompt explicitly disclaims the other axis so the two can be combined cleanly, including the "unnatural" cells (e.g., tentative-yet-brash). A prior single-axis design (tone: strong vs. weak) is retained under `generation/`; the active design lives in `generation_2x2/`.
- **Sample:** Stimuli only. The production set `generation_2x2/essays_final_2x2.json` contains 320 essays (20 quads × 2 issues × 2 political orientations × 4 cells; seed 2026, Claude Opus 4.6). Participant data collection happens in a separate Qualtrics survey that fetches this JSON (served via GitHub Pages at `https://Zachary-Wojtowicz.github.io/persuasion_experiments/generation_2x2/essays_final_2x2.json`) and assigns one matching essay per participant via `generation_2x2/qualtrics_essay_loader.js`. A follow-up bundle `generation_followup/qualtrics_essays.json` (self-describing: `schema_version`, `corpus_sha256`, `essay_count`) contains 112 essays that carry the same confidence × brashness 2×2 to 7 numeric-estimation claims (e.g. antidepressant placebo improvement, unarmed police killings, divorce regret), each argued in two directions (higher vs. lower) across 2 quads.
- **Measures:** Per-essay metadata in the generated JSON (`confidence`, `brashness`, `confidence_variant`, `brashness_variant`, `rhetorical_approach`, `prompt`). The Qualtrics loader writes embedded fields on each participant's response: `essay`, `essay_issue`, `essay_politics`, `essay_confidence`, `essay_brashness`, `essay_quad_index`, `essay_stance`, `essay_contrary_stance` (the prior single `tone` field is replaced by `confidence` + `brashness`). Outcome and attention-check items live in the Qualtrics survey, not in this repo.
- **Analysis approach:** Not in this repo — analysis lives downstream of the Qualtrics survey. `analyze_essays.py` performs only stimulus-side manipulation checks (per-cell booster/hedge/brashness marker counts, opening diversity).

## Reproducing the analysis

```bash
cd generation_2x2
pip install -r ../generation/requirements.txt   # anthropic, openai, python-dotenv

# Provide API key(s) via a .env file in this directory:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...

# Regenerate the full stimulus set (defaults: Claude Opus 4.6, all issues × politics,
# confidence × brashness 2x2, temperature 1.0). The committed set used 20 quads, seed 2026.
python generate_arguments.py --num-essays 20 --seed 2026 --out essays_final_2x2.json

# Inspect / manipulation-check a generated set.
python analyze_essays.py essays_final_2x2.json

# Small inspection sample (defaults: guns, both orientations, 1 quad each -> 8 essays).
./run_sample.sh

# Single-essay mode: pin one cell.
python generate_arguments.py --confidence high --brashness low --issue guns --politics liberal
```

Key CLI flags: `--issue {guns,abortion}`, `--politics {liberal,conservative}`, `--confidence {low,high}`, `--brashness {low,high}`, `--num-essays` (quads per cell, default 3; each quad = 4 essays), `--provider {anthropic,openai}`, `--model`, `--temperature` (default 1.0), `--seed`, `--pool-size` (default 15), `--max-concurrent` (default 20), `--out`.

## Repository structure

```
generation_2x2/                  # active design: confidence x brashness 2x2
├── generate_arguments.py        # stimulus generation script
├── analyze_essays.py            # per-cell manipulation-check inspector
├── qualtrics_essay_loader.js    # client-side loader run by the Qualtrics survey
├── run_sample.sh                # convenience wrapper for a small inspection sample
├── essays_final_2x2.{json,md}   # production stimulus set served via GitHub Pages
├── full_pilot.{json,md}         # pilot stimulus set
└── sample_guns.{json,md}        # small inspection sample
generation_followup/             # follow-up stimulus bundle for a second Qualtrics survey
└── qualtrics_essays.json        # 112 essays: confidence x brashness 2x2 on 7 numeric-estimation claims
generation/                      # prior single-axis design (tone: strong vs. weak)
├── generate_arguments.py
├── qualtrics_essay_loader.js
├── requirements.txt             # shared dependency list
└── essays_final.{json,md,pdf}   # frozen single-axis stimulus snapshot
.nojekyll                        # keeps GitHub Pages from filtering these directories
```
