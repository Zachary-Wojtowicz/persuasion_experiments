# Persuasion Arguments Generation

Generates paired persuasive essays that vary in tone (strong vs. weak) while holding factual content constant, providing stimuli for an experiment on how the confidence/strength of an argument's tone affects persuasion on contested political claims (gun background checks; abortion regret).

![Status](https://img.shields.io/badge/status-data_collection-yellow)
![Preregistration](https://img.shields.io/badge/preregistered-no-lightgrey)

## Methods
- **Design:** 2 (tone: strong vs. weak) × 2 (issue: guns vs. abortion) × 2 (political orientation of stance: liberal vs. conservative) stimulus set. Within each pair, both essays share the same 5 argument points sampled from a per-stance pool of 15 (C(15,5) = 3,003 combinations) and a single rhetorical approach drawn from 8 structural options; only confidence-signaling language (boosters vs. hedges) differs. The tone variant, rhetorical approach, and full LLM prompt are saved alongside each essay for traceability.
- **Sample:** Stimuli only. Participant data collection happens in a separate Qualtrics survey that fetches `generation/all_essays.json` (served via GitHub Pages at `https://Zachary-Wojtowicz.github.io/persuasion_experiments/generation/all_essays.json`) and assigns one matching essay per participant via `generation/qualtrics_essay_loader.js`.
- **Measures:** Per-essay metadata in `generation/all_essays.json` (`tone`, `tone_variant`, `rhetorical_approach`, `points`, `prompt`). The Qualtrics loader writes embedded fields on each participant's response: `essay`, `essay_issue`, `essay_politics`, `essay_tone`, `essay_pair_index`, `essay_stance`, `essay_contrary_stance`. Outcome and attention-check items live in the Qualtrics survey, not in this repo.
- **Analysis approach:** Not in this repo — analysis lives downstream of the Qualtrics survey.

## Reproducing the analysis

```bash
cd generation
pip install -r requirements.txt

# Provide API key(s) via a .env file in the project root:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...

# Regenerate the full stimulus set (defaults: Claude Opus 4.6, all issues × politics × tones,
# 3 essay pairs per cell, temperature 1.0).
python generate_arguments.py --out all_essays.json

# Reproducible run with a fixed seed.
python generate_arguments.py --seed 42 --out all_essays.json

# Restrict to one cell or one tone (single-essay mode).
python generate_arguments.py --issue guns --politics liberal --tone weak
```

Key CLI flags: `--issue {guns,abortion}`, `--politics {liberal,conservative}`, `--tone {strong,weak}`, `--num-essays` (pairs per cell, default 3), `--provider {anthropic,openai}`, `--model`, `--temperature` (default 1.0), `--seed`, `--pool-size` (default 15), `--out`.

## Repository structure

```
generation/
├── generate_arguments.py        # stimulus generation script
├── qualtrics_essay_loader.js    # client-side loader run by the Qualtrics survey
├── requirements.txt
├── all_essays.json              # current stimulus set served via GitHub Pages
├── essays_final.{json,md,pdf}   # frozen stimulus set snapshot
.nojekyll                        # keeps GitHub Pages from filtering the generation/ directory
```
