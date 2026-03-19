# Persuasion Arguments Generation

Generate paired persuasive essays that vary in tone (strong vs. weak) while holding factual content constant. Designed to produce stimuli for experiments studying how argument strength affects persuasion.

## How it works

The generation follows a two-step process:

1. **Point pool generation** — For each issue/stance, the script generates a large pool of argument points (default 15) using a randomly selected prompt variant for diversity.
2. **Essay generation** — For each essay pair, 5 points are randomly sampled from the pool and turned into two essays: one strong-toned and one weak-toned. Each essay also receives a randomly selected rhetorical approach (e.g., "open with a question," "lead with strongest evidence") and tone instruction variant to maximize stylistic diversity across essays.

### Variation mechanisms

The script uses several strategies to ensure essays differ from each other:

- **Point generation prompt variants** — Multiple differently-worded prompts for generating argument points, selected at random.
- **Point pool sampling** — A pool of 15 points is generated once per stance; each essay pair draws a random subset of 5, giving C(15,5) = 3,003 possible combinations.
- **Tone instruction variants** — Each tone (strong/weak) has 5 different phrasings that convey the same confidence level but suggest different rhetorical registers (e.g., "passionate advocate" vs. "confident policy analyst").
- **Rhetorical approach rotation** — 8 structural approaches (e.g., "open with a scenario," "begin by conceding common ground") are randomly assigned per essay.
- **Anti-formula nudge** — Every essay prompt includes an instruction to avoid formulaic patterns and vary transitions.

### Experimental design

The primary experimental variable is **tone** (`strong` / `weak`). Within each pair, both essays share the same 5 argument points — only the tone differs. The specific tone variant, rhetorical approach, and full prompt used are saved alongside each essay in the output for traceability.

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

## Issues and stances

| Issue | Liberal stance | Conservative stance |
|-------|---------------|-------------------|
| Guns | Many guns sold in the US are sold without a background check | Few guns sold in the US are sold without a background check |
| Abortion | Few women regret having had an elective abortion | Many women regret having had an elective abortion |
