#!/usr/bin/env python3
"""
Generate persuasive 5-paragraph essays that vary in tone (weak vs strong)
while holding factual content constant. Two-step process:
  1. Generate a pool of key points for a stance.
  2. Sample 5 points per pair and turn them into paired essays (one strong, one weak).

Each pair of essays shares a unique set of 5 points. The --num-essays argument
controls how many pairs are generated per combination (each pair = 1 strong + 1 weak essay).

The 'politics' variable (liberal / conservative) determines which stance
is argued *for* and which is the contrary stance on each topic.

Usage:
  # Default: all topics, both politics, 3 essay pairs each (Anthropic claude-opus-4-6)
  python generate_arguments.py --out essays.json

  # Generate 25 essays per combination
  python generate_arguments.py --num-essays 25 --out essays.json

  # Restrict to one issue or one political orientation
  python generate_arguments.py --issue guns --out essays.json
  python generate_arguments.py --politics liberal --out essays.json

  # Use OpenAI instead
  python generate_arguments.py --provider openai --model gpt-4o --out essays.json

  # Single essay mode (specify --tone to generate just one essay)
  python generate_arguments.py --tone weak --issue guns --politics liberal

  # Set a seed for reproducible random sampling
  python generate_arguments.py --seed 42 --out essays.json

Environment variables:
  ANTHROPIC_API_KEY - Required when using --provider anthropic (default)
  OPENAI_API_KEY    - Required when using --provider openai
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import random
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Config: issues, stances, and tone statements
# ---------------------------------------------------------------------------

ISSUES = {
    "guns": {
        "liberal": "Many guns that are sold in the United States are sold without a background check.",
        "conservative": "Few guns that are sold in the United States are sold without a background check.",
    },
    "abortion": {
        "liberal": "Few women regret having had an elective abortion.",
        "conservative": "Many women regret having had an elective abortion.",
    },
}

POLITICS = ("liberal", "conservative")

TONE_VARIANTS = {
    "strong": [
        "Your very first sentence must make a bold, confident claim that signals certainty immediately — do not open with 'Consider' or a question. Write in a maximally confident style throughout. Use intensifiers and boosters: 'clearly,' 'undeniably,' 'the evidence proves,' 'without question,' 'it is obvious,' 'there is no doubt.' NEVER use hedges like 'might,' 'perhaps,' 'it seems,' 'some evidence suggests,' 'it is possible,' or 'arguably.' Present every claim as established fact.",
        "Open with a direct, declarative statement of fact — no questions, no 'consider,' no hedging. Write with absolute conviction throughout. Use language of certainty: 'the data conclusively shows,' 'it is beyond dispute,' 'any honest reading makes clear,' 'the facts speak for themselves.' Avoid ALL epistemic hedging — no 'could,' 'may,' 'tends to,' 'on balance,' or 'it appears.'",
        "Begin with an unqualified assertion that leaves no doubt about where you stand. Write as though the case is completely settled throughout. Use declarative, unqualified assertions: 'the reality is,' 'the truth is straightforward,' 'the evidence is overwhelming.' Strip out any tentativeness — no 'roughly,' 'around,' 'in some cases,' or 'one might argue.'",
        "Start with a strong, definitive claim — not a question or an invitation to 'consider.' Write with forceful directness throughout. Use strong epistemic markers: 'we know,' 'it has been proven,' 'no serious person disputes,' 'the research is unambiguous.' Do not use approximators, shields, or any concessive language.",
        "Lead with an assertive, unhedged statement of your position. Write with the confidence of someone presenting irrefutable evidence throughout. Use boosters freely: 'plainly,' 'unquestionably,' 'in fact,' 'demonstrably,' 'the record is clear.' Purge all hedging devices — no 'might,' 'perhaps,' 'possibly,' 'it could be argued.'",
    ],
    "weak": [
        "Your very first sentence must signal tentativeness — use a hedge like 'it seems,' 'it appears,' 'one might argue,' or 'there are reasons to think.' Write in a maximally tentative style throughout. Use hedges and qualifiers: 'it seems,' 'the evidence suggests,' 'one might argue,' 'it is possible that,' 'perhaps,' 'arguably.' NEVER use boosters like 'clearly,' 'undeniably,' 'without question,' or 'the facts prove.'",
        "Open with an explicitly cautious or uncertain framing — signal from the very first words that this is a tentative argument. Write with careful epistemic humility throughout. Use language of uncertainty: 'the data appears to indicate,' 'there is reason to think,' 'on balance,' 'it could be that.' Avoid ALL certainty markers — no 'proves,' 'demonstrates conclusively,' 'beyond dispute.'",
        "Begin with a hedged, qualified claim that acknowledges uncertainty from the outset. Write as though presenting preliminary findings throughout. Use shields and approximators: 'roughly,' 'approximately,' 'in many cases,' 'tends to,' 'some research suggests.' Strip out any language of certainty — no 'the truth is,' 'it has been proven,' or 'unquestionably.'",
        "Start with a tentative observation or a question that invites reflection rather than stating a conclusion. Write with measured restraint throughout. Use epistemic qualifiers: 'may,' 'could,' 'it appears that,' 'the available evidence points toward,' 'while not definitive.' Do not use forceful assertions or declarative certainty.",
        "Lead with a careful, hedged opening that makes clear you are exploring rather than declaring. Write with the caution of someone who respects the complexity of this issue throughout. Use tentative phrasing: 'it is worth considering,' 'there are grounds to believe,' 'the picture that emerges,' 'on one reading of the evidence.' Purge all boosters.",
    ],
}



def stances_for(issue_key: str, politics: str) -> tuple[str, str]:
    """Return (stance, contrary_stance) for a given issue and political orientation."""
    issue = ISSUES[issue_key]
    contrary = "conservative" if politics == "liberal" else "liberal"
    return issue[politics], issue[contrary]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

POINT_GENERATION_PROMPTS = [
    """Please list {num_points} of the strongest points that can be made arguing the following issue.

Issue: {stance}

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """Please identify {num_points} compelling arguments in favor of the following position.

Position: {stance}

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """What are {num_points} persuasive reasons someone might give to support the following claim?

Claim: {stance}

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """Please provide {num_points} distinct pieces of evidence or reasoning that support the following stance.

Stance: {stance}

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """List {num_points} different arguments — drawing on facts, statistics, and logical reasoning — for the following position.

Position: {stance}

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",
]

SYSTEM_PROMPT = """You are writing persuasive essays for a research study on communication styles. You must strictly follow the stylistic constraints given in each request. Do not break character, add meta-commentary, or explain what you are doing. Do not include a title, header, or any formatting beyond paragraph breaks."""

ESSAY_PROMPT = """Write a short persuasive essay of about five paragraphs that makes the following 5 points. Your audience believes that {contrary_stance}. Your goal is to persuade them that {stance}.

1. {point_1}
2. {point_2}
3. {point_3}
4. {point_4}
5. {point_5}

Each paragraph should be separated by a blank line. The essay should be approximately 500 words. Use a distinctive opening, varied paragraph transitions, and do not simply enumerate points in order.

IMPORTANT CONSTRAINTS that apply regardless of stylistic instructions below:
- Do NOT use hostile, aggressive, dismissive, or emotionally charged language (no 'dangerous,' 'reckless,' 'alarming,' 'devastating,' 'shocking,' 'absurd'). Maintain a respectful, engaged tone toward the reader.
- Do NOT add caveats about methodology, sample sizes, or evidence quality beyond what the points themselves state.
- Present the same scope of claims regardless of confidence level — do not inflate or deflate the magnitude of evidence (e.g., do not change 'many' to 'vast' or 'some' to 'a few').
- Use a similar mix of sentence lengths regardless of confidence level — do not write exclusively in short punchy sentences or exclusively in long complex ones.

STYLISTIC CONSTRAINT (follow this precisely): {tone_statement}"""

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY = 15  # seconds (doubles each attempt: 15, 30, 60, 120, 240)


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    async def complete(self, prompt: str, model: str, temperature: float = 0.7, system: str | None = None) -> str:
        """Send a prompt and return the completion text."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client wrapper."""

    def __init__(self):
        import openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self._client = openai.AsyncOpenAI()
        self._transient_errors = (
            openai.InternalServerError,
            openai.RateLimitError,
            openai.APIConnectionError,
        )

    async def complete(self, prompt: str, model: str, temperature: float = 0.7, system: str | None = None) -> str:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                messages = [{"role": "user", "content": prompt}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except self._transient_errors as e:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, _RETRY_BASE_DELAY * 2)
                print(f"[retry {attempt + 1}/{_RETRY_ATTEMPTS}] {type(e).__name__} — waiting {delay:.0f}s before retrying")
                await asyncio.sleep(delay)


class AnthropicClient(LLMClient):
    """Anthropic API client wrapper."""

    def __init__(self):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")
        self._client = anthropic.AsyncAnthropic()
        self._transient_errors = (
            anthropic.InternalServerError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        )

    async def complete(self, prompt: str, model: str, temperature: float = 0.7, system: str | None = None) -> str:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                kwargs = dict(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                if system:
                    kwargs["system"] = system
                resp = await self._client.messages.create(**kwargs)
                return resp.content[0].text.strip()
            except self._transient_errors as e:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, _RETRY_BASE_DELAY * 2)
                print(f"[retry {attempt + 1}/{_RETRY_ATTEMPTS}] {type(e).__name__} — waiting {delay:.0f}s before retrying")
                await asyncio.sleep(delay)


PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-6",
}


def get_client(provider: str = "openai") -> LLMClient:
    """Get an LLM client for the specified provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider]()

# ---------------------------------------------------------------------------
# Point parsing
# ---------------------------------------------------------------------------

_NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*(.+)")


def _parse_points(text: str, num_points: int) -> list[str]:
    """Parse numbered points from LLM output, handling multi-line points."""
    points: list[str] = []
    current: str | None = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            if current is not None:
                points.append(current)
            current = m.group(1)
        elif current is not None:
            # Continuation line — append to current point
            current += " " + line
    if current is not None:
        points.append(current)
    return points[:num_points]

# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------

BOOSTERS = {
    "clearly", "undeniably", "without question", "obviously", "plainly",
    "unquestionably", "demonstrably", "in fact", "without doubt", "beyond dispute",
    "irrefutable", "indisputable", "undoubtedly", "conclusively", "incontrovertible",
}

HEDGES = {
    "might", "perhaps", "possibly", "arguably", "seemingly",
    "it seems", "it appears", "it is possible", "on balance", "tends to",
    "suggests", "may", "could", "roughly", "approximately",
}


def score_confidence(text: str) -> dict:
    """Count boosters and hedges in an essay. Returns counts and a simple score."""
    lower = text.lower()
    booster_hits = [w for w in BOOSTERS if w in lower]
    hedge_hits = [w for w in HEDGES if w in lower]
    return {
        "booster_count": len(booster_hits),
        "hedge_count": len(hedge_hits),
        "boosters_found": booster_hits,
        "hedges_found": hedge_hits,
    }


def validate_pair(strong_essay: str, weak_essay: str) -> dict:
    """Check that a strong essay has more boosters and fewer hedges than the weak one."""
    s = score_confidence(strong_essay)
    w = score_confidence(weak_essay)
    passed = (s["booster_count"] > w["booster_count"]) or (w["hedge_count"] > s["hedge_count"])
    return {
        "passed": passed,
        "strong": s,
        "weak": w,
    }

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


async def generate_points(
    client: LLMClient, stance: str, num_points: int = 5, model: str = "claude-opus-4-6", temperature: float = 0.7
) -> tuple[list[str], str]:
    """Generate key points for the given stance using a randomly selected prompt variant. Returns (points, prompt_used)."""
    prompt_template = random.choice(POINT_GENERATION_PROMPTS)
    prompt = prompt_template.format(stance=stance, num_points=num_points)
    text = await client.complete(prompt, model, temperature=temperature)
    return _parse_points(text, num_points), prompt


async def generate_essay(
    client: LLMClient,
    stance: str,
    contrary_stance: str,
    points: list[str],
    tone_statement: str,
    model: str = "claude-opus-4-6",
    temperature: float = 0.7,
) -> tuple[str, str]:
    """Turn 5 points into a 5-paragraph essay with the given tone. Returns (essay, prompt_used)."""
    if len(points) < 5:
        raise ValueError(f"Need exactly 5 points, got {len(points)}")
    prompt = ESSAY_PROMPT.format(
        stance=stance,
        contrary_stance=contrary_stance,
        point_1=points[0],
        point_2=points[1],
        point_3=points[2],
        point_4=points[3],
        point_5=points[4],
        tone_statement=tone_statement,
    )
    text = await client.complete(prompt, model, temperature=temperature, system=SYSTEM_PROMPT)
    return text, prompt


async def run_one(
    client: LLMClient,
    issue_key: str,
    tone_key: str,
    politics: str,
    model: str = "claude-opus-4-6",
    temperature: float = 0.7,
) -> dict:
    """Generate one essay: one issue, one tone, one political orientation."""
    stance, contrary_stance = stances_for(issue_key, politics)
    tone_statement = random.choice(TONE_VARIANTS[tone_key])

    points, point_prompt = await generate_points(client, stance, model=model, temperature=temperature)
    essay, essay_prompt = await generate_essay(
        client, stance, contrary_stance, points, tone_statement,
        model=model, temperature=temperature
    )

    return {
        "issue": issue_key,
        "politics": politics,
        "tone": tone_key,
        "tone_variant": tone_statement,
        "stance": stance,
        "contrary_stance": contrary_stance,
        "points": points,
        "point_generation_prompt": point_prompt,
        "essay": essay,
        "essay_generation_prompt": essay_prompt,
    }


async def _generate_pair(
    client: LLMClient,
    issue_key: str,
    politics: str,
    stance: str,
    contrary_stance: str,
    points: list[str],
    pair_idx: int,
    num_essays: int,
    model: str,
    temperature: float = 0.7,
) -> dict:
    """Generate one pair: strong/weak essays from pre-selected points (in parallel)."""

    async def _gen_essay(tone_key: str) -> dict:
        tone_statement = random.choice(TONE_VARIANTS[tone_key])
        print(f"  [{issue_key} / {politics}] Pair {pair_idx + 1}/{num_essays}: Generating {tone_key} essay ...", flush=True)
        essay, essay_prompt = await generate_essay(
            client, stance, contrary_stance, points, tone_statement,
            model=model, temperature=temperature
        )
        return {
            "tone": tone_key,
            "tone_variant": tone_statement,
            "essay": essay,
            "prompt": essay_prompt,
        }

    essays = await asyncio.gather(*[_gen_essay(tk) for tk in TONE_VARIANTS])

    return {
        "pair_index": pair_idx,
        "points": points,
        "essays": list(essays),
    }


async def run_topic(
    client: LLMClient,
    issue_key: str,
    politics_list: list[str] | None = None,
    num_essays: int = 3,
    model: str = "claude-opus-4-6",
    temperature: float = 0.7,
    pool_size: int = 15,
) -> dict:
    """
    For one topic: generate `num_essays` pairs of essays per orientation.
    Each pair shares a unique set of 5 points, with one strong and one weak essay.

    politics_list controls which orientations to generate for (default: both).
    Orientations and pairs are generated in parallel.
    """
    if politics_list is None:
        politics_list = list(POLITICS)

    async def _run_orientation(politics: str) -> dict:
        stance, contrary_stance = stances_for(issue_key, politics)

        # Step 1: Generate a large pool of diverse points
        print(f"  [{issue_key} / {politics}] Generating pool of {pool_size} points ...", flush=True)
        point_pool, point_prompt = await generate_points(
            client, stance, num_points=pool_size, model=model, temperature=temperature
        )
        print(f"  [{issue_key} / {politics}] Got {len(point_pool)} points in pool.", flush=True)

        if len(point_pool) < 5:
            raise ValueError(f"Need at least 5 points in pool for {issue_key}/{politics}, got {len(point_pool)}")

        # Step 2: Sample 5 points per pair and generate essays in parallel
        pair_tasks = []
        for pair_idx in range(num_essays):
            sampled = random.sample(point_pool, 5)
            pair_tasks.append(
                _generate_pair(
                    client, issue_key, politics, stance, contrary_stance,
                    sampled, pair_idx, num_essays, model, temperature
                )
            )
        pairs = await asyncio.gather(*pair_tasks)
        pairs = sorted(pairs, key=lambda p: p["pair_index"])

        return {
            "politics": politics,
            "stance": stance,
            "contrary_stance": contrary_stance,
            "point_pool": point_pool,
            "point_pool_prompt": point_prompt,
            "pairs": pairs,
        }

    stance_runs = await asyncio.gather(*[_run_orientation(p) for p in politics_list])
    return {"issue": issue_key, "stance_runs": list(stance_runs)}


async def async_main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate persuasive essays (2-step: points -> essay).")
    parser.add_argument("--issue", choices=list(ISSUES), default=None, help="Restrict to one issue (default: all)")
    parser.add_argument("--tone", choices=list(TONE_VARIANTS), default=None, help="Single-essay mode: generate one essay with this tone")
    parser.add_argument("--politics", choices=list(POLITICS), default=None, help="Restrict to one political orientation (default: both)")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="anthropic", help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default=None, help="Model name (default: claude-opus-4-6)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default 0.8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out", default=None, help="Optional JSON output file")
    parser.add_argument("--num-essays", type=int, default=3, metavar="N", help="Number of essay pairs per issue/politics combination (default 3)")
    parser.add_argument("--pool-size", type=int, default=15, metavar="N", help="Number of points to generate in the pool before sampling 5 per pair (default 15)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from dotenv import load_dotenv
    load_dotenv()

    model = args.model or DEFAULT_MODELS[args.provider]
    client = get_client(args.provider)

    # Single-essay mode: only when --tone is explicitly specified
    if args.tone:
        issue = args.issue or "guns"
        politics = args.politics or "liberal"
        result = await run_one(client, issue, args.tone, politics, model=model, temperature=args.temperature)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote to {args.out}")
        return

    # Batch mode (default): all issues x all politics x both tones
    issue_keys = [args.issue] if args.issue else list(ISSUES)
    politics_list = [args.politics] if args.politics else list(POLITICS)

    for issue_key in issue_keys:
        print(f"Topic: {issue_key} — politics={politics_list}, generating {args.num_essays} essay pairs per orientation (in parallel) ...")

    results = await asyncio.gather(*[
        run_topic(
            client, issue_key,
            politics_list=politics_list,
            num_essays=args.num_essays,
            model=model,
            temperature=args.temperature,
            pool_size=args.pool_size,
        )
        for issue_key in issue_keys
    ])
    results = list(results)

    # Validate confidence manipulation
    total_pairs = 0
    failed_pairs = 0
    for result in results:
        for sr in result["stance_runs"]:
            for pair in sr["pairs"]:
                total_pairs += 1
                essays_by_tone = {e["tone"]: e["essay"] for e in pair["essays"]}
                if "strong" in essays_by_tone and "weak" in essays_by_tone:
                    v = validate_pair(essays_by_tone["strong"], essays_by_tone["weak"])
                    pair["validation"] = v
                    if not v["passed"]:
                        failed_pairs += 1
    print(f"\nValidation: {total_pairs - failed_pairs}/{total_pairs} pairs passed confidence check.")
    if failed_pairs:
        print(f"  WARNING: {failed_pairs} pair(s) may have weak confidence separation — consider regenerating.")

    for result in results:
        result_pairs = sum(len(sr["pairs"]) for sr in result["stance_runs"])
        num_orientations = len(result["stance_runs"])
        print(f"  [{result['issue']}] -> {result_pairs} pairs ({num_orientations} orientations x {args.num_essays} pairs, each with 1 strong + 1 weak)")
    if args.out:
        output = {
            "config": {
                "provider": args.provider,
                "model": model,
                "temperature": args.temperature,
                "seed": args.seed,
                "pool_size": args.pool_size,
                "num_essays": args.num_essays,
            },
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote to {args.out}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
