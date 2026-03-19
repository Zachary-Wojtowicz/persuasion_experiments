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
        "Your task is to write in as confident and self-assured a style as possible. You know without a doubt that you are right and the other side is wrong.",
        "Write as a passionate advocate who is deeply convinced of this position. Present your case with forceful certainty.",
        "Write as though you are presenting irrefutable evidence in a debate. Your tone should convey absolute conviction and leave no room for doubt.",
        "Write with the directness and authority of someone who has studied this issue thoroughly and finds the opposing view indefensible.",
        "Write as a confident policy analyst delivering findings you consider beyond dispute. Be assertive and unwavering.",
    ],
    "weak": [
        "Your task is to write in as tentative and careful a style as possible. You understand that this is a delicate issue, and that it's important not to overclaim.",
        "Write as a thoughtful academic who weighs evidence carefully and acknowledges uncertainty. Be measured and cautious in your claims.",
        "Write as a careful journalist presenting evidence to a skeptical audience. Hedge appropriately and respect the complexity of the issue.",
        "Write with the restraint of someone who genuinely respects the difficulty of this topic and wants to be fair to all perspectives while still making a case.",
        "Write as though presenting preliminary findings at a research seminar — be honest about limitations and careful not to overstate conclusions.",
    ],
}

RHETORICAL_APPROACHES = [
    "Open with a concrete example or real-world scenario, and use it as a thread throughout the essay.",
    "Open with a direct question to the reader that challenges their assumption, then answer it through the essay.",
    "Lead with the single strongest piece of evidence first, then build supporting arguments around it.",
    "Open by finding genuine common ground with the reader's position, then gradually pivot to show where it falls short.",
    "Structure the essay as a series of increasingly specific and surprising observations, building toward the strongest point at the end.",
    "Open with a historical or comparative framing that puts the issue in broader context.",
    "Open by vividly describing a specific situation or setting where the issue plays out in practice.",
    "Begin by conceding what the opposing view gets right, then show why the full picture leads to a different conclusion.",
]


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

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """Please identify {num_points} compelling arguments in favor of the following position.

Position: {stance}

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """What are {num_points} persuasive reasons someone might give to support the following claim?

Claim: {stance}

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """Please provide {num_points} distinct pieces of evidence or reasoning that support the following stance.

Stance: {stance}

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",

    """List {num_points} different arguments — drawing on facts, statistics, and logical reasoning — for the following position.

Position: {stance}

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text.""",
]

ESSAY_PROMPT = """Please write a short, five-paragraph essay that makes the following 5 points. Your audience believes that {contrary_stance}. Your goal is to persuade them that {stance}.

1. {point_1}
2. {point_2}
3. {point_3}
4. {point_4}
5. {point_5}

Each paragraph should be separated by a blank line, but do not include a header or any other formatting. Avoid formulaic essay patterns — use a distinctive opening, varied paragraph transitions, and do not simply enumerate points in order. {rhetorical_approach} {tone_statement}"""

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 5
_RETRY_DELAY = 10  # seconds


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    async def complete(self, prompt: str, model: str, temperature: float = 0.7) -> str:
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

    async def complete(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except self._transient_errors:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_RETRY_DELAY)


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

    async def complete(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await self._client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp.content[0].text.strip()
            except self._transient_errors:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_RETRY_DELAY)


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
# Generation helpers
# ---------------------------------------------------------------------------


async def generate_points(
    client: LLMClient, stance: str, num_points: int = 5, model: str = "gpt-4o-mini", temperature: float = 0.7
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
    rhetorical_approach: str,
    model: str = "gpt-4o-mini",
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
        rhetorical_approach=rhetorical_approach,
        tone_statement=tone_statement,
    )
    text = await client.complete(prompt, model, temperature=temperature)
    return text, prompt


async def run_one(
    client: LLMClient,
    issue_key: str,
    tone_key: str,
    politics: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> dict:
    """Generate one essay: one issue, one tone, one political orientation."""
    stance, contrary_stance = stances_for(issue_key, politics)
    tone_statement = random.choice(TONE_VARIANTS[tone_key])
    rhetorical_approach = random.choice(RHETORICAL_APPROACHES)

    points, point_prompt = await generate_points(client, stance, model=model, temperature=temperature)
    essay, essay_prompt = await generate_essay(
        client, stance, contrary_stance, points, tone_statement, rhetorical_approach,
        model=model, temperature=temperature
    )

    return {
        "issue": issue_key,
        "politics": politics,
        "tone": tone_key,
        "tone_variant": tone_statement,
        "rhetorical_approach": rhetorical_approach,
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
        rhetorical_approach = random.choice(RHETORICAL_APPROACHES)
        print(f"  [{issue_key} / {politics}] Pair {pair_idx + 1}/{num_essays}: Generating {tone_key} essay ...", flush=True)
        essay, essay_prompt = await generate_essay(
            client, stance, contrary_stance, points, tone_statement, rhetorical_approach,
            model=model, temperature=temperature
        )
        return {
            "tone": tone_key,
            "tone_variant": tone_statement,
            "rhetorical_approach": rhetorical_approach,
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
    model: str = "gpt-4o-mini",
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
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default 1.0)")
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

    results = []
    for issue_key in issue_keys:
        print(f"Topic: {issue_key} — politics={politics_list}, generating {args.num_essays} essay pairs per orientation (in parallel) ...")
        result = await run_topic(
            client,
            issue_key,
            politics_list=politics_list,
            num_essays=args.num_essays,
            model=model,
            temperature=args.temperature,
            pool_size=args.pool_size,
        )
        results.append(result)
        total_pairs = sum(len(sr["pairs"]) for sr in result["stance_runs"])
        num_orientations = len(result["stance_runs"])
        print(f"  -> {total_pairs} pairs ({num_orientations} orientations x {args.num_essays} pairs, each with 1 strong + 1 weak)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote to {args.out}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
