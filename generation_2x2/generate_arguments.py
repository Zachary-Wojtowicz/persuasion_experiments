#!/usr/bin/env python3
"""
Generate persuasive 5-paragraph essays that vary on TWO orthogonal stylistic
axes — CONFIDENCE (how certain the writer is that the claims are true) and
BRASHNESS (how the writer treats people who disagree) — while holding factual
content constant. Two-step process:
  1. Generate a pool of key points for a stance.
  2. Sample 5 points per quad and turn them into a matched set of 4 essays
     (one per cell of the confidence x brashness 2x2).

Each *quad* of essays shares a unique set of 5 points AND a single rhetorical
approach, so the only things that vary within a quad are the two stylistic
axes. The --num-essays argument controls how many quads are generated per
combination (each quad = 4 essays: {low,high} confidence x {low,high} brashness).

The 'politics' variable (liberal / conservative) determines which stance
is argued *for* and which is the contrary stance on each topic.

Usage:
  # Default: all topics, both politics, 3 quads each (Anthropic claude-opus-4-6)
  python generate_arguments.py --out essays.json

  # Generate 25 quads per combination (= 100 essays per combination)
  python generate_arguments.py --num-essays 25 --out essays.json

  # Restrict to one issue or one political orientation
  python generate_arguments.py --issue guns --out essays.json
  python generate_arguments.py --politics liberal --out essays.json

  # Use OpenAI instead
  python generate_arguments.py --provider openai --model gpt-4o --out essays.json

  # Single essay mode (specify --confidence and/or --brashness for one essay)
  python generate_arguments.py --confidence high --brashness low --issue guns --politics liberal

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
# Config: issues, stances, and the two stylistic axes
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

# --- Axis 1: CONFIDENCE -----------------------------------------------------
# Epistemic stance toward the PROPOSITION: how sure the writer is that the
# claims are true. Governed by boosters vs. hedges. Says nothing about how the
# writer treats people who disagree (that is the brashness axis). Each variant
# explicitly disclaims the other axis so the two can be combined cleanly,
# including in the "unnatural" cells (e.g., tentative-yet-brash).
CONFIDENCE_VARIANTS = {
    "high": [
        "You are maximally confident in the TRUTH of the claims you are making and want to avoid under-asserting yourself. Your very first sentence must signal certainty about the facts immediately. Write in a maximally certain style throughout: present every claim as established fact. Use intensifiers and epistemic boosters: 'clearly,' 'undeniably,' 'the evidence proves,' 'without question,' 'it is obvious,' 'there is no doubt.' NEVER use hedges like 'might,' 'perhaps,' 'it seems,' 'some evidence suggests,' 'it is possible,' or 'arguably.' This constraint governs ONLY how certain you are that the claims are true — it says nothing about how warmly or harshly you treat people who disagree."
    ],
    "low": [
        "You are tentative and epistemically cautious about the TRUTH of the claims you are making, and want to avoid over-asserting yourself. Your very first sentence must signal open-mindedness about whether the claim is correct. Write in a maximally tentative style throughout: present claims as provisional. Use hedges and qualifiers: 'it seems,' 'the evidence suggests,' 'one might argue,' 'it is possible that,' 'perhaps,' 'arguably.' NEVER use boosters like 'clearly,' 'undeniably,' 'without question,' or 'the facts prove.' This constraint governs ONLY how certain you are that the claims are true — it says nothing about how warmly or harshly you treat people who disagree."
    ],
}

# --- Axis 2: BRASHNESS ------------------------------------------------------
# Interpersonal stance toward people who hold the opposing view (audience-
# directed). The low level is ACTIVE warmth/charity, not mere neutrality —
# this is deliberate, to counteract the dismissiveness that confidence boosters
# tend to smuggle in. Says nothing about how certain the writer is (that is the
# confidence axis).
BRASHNESS_VARIANTS = {
    "high": [
        "Adopt a brash, dismissive interpersonal stance toward the reader and anyone who holds the opposing view. Treat the opposing position as not worth taking seriously, and make clear you think people who hold it — including the reader — are mistaken and ought to know better. Address the reader directly and tell them their current view is wrong (e.g., 'if you believe otherwise, you've been misled,' 'anyone still clinging to this hasn't thought it through'). Be contemptuous and condescending toward the opposing side. This constraint governs ONLY your interpersonal stance toward people who disagree — it says nothing about how certain you are that your claims are true."
    ],
    "low": [
        "Adopt a warm, respectful, and charitable interpersonal stance toward the reader and anyone who holds the opposing view. Explicitly acknowledge that reasonable, intelligent people hold the opposing position for understandable reasons, and validate the reader's perspective even as you make your case (e.g., 'it's completely understandable to see it that way,' 'thoughtful people land on the other side of this'). Never belittle, mock, or talk down to the reader or the opposing side. This constraint governs ONLY your interpersonal stance toward people who disagree — it says nothing about how certain you are that your claims are true."
    ],
}


# Rhetorical approaches are shared across all cells of a quad to avoid
# confounding argument structure with the stylistic axes. All phrasing is
# tone-neutral: it specifies WHAT to open with, not HOW confidently or how
# warmly to say it.
RHETORICAL_APPROACHES = [
    "Open with a concrete example or real-world scenario, and use it as a thread throughout the essay.",
    "Open with a direct question to the reader, then answer it through the essay.",
    "Lead with the single strongest piece of evidence first, then build supporting arguments around it.",
    "Structure the essay as a series of increasingly specific observations, building toward the strongest point at the end.",
    "Open with a historical or comparative framing that puts the issue in broader context.",
    "Open by vividly describing a specific situation or setting where the issue plays out in practice.",
    "Open with a clear statement of your position, then support it.",
    "Open by noting what the opposing view gets right, then make the case for why the full picture leads elsewhere.",
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

IMPORTANT: Do NOT include specific percentages, fractions, or frequencies (e.g., "22%", "one in five", "nearly a quarter"). Arguments should rely on reasoning, mechanisms, and qualitative evidence rather than citing specific numerical estimates.

Respond with exactly {num_points} numbered points (1. ... 2. ... etc.). No other text."""
]

SYSTEM_PROMPT = """You are writing persuasive essays for a research study on communication styles. You must strictly follow the stylistic constraints given in each request. Do not break character, add meta-commentary, or explain what you are doing. Do not include a title, header, or any formatting beyond paragraph breaks."""

ESSAY_PROMPT = """Write a short persuasive essay of about five paragraphs that makes the following 5 points. Your audience believes that {contrary_stance}. Your goal is to persuade them that {stance}.

1. {point_1}
2. {point_2}
3. {point_3}
4. {point_4}
5. {point_5}

Each paragraph should be separated by a blank line. The essay should be approximately 500 words. Do not simply enumerate points in order.

IMPORTANT CONSTRAINTS:
- Do NOT add caveats about methodology, sample sizes, or evidence quality beyond what the points themselves state.
- Present the same scope of claims regardless of the stylistic constraints below — do not inflate or deflate the magnitude of evidence (e.g., do not change 'many' to 'vast' or 'some' to 'a few').
- Use a similar mix of sentence lengths regardless of the stylistic constraints below — do not write exclusively in short punchy sentences or exclusively in long complex ones.

The following two stylistic constraints are INDEPENDENT dimensions. Honor BOTH fully and simultaneously, even when they feel unusual in combination (e.g., tentative about the facts while still being dismissive of opponents, or highly certain about the facts while still being warm toward opponents).

STYLISTIC CONSTRAINT — CERTAINTY (how sure you are that the claims are true): {confidence_statement}

STYLISTIC CONSTRAINT — INTERPERSONAL STANCE (how you treat people who disagree): {brashness_statement}

STRUCTURAL APPROACH (follow this precisely — it determines how you open and organize the essay): {rhetorical_approach}"""


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

    def __init__(self, max_concurrent: int | None = None):
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
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def complete(self, prompt: str, model: str, temperature: float = 0.7, system: str | None = None) -> str:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                messages = [{"role": "user", "content": prompt}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})
                # Cap concurrent in-flight requests so we stay under the
                # provider's RPM ceiling. Held only during the request itself
                # (not during retry backoff).
                if self._sem is not None:
                    async with self._sem:
                        resp = await self._client.chat.completions.create(
                            model=model, messages=messages, temperature=temperature,
                        )
                else:
                    resp = await self._client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
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

    def __init__(self, max_concurrent: int | None = None):
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
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

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
                # Cap concurrent in-flight requests so we stay under the
                # provider's RPM ceiling. Held only during the request itself
                # (not during retry backoff).
                if self._sem is not None:
                    async with self._sem:
                        resp = await self._client.messages.create(**kwargs)
                else:
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


def get_client(provider: str = "openai", max_concurrent: int | None = None) -> LLMClient:
    """Get an LLM client for the specified provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider](max_concurrent=max_concurrent)

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
# Manipulation-check validation (automated pre-check, not the human rating)
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

# Markers of a brash / dismissive / audience-directed interpersonal stance.
# Heuristic only — the real manipulation check is the human-rated brashness
# items. Used here as a cheap separation pre-check before fielding.
BRASH_MARKERS = {
    "foolish", "naive", "ignorant", "misguided", "deluded", "delusion",
    "absurd", "ridiculous", "nonsense", "laughable", "obtuse", "willful",
    "willfully", "kidding themselves", "fooling themselves", "haven't thought",
    "hasn't thought", "been misled", "you've been misled", "should know better",
    "wake up", "frankly", "let's be honest", "let us be honest", "anyone who",
    "if you still", "if you believe", "refuse to see", "refuse to accept",
    "don't understand", "do not understand", "fail to grasp", "fails to grasp",
    "no excuse", "brainwashed", "propaganda", "so-called", "supposedly",
    "talking point", "talking points", "cling to", "clinging to",
}


def score_confidence(text: str) -> dict:
    """Count boosters and hedges in an essay. Returns counts and the matches."""
    lower = text.lower()
    booster_hits = [w for w in BOOSTERS if w in lower]
    hedge_hits = [w for w in HEDGES if w in lower]
    return {
        "booster_count": len(booster_hits),
        "hedge_count": len(hedge_hits),
        "boosters_found": booster_hits,
        "hedges_found": hedge_hits,
    }


def score_brashness(text: str) -> dict:
    """Count brash/dismissive markers in an essay. Returns count and matches."""
    lower = text.lower()
    brash_hits = [w for w in BRASH_MARKERS if w in lower]
    return {
        "brash_count": len(brash_hits),
        "brash_found": brash_hits,
    }


def validate_quad(essays_by_cell: dict[tuple[str, str], str]) -> dict:
    """Check that confidence and brashness separate as intended across a quad.

    essays_by_cell maps (confidence_level, brashness_level) -> essay text for
    the four cells. We confirm (a) high-confidence cells carry more boosters
    (or low-confidence cells carry more hedges) and (b) high-brashness cells
    carry more brash markers — pooling over the orthogonal axis.
    """
    cells: dict[str, dict] = {}
    for (conf, brash), text in essays_by_cell.items():
        c = score_confidence(text)
        b = score_brashness(text)
        cells[f"{conf}_{brash}"] = {
            "confidence": conf,
            "brashness": brash,
            "booster_count": c["booster_count"],
            "hedge_count": c["hedge_count"],
            "boosters_found": c["boosters_found"],
            "hedges_found": c["hedges_found"],
            "brash_count": b["brash_count"],
            "brash_found": b["brash_found"],
        }

    def pooled(axis: str, level: str, metric: str) -> float:
        vals = [v[metric] for v in cells.values() if v[axis] == level]
        return sum(vals) / len(vals) if vals else 0.0

    hi_boost, lo_boost = pooled("confidence", "high", "booster_count"), pooled("confidence", "low", "booster_count")
    hi_hedge, lo_hedge = pooled("confidence", "high", "hedge_count"), pooled("confidence", "low", "hedge_count")
    passed_confidence = (hi_boost > lo_boost) or (lo_hedge > hi_hedge)

    hi_brash, lo_brash = pooled("brashness", "high", "brash_count"), pooled("brashness", "low", "brash_count")
    passed_brashness = hi_brash > lo_brash

    return {
        "passed": passed_confidence and passed_brashness,
        "passed_confidence": passed_confidence,
        "passed_brashness": passed_brashness,
        "cells": cells,
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
    confidence_statement: str,
    brashness_statement: str,
    rhetorical_approach: str,
    model: str = "claude-opus-4-6",
    temperature: float = 0.7,
) -> tuple[str, str]:
    """Turn 5 points into a 5-paragraph essay with the given confidence and
    brashness. Returns (essay, prompt_used)."""
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
        confidence_statement=confidence_statement,
        brashness_statement=brashness_statement,
        rhetorical_approach=rhetorical_approach,
    )
    text = await client.complete(prompt, model, temperature=temperature, system=SYSTEM_PROMPT)
    return text, prompt


async def run_one(
    client: LLMClient,
    issue_key: str,
    confidence_key: str,
    brashness_key: str,
    politics: str,
    model: str = "claude-opus-4-6",
    temperature: float = 0.7,
) -> dict:
    """Generate one essay: one issue, one confidence level, one brashness level, one orientation."""
    stance, contrary_stance = stances_for(issue_key, politics)
    confidence_statement = random.choice(CONFIDENCE_VARIANTS[confidence_key])
    brashness_statement = random.choice(BRASHNESS_VARIANTS[brashness_key])
    rhetorical_approach = random.choice(RHETORICAL_APPROACHES)

    points, point_prompt = await generate_points(client, stance, model=model, temperature=temperature)
    essay, essay_prompt = await generate_essay(
        client, stance, contrary_stance, points,
        confidence_statement, brashness_statement, rhetorical_approach,
        model=model, temperature=temperature
    )

    return {
        "issue": issue_key,
        "politics": politics,
        "confidence": confidence_key,
        "brashness": brashness_key,
        "confidence_variant": confidence_statement,
        "brashness_variant": brashness_statement,
        "rhetorical_approach": rhetorical_approach,
        "stance": stance,
        "contrary_stance": contrary_stance,
        "points": points,
        "point_generation_prompt": point_prompt,
        "essay": essay,
        "essay_generation_prompt": essay_prompt,
    }


# The four cells of the confidence x brashness 2x2.
QUAD_CELLS = [(conf, brash) for conf in CONFIDENCE_VARIANTS for brash in BRASHNESS_VARIANTS]


async def _generate_quad(
    client: LLMClient,
    issue_key: str,
    politics: str,
    stance: str,
    contrary_stance: str,
    points: list[str],
    quad_idx: int,
    num_quads: int,
    model: str,
    temperature: float = 0.7,
) -> dict:
    """Generate one quad: the 4 confidence x brashness essays from pre-selected
    points and a single shared rhetorical approach (all 4 in parallel)."""

    # Every cell of the quad shares the SAME rhetorical approach and points,
    # so only the two stylistic axes vary within the quad.
    rhetorical_approach = random.choice(RHETORICAL_APPROACHES)

    async def _gen_essay(confidence_key: str, brashness_key: str) -> dict:
        confidence_statement = random.choice(CONFIDENCE_VARIANTS[confidence_key])
        brashness_statement = random.choice(BRASHNESS_VARIANTS[brashness_key])
        print(
            f"  [{issue_key} / {politics}] Quad {quad_idx + 1}/{num_quads}: "
            f"Generating conf={confidence_key}/brash={brashness_key} essay ...",
            flush=True,
        )
        essay, essay_prompt = await generate_essay(
            client, stance, contrary_stance, points,
            confidence_statement, brashness_statement, rhetorical_approach,
            model=model, temperature=temperature
        )
        return {
            "confidence": confidence_key,
            "brashness": brashness_key,
            "confidence_variant": confidence_statement,
            "brashness_variant": brashness_statement,
            "rhetorical_approach": rhetorical_approach,
            "essay": essay,
            "prompt": essay_prompt,
        }

    essays = await asyncio.gather(*[_gen_essay(c, b) for (c, b) in QUAD_CELLS])

    return {
        "quad_index": quad_idx,
        "points": points,
        "rhetorical_approach": rhetorical_approach,
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
    For one topic: generate `num_essays` quads of essays per orientation.
    Each quad shares a unique set of 5 points and one rhetorical approach,
    with one essay per cell of the confidence x brashness 2x2.

    politics_list controls which orientations to generate for (default: both).
    Orientations and quads are generated in parallel.
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

        # Step 2: Sample 5 points per quad and generate the 4 essays in parallel
        quad_tasks = []
        for quad_idx in range(num_essays):
            sampled = random.sample(point_pool, 5)
            quad_tasks.append(
                _generate_quad(
                    client, issue_key, politics, stance, contrary_stance,
                    sampled, quad_idx, num_essays, model, temperature
                )
            )
        quads = await asyncio.gather(*quad_tasks)
        quads = sorted(quads, key=lambda q: q["quad_index"])

        return {
            "politics": politics,
            "stance": stance,
            "contrary_stance": contrary_stance,
            "point_pool": point_pool,
            "point_pool_prompt": point_prompt,
            "quads": quads,
        }

    stance_runs = await asyncio.gather(*[_run_orientation(p) for p in politics_list])
    return {"issue": issue_key, "stance_runs": list(stance_runs)}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def async_main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate persuasive essays (2-step: points -> confidence x brashness 2x2).")
    parser.add_argument("--issue", choices=list(ISSUES), default=None, help="Restrict to one issue (default: all)")
    parser.add_argument("--confidence", choices=list(CONFIDENCE_VARIANTS), default=None, help="Single-essay mode: confidence level (low/high)")
    parser.add_argument("--brashness", choices=list(BRASHNESS_VARIANTS), default=None, help="Single-essay mode: brashness level (low/high)")
    parser.add_argument("--politics", choices=list(POLITICS), default=None, help="Restrict to one political orientation (default: both)")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="anthropic", help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default=None, help="Model name (default: claude-opus-4-6)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out", default=None, help="Optional JSON output file")
    parser.add_argument("--num-essays", type=int, default=3, metavar="N", help="Number of essay quads per issue/politics combination; each quad = 4 essays (default 3)")
    parser.add_argument("--pool-size", type=int, default=15, metavar="N", help="Number of points to generate in the pool before sampling 5 per quad (default 15)")
    parser.add_argument("--max-concurrent", type=int, default=20, metavar="N", help="Cap on concurrent in-flight API requests (default 20; lower this if you hit RPM limits, raise it if you have a higher tier)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from dotenv import load_dotenv
    load_dotenv()

    model = args.model or DEFAULT_MODELS[args.provider]
    client = get_client(args.provider, max_concurrent=args.max_concurrent)

    # Single-essay mode: triggered when --confidence and/or --brashness is given.
    # The unspecified axis defaults (confidence -> high, brashness -> low).
    if args.confidence or args.brashness:
        issue = args.issue or "guns"
        politics = args.politics or "liberal"
        confidence = args.confidence or "high"
        brashness = args.brashness or "low"
        result = await run_one(client, issue, confidence, brashness, politics, model=model, temperature=args.temperature)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote to {args.out}")
        else:
            print(json.dumps(result, indent=2))
        return

    # Batch mode (default): all issues x all politics x the full 2x2
    issue_keys = [args.issue] if args.issue else list(ISSUES)
    politics_list = [args.politics] if args.politics else list(POLITICS)

    for issue_key in issue_keys:
        print(f"Topic: {issue_key} — politics={politics_list}, generating {args.num_essays} essay quads per orientation (in parallel) ...")

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

    # Validate the confidence x brashness manipulation per quad
    total_quads = 0
    failed_conf = 0
    failed_brash = 0
    for result in results:
        for sr in result["stance_runs"]:
            for quad in sr["quads"]:
                total_quads += 1
                essays_by_cell = {(e["confidence"], e["brashness"]): e["essay"] for e in quad["essays"]}
                v = validate_quad(essays_by_cell)
                quad["validation"] = v
                if not v["passed_confidence"]:
                    failed_conf += 1
                if not v["passed_brashness"]:
                    failed_brash += 1

    print(f"\nValidation: {total_quads} quads")
    print(f"  Confidence separation: {total_quads - failed_conf}/{total_quads} passed")
    print(f"  Brashness separation:  {total_quads - failed_brash}/{total_quads} passed")
    if failed_conf or failed_brash:
        print("  WARNING: some quads show weak separation on an axis — consider regenerating.")

    # Crossed manipulation-check table: a clean 2x2 has boosters tracking the
    # confidence axis only, and brash markers tracking the brashness axis only.
    cell_agg: dict[str, dict[str, list]] = {}
    for result in results:
        for sr in result["stance_runs"]:
            for quad in sr["quads"]:
                for ck, cv in quad["validation"]["cells"].items():
                    a = cell_agg.setdefault(ck, {"boost": [], "hedge": [], "brash": []})
                    a["boost"].append(cv["booster_count"])
                    a["hedge"].append(cv["hedge_count"])
                    a["brash"].append(cv["brash_count"])

    print("\n--- Crossed manipulation check (mean markers per essay) ---")
    print(f"  {'cell (conf_brash)':<20} {'boosters':>9} {'hedges':>7} {'brash':>6}")
    for ck in ("low_low", "low_high", "high_low", "high_high"):
        if ck in cell_agg:
            a = cell_agg[ck]
            print(f"  {ck:<20} {_mean(a['boost']):>9.1f} {_mean(a['hedge']):>7.1f} {_mean(a['brash']):>6.1f}")

    for result in results:
        result_quads = sum(len(sr["quads"]) for sr in result["stance_runs"])
        num_orientations = len(result["stance_runs"])
        print(f"  [{result['issue']}] -> {result_quads} quads "
              f"({num_orientations} orientations x {args.num_essays} quads, each with 4 essays: confidence x brashness)")
    if args.out:
        output = {
            "config": {
                "provider": args.provider,
                "model": model,
                "temperature": args.temperature,
                "seed": args.seed,
                "pool_size": args.pool_size,
                "num_essays": args.num_essays,
                "design": "2x2 confidence x brashness",
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
