#!/usr/bin/env python3
"""
Generate persuasive 5-paragraph essays that vary in tone (weak vs strong)
while holding factual content constant. Two-step process:
  1. Generate 5 key points for a stance.
  2. Turn those points into paired essays (one strong, one weak) using the same points.

Each pair of essays shares a unique set of 5 points. The --essays-per-tone argument
controls how many pairs are generated (each pair = 1 strong + 1 weak essay).

The 'politics' variable (liberal / conservative) determines which stance
is argued *for* and which is the contrary stance on each topic.

Usage (generate n pairs per orientation for all topics & both political orientations):
  python generate_arguments.py --all --both-politics --essays-per-tone 5 --out all_essays.json --model gpt-5.2
"""

import os
import json
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config: issues, stances, and tone statements (from your spec)
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

# TONES = {
#     "strong": "Your task is to write in a style that is as combative and unsparing of the other side as possible.",
#     "weak": "Your task is to write in a style that is as friendly and sympathetic to the other side as possible.",
# }

TONES = {
    "strong": "Your task is to write in as confident and self-assured a style as possible. You know without a doubt that you are right and the other side is wrong.",
    "weak": "Your task is to write in as tentative and careful a style as possible. You understand that this is a delicate issue, and that it's important not to overclaim.",
}


def stances_for(issue_key: str, politics: str) -> tuple[str, str]:
    """Return (stance, contrary_stance) for a given issue and political orientation."""
    issue = ISSUES[issue_key]
    contrary = "conservative" if politics == "liberal" else "liberal"
    return issue[politics], issue[contrary]

# ---------------------------------------------------------------------------
# Prompts (from your spec)
# ---------------------------------------------------------------------------

POINT_GENERATION_PROMPT = """Please list five of the strongest points that can be made arguing the following issue.

Issue: {stance}

Respond with exactly 5 numbered points (1. ... 2. ... etc.). No other text."""

ESSAY_PROMPT = """Please write a short, five-paragraph essay that makes the following 5 points. Your audience believes that {contrary_stance}. Your goal is to persuade them that {stance}.

1. {point_1}
2. {point_2}
3. {point_3}
4. {point_4}
5. {point_5}

Each paragraph should be separated by a blank line, but do not include a header or any other formatting. {tone_statement}"""

def get_client():
    """OpenAI client; uses OPENAI_API_KEY from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return OpenAI()


def generate_points(client: OpenAI, stance: str, model: str = "gpt-4o-mini") -> tuple[list[str], str]:
    """Step 1: Get 5 key points for the given stance. Returns (points, prompt_used)."""
    prompt = POINT_GENERATION_PROMPT.format(stance=stance)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    # Parse "1. ... 2. ..." into a list (robust to minor formatting)
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading "1.", "2.", etc.
        for i in range(1, 10):
            prefix = f"{i}."
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line:
            points.append(line)
    return points[:5], prompt


def generate_essay(
    client: OpenAI,
    stance: str,
    contrary_stance: str,
    points: list[str],
    tone_statement: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """Step 2: Turn the 5 points into a 5-paragraph essay with the given tone. Returns (essay, prompt_used)."""
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
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip(), prompt


def run_one(
    client: OpenAI,
    issue_key: str,
    tone_key: str,
    politics: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Generate one essay: one issue, one tone, one political orientation."""
    stance, contrary_stance = stances_for(issue_key, politics)
    tone_statement = TONES[tone_key]

    points, point_prompt = generate_points(client, stance, model=model)
    essay, essay_prompt = generate_essay(
        client, stance, contrary_stance, points, tone_statement, model=model
    )

    return {
        "issue": issue_key,
        "politics": politics,
        "tone": tone_key,
        "stance": stance,
        "contrary_stance": contrary_stance,
        "points": points,
        "point_generation_prompt": point_prompt,
        "essay": essay,
        "essay_generation_prompt": essay_prompt,
    }


def run_topic(
    client: OpenAI,
    issue_key: str,
    politics_list: list[str] | None = None,
    essays_per_tone: int = 3,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    For one topic: generate `essays_per_tone` pairs of essays.
    Each pair shares a unique set of 5 points, with one strong and one weak essay.

    politics_list controls which orientations to generate for (default: both).
    """
    if politics_list is None:
        politics_list = list(POLITICS)

    stance_runs = []
    for politics in politics_list:
        stance, contrary_stance = stances_for(issue_key, politics)

        # Generate N pairs: each pair has its own set of points + one strong & one weak essay
        pairs = []
        for pair_idx in range(essays_per_tone):
            print(f"  [{issue_key} / {politics}] Pair {pair_idx + 1}/{essays_per_tone}: Generating 5 points ...", flush=True)
            points, point_prompt = generate_points(client, stance, model=model)
            if len(points) < 5:
                raise ValueError(f"Expected 5 points for {issue_key} / {politics}, got {len(points)}")
            print(f"  [{issue_key} / {politics}] Pair {pair_idx + 1}/{essays_per_tone}: Got {len(points)} points.", flush=True)

            # Generate one essay per tone (strong and weak) using these points
            essays = []
            for tone_key in TONES:
                tone_statement = TONES[tone_key]
                print(f"  [{issue_key} / {politics}] Pair {pair_idx + 1}/{essays_per_tone}: Generating {tone_key} essay ...", flush=True)
                essay, essay_prompt = generate_essay(
                    client, stance, contrary_stance, points, tone_statement, model=model
                )
                essays.append({"tone": tone_key, "essay": essay, "prompt": essay_prompt})

            pairs.append({
                "pair_index": pair_idx,
                "points": points,
                "point_generation_prompt": point_prompt,
                "essays": essays,
            })

        stance_runs.append({
            "politics": politics,
            "stance": stance,
            "contrary_stance": contrary_stance,
            "pairs": pairs,
        })

    if len(stance_runs) == 1:
        r = stance_runs[0]
        return {
            "issue": issue_key,
            "politics": r["politics"],
            "stance": r["stance"],
            "contrary_stance": r["contrary_stance"],
            "pairs": r["pairs"],
        }
    return {"issue": issue_key, "stance_runs": stance_runs}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate persuasive essays (2-step: points → essay).")
    parser.add_argument("--issue", choices=list(ISSUES), default="guns", help="Issue: guns or abortion")
    parser.add_argument("--tone", choices=list(TONES), default="weak", help="Tone: weak or strong (used only without --per-topic)")
    parser.add_argument("--politics", choices=list(POLITICS), default="liberal", help="Political orientation: liberal or conservative")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--out", default=None, help="Optional JSON output file")
    parser.add_argument("--all", action="store_true", help="Run per-topic for all issues (one point set per topic, then N essays per tone)")
    parser.add_argument("--per-topic", action="store_true", help="One point set for this topic, then N essays per tone (weak and strong)")
    parser.add_argument("--both-politics", action="store_true", help="With --all or --per-topic: generate for both political orientations per topic")
    parser.add_argument("--essays-per-tone", type=int, default=3, metavar="N", help="With --per-topic or --all: generate N pairs of essays, each pair sharing unique points (default 3)")
    args = parser.parse_args()

    client = get_client()

    # Per-topic mode: one set of points per stance, then N essays per tone
    if args.all or args.per_topic:
        if args.all:
            issue_keys = list(ISSUES)
        else:
            issue_keys = [args.issue]

        politics_list = list(POLITICS) if args.both_politics else [args.politics]

        results = []
        for issue_key in issue_keys:
            print(f"Topic: {issue_key} — politics={politics_list}, generating {args.essays_per_tone} pairs per orientation ...")
            result = run_topic(
                client,
                issue_key,
                politics_list=politics_list,
                essays_per_tone=args.essays_per_tone,
                model=args.model,
            )
            results.append(result)
            if "stance_runs" in result:
                total_pairs = sum(len(sr["pairs"]) for sr in result["stance_runs"])
                print(f"  → {total_pairs} pairs ({len(result['stance_runs'])} orientations × {args.essays_per_tone} pairs, each with 1 strong + 1 weak)")
            else:
                print(f"  → {len(result['pairs'])} pairs ({args.essays_per_tone} pairs, each with 1 strong + 1 weak)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Wrote to {args.out}")
        for r in results:
            if "stance_runs" in r:
                for sr in r["stance_runs"]:
                    for pair in sr["pairs"]:
                        print("\n" + "=" * 60)
                        s = sr["stance"]
                        print(f"Issue: {r['issue']} | Politics: {sr['politics']} | Pair: {pair['pair_index'] + 1}")
                        print(f"Stance: {s[:60] + '...' if len(s) > 60 else s}")
                        print("Points:")
                        for i, p in enumerate(pair["points"], 1):
                            print(f"  {i}. {p}")
                        print("=" * 60)
                        for e in pair["essays"]:
                            print(f"\n--- {e['tone'].upper()} ---\n{e['essay']}\n")
            else:
                for pair in r["pairs"]:
                    print("\n" + "=" * 60)
                    print(f"Issue: {r['issue']} | Politics: {r['politics']} | Pair: {pair['pair_index'] + 1}")
                    print(f"Stance: {r['stance']}")
                    print("Points:")
                    for i, p in enumerate(pair["points"], 1):
                        print(f"  {i}. {p}")
                    print("=" * 60)
                    for e in pair["essays"]:
                        print(f"\n--- {e['tone'].upper()} ---\n{e['essay']}\n")
        return

    # Single-essay mode (original behavior)
    result = run_one(client, args.issue, args.tone, args.politics, model=args.model)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote to {args.out}")
    print("Points:")
    for i, p in enumerate(result["points"], 1):
        print(f"  {i}. {p}")
    print("\nEssay:")
    print(result["essay"])


if __name__ == "__main__":
    main()
