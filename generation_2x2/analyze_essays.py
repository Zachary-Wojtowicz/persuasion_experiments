"""Post-hoc inspector for a 2x2 (confidence x brashness) essay set.

Usage:  python analyze_essays.py <essays.json>

Reads the JSON produced by generate_arguments.py and reports:
  - per (issue, politics) quad counts and per-axis validation pass rates,
  - rhetorical-approach distribution,
  - the crossed marker table (boosters / hedges / brash per cell),
  - opening diversity per cell (a sanity check on within-cell variety).
"""
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

print(f"Config: {json.dumps(data['config'], indent=2)}\n")

ORDER = [("low", "low"), ("low", "high"), ("high", "low"), ("high", "high")]

total_quads = 0
passed_conf = 0
passed_brash = 0
approaches_used: dict[str, int] = {}
cell_openings: dict[tuple[str, str], list[tuple[str, str]]] = {}
cell_markers: dict[str, dict[str, list[int]]] = {}

for result in data["results"]:
    for sr in result["stance_runs"]:
        path = f"{result['issue']}/{sr['politics']}"
        quads = sr["quads"]
        sr_passed_conf = 0
        sr_passed_brash = 0

        for quad in quads:
            total_quads += 1
            v = quad.get("validation", {})
            if v.get("passed_confidence"):
                passed_conf += 1
                sr_passed_conf += 1
            if v.get("passed_brashness"):
                passed_brash += 1
                sr_passed_brash += 1

            ra = quad.get("rhetorical_approach", "N/A")
            approaches_used[ra[:50]] = approaches_used.get(ra[:50], 0) + 1

            for e in quad["essays"]:
                key = (e["confidence"], e["brashness"])
                first_sent = e["essay"].split(".")[0][:100]
                cell_openings.setdefault(key, []).append((path, first_sent))

            for ck, cv in v.get("cells", {}).items():
                m = cell_markers.setdefault(ck, {"boost": [], "hedge": [], "brash": []})
                m["boost"].append(cv["booster_count"])
                m["hedge"].append(cv["hedge_count"])
                m["brash"].append(cv["brash_count"])

        print(
            f"{path}: {len(quads)} quads, "
            f"conf={sr_passed_conf}/{len(quads)}, "
            f"brash={sr_passed_brash}/{len(quads)}"
        )

print(f"\nTOTAL: {total_quads} quads")
if total_quads:
    print(f"  Confidence separation: {passed_conf}/{total_quads} ({100*passed_conf/total_quads:.0f}%)")
    print(f"  Brashness  separation: {passed_brash}/{total_quads} ({100*passed_brash/total_quads:.0f}%)")

print("\n--- Rhetorical approach distribution ---")
for k, v in sorted(approaches_used.items(), key=lambda x: -x[1]):
    print(f"  {v:3d}x  {k}...")


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


print("\n--- Crossed manipulation check (mean markers per essay) ---")
print(f"  {'cell (conf_brash)':<20} {'boosters':>9} {'hedges':>7} {'brash':>6}  {'n':>4}")
for (c, b) in ORDER:
    ck = f"{c}_{b}"
    m = cell_markers.get(ck, {"boost": [], "hedge": [], "brash": []})
    n = len(m["boost"])
    print(
        f"  {ck:<20} "
        f"{_mean(m['boost']):>9.1f} "
        f"{_mean(m['hedge']):>7.1f} "
        f"{_mean(m['brash']):>6.1f}  "
        f"{n:>4}"
    )

print("\n--- Opening diversity per cell (first 40 chars of first sentence) ---")
for (c, b) in ORDER:
    key = (c, b)
    if key not in cell_openings:
        continue
    openings = cell_openings[key]
    seen: dict[str, int] = {}
    for _path, opener in openings:
        k = opener[:40]
        seen[k] = seen.get(k, 0) + 1
    print(f"\n  conf={c} brash={b} — {len(seen)} unique clusters / {len(openings)} essays")
    for k, v in sorted(seen.items(), key=lambda x: -x[1])[:6]:
        print(f"    {v:3d}x  \"{k}...\"")
