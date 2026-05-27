import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

print(f"Config: {json.dumps(data['config'], indent=2)}\n")

total_pairs = 0
passed = 0
strong_openings = []
weak_openings = []
approaches_used = {}

for result in data['results']:
    for sr in result['stance_runs']:
        cell = f"{result['issue']}/{sr['politics']}"
        cell_pairs = len(sr['pairs'])
        cell_passed = 0

        for pair in sr['pairs']:
            total_pairs += 1
            v = pair.get('validation', {})
            if v.get('passed'):
                passed += 1
                cell_passed += 1

            ra = pair.get('rhetorical_approach', 'N/A')
            approaches_used[ra[:50]] = approaches_used.get(ra[:50], 0) + 1

            for e in pair['essays']:
                first_sent = e['essay'].split('.')[0][:100]
                if e['tone'] == 'strong':
                    strong_openings.append((cell, first_sent))
                else:
                    weak_openings.append((cell, first_sent))

        print(f"{cell}: {cell_pairs} pairs, {cell_passed}/{cell_pairs} passed validation")

print(f"\nTOTAL: {total_pairs} pairs, {passed}/{total_pairs} passed validation ({100*passed/total_pairs:.0f}%)")

print(f"\n--- Rhetorical approach distribution ---")
for k, v in sorted(approaches_used.items(), key=lambda x: -x[1]):
    print(f"  {v:2d}x  {k}...")

print(f"\n--- Strong opening diversity (first 40 chars) ---")
seen = {}
for cell, opener in strong_openings:
    key = opener[:40]
    seen[key] = seen.get(key, 0) + 1
print(f"  {len(seen)} unique clusters (out of {len(strong_openings)} essays)")
for k, v in sorted(seen.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v:2d}x  \"{k}...\"")

print(f"\n--- Weak opening diversity (first 40 chars) ---")
seen = {}
for cell, opener in weak_openings:
    key = opener[:40]
    seen[key] = seen.get(key, 0) + 1
print(f"  {len(seen)} unique clusters (out of {len(weak_openings)} essays)")
for k, v in sorted(seen.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v:2d}x  \"{k}...\"")

strong_b, strong_h, weak_b, weak_h = [], [], [], []
for result in data['results']:
    for sr in result['stance_runs']:
        for pair in sr['pairs']:
            v = pair.get('validation', {})
            if v:
                strong_b.append(v['strong']['booster_count'])
                strong_h.append(v['strong']['hedge_count'])
                weak_b.append(v['weak']['booster_count'])
                weak_h.append(v['weak']['hedge_count'])

print(f"\n--- Booster/hedge counts (mean [min-max]) ---")
print(f"  Strong: boosters={sum(strong_b)/len(strong_b):.1f} [{min(strong_b)}-{max(strong_b)}], hedges={sum(strong_h)/len(strong_h):.1f} [{min(strong_h)}-{max(strong_h)}]")
print(f"  Weak:   boosters={sum(weak_b)/len(weak_b):.1f} [{min(weak_b)}-{max(weak_b)}], hedges={sum(weak_h)/len(weak_h):.1f} [{min(weak_h)}-{max(weak_h)}]")
