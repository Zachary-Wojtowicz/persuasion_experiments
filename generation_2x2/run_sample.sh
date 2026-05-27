#!/usr/bin/env bash
# Generate a small confidence x brashness (2x2) sample set for inspection.
# Each "quad" = 4 essays (one per cell). Overridable via env vars, e.g.:
#   NUM_ESSAYS=3 ISSUE=abortion POLITICS=conservative ./run_sample.sh
#
# Defaults: guns, BOTH orientations, 1 quad each -> 8 essays (2 pol x 1 quad x 4 cells).
# Requires ANTHROPIC_API_KEY in the environment or a .env in this directory.
set -euo pipefail
cd "$(dirname "$0")"

ISSUE="${ISSUE:-guns}"
NUM_ESSAYS="${NUM_ESSAYS:-1}"
SEED="${SEED:-42}"
MODEL="${MODEL:-claude-opus-4-6}"
OUT="${OUT:-sample_${ISSUE}.json}"

ARGS=(--issue "$ISSUE" --num-essays "$NUM_ESSAYS" --seed "$SEED" --model "$MODEL" --out "$OUT")
# Restrict to one orientation only if POLITICS is set; otherwise generate both.
[ -n "${POLITICS:-}" ] && ARGS+=(--politics "$POLITICS")

echo "Running: python generate_arguments.py ${ARGS[*]}"
python generate_arguments.py "${ARGS[@]}"
echo "Wrote sample to generation_2x2/$OUT"
