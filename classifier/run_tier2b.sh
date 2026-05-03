#!/usr/bin/env bash
# Tier-2b: L-universality head-to-head.
# Same 4 candidates as tier-1; sweep over L; modest seeds + trials per L
# so the total wall budget stays under ~2.5 h.
set -e
source .venv/bin/activate
mkdir -p results/density_classification/2026-04-30

for L in 64 96 192 256; do
  case $L in
    64)
      SEEDS=8; TRIALS=64; MULT=32
      ;;
    96)
      SEEDS=8; TRIALS=64; MULT=32
      ;;
    192)
      SEEDS=4; TRIALS=32; MULT=32
      ;;
    256)
      SEEDS=4; TRIALS=32; MULT=32
      ;;
  esac
  echo "================================================================"
  echo "L=$L  seeds=$SEEDS  trials/side=$TRIALS  max-mult=$MULT"
  echo "================================================================"
  python3 -u classifier/head_to_head_benchmark.py \
    --candidates-file classifier/h2h_candidates_tier1.json \
    --L $L --delta 0.02 \
    --n-seeds $SEEDS --n-random-per-side $TRIALS \
    --adv-densities 0.01 0.02 0.05 \
    --max-steps-mult $MULT \
    --output results/density_classification/2026-04-30/h2h_tier2b_L${L}.json
done
echo "tier-2b complete"
