#!/bin/bash
# Run full Phase 1 analysis on all downloaded LEMON subjects
# Usage: bash scripts/run_full_analysis.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Checking data ==="
N_SUBS=$(ls -d data/lemon/sub-*/RSEEG/*.vhdr 2>/dev/null | wc -l | tr -d ' ')
echo "Found $N_SUBS subjects with EEG data"

if [ "$N_SUBS" -lt 50 ]; then
    echo "ERROR: Only $N_SUBS subjects found. Expected 200+. Is download complete?"
    exit 1
fi

echo ""
echo "=== Running Phase 1 PTA pipeline ==="
echo "Output: results/phase1_full/"
echo "Started: $(date)"
echo ""

python3 -m src.phase1.run_pta \
    --data-dir data/lemon/ \
    --output results/phase1_full/ \
    2>&1 | tee results/phase1_full/pipeline.log

echo ""
echo "=== Completed: $(date) ==="
