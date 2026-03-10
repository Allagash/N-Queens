#!/usr/bin/env bash
set -euo pipefail

N=19
BINARY="${1:-./nq}"
OUTDIR="${2:-bench}"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

run_case() {
  local label="$1"
  local persist_blocks="$2"
  local log="$OUTDIR/n19_${label}.log"

  echo "=== N=19 $label ==="
  if [[ -z "$persist_blocks" ]]; then
    {
      echo "=== N=19 $label ==="
      NQ_SOLVER_MODE=gpu \
      NQ_GPU_TIMING=1 \
      NQ_FRONTIER_DEPTH=6 \
      NQ_GPU_BATCH_TASKS=500000 \
      "$BINARY" "$N"
    } 2>&1 | tee "$log"
  else
    {
      echo "=== N=19 $label ==="
      NQ_SOLVER_MODE=gpu \
      NQ_GPU_TIMING=1 \
      NQ_FRONTIER_DEPTH=6 \
      NQ_GPU_BATCH_TASKS=500000 \
      NQ_GPU_PERSIST_BLOCKS="$persist_blocks" \
      "$BINARY" "$N"
    } 2>&1 | tee "$log"
  fi
}

# Baseline (auto persistent block count from device properties).
run_case "persist_auto" ""

# Persistent-kernel block override sweeps.
run_case "persist_64" "64"
run_case "persist_128" "128"
run_case "persist_256" "256"

echo
echo "Logs written to: $OUTDIR"
echo "Quick compare:"
grep -H "GPU timing\|Calculations took" "$OUTDIR"/n19_persist_*.log || true
