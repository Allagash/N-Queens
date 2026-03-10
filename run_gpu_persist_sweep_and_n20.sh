#!/usr/bin/env bash
set -euo pipefail

BINARY="${1:-./nq}"
OUTDIR="${2:-bench}"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

run_n19_case() {
  local label="$1"
  local persist_blocks="$2"
  local log="$OUTDIR/n19_persist_${label}.log"

  echo "=== N=19 persist=$label ==="
  if [[ "$label" == "auto" ]]; then
    {
      echo "=== N=19 persist=$label ==="
      NQ_SOLVER_MODE=gpu \
      NQ_GPU_TIMING=1 \
      NQ_FRONTIER_DEPTH=6 \
      NQ_GPU_BATCH_TASKS=500000 \
      "$BINARY" 19
    } 2>&1 | tee "$log"
  else
    {
      echo "=== N=19 persist=$label ==="
      NQ_SOLVER_MODE=gpu \
      NQ_GPU_TIMING=1 \
      NQ_FRONTIER_DEPTH=6 \
      NQ_GPU_BATCH_TASKS=500000 \
      NQ_GPU_PERSIST_BLOCKS="$persist_blocks" \
      "$BINARY" 19
    } 2>&1 | tee "$log"
  fi
}

for persist in auto 64 128 192 256 384; do
  run_n19_case "$persist" "$persist"
done

log="$OUTDIR/n20_default.log"
echo "=== N=20 default ==="
{
  echo "=== N=20 default ==="
  NQ_SOLVER_MODE=gpu \
  NQ_GPU_TIMING=1 \
  NQ_FRONTIER_DEPTH=6 \
  NQ_GPU_BATCH_TASKS=500000 \
  "$BINARY" 20
} 2>&1 | tee "$log"

echo
echo "Logs written to: $OUTDIR"
echo "Quick compare:"
grep -H "GPU timing\|Calculations took" \
  "$OUTDIR"/n19_persist_*.log \
  "$OUTDIR"/n20_default.log || true
