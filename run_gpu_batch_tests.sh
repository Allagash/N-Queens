#!/usr/bin/env bash
set -euo pipefail

N="${1:-18}"
BINARY="${2:-./nq}"
OUTDIR="${3:-bench}"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

for batch in 100000 250000 500000; do
  log="$OUTDIR/n${N}_batch_${batch}.log"
  echo "=== N=$N batch=$batch ==="
  {
    echo "=== N=$N batch=$batch ==="
    NQ_SOLVER_MODE=gpu \
    NQ_GPU_TIMING=1 \
    NQ_GPU_BATCH_TASKS="$batch" \
    "$BINARY" "$N"
  } 2>&1 | tee "$log"
done

echo
echo "Logs written to: $OUTDIR"
echo "Quick compare:"
grep -H "GPU timing\|Calculations took" "$OUTDIR"/n${N}_batch_*.log || true
