#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./bench_frontier_depth.sh [options]

Options:
  --n <int>            Board size (default: 18)
  --depths "<list>"    Space-separated frontier depths (default: "4 5 6 7 8 9 10")
  --repeats <int>      Runs per depth (default: 3)
  --outdir <path>      Output log directory (default: bench)
  --bin <path>         Solver binary (default: ./nq)
  --mode <cpu|gpu|auto>  NQ_SOLVER_MODE (default: gpu)
  --gpu-timing <0|1>   NQ_GPU_TIMING (default: 1)
  --skip-warmup        Skip initial warm-up run
  -h, --help           Show this help

Example:
  ./bench_frontier_depth.sh --n 18 --depths "5 6 7 8 9" --repeats 3
EOF
}

N=18
DEPTHS="4 5 6 7 8 9 10"
REPEATS=3
OUTDIR="bench"
BINARY="./nq"
SOLVER_MODE="gpu"
GPU_TIMING="1"
SKIP_WARMUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N="$2"
      shift 2
      ;;
    --depths)
      DEPTHS="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --bin)
      BINARY="$2"
      shift 2
      ;;
    --mode)
      SOLVER_MODE="$2"
      shift 2
      ;;
    --gpu-timing)
      GPU_TIMING="$2"
      shift 2
      ;;
    --skip-warmup)
      SKIP_WARMUP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

if [[ "$SKIP_WARMUP" -eq 0 ]]; then
  WARMUP_DEPTH="$(echo "$DEPTHS" | awk '{print $1}')"
  echo "Warming up CUDA/context: N=$N depth=$WARMUP_DEPTH"
  NQ_SOLVER_MODE="$SOLVER_MODE" \
  NQ_GPU_TIMING="$GPU_TIMING" \
  NQ_FRONTIER_DEPTH="$WARMUP_DEPTH" \
  "$BINARY" "$N" >/dev/null 2>&1 || true
fi

for d in $DEPTHS; do
  for r in $(seq 1 "$REPEATS"); do
    LOG="$OUTDIR/n${N}_d${d}_r${r}.log"
    echo "=== N=$N depth=$d run=$r ==="
    {
      echo "=== N=$N depth=$d run=$r ==="
      NQ_SOLVER_MODE="$SOLVER_MODE" \
      NQ_GPU_TIMING="$GPU_TIMING" \
      NQ_FRONTIER_DEPTH="$d" \
      /usr/bin/time -f "wall=%e user=%U sys=%S" "$BINARY" "$N"
    } 2>&1 | tee "$LOG"
  done
done

SUMMARY="$OUTDIR/n${N}_summary.txt"
{
  echo "Summary for N=$N"
  echo
  grep -H "GPU timing\|wall=" "$OUTDIR"/n${N}_d*_r*.log || true
  echo
  echo "Average wall time by depth:"
  for d in $DEPTHS; do
    awk -v depth="$d" '
      /wall=/ {
        split($0, a, "wall=")
        split(a[2], b, " ")
        sum += b[1]
        n++
      }
      END {
        if (n > 0) {
          printf "depth=%s avg_wall=%.3fs runs=%d\n", depth, sum / n, n
        }
      }
    ' "$OUTDIR"/n${N}_d${d}_r*.log
  done
} | tee "$SUMMARY"

echo
echo "Wrote logs to: $OUTDIR"
echo "Wrote summary to: $SUMMARY"
