#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./bench_modes.sh [options]

Options:
  --ns "<list>"          Space-separated board sizes (default: "16 17 18 19 20")
  --modes "<list>"       Space-separated modes (default: "cpu gpu auto")
  --repeats <int>        Runs per (N, mode) pair (default: 3)
  --bin <path>           Solver binary (default: ./nq)
  --outdir <path>        Output directory (default: bench)
  --gpu-timing <0|1>     Set NQ_GPU_TIMING during runs (default: 1)
  --frontier-depth <int> Override NQ_FRONTIER_DEPTH for all runs
  --gpu-batch <int>      Override NQ_GPU_BATCH_TASKS for all runs
  --gpu-min-tasks <int>  Override NQ_GPU_MIN_TASKS for all runs
  --skip-warmup          Skip initial warm-up run
  -h, --help             Show this help

Outputs:
  <outdir>/bench_modes_raw.csv
  <outdir>/bench_modes_summary.csv
EOF
}

NS="16 17 18 19 20"
MODES="cpu gpu auto"
REPEATS=3
BINARY="./nq"
OUTDIR="bench"
GPU_TIMING="1"
FRONTIER_DEPTH=""
GPU_BATCH=""
GPU_MIN_TASKS=""
SKIP_WARMUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns)
      NS="$2"
      shift 2
      ;;
    --modes)
      MODES="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --bin)
      BINARY="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --gpu-timing)
      GPU_TIMING="$2"
      shift 2
      ;;
    --frontier-depth)
      FRONTIER_DEPTH="$2"
      shift 2
      ;;
    --gpu-batch)
      GPU_BATCH="$2"
      shift 2
      ;;
    --gpu-min-tasks)
      GPU_MIN_TASKS="$2"
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
RAW_CSV="$OUTDIR/bench_modes_raw.csv"
SUMMARY_CSV="$OUTDIR/bench_modes_summary.csv"

echo "n,mode,run,wall_s,user_s,sys_s,calc_seconds,solutions" > "$RAW_CSV"

if [[ "$SKIP_WARMUP" -eq 0 ]]; then
  WARM_N="$(echo "$NS" | awk '{print $1}')"
  echo "Warming up: N=$WARM_N mode=gpu"
  WARM_ENV=("NQ_SOLVER_MODE=gpu" "NQ_GPU_TIMING=0")
  if [[ -n "$FRONTIER_DEPTH" ]]; then
    WARM_ENV+=("NQ_FRONTIER_DEPTH=$FRONTIER_DEPTH")
  fi
  if [[ -n "$GPU_BATCH" ]]; then
    WARM_ENV+=("NQ_GPU_BATCH_TASKS=$GPU_BATCH")
  fi
  if [[ -n "$GPU_MIN_TASKS" ]]; then
    WARM_ENV+=("NQ_GPU_MIN_TASKS=$GPU_MIN_TASKS")
  fi
  env "${WARM_ENV[@]}" "$BINARY" "$WARM_N" >/dev/null 2>&1 || true
fi

for n in $NS; do
  for mode in $MODES; do
    for run in $(seq 1 "$REPEATS"); do
      log="$OUTDIR/mode_${mode}_n${n}_r${run}.log"
      echo "=== N=$n mode=$mode run=$run ==="
      RUN_ENV=("NQ_SOLVER_MODE=$mode" "NQ_GPU_TIMING=$GPU_TIMING")
      if [[ -n "$FRONTIER_DEPTH" ]]; then
        RUN_ENV+=("NQ_FRONTIER_DEPTH=$FRONTIER_DEPTH")
      fi
      if [[ -n "$GPU_BATCH" ]]; then
        RUN_ENV+=("NQ_GPU_BATCH_TASKS=$GPU_BATCH")
      fi
      if [[ -n "$GPU_MIN_TASKS" ]]; then
        RUN_ENV+=("NQ_GPU_MIN_TASKS=$GPU_MIN_TASKS")
      fi

      {
        echo "=== N=$n mode=$mode run=$run ==="
        TIMEFORMAT='wall=%R user=%U sys=%S'
        time env "${RUN_ENV[@]}" "$BINARY" "$n"
      } 2>&1 | tee "$log"

      wall="$(sed -n 's/^wall=\([0-9.][0-9.]*\).*/\1/p' "$log" | head -n1)"
      user="$(sed -n 's/^wall=[0-9.][0-9.]* user=\([0-9.][0-9.]*\).*/\1/p' "$log" | head -n1)"
      sys="$(sed -n 's/^wall=[0-9.][0-9.]* user=[0-9.][0-9.]* sys=\([0-9.][0-9.]*\).*/\1/p' "$log" | head -n1)"
      calc_seconds="$(sed -n 's/.*Calculations took \([0-9][0-9]*\) second.*/\1/p' "$log" | head -n1)"
      solutions="$(sed -n 's/.*For board size [0-9][0-9]*, \([0-9][0-9]*\) solution.*/\1/p' "$log" | head -n1)"

      if [[ -z "$wall" ]]; then wall="NA"; fi
      if [[ -z "$user" ]]; then user="NA"; fi
      if [[ -z "$sys" ]]; then sys="NA"; fi
      if [[ -z "$calc_seconds" ]]; then calc_seconds="NA"; fi
      if [[ -z "$solutions" ]]; then
        if grep -q "No solutions found." "$log"; then
          solutions="0"
        else
          solutions="NA"
        fi
      fi

      printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$n" "$mode" "$run" "$wall" "$user" "$sys" "$calc_seconds" "$solutions" >> "$RAW_CSV"
    done
  done
done

awk -F, '
  NR == 1 { next }
  $4 != "NA" {
    key = $1 SUBSEP $2
    n[key] = $1
    mode[key] = $2
    runs[key]++
    wall_sum[key] += $4
    user_sum[key] += ($5 == "NA" ? 0 : $5)
    sys_sum[key] += ($6 == "NA" ? 0 : $6)
    if (!(key in min_wall) || $4 < min_wall[key]) min_wall[key] = $4
    if (!(key in max_wall) || $4 > max_wall[key]) max_wall[key] = $4
  }
  END {
    print "n,mode,runs,avg_wall_s,min_wall_s,max_wall_s,avg_user_s,avg_sys_s,speedup_vs_cpu"
    for (k in runs) {
      avg_wall[k] = wall_sum[k] / runs[k]
    }
    for (k in runs) {
      split(k, parts, SUBSEP)
      board = parts[1]
      m = parts[2]
      cpu_key = board SUBSEP "cpu"
      speedup = ""
      if (cpu_key in avg_wall && avg_wall[k] > 0) {
        speedup = sprintf("%.6f", avg_wall[cpu_key] / avg_wall[k])
      }
      printf "%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
             n[k], mode[k], runs[k], avg_wall[k], min_wall[k], max_wall[k],
             user_sum[k] / runs[k], sys_sum[k] / runs[k], speedup
    }
  }
' "$RAW_CSV" | {
  read -r header
  echo "$header"
  sort -t, -k1,1n -k2,2
} > "$SUMMARY_CSV"

echo
echo "Wrote raw CSV: $RAW_CSV"
echo "Wrote summary CSV: $SUMMARY_CSV"
