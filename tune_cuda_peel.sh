#!/usr/bin/env bash
# Step 2 sweep for the peeled CUDA kernels on the NVIDIA box: block count,
# frames in shared vs thread-local memory, pipeline slots and frontier depth.
#
# Usage: ./tune_cuda_peel.sh [--with-n20] [path-to-nq]
#   Build first with `make cuda`. N=19 sweep takes about 4 minutes;
#   --with-n20 adds about 8 more.

set -u

WITH_N20=0
NQ=./nq
for arg in "$@"; do
    case "$arg" in
        --with-n20) WITH_N20=1 ;;
        *) NQ=$arg ;;
    esac
done

if [[ ! -x "$NQ" ]]; then
    echo "$NQ not found or not executable; run 'make cuda' first." >&2
    exit 2
fi

declare -A EXPECTED=([18]=666090624 [19]=4968057848 [20]=39029188884)

# run N kernel blocks [extra VAR=value ...]   (blocks=0 means the default)
run() {
    local n=$1 kernel=$2 blocks=$3
    shift 3
    local envs=(NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_GPU_KERNEL="$kernel" "$@")
    [[ $blocks -gt 0 ]] && envs+=(NQ_GPU_PERSIST_BLOCKS="$blocks")
    local out got ms used status=ok
    out=$(env "${envs[@]}" "$NQ" "$n" 2>&1)
    got=$(echo "$out" | grep -o '[0-9]* solution' | grep -o '[0-9]*')
    ms=$(echo "$out" | grep -o 'kernel+reduce=[0-9.]*' | cut -d= -f2)
    used=$(echo "$out" | grep -o 'max_blocks=[0-9]*' | cut -d= -f2)
    [[ "${got:-0}" != "${EXPECTED[$n]}" ]] && status="WRONG COUNT ${got:-none}"
    printf 'N=%s %-10s blocks=%-4s kernel+reduce=%10s ms  [%s]  %s\n' \
        "$n" "$kernel" "${used:-?}" "${ms:-?}" "$status" "$*"
    if [[ "$status" != ok ]]; then
        echo "$out" | grep -i 'cuda' | head -3
    fi
}

echo "== N=19: block count"
run 19 legacy 0
run 19 legacy 144
run 19 peel 288
run 19 peel 576
run 19 peel-local 288
run 19 peel-local 576
run 19 peel-local 960

echo
echo "== N=19: one pipeline slot instead of two"
run 19 peel 288 NQ_GPU_PIPELINE_SLOTS=1
run 19 peel-local 576 NQ_GPU_PIPELINE_SLOTS=1

echo
echo "== N=19: frontier depth 7 (default is 6)"
run 19 peel 288 NQ_FRONTIER_DEPTH=7
run 19 peel-local 576 NQ_FRONTIER_DEPTH=7

if [[ $WITH_N20 -eq 1 ]]; then
    echo
    echo "== N=20 (legacy at default settings measured 244 s)"
    run 20 peel 288
    run 20 peel-local 576
    run 20 peel 288 NQ_FRONTIER_DEPTH=7
    run 20 peel-local 576 NQ_FRONTIER_DEPTH=7
fi
