#!/usr/bin/env bash
# Compare the two CUDA subtree kernels (NQ_GPU_KERNEL=peel|legacy, peel is the
# default) on the NVIDIA box at default settings: correctness first, then timing.
#
# Usage: ./bench_cuda_kernels.sh [--with-n20] [path-to-nq]
#   Build first with `make cuda`. Default binary is ./nq.
#   --with-n20 adds N=20 (about 3.5 minutes for the legacy kernel on an RTX 4060).

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

EXPECTED=(0 0 0 0 2 10 4 40 92 352 724 2680 14200 73712 365596 2279184 14772512 95815104 666090624 4968057848 39029188884)

count_of() {  # prints the solution count from a run's output
    grep -o '[0-9]* solution' | grep -o '[0-9]*'
}

echo "== Correctness: forced GPU, both kernels, N=8..18"
fail=0
for kernel in legacy peel; do
    for n in 8 9 10 11 12 13 14 15 16 17 18; do
        out=$(NQ_SOLVER_MODE=gpu NQ_GPU_KERNEL=$kernel "$NQ" "$n" 2>&1)
        got=$(echo "$out" | count_of)
        if [[ "${got:-0}" != "${EXPECTED[$n]}" ]]; then
            echo "FAIL  kernel=$kernel N=$n: got ${got:-none}, expected ${EXPECTED[$n]}"
            echo "$out" | grep -i "cuda" | head -3
            fail=1
        fi
    done
    # Shallow and deep frontiers exercise different entry branches of the peel kernel.
    for depth in 2 4 9 12; do
        out=$(NQ_SOLVER_MODE=gpu NQ_GPU_KERNEL=$kernel NQ_FRONTIER_DEPTH=$depth "$NQ" 14 2>&1)
        got=$(echo "$out" | count_of)
        if [[ "${got:-0}" != "${EXPECTED[14]}" ]]; then
            echo "FAIL  kernel=$kernel N=14 depth=$depth: got ${got:-none}, expected ${EXPECTED[14]}"
            fail=1
        fi
    done
done
if [[ $fail -ne 0 ]]; then
    echo "Correctness FAILED; skipping timing."
    exit 1
fi
echo "All counts correct."

echo
echo "== Timing (kernel+reduce ms is the number to compare)"
sizes="18 19"
[[ $WITH_N20 -eq 1 ]] && sizes="18 19 20"
for n in $sizes; do
    for kernel in legacy peel; do
        out=$(NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_GPU_KERNEL=$kernel "$NQ" "$n" 2>&1)
        got=$(echo "$out" | count_of)
        status=ok
        [[ "${got:-0}" != "${EXPECTED[$n]}" ]] && status="WRONG COUNT ${got:-none}"
        echo "N=$n kernel=$kernel [$status]"
        echo "  $(echo "$out" | grep 'GPU timing')"
    done
done
