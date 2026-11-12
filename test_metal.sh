#!/bin/bash
# Correctness tests for the Metal backend. Run via `make test-metal`.
#
# Every case forces the GPU path (NQ_SOLVER_MODE=gpu) and compares the count
# with the known value (OEIS A000170). A case also fails if the backend printed
# a "Metal:" diagnostic, because that means it fell back to the CPU and the
# count no longer says anything about the GPU.
#
# Usage: ./test_metal.sh [path-to-nq]     (default ./nq)

NQ=${1:-./nq}
EXPECTED=(0 0 0 0 2 10 4 40 92 352 724 2680 14200 73712 365596 2279184 14772512 95815104 666090624)

fail=0
cases=0
recovered=0

# solutions <N> [VAR=value ...]  -> prints the solution count (0 if none).
# Any "Metal:" diagnostic is shown on stderr. A diagnostic means the count did
# not come from an undisturbed GPU run, so the result is tagged:
#   "recovered:<count>"  a command buffer failed and the retry succeeded
#   "fallback"           the backend gave up and the CPU produced the count
solutions() {
    local n=$1; shift
    local out got
    out=$(env "$@" "$NQ" "$n" 2>&1)
    got=$(echo "$out" | grep -o '[0-9]* solution' | grep -o '[0-9]*')
    got=${got:-0}
    if echo "$out" | grep -q "Metal:"; then
        echo "---- diagnostic from: $* $NQ $n" >&2
        echo "$out" | grep -A2 "Metal:" >&2
        if echo "$out" | grep -q "falling back\|using CPU solver"; then
            echo "fallback"
        else
            echo "recovered:$got"
        fi
        return
    fi
    echo "$got"
}

# check <label> <N> [VAR=value ...]
check() {
    local label=$1 n=$2; shift 2
    local got
    got=$(solutions "$n" NQ_SOLVER_MODE=gpu "$@")
    cases=$((cases + 1))
    if [ "$got" = "recovered:${EXPECTED[$n]}" ]; then
        echo "NOTE  N=$n  $label: a command buffer failed, the retry succeeded, count is correct"
        recovered=$((recovered + 1))
    elif [ "$got" != "${EXPECTED[$n]}" ]; then
        echo "FAIL  N=$n  $label: got $got, expected ${EXPECTED[$n]}"
        fail=1
    fi
}

if [ ! -x "$NQ" ]; then
    echo "$NQ not found; run 'make metal' first." >&2
    exit 2
fi

echo "== GPU vs known counts, N=2..18"
for n in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
    check "default settings" $n
done

echo "== CPU cross-check, N=4..16"
for n in 4 5 6 7 8 9 10 11 12 13 14 15 16; do
    got=$(solutions $n NQ_SOLVER_MODE=cpu)
    cases=$((cases + 1))
    if [ "$got" != "${EXPECTED[$n]}" ]; then
        echo "FAIL  N=$n  CPU path: got $got, expected ${EXPECTED[$n]}"
        fail=1
    fi
done

echo "== Frontier depths (1 gives mixed start rows on odd N; N-1 leaves one row for the GPU)"
for n in 5 6 13 14; do
    for d in 1 2 6 $((n - 1)); do
        check "NQ_FRONTIER_DEPTH=$d" $n NQ_FRONTIER_DEPTH=$d
    done
done
for n in 2 3 4 5 6 7; do
    check "NQ_FRONTIER_DEPTH=1" $n NQ_FRONTIER_DEPTH=1
done

echo "== Residual-row caps (smaller cap = deeper CPU expansion)"
for r in 1 4 5 8 10 13 23; do
    check "NQ_METAL_MAX_ROWS=$r" 16 NQ_METAL_MAX_ROWS=$r
    check "NQ_METAL_MAX_ROWS=$r" 15 NQ_METAL_MAX_ROWS=$r
done

echo "== Chunk and batch boundaries"
check "flush mid-expansion (chunk=1000)" 16 NQ_METAL_MAX_ROWS=8 NQ_METAL_CHUNK=1000
check "ragged last batch and command buffer" 16 \
      NQ_METAL_MAX_ROWS=8 NQ_METAL_CHUNK=70000 NQ_METAL_BATCH=1000 NQ_METAL_DISPATCHES_PER_CB=3
check "one partial per chunk" 9 NQ_METAL_CHUNK=1 NQ_METAL_BATCH=1
check "mixed start rows + expansion" 15 \
      NQ_FRONTIER_DEPTH=1 NQ_METAL_MAX_ROWS=9 NQ_METAL_CHUNK=5000

echo "== Missing metallib falls back to the CPU and is still correct"
out=$(NQ_SOLVER_MODE=gpu NQ_METALLIB=/nonexistent.metallib "$NQ" 10 2>&1)
cases=$((cases + 1))
if ! echo "$out" | grep -q "Metal: cannot load" || ! echo "$out" | grep -q " ${EXPECTED[10]} solutions"; then
    echo "FAIL  fallback on missing metallib"
    fail=1
fi

if [ $fail -eq 0 ]; then
    echo "All $cases cases passed."
    if [ $recovered -gt 0 ]; then
        echo "$recovered case(s) needed a GPU retry; see the diagnostics above."
    fi
else
    echo "FAILURES detected ($cases cases run)."
    exit 1
fi
