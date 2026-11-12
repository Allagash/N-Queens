# Next-Round Speedup Optimizations for nq_metal

## Context

After implementing 3-row leaf peeling, MAX_STACK_DEPTH reduction (17), and K tuning,
the solver achieves ~2.5x speedup over the original but is bottlenecked by checkpoint
overhead. The low ITER_BUDGET (200K, forced by watchdog + leaf peeling interaction)
causes frequent checkpointing, and each checkpoint round requires a full host↔GPU
round-trip. For N=22, thousands of batches × dozens of resume rounds = massive
dispatch overhead.

Goal: further reduce N=22 from 5h19m and make N=23 practical.

## Current Performance

| N | Solutions | Config | Time | vs Original |
|---|-----------|--------|------|-------------|
| 20 | 39,029,188,884 | K=6, 3-row peel, budget=200K | 4m16s | 2.4x (was 10m17s) |
| 21 | 314,666,222,712 | K=6, 3-row peel, budget=200K | ~35m (est) | ~2.5x (was 1h28m) |
| 22 | 2,691,008,701,644 | K=6, 3-row peel, budget=200K | 5h19m | 2.6x (was 14h1m) |

## Optimizations (in priority order)

### 1. Eliminate Checkpointing — Raise K to avoid watchdog (est. 1.5-2x)
**Files:** `nq_metal.m`, `nq.metal`
**Effort:** Low
**Risk:** Low

The entire checkpoint system exists because K=6 threads are too heavy for the GPU
watchdog. But with 3-row leaf peeling, threads complete much faster per tree node.
Raising K to 7 or 8 gives each thread only 14-15 rows to solve (vs 16 at K=6),
which may fit within the watchdog without any checkpointing.

**What changes:**
- Set K=7 or K=8 for N≥20 (test both)
- Remove the `iters` counter and ITER_BUDGET check from the hot loop
- Remove checkpoint buffer allocation, resume kernel dispatch, drain_checkpoints()
- Remove the entire `nqueen_resume` kernel (or keep it but never call it)

**Why this is high-value:** Removes ALL checkpoint round-trip overhead. The hot loop
becomes tighter (no iters increment, no budget comparison per iteration). No resume
dispatches. The GPU runs uninterrupted.

**Risk mitigation:** The original solver ran K=8 (N=20) and K=10 (N=22) without
checkpoints and completed successfully. With 3-row leaf peeling making threads
faster, K=7 or K=8 should be safe. Test with N=20 first (fast turnaround).

**Fallback:** If K=8 still hits the watchdog for N=22, keep checkpointing but use
a higher ITER_BUDGET (say 500K) since K=8 threads are lighter. Or use K=9.

### 2. Progress Reporting (not a speedup, but essential)
**Files:** `nq_metal.m`
**Effort:** Low
**Risk:** None

For multi-hour runs, print periodic progress every 60 seconds:
- Elapsed time
- Batches completed / estimated total
- Partial solution count so far

Helps distinguish "running fine" from "stuck/crashed" without needing `ps`.

### 3. Multi-Dispatch Command Buffers (est. 1.1-1.2x)
**Files:** `nq_metal.m`
**Effort:** Medium
**Risk:** Low (only if checkpointing is eliminated first)

If checkpointing is eliminated (opt #1), multiple batch dispatches can be encoded
into a single command buffer. Instead of:
```
for each batch:
    create CB → encode → commit → waitUntilCompleted → collect
```
Do:
```
create CB → encode batch 0 → encode batch 1 → ... → encode batch N → commit → waitUntilCompleted → collect all
```

This eliminates per-batch CB creation/commit/wait overhead. Encode up to 250 batches
(one full chunk) into a single CB. The GPU processes them back-to-back.

**Prerequisite:** Only works without checkpointing, since checkpoints require
per-batch drain between dispatches.

### 4. Increase BATCH_SIZE (est. 1.05-1.1x)
**Files:** `nq_metal.m`
**Effort:** Trivial
**Risk:** Low

BATCH_SIZE=32,000 matches M1 Max's ~32K concurrent threads. But Apple GPUs can
queue more work than they execute concurrently. Increasing to 64K or 128K reduces
the number of dispatches by 2-4x, cutting per-dispatch overhead.

Test with N=20 to validate: should see modest improvement from fewer dispatches.

### 5. Rotational Symmetry (est. 1.5-1.8x effective)
**Files:** `nq_metal.m` (CPU enumeration)
**Effort:** High
**Risk:** Medium (correctness is subtle)

Currently exploits only Y-axis mirror symmetry (2x). The full symmetry group of
the chessboard is the dihedral group D4 with 8 elements (4 rotations × 2 reflections).
Most solutions have an orbit of size 8, so only ~1/8 need to be explored explicitly.

Implementation requires:
- Canonical-form detection for first-row placements
- Orbit-size calculation (some solutions are self-symmetric: orbit < 8)
- Only enumerate the canonical representative from each orbit
- Multiply each representative's count by its orbit size

Effective speedup is ~3.5-3.8x (not 8x, because: half already handled by mirror,
and orbits of size < 8 need special counting).

This is the highest theoretical speedup but requires careful implementation and
extensive correctness testing against known solution counts.

### 6. First-Two-Row Restriction (est. 1.1-1.2x)
**Files:** `nq_metal.m`
**Effort:** Low
**Risk:** Low

Refine the CPU enumeration to use known restrictions on the first two rows.
When the row-0 queen is in column 0, the row-1 queen's valid range is reduced
by symmetry arguments. This prunes ~10-20% of partial placements without
changing the GPU kernel at all.

## Implementation Order

```
Phase 1 (quick wins):
  #1  Eliminate checkpointing (K=7 or K=8)     ← biggest single impact
  #2  Progress reporting                        ← essential for long runs

Phase 2 (polish, depends on Phase 1):
  #3  Multi-dispatch command buffers            ← only if #1 succeeds
  #4  Increase BATCH_SIZE                       ← trivial, test empirically

Phase 3 (algorithmic, independent):
  #5  Rotational symmetry                       ← high effort, high reward
  #6  First-two-row restriction                 ← low effort, modest reward
```

## Files to Modify
- `nq.metal` — Remove checkpoint/resume code from hot loop (opt #1), or keep as-is
- `nq_metal.m` — K tuning, remove checkpoint dispatch, progress reporting, multi-dispatch

## Verification
1. `make clean && make` — build succeeds
2. `make test` — correctness for N=4..12
3. Benchmark N=18: ≤4s (regression)
4. Benchmark N=20: compare against 4m16s baseline — target ~2-3m
5. Benchmark N=22: compare against 5h19m — target ~3-4h
6. (Long run) N=23: first attempt, target completion within ~2 days
