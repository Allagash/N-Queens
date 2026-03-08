# GPU Optimization Plan for N-Queens

## Goal
- Accelerate the current multithreaded CPU N-Queens solver using GPU compute while preserving exact solution counts.
- Keep the existing symmetry reduction logic and CPU threading as part of a hybrid pipeline.

## Scope and Constraints
- Preserve current algorithmic semantics (bitboard backtracking, symmetry handling).
- Use 64-bit masks (`col`, `pos_diag`, `neg_diag`) to support current board-size goals.
- Optimize for throughput on large `N` where runtime is dominated by deep subtree exploration.
- Maintain deterministic correctness (same counts as CPU reference).

## High-Level Strategy
1. Keep early search expansion on CPU.
2. Cut the tree at a configurable depth `d` (frontier generation).
3. Send frontier states (independent subtrees) to GPU.
4. Solve each subtree on GPU with iterative DFS and local stacks.
5. Reduce per-thread counts into global totals.
6. Overlap CPU frontier generation with GPU execution using batches/streams.

## Phase 1: Baseline and Instrumentation
1. Freeze a CPU baseline:
- Keep current threaded solver as correctness oracle.
- Record wall-clock and throughput for representative inputs (`N=14..20` as feasible).

2. Add measurement hooks:
- Time split: frontier generation, host->device copy, kernel, device->host copy, reduction.
- Emit CSV/JSON benchmark rows for reproducible comparisons.

## Phase 2: Frontier Generation on CPU
1. Add a frontier-state struct:
```c
typedef struct {
    uint64_t col;
    uint64_t pos_diag;
    uint64_t neg_diag;
    uint64_t bitfield;   // candidates at current row
    uint8_t  numrows;    // current depth
} FrontierTask;
```

2. Generate tasks at depth `d`:
- Reuse current iterative backtracking.
- Stop expansion at `numrows == d`.
- Push each partial state into a task vector.

3. Apply symmetry on CPU before task emission:
- Keep existing half-board + odd-center treatment.
- Ensure GPU only sees unique mirrored-half work.

4. Tune `d`:
- Start with `d=6` for smaller `N`, increase to `d=8..10` for larger `N`.
- Objective: enough tasks to saturate GPU without excessive generation overhead.

## Phase 3: GPU Kernel Design
1. Execution model:
- One thread processes one subtree task (initial implementation).
- Later optimization: one warp per task if per-thread stack pressure is too high.

2. Core kernel loop:
- Use iterative DFS (no recursion).
- Maintain thread-local stack arrays for `bitfield` and row state.
- Compute LSB with `lsb = bits & -bits`.
- Update masks exactly like CPU logic.

3. Counting:
- Keep per-thread local solution counter in registers.
- End of kernel: block-level reduction in shared memory.
- One global atomic add per block to reduce contention.

4. Data movement:
- Copy batches of `FrontierTask` to device memory.
- Return only aggregate counts (or per-block counts) to host.

5. Precision:
- Use `uint64_t` for masks and counts in kernel.
- If targeting very large totals, support two-level accumulation:
- Device partials in 64-bit.
- Final host accumulation in `unsigned __int128` (or checked overflow path).

## Phase 4: Load Balancing and Divergence Control
1. Start with static mapping (thread `i` -> task `i`) for simplicity.
2. Upgrade to persistent kernel + global work queue:
- Each thread/block atomically fetches next task index.
- Prevents idle SMs when tasks have uneven runtimes.

3. Bucket tasks by estimated hardness:
- Heuristic: remaining depth and initial candidate count (`popcount(bitfield)`).
- Launch similar-cost tasks together to reduce warp divergence.

4. Batch sizing:
- Use moderate batch sizes (e.g., 50k–500k tasks) to balance occupancy and transfer overhead.
- Auto-tune by benchmarking on startup or first run.

## Phase 5: Hybrid CPU+GPU Pipeline
1. Double-buffered task flow:
- CPU thread pool generates batch `k+1` while GPU solves batch `k`.
- Use at least 2 CUDA streams for overlap.

2. Work stealing fallback:
- If GPU queue drains, CPU can solve remaining tasks directly.
- Useful on smaller `N` where GPU launch/transfer overhead dominates.

3. Adaptive mode selection:
- CPU-only path for small boards (`N <= threshold`).
- Hybrid path for larger boards.
- Threshold chosen empirically.

## Phase 6: Validation and Correctness
1. Deterministic regression set:
- Validate exact counts for known values (`N=8..16`, and higher as practical).

2. Cross-check modes:
- CPU-only vs GPU-only vs hybrid must match exactly.

3. Stress tests:
- Randomized frontier splits (`d` variations) to ensure invariance of totals.
- Repeated runs to catch race bugs in reductions/work queues.

## Phase 7: Performance Targets
1. Define success metrics:
- End-to-end speedup vs current threaded CPU baseline.
- Kernel occupancy and achieved throughput.
- PCIe transfer overhead as percentage of total runtime.

2. Initial target:
- 2x+ speedup at `N=16` (if enough GPU resources).
- Higher gains expected at larger `N` once load balancing is tuned.

## Implementation Milestones
1. `M1`: CPU frontier generator + unchanged CPU subtree solver.
2. `M2`: Minimal GPU kernel, static task mapping, correct counts.
3. `M3`: Batched transfer and block reductions.
4. `M4`: Persistent-kernel work queue and load balancing.
5. `M5`: Overlapped CPU/GPU pipeline and auto-tuned depth/batch sizing.
6. `M6`: Benchmark report + default runtime policy.

## Practical Notes for This Codebase
- Keep existing `Nqueen()` as orchestrator entrypoint.
- Split solver into:
- `generate_frontier(board_size, depth, tasks)`
- `solve_tasks_cpu(tasks)`
- `solve_tasks_gpu(tasks)`
- Preserve mirror-doubling logic at the orchestration layer (not inside kernel).
- Keep a compile-time switch:
- `-DUSE_CUDA=1` enables GPU path.
- Otherwise build remains pure CPU and portable.

## Risks and Mitigations
1. Risk: GPU underutilization from branch divergence.
- Mitigation: task bucketing + persistent queue.

2. Risk: stack/register pressure lowering occupancy.
- Mitigation: cap frontier depth, test thread-vs-warp task mapping, tune launch config.

3. Risk: transfer overhead erasing gains on small `N`.
- Mitigation: adaptive CPU-only threshold.

4. Risk: count overflow at very large `N`.
- Mitigation: explicit overflow checks and wider host accumulator path.

## Deliverables
1. CUDA-enabled solver path with deterministic correctness tests.
2. Benchmark script + result table comparing CPU and hybrid modes.
3. Runtime knobs:
- `--frontier-depth`
- `--gpu-batch-size`
- `--mode=cpu|gpu|hybrid|auto`
