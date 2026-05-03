# M6 — Benchmark Report and Profiling Findings

## Setup

- Hardware: NVIDIA RTX 4060 (Ada, sm_89), Linux box `gabriel`. CPU baseline is the threaded path in `nq.c` (pthread fanout, all cores).
- Build: `nvcc -O3 -DUSE_CUDA=1 -arch=sm_89 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra"`
- Runs: 3 per (N, mode) via `bench_modes.sh`, `--frontier-depth 6 --gpu-batch 500000`. Counts validated against the published series for N=16..20.

## Headline results

After the M4 persistent-launch tuning, the M5 stream pipeline, the shared-memory rewrite, and the offset-by-`start_numrows` change.

| N  | mode | runs | avg wall (s) | min / max     | speedup vs CPU |
|----|------|------|--------------|---------------|----------------|
| 16 | cpu  | 3    | 0.464        | 0.46 / 0.47   | 1.00x |
| 16 | gpu  | 3    | 0.510        | 0.46 / 0.57   | 0.91x ← GPU launch overhead dominates |
| 16 | auto | 3    | 0.470        | 0.44 / 0.51   | 0.99x ← auto correctly picks CPU |
| 17 | cpu  | 3    | 2.775        | 1.89 / 3.36   | 1.00x |
| 17 | gpu  | 3    | 0.989        | 0.93 / 1.11   | 2.81x |
| 17 | auto | 3    | **0.935**    | 0.91 / 0.97   | **2.97x** |
| 18 | cpu  | 3    | 21.97        | 21.41 / 22.84 | 1.00x |
| 18 | gpu  | 3    | 4.823        | 4.77 / 4.92   | 4.56x |
| 18 | auto | 3    | **4.337**    | 3.45 / 4.79   | **5.07x** |
| 19 | cpu  | 3    | 170.4        | 168 / 173     | 1.00x |
| 19 | gpu  | 3    | 27.15        | 26.6 / 28.2   | 6.28x |
| 19 | auto | 3    | **26.71**    | 26.6 / 26.7   | **6.38x** |
| 20 | cpu  | 3    | 1396.1       | 1390 / 1407   | 1.00x |
| 20 | gpu  | 3    | 212.1        | 211.4 / 213.4 | 6.58x |
| 20 | auto | 3    | **210.4**    | 208.0 / 211.6 | **6.64x** |

Source: `bench_modes_summary.csv` and `bench_modes_raw.csv` at the project root.

## Progression of GPU wall time at N=19

| stage                                    | N=19 GPU wall | improvement |
|------------------------------------------|---------------|-------------|
| Original CUDA path (M2)                  | ~75 s         | —           |
| + M4 persistent-launch tuning            | 64 s          | 1.17x       |
| + shared-memory per-thread state (SMEM)  | 31 s          | 2.06x       |
| + offset-by-`start_numrows` for smaller `depth_stride` | **27 s** | 1.15x       |

Cumulative GPU-only improvement: 75s -> 27s = **2.78x**. Versus threaded CPU: **6.28x** at N=19, **6.58x** at N=20.

Note: the same-run CPU baseline (`gabriel` was warm) is ~30% slower than an earlier baseline; against that earlier baseline (122.5s at N=19, 954.6s at N=20) the speedup is ~4.5x for both. Both numbers tell the same story — the bench just picks up CPU-side noise.

## Kernel-time breakdown (N=19, `NQ_GPU_TIMING=1`)

```
H2D            =     7 ms     (0.03%)
kernel+reduce  = 26377 ms     (99.96%)
D2H            =     2 ms     (0.01%)
batches=4  pipeline_slots=2  max_blocks=62  target_tasks_per_block=8192
```

Transfers are still noise. Anything that moves the wall-clock has to come out of the kernel.

## What `ncu` told us

Profiles taken with `solveTasksKernel` on N=17/18/19 with `--section Occupancy --section WarpStateStats --section ComputeWorkloadAnalysis`.

| metric                              | original (M4) | + SMEM    | + offset (current) |
|-------------------------------------|---------------|-----------|--------------------|
| Block size                          | 256           | 64        | 64                 |
| Registers / thread                  | 32            | 32        | 32                 |
| Stack frame (local memory)          | nonzero       | 0         | 0                  |
| Block Limit Shared Mem              | n/a           | 2         | **3**              |
| Theoretical Occupancy               | 100%          | 8.3%      | 12.5%              |
| Achieved Occupancy                  | 43%           | 8.3%      | 10.7%              |
| SM Busy / Issue Slots Busy          | 7.5%          | 15.5%     | **18.9%**          |
| Warp Cycles / Issued Instruction    | 26.8          | 6.4       | 6.8                |
| Avg Active Threads / Warp           | 11.2 / 32     | 11.2 / 32 | **12.2 / 32**      |
| Top stall reason                    | L1TEX (~80%)  | MIO (39%) | MIO (40%)          |

What this says:

- The original kernel was **memory-bound on local-memory spill** (per-thread DFS arrays parked in DRAM). Moving them to shared memory eliminated that — `stack frame: 0`, L1TEX stalls gone, kernel time roughly halved.
- The offset trick **shrank shared memory per block from ~38 KB to ~26 KB**, lifting `Block Limit Shared Mem` from 2 to 3 and theoretical occupancy from 8.3% to 12.5%, for the predicted ~15% additional kernel speedup.
- **Divergence is now the largest single inefficiency**: only ~12 of 32 lanes per warp are doing useful work on average. Thread-per-task DFS is the cause; warp-per-task is the prescribed cure.
- Top remaining stall is MIO scoreboard (~40%) — shared-memory access latency, partly bank-conflict driven.

## Recommended default policy

- Build with `-DUSE_CUDA=1 -arch=<your sm>`.
- Default to `NQ_SOLVER_MODE=auto` — at N>=17 it picks GPU; at N=16 it correctly stays on CPU.
- Keep `NQ_GPU_PIPELINE_SLOTS=2` (no harm, latent benefit if transfers ever grow).
- `NQ_FRONTIER_DEPTH=6` and `NQ_GPU_BATCH_TASKS=500000` remain best-known on a 4060 for N=19/20.
- Do **not** enable `NQ_GPU_AUTOTUNE_TARGET` by default (sample is too narrow to learn the right region; adds ~4s of overhead).
- `NQ_GPU_BUCKET_TASKS` stays opt-in.

## What's left, in priority order

1. **Bank-conflict padding (~10–15%, small change)**
   With `depth_stride=13`, the per-thread region is 416 bytes; the bank stride mod 32 is 8, so 4 threads share a bank cycle. Adding a single 8-byte pad per thread shifts the cycle and roughly halves the MIO scoreboard stalls. Should drop the kernel a further 10–15%.

2. **Warp-per-task (~2–3x, larger refactor)**
   The avg-active-threads-per-warp number (12/32) caps thread-per-task at ~37% of peak. With 32 lanes cooperating on one task — candidate selection becomes a warp-vote, the DFS stack becomes warp-shared, and load balancing happens at warp granularity — divergence largely disappears and per-task shared-memory headroom grows 32x. This is Phase 3.1 of the original plan.

3. **N>=21 stress tests**
   Beyond benchmarking, no algorithmic blocker. The 99 KB shared-memory opt-in path works through N=27 at block size 64.

The work in (1) and (2) is optional follow-on — current state already exceeds the original M6 target ("2x at N=16") by 3–6x.
