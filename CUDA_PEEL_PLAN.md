# Can the NVIDIA (RTX 4060) build be made meaningfully faster?

Status: **Step 1 implemented on branch `cuda-peel`, awaiting measurement on the
NVIDIA box** (`./bench_cuda_kernels.sh` after `make cuda`) · 2026-11-13

Step 1 as built: `nq_peel.h` (solver), `test_peel.c` + `make test-peel` (host test,
97 cases, passes), `NQ_GPU_KERNEL=peel` in `nq_cuda.cu` (default still `legacy`).
Per-thread shared memory in peel mode is 16 B x (rows - 5): 128 B at N=19 versus
416 B. The CUDA source could only be type-checked against a stub header on the Mac;
it has not been compiled by `nvcc` or run on a GPU yet.

## Context

The CUDA backend on the RTX 4060 box solves N=19 in ~27 s and N=20 in ~212 s. The
question is whether a worthwhile speedup remains, or only minor gains. The Metal
integration just finished gives a new reference point that did not exist when the
CUDA kernel was tuned (M4–M6), and it changes the answer.

## Conclusion

**Yes, and it is not minor. Estimated 2–4x, from a change confined to one kernel
function.** The estimate is an inference, not a measurement, so the plan starts with
a cheap experiment that proves or kills it before anything else is built.

## Evidence

1. **The M1 Max now beats the 4060 on this problem.** Same algorithm, same frontier:

   | N  | RTX 4060, CUDA kernel | M1 Max, Metal kernel |
   |----|-----------------------|----------------------|
   | 18 | 4.8 s                 | 2.4 s                |
   | 20 | 212 s                 | 157 s                |

   An RTX 4060 is the stronger GPU for integer/bit-twiddling work (from memory,
   roughly 2.5–3x an M1 Max on hash-cracking benchmarks; treat that ratio as
   approximate). Losing by 1.35–2x to a weaker GPU means the kernel, not the
   hardware, is the limit.

2. **The CUDA kernel lacks every optimization that made the Metal kernel fast.**
   `solveTaskDevice` (`nq_cuda.cu:18-77`) versus `nqueen` (`nq_metal.metal`):

   | | CUDA today | Metal |
   |---|---|---|
   | Mask width | 64-bit (`BITFIELDTYPE`) on a 32-bit datapath | 32-bit |
   | Last rows | every leaf visited one at a time, `++solutions` | last 4 rows peeled into register-only loops, final row counted with one `popcount` |
   | Per-node state | 4 arrays in shared memory, read/written every node | registers; small stack touched only above the last 4 rows |
   | Per-thread memory | 416 B (N=19) | ≤ 144 B |

   In nq_metal's own history, leaf peeling and the changes that came with it took
   N=20 from 10m17s to 2m37s on unchanged hardware (`METAL_RESULTS.md`). Those
   numbers mix in other changes (checkpoint removal, depth tuning), so they show the
   direction and rough size, not an exact factor for peeling alone.

3. **Your own `ncu` profile shows the 4060 mostly idle**
   (`GPU_M6_BENCHMARK_AND_PROFILING.md`): issue slots 18.9% busy, theoretical
   occupancy 12.5% (shared memory allows only 3 blocks of 64 threads per SM), top
   stall is shared-memory access (MIO, 40%). The N=19 timing line shows
   `max_blocks=62`: about 4,000 threads on a GPU that can hold ~37,000. Shrinking
   per-thread state attacks exactly these limits.

## What I would not do

- **Bank-conflict padding** (M6 item 1): 10–15%. Minor; superseded by the change below.
- **Warp-per-task** (M6 item 2): claimed 2–3x but a large, risky refactor. Revisit
  only if the experiment below disappoints.
- **Orbit-weighted symmetry on the GPU** (`SYMMETRY_OPTIMIZATION_PLAN.md`): it needs
  per-leaf queen positions, which is incompatible with popcount peeling, the larger
  win. The CPU experience is also discouraging: forced-CPU N=19 on the box is 178 s
  now versus 170 s in the M6 report, i.e. orbit weighting bought nothing there.

## Plan

### Step 1 — Experiment and decision gate (small)

Add a second device function next to the existing one, selectable at run time, so
both kernels can be compared in one binary on the same day.

- **New file `nq_peel.h`**: a 32-bit, 4-row-peeled DFS, a direct port of the loop in
  `nq_metal.metal` (push while `row + 5 <= last`; branches for 4, 3, 2, 1, 0 rows
  left; `popcount` at the bottom). Written as plain C with two macros (function
  qualifier, popcount) so the identical source compiles as a CUDA `__device__`
  function and as a host function.
- **`nq_cuda.cu`**: `solveTasksKernel` (`:79-125`) calls the peeled function when a
  kernel argument says so. Frames stay in the existing dynamic shared-memory region
  but as 4 x `uint32_t` per row, and `depth_stride` shrinks by the 4 peeled rows
  (`:693-699`). The work queue, batching, streams and reduction are untouched.
  `NQ_GPU_KERNEL=legacy|peel` picks the kernel; default stays `legacy` until the gate
  passes. Narrowing 64→32 bits is exact for N ≤ 31, same argument as in
  `nq_metal.m`; N > 31 falls back to `legacy`.
- **Host test on the Mac** (no GPU needed): a small harness compiles `nq_peel.h` as
  C and checks counts for N = 4..16 across start rows, since CUDA code cannot be
  built here. Add it as `make test-peel`.

**Gate:** on the 4060, `NQ_GPU_KERNEL=peel` must be correct for N = 8..19 and cut
N=19 kernel time by at least 1.5x (27 s → ≤ 18 s). If not, stop and keep `legacy`;
the cost was one function and a test.

### Step 2 — Use the freed capacity (only if the gate passes; mostly tuning)

- Per-thread shared memory drops from 416 B to about 160 B, so more blocks fit per
  SM. Re-sweep the existing knobs, no new code: `NQ_GPU_PERSIST_BLOCKS`,
  `NQ_GPU_TARGET_TASKS_PER_BLOCK`, and `CUDA_BLOCK_SIZE` (`nq_cuda.cu:10`).
  `run_gpu_persist_sweep_and_n20.sh` already does this kind of sweep.
- Try `NQ_FRONTIER_DEPTH=7` at N=20 (13 rows per thread instead of 14, 22.8M tasks,
  1.6 GB of host memory) and keep whichever depth is faster.
- Try frames in thread-local memory instead of shared memory. M6 found local memory
  slow, but that was with 640 B of 64-bit state touched on every node; here it is
  ≤ 144 B touched only above the peeled rows.
- Make `peel` the default, record results in `GPU_M6_BENCHMARK_AND_PROFILING.md`
  (or an M7 report) and `readme.md`.

### Expected outcome

N=20 from ~212 s to somewhere between 55 s and 105 s. The low end assumes the 4060
reaches its hash-benchmark advantage over the M1 Max; the high end assumes it only
matches it at 2x the current speed.

## Files

- Add: `nq_peel.h`, a host test harness, `make test-peel` target.
- Edit: `nq_cuda.cu` (kernel dispatch, `depth_stride`, one env knob), `Makefile`,
  docs.
- Unchanged: `nq.c`, `nq_types.h`, the Metal backend, CUDA pipeline/batching code.

## Verification

1. Mac: `make test-peel` passes; `make check-cuda-syntax` and `make test-metal` still pass.
2. NVIDIA box: `make cuda`, then for N = 8..19
   `NQ_SOLVER_MODE=gpu NQ_GPU_KERNEL=peel ./nq N` matches the known counts, and
   `NQ_GPU_KERNEL=legacy` still does.
3. Timing with `NQ_GPU_TIMING=1` at N=18, 19, 20 for both kernels, same session.
4. `ncu` occupancy / issue-slot numbers for the peel kernel, compared with the M6 table.

## Constraints

- CUDA cannot be built or run from the Mac. Step 1's kernel logic is tested on the
  host, but every GPU measurement has to be run on the NVIDIA box.
- The Mac's disk has been intermittently full; implementation needs free space.
