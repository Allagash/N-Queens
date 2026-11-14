# nq — N-Queens solution counter

Counts the solutions to the N-Queens problem for an N x N board: `./nq <N>`.

## Building

One source tree, three builds. Each writes `./nq`.

| Build | Command | Platform | Needs |
|-------|---------|----------|-------|
| CPU only | `make cpu` | anything with a C compiler and pthreads | — |
| CUDA GPU | `make cuda` | Linux / NVIDIA | CUDA toolkit (`nvcc`) |
| Metal GPU | `make metal` | macOS / Apple Silicon | full Xcode (for the `metal` shader compiler) |

The CUDA and Metal backends cannot be combined in one binary.

The underlying commands, if you would rather not use `make`:

    cc -O3 -Wall -Wextra -pthread nq.c -o nq
    cc -O3 -march=native -Wall -Wextra -pthread nq.c -o nq     # machine-specific, less portable
    nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq

`make metal` also writes `nq_metal.metallib`, the compiled GPU kernel. It must stay
in the same directory as `nq`, or be named with `NQ_METALLIB=<path>`.

## Choosing CPU or GPU at run time (both GPU builds)

- default / auto: uses the GPU when the problem is big enough, otherwise the CPU.
  The GPU is used from N=17 on the CUDA build and from N=15 on the Metal build.
- force CPU: `NQ_SOLVER_MODE=cpu ./nq 16`
- force GPU attempt: `NQ_SOLVER_MODE=gpu ./nq 16`
- auto-mode threshold in frontier tasks (default 600000 for CUDA, 200000 for Metal):
  `NQ_GPU_MIN_TASKS=8192 ./nq 16`
- one-line timing summary on stderr: `NQ_GPU_TIMING=1 ./nq 18`
- frontier depth (default 6): `NQ_FRONTIER_DEPTH=<depth> ./nq <N>`

## Metal build

    make metal
    make test-metal      # 71 correctness cases, about a minute
    ./nq 20

Results on an Apple M1 Max (more in `METAL_RESULTS.md`, design in
`METAL_INTEGRATION_PLAN.md`):

| N  | Solutions         | Time      |
|----|-------------------|-----------|
| 18 | 666,090,624       | 2.4 s     |
| 20 | 39,029,188,884    | 2m 38s    |
| 21 | 314,666,222,712   | 24m 01s   |
| 22 | 2,691,008,701,644 | 3h 55m    |

Runs longer than a minute print a progress line on stderr every 60 seconds.

Tuning knobs. The defaults were measured on an M1 Max; other Apple GPUs may
tolerate more or need less:

- `NQ_METAL_MAX_ROWS` (default 13): most rows one GPU thread may solve. The CPU
  expands the frontier deeper to keep threads under this, so long-running threads
  do not trip the GPU watchdog.
- `NQ_METAL_BATCH` (default 65536): GPU threads per dispatch.
- `NQ_METAL_DISPATCHES_PER_CB` (default 16): dispatches per command buffer.
- `NQ_METAL_CHUNK` (default 8000000): partial boards buffered before each GPU
  flush; 24 bytes of memory each.

If the GPU reports an error, the failed work is retried once. If it fails again,
board sizes of 20 and up stop with an error rather than silently restarting on the
CPU, which would take hours to days; smaller boards fall back to the CPU.

## CUDA build

Run-time options specific to the CUDA build:
- GPU batch size (default 500000 tasks): `NQ_GPU_BATCH_TASKS=200000 ./nq 18`
- persistent-kernel block override (optional): `NQ_GPU_PERSIST_BLOCKS=256 ./nq 19`
- auto persist tuning target (tasks per block, default 8192): `NQ_GPU_TARGET_TASKS_PER_BLOCK=8192 ./nq 19`
- optional startup auto-tune of target (default off): `NQ_GPU_AUTOTUNE_TARGET=1 ./nq 19`
- auto-tune sample tasks (default 8192; use larger values like 65536+ on larger GPUs): `NQ_GPU_AUTOTUNE_SAMPLE_TASKS=16384 ./nq 19`
- GPU pipeline slots for overlapped batch execution (default 2, max 2): `NQ_GPU_PIPELINE_SLOTS=1 ./nq 19`
- GPU task bucketing (default off, opt-in): `NQ_GPU_BUCKET_TASKS=1 ./nq 19`
- subtree kernel: `peel` is the default for boards up to 31 (on an RTX 4060: N=19 in 12 s,
  N=20 in 103 s, 2.4-2.6x faster than `legacy`; see `CUDA_PEEL_PLAN.md`); it picks its own
  block count to fill the GPU unless `NQ_GPU_PERSIST_BLOCKS` is set. The old
  kernel: `NQ_GPU_KERNEL=legacy ./nq 19`

Mode benchmark harness (CPU vs GPU vs auto, CSV output):
- `./bench_modes.sh --ns "16 17 18 19 20" --repeats 3 --gpu-batch 500000 --frontier-depth 6`

## CUDA working notes

M4
nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq
baseline:
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 ./nq 19

Tune:
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 NQ_GPU_PERSIST_BLOCKS=64 ./nq 19
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 NQ_GPU_PERSIST_BLOCKS=128 ./nq 19
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 NQ_GPU_PERSIST_BLOCKS=256 ./nq 19

Created script: run_n19_m4_tests.sh

It runs all N=19 M4 commands:

baseline persistent auto blocks
NQ_GPU_PERSIST_BLOCKS=64
NQ_GPU_PERSIST_BLOCKS=128
NQ_GPU_PERSIST_BLOCKS=256

3/9
On your NVIDIA box, compare with and without bucketing:
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 NQ_GPU_PERSIST_BLOCKS=64 NQ_GPU_BUCKET_TASKS=1 ./nq 19
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 NQ_FRONTIER_DEPTH=6 NQ_GPU_BATCH_TASKS=500000 NQ_GPU_PERSIST_BLOCKS=64 NQ_GPU_BUCKET_TASKS=0 ./nq 19

to compile:

cd ~/projects/cuda
nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq


If you want architecture tuning for your RTX 4060 (Ada, sm_89), use:

nvcc -O3 -DUSE_CUDA=1 -arch=sm_89 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq

quick sanity run
NQ_SOLVER_MODE=gpu NQ_GPU_TIMING=1 ./nq 19

allagash98@Jeffs-MacBook-Pro-2 nqueens (main) $ ./nq 17
N Queens program by Jeff Somers.
  allagash98@yahoo.com or jsomers@alumni.williams.edu
Start:   Fri Mar  6 20:51:03 2026
End:  Fri Mar  6 20:51:06 2026
Calculations took 3 seconds.
For board size 17, 95815104 solutions found.
allagash98@Jeffs-MacBook-Pro-2 nqueens (main) $ ./nq 16
N Queens program by Jeff Somers.
  allagash98@yahoo.com or jsomers@alumni.williams.edu
Start:   Fri Mar  6 20:51:09 2026
End:  Fri Mar  6 20:51:10 2026
Calculations took 1 second.
For board size 16, 14772512 solutions found.
allagash98@Jeffs-MacBook-Pro-2 nqueens (main) $ ./nq 18
N Queens program by Jeff Somers.
  allagash98@yahoo.com or jsomers@alumni.williams.edu
Start:   Fri Mar  6 20:51:13 2026
End:  Fri Mar  6 20:51:37 2026
Calculations took 24 seconds.
For board size 18, 666090624 solutions found.
allagash98@Jeffs-MacBook-Pro-2 nqueens (main) $ ./nq 19
N Queens program by Jeff Somers.
  allagash98@yahoo.com or jsomers@alumni.williams.edu
Start:   Fri Mar  6 20:51:43 2026
End:  Fri Mar  6 20:54:56 2026
Calculations took 193 seconds.
Equals 3 minutes and 13 seconds.
For board size 19, 4968057848 solutions found.
allagash98@Jeffs-MacBook-Pro-2 nqueens (main) $ ./nq 20
N Queens program by Jeff Somers.
  allagash98@yahoo.com or jsomers@alumni.williams.edu
Start:   Fri Mar  6 20:56:44 2026

End:  Fri Mar  6 21:23:07 2026
Calculations took 1583 seconds.
Equals 26 minutes and 23 seconds.
For board size 20, 39029188884 solutions found.
