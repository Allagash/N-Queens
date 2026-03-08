Compile with cc -O3 -Wall -Wextra -pthread nq.c -o nq

For machine-specific, and less portable:
cc -O3 -march=native  -Wall -Wextra -pthread nq.c -o nq

CUDA build (M2 path, requires CUDA toolkit + GPU):
nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq

Runtime mode selection with CUDA build:
- default / auto: tries GPU, falls back to CPU
- auto-mode GPU threshold (default 4096 tasks): `NQ_GPU_MIN_TASKS=8192 ./nq 16`
- GPU batch size (default 500000 tasks): `NQ_GPU_BATCH_TASKS=200000 ./nq 18`
- force CPU: `NQ_SOLVER_MODE=cpu ./nq 16`
- force GPU attempt: `NQ_SOLVER_MODE=gpu ./nq 16`
- enable GPU timing metrics: `NQ_GPU_TIMING=1 ./nq 16`

Frontier depth tuning:
- `NQ_FRONTIER_DEPTH=<depth> ./nq <N>`

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
