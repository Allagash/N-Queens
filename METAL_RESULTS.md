# N-Queens Metal Solver — Timing Results

Hardware: Apple M1 Max, 32 GB unified memory

## Integrated backend (`make metal` in this repo, November 2026)

Same kernel and settings as the standalone "Current" table below; the frontier comes
from `nq.c` and is expanded to row K inside `nq_metal.m`.

| N  | Solutions             | K  | Time       | vs standalone |
|----|-----------------------|----|------------|---------------|
| 18 | 666,090,624           | 6  | 2.4s       | —             |
| 20 | 39,029,188,884        | 7  | 2m 38s     | +0.6%         |
| 21 | 314,666,222,712       | 8  | 24m 01s    | +1.1%         |
| 22 | 2,691,008,701,644     | 9  | 3h 55m     | +2.3%         |

The N=22 run hit two transient command-buffer failures (Internal Error 0000000e)
about six minutes in; both were retried successfully. The standalone solver exits on
that error.

The tables below are from the standalone `nq_metal` repository, before integration.

## Current (4-row leaf peeling, no checkpointing, K=N-13)

| N  | Solutions             | K  | Time       | vs Original |
|----|-----------------------|----|------------|-------------|
| 20 | 39,029,188,884        | 7  | 2m 37s     | 3.9x        |
| 21 | 314,666,222,712       | 8  | 23m 45s    | 3.7x        |
| 22 | 2,691,008,701,644     | 9  | 3h 50m     | 3.7x        |
| 23 | 24,233,937,684,440    | 10 | 37h 35m    | —           |

## Original (3-row leaf peeling, checkpointing, fixed K)

| N  | Solutions             | K  | Time         |
|----|-----------------------|----|--------------|
| 20 | 39,029,188,884        | 8  | 10m 17s      |
| 21 | 314,666,222,712       | 8  | 1h 27m 44s   |
| 22 | 2,691,008,701,644     | 10 | 14h 1m 22s   |

**K** is the number of rows pre-enumerated by CPU; GPU threads solve the remaining N-K rows each.
