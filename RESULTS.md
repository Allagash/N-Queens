# N-Queens Metal Solver — Timing Results

Hardware: Apple M1 Max, 32 GB unified memory

| N  | Solutions             | K  | Time         |
|----|-----------------------|----|--------------|
| 20 | 39,029,188,884        | 8  | 10m 17s      |
| 21 | 314,666,222,712       | 8  | 1h 27m 44s   |
| 22 | 2,691,008,701,644     | 10 | 14h 1m 22s   |

**K** is the number of rows pre-enumerated by CPU; GPU threads solve the remaining N−K rows each.
