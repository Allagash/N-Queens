# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file C program (`nq.c`) that solves the **N-Queens problem** — finding all ways to place N queens on an N×N chessboard such that no two queens attack each other.

**Usage**: `./nq N` where N is the board size (2–18 on Unix/Linux, 2–21 on Windows)

## Building

No build system is configured. Compile directly:

```bash
gcc -O2 -o nq nq.c
# or
clang -O2 -o nq nq.c
```

## Architecture

The entire implementation lives in `nq.c` (~404 lines). Key functions:

- **`main()`** — validates input, invokes solver, reports timing via `time()`/`difftime()`
- **`Nqueen(int board_size)`** — core solver: iterative backtracking with an explicit manual stack (not recursive)
- **`printResults()`** — formats and prints solution count and elapsed time

### Algorithm

Uses **bit manipulation** throughout for performance:
- `aQueenBitCol[]`, `aQueenBitPosDiag[]`, `aQueenBitNegDiag[]` — bitfields tracking occupied columns and diagonals
- LSB extraction via 2's-complement trick: `lsb = -((signed)bitfield) & bitfield`
- **Symmetry optimization**: only solves for half the board width in the first row, then doubles the count (mirrors across Y-axis)

### Platform differences

```c
#ifdef WIN32
  #define MAX_BOARDSIZE 21        // 64-bit: unsigned __int64
#else
  #define MAX_BOARDSIZE 18        // 32-bit: unsigned long
#endif
```

No external dependencies — only `<stdio.h>`, `<stdlib.h>`, `<time.h>`.

## Critical Invariants

### No sliding window dispatch on a single Metal command queue

A single `MTLCommandQueue` serializes command buffers — queuing multiple batches
does **not** give GPU parallelism. Worse, if any command buffer fails (watchdog
timeout, internal error), **all subsequently queued command buffers on the same
queue also fail**, producing misleading "Internal Error (0000000e)" messages on
whichever CB you happen to `waitUntilCompleted` first.

This was observed on 2026-03-19: a sliding-window scheme (WINDOW_SIZE=6) that
queued multiple primary batches before waiting crashed reliably on N=21 with
`GPU resume error (slot 0): Internal Error`. The resume kernel's CB was queued
behind other primary CBs; when an earlier one failed, the failure cascaded.

**Rule**: always dispatch a batch, `waitUntilCompleted`, collect results
**before** submitting the next batch. The current sequential loop in
`flush_to_gpu()` follows this pattern (with multi-dispatch encoding up to
DISPATCHES_PER_CB batches per CB, which is safe since it's a single CB).
Do not re-introduce a sliding window without first using **separate command
queues** per slot.

### Checkpointing was removed — use K tuning instead

As of 2026-03-20, checkpointing (ITER_BUDGET, nqueen_resume kernel,
drain_checkpoints) was removed. Instead, K is raised high enough that
individual GPU threads complete within the watchdog window:

- For N≥20: K = N - 13 (each thread handles exactly 13 rows)
- For N≤19: K=6 or less (threads are light enough at any depth)

**The max safe thread depth is 13 rows** (9 stack + 4 peeled). 14 rows
crashes reliably:
- K=7 for N=21 → 14 rows → crashed at 14m (2026-03-20)
- K=8 for N=22 → 14 rows → crashed at 1m22s (2026-03-20)
- K=6 for N=20 → 14 rows → crashed at 2m51s (2026-03-20)

K = N - 13 gives: K=7 (N=20), K=8 (N=21), K=9 (N=22), K=10 (N=23).

BATCH_SIZE=65536 with DISPATCHES_PER_CB=16 keeps each CB under the
watchdog. BATCH_SIZE=131072 crashes — too many threads per dispatch.

Historical note: ITER_BUDGET as a `[[function_constant]]` crashed even at
values that worked as `#define`. The Metal compiler generates different code
for function constants vs compile-time literals.
