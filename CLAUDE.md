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

**Rule**: always dispatch a batch, `waitUntilCompleted`, collect results, and
drain checkpoints **before** submitting the next batch. The current sequential
loop in `flush_to_gpu()` follows this pattern. Do not re-introduce a sliding
window without first using **separate command queues** per slot (each queue
serializes independently, so failures don't cascade).

### ITER_BUDGET must be a static `#define`, not a function constant

`ITER_BUDGET` controls how many main-loop iterations a GPU thread runs before
checkpointing. It **must** be a `#define` in `nq.metal`, not a Metal function
constant (`[[function_constant]]`).

When ITER_BUDGET was changed to a `constant uint [[function_constant(2)]]`,
N=20 crashed with `Internal Error (0000000e)` at a budget of 1,000,000 — even
though it ran correctly with `#define ITER_BUDGET 2000000u` and `500000u`.
The Metal compiler generates different (less optimized) code for function
constants vs compile-time literals, changing GPU timing and resource usage
in ways that trigger the watchdog or other internal errors.

### ITER_BUDGET must account for leaf-peeling inner work

The `iters` counter only counts main-loop iterations, **not** the O(N²) inner
work in 3-row leaf-peeling cases. With 3-row peeling, the actual GPU work per
thread is roughly `ITER_BUDGET × N²`.

- `ITER_BUDGET=2,000,000`: N=21 crashed at 14m (watchdog on heavy batch)
- `ITER_BUDGET=500,000`: N=22 crashed at 23m (still too high for N=22)
- `ITER_BUDGET=200,000`: N=20 completes in 4m16s (correct, ~2x vs baseline 218s)

**Rule**: when adding deeper leaf peeling (e.g. 4-row), reduce ITER_BUDGET
proportionally. The product `ITER_BUDGET × (leaf inner work)` must keep the
worst-case batch under ~30 seconds (half the watchdog, for safety margin).
Current values: 3-row peeling with `#define ITER_BUDGET 200000u`.
