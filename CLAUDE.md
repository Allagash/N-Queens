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
