# Metal Backend Integration Plan

**Goal:** Fold `~/Development/nq_metal` back into this repo as a second GPU backend
(`USE_METAL`) behind the **same dispatch seam** the CUDA backend already uses.

**Outcome:** One codebase builds to a CPU-only binary everywhere, a CUDA binary on
Linux/NVIDIA (`-DUSE_CUDA`), or a Metal binary on macOS/Apple Silicon (`-DUSE_METAL`).
CUDA and Metal are mutually exclusive (enforced with `#error` if both are defined).

Status: **Phases 0 and 1 implemented on branch `metal-backend`; Phase 2 next** · Last updated 2026-11-11

Implementation notes (2026-11-11, M1 Max):
- The history import was done as a plain `git merge --allow-unrelated-histories`
  rather than `git subtree add`: a subtree merge renames every file inside the merge
  commit, which plain `git log --follow` does not trace. `nq.metal` and `nq_metal.m`
  kept their original paths through the merge and were renamed afterwards.
- Forced-GPU counts are correct for N = 2..18, and for the Phase 2 edge matrix
  (frontier depths 1/2/6/N−1, `NQ_METAL_MAX_ROWS` 1..23, tiny chunk and batch sizes).
- N=20 → 39,029,188,884 in 2m38s (standalone nq_metal: 2m37s). N=18 in 2.4 s.
- Not yet exercised: the GPU-failure retry/exit path (no failure has occurred), a
  real CUDA build, and the N=21/N=22 long runs.
- The built `./nq` binary is no longer tracked; `NQ_METAL_CHUNK` was added as a
  fourth tuning knob, mainly for tests.

### What changed in revision 2

Revision 1 was checked against `nq.c`, `nq_cuda.cu` and the nq_metal sources. Three
of its assumptions did not hold, and two of its four open decisions dissolved:

| Rev 1 said | What the code shows | Consequence |
|---|---|---|
| Front-end frontier feeds the kernel directly; nq_metal's `enumerate()` is dropped | The frontier is fully materialized at 72 B/task. At Metal's required depth that is 1.6 GB (N=20), 19.7 GB (N=21), 238 GB (N=22), ~2.9 TB (N=23) | **Two-level split** (§3): keep the shallow depth-6 frontier, expand to kernel depth *inside* the backend in bounded chunks. `enumerate()` is kept, not dropped |
| Backend asserts ≤ 13 residual rows or declines to CPU | Default frontier depth is 6 (`nq.c:106`), so every N ≥ 20 would have 14+ residual rows and always decline | Backend picks its own target depth; never declines for depth reasons |
| Kernel has a fixed compile-time `START_ROW`; needs generalizing (old 3.1a/b, Phase 3) | `START_ROW` is a Metal *function constant*, specialized at pipeline creation at runtime (`nq_metal.m:244-248`) | No kernel generalization needed. Old decision 1 and old Phase 3 removed |
| 32-bit kernel caps Metal at N ≤ 23; 64-bit kernel is an option | 32-bit `uint` masks work to N = 31; the `23` is only a stack-array define, and stack use is bounded by residual rows (≤ 10 frames), not N | No N cap below this repo's `MAX_BOARDSIZE` 27. Old decision 4 removed |

---

## 1. The seam (verified)

- **Declaration** — `nq.c:182-188`, `solveTasksGpuCuda(board_minus, mask, tasks, task_count, &total)`.
  Returns nonzero if it handled the work, 0 to decline (→ CPU).
- **Call site** — `nq.c:944-966` in `Nqueen()`: generate frontier → optionally bucket →
  try GPU → fall back to CPU.
- **Scaling** — `nq.c:969-976`: GPU path returns one count per leaf and `Nqueen()`
  multiplies by 2; CPU path returns orbit-weighted counts and divides by 12.
- **Interface struct** — `NqueenTask` (`nq_types.h:24-33`), 72 bytes. The GPU only
  needs `start_numrows`, `col`, `pos_diag`, `neg_diag`, `bitfield`; `col_0_idx` and
  `seed_positions` are CPU-symmetry fields and are ignored by Metal.

The guards around the GPU helpers (`shouldTryGpuSolver`, `getGpuMinTasks`, bucketing;
`nq.c:436-525`, `836-894`) are `#ifdef USE_CUDA` today. They become
`#if defined(USE_CUDA) || defined(USE_METAL)` via one `NQ_HAVE_GPU` macro, and the
call site dispatches through a single `solveTasksGpu` macro so there is one
`gpu_path_taken` branch, not two.

Semantics match exactly between the two repos: `start_numrows` = number of rows
already placed = index of the row being filled, same as the kernel's `numrows`;
diagonal shift directions are identical (`pos << 1`, `neg >> 1`); the odd-N
centre-column seeding is equivalent (`nq.c:685-719` vs `nq_metal.m:286-302`).

---

## 2. What each repo contributes

| Piece | Source | Role after integration |
|-------|--------|------------------------|
| Shallow frontier (`generateFrontier`, depth 6), mode selection, final ×2 | this repo `nq.c` | **Kept** — shared by all backends |
| `NqueenTask` | this repo `nq_types.h` | **Kept** — the backend contract, unchanged |
| CUDA backend | this repo `nq_cuda.cu` | Kept untouched |
| MSL kernel `nqueen` (4-row leaf peel, 32-bit) | nq_metal `nq.metal` | **Moved unchanged** to `nq_metal.metal` |
| `flush_to_gpu`, chunk/batch buffers, progress | nq_metal `nq_metal.m` | **Ported** into `nq_metal.m` behind `solveTasksGpuMetal` |
| `enumerate()` | nq_metal `nq_metal.m:156-180` | **Kept, re-rooted**: starts from each `NqueenTask` instead of row 0 |
| Depth rule `K = N − 13` | nq_metal `nq_metal.m:203-209` | **Kept inside the backend** as the target-depth rule (§3) |
| Row-0 seeding, `main()`, `g_total *= 2`, argv-based metallib path | nq_metal | **Dropped** — owned by `nq.c` |
| nq_metal's `nq.c` (2002 reference solver) | nq_metal | Dropped |

---

## 3. Design: two-level split

```
nq.c                      nq_metal.m (host)                         nq_metal.metal
generateFrontier(depth 6) → for each NqueenTask:                    
  ≤ 9.5M tasks, ≤ 0.7 GB      expand to target depth T (enumerate)  
                              → 16-byte PartialPlacement             
                              → chunk buffer (8M entries, 192 MB)    
                              → when full: flush_to_gpu ───────────→ nqueen kernel
                                 (64K-thread batches, ≤16 per CB,    one thread per
                                  wait, sum results on CPU)          partial
```

**Target depth.** `T = max(max start_numrows over tasks, N − NQ_METAL_MAX_ROWS)`,
clamped to `N − 1`; `NQ_METAL_MAX_ROWS` defaults to 13. This reproduces nq_metal's
validated configurations exactly: T = 6 for N ≤ 19 (no expansion, straight copy),
T = 7/8/9/10 for N = 20/21/22/23. `T` is passed to the kernel as the `START_ROW`
function constant, so every partial in a run has the same depth. Tasks whose
`start_numrows == T` are copied; shallower ones are expanded; partials with an empty
`bitfield` are skipped.

**Why this shape.**
- Memory stays bounded at the frontier (0.21 GB at N=20, 0.68 GB at N=23) plus the
  192 MB chunk, regardless of N. Measured frontier sizes at depth 6: 2.97M (N=20),
  4.47M (N=21), 6.57M (N=22), 9.45M (N=23).
- `nq.c` needs no depth awareness; `NQ_FRONTIER_DEPTH` keeps working for any value,
  including the mixed-depth case (depth 1 with odd N emits `start_numrows` 1 and 2).
- The watchdog cap lives next to the code it protects.
- Shallow tasks give a real progress denominator: "tasks consumed / task_count"
  replaces nq_metal's unbounded batch counter with a percentage.

**64 → 32-bit narrowing.** Truncating `col`/`pos_diag`/`neg_diag`/`bitfield` to
`uint32_t` is exact for N ≤ 31: the only bits lost are `pos_diag` bits ≥ 32, which
have been shifted past the board and can never shift back. Guard: decline (return 0)
if `board_minus + 1 > 27` — unreachable today, but keeps the backend honest if
`MAX_BOARDSIZE` is ever raised past 31.

**Kernel stack.** `stk[MAX_BOARDSIZE + 2]` stays as is (25 frames). Frames are only
pushed while `numrows + 5 <= board_minus`, so depth ≤ `NQ_METAL_MAX_ROWS − 3`; the
host clamps `NQ_METAL_MAX_ROWS` to ≤ 23 so this can never overflow. Do not resize the
array in Phase 1 — nq_metal's notes record that stack size affected performance.

---

## 4. Correctness contract: symmetry convention

This is the highest-risk item: a mismatch produces a wrong count that looks plausible.

- The frontier emitted to GPUs is **vertical-mirror-only**: row-0 queen in the lower
  half (plus the odd-N centre seeds), with **no** `d_main` prune
  (`nq.c:619-625`). The kernel counts every leaf once; `Nqueen()` doubles.
- Therefore `solveTasksGpuMetal` returns the raw leaf sum. **nq_metal's
  `g_total *= 2` (`nq_metal.m:308`) must not be ported.**
- **Standing constraint on future work:** the comment at `nq.c:619-625` plans to move
  the `d_main` prune into frontier generation once the CUDA kernel adopts orbit
  weighting (`SYMMETRY_OPTIMIZATION_PLAN.md` Phase 1 step 5). The Metal kernel cannot
  follow cheaply — its speed comes from `popcount` leaf peeling, which never
  materializes per-leaf positions. When that migration happens, frontier generation
  must keep a mirror-only mode for Metal, and the `gpu_path_taken` scaling must
  become per-backend. Add a comment at `nq.c:619` saying so in Phase 0.

---

## 5. Metal invariants to carry over

Copy these into the header comment of the new `nq_metal.m` (source: nq_metal
`CLAUDE.md`, measured on M1 Max, 2026-03-19/20):

1. **No sliding window on one command queue.** Commit → `waitUntilCompleted` →
   collect → next. A failed command buffer fails every CB queued behind it on the
   same queue. Multiple dispatches inside one CB (≤ `DISPATCHES_PER_CB` = 16) are fine.
2. **13 residual rows maximum per thread.** 14 rows hit the GPU watchdog at N=20, 21
   and 22. `BATCH_SIZE` 65536 is safe; 131072 crashes.
3. **Function constants change codegen.** Keep `BOARD_SIZE`/`START_ROW` as the only
   function constants; anything loop-bounding stays a `#define`.

These limits were measured on one machine. Expose `NQ_METAL_MAX_ROWS`,
`NQ_METAL_BATCH` and `NQ_METAL_DISPATCHES_PER_CB` as env overrides (same style as the
`NQ_GPU_*` knobs) so other Apple GPUs can be tuned without a rebuild.

New in the port (nq_metal did not need these as a standalone tool):

- **ARC + `@autoreleasepool` per command buffer.** nq_metal builds without
  `-fobjc-arc` and has no pool, so every CB/encoder leaks for the life of the run.
  Build with `-fobjc-arc` and wrap each CB iteration in `@autoreleasepool`.
- **No partial totals, and `exit()` only for a repeated GPU failure at N ≥ 20.** See
  decision 3 in §8.
- **Metal state is created per call and released on return** — no file-scope globals
  outliving `solveTasksGpuMetal`.

---

## 6. Build

- Keep the host file as **`nq_metal.m`** (Objective-C), not `.mm`. nq_metal's host is
  already plain ObjC; `.m` gives C linkage for `solveTasksGpuMetal` with no
  `extern "C"` and no libc++ link.
- New `Makefile` with `cpu`, `cuda`, `metal`, `test-metal`, `clean`:
  ```make
  METAL    = xcrun -sdk macosx metal
  METALLIB = xcrun -sdk macosx metallib

  cpu:
  	cc -O3 -Wall -Wextra -pthread nq.c -o nq
  cuda:
  	nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq
  nq_metal.metallib: nq_metal.metal
  	$(METAL) -O2 -c $< -o nq_metal.air && $(METALLIB) nq_metal.air -o $@
  metal: nq_metal.metallib
  	clang -O3 -DUSE_METAL=1 -fobjc-arc -Wall -Wextra -pthread nq.c nq_metal.m \
  	      -framework Metal -framework Foundation -o nq
  ```
  Kernel flags stay at `-O2`, identical to nq_metal, so timings are comparable.
- The `metal` compiler ships with full Xcode, not the Command Line Tools. It is
  present on this machine (metal 32023.864).
- **metallib location:** resolve next to the executable with `_NSGetExecutablePath`
  (the backend has no `argv`), overridable with `NQ_METALLIB=<path>`. If it cannot be
  loaded, print why and return 0.
- `.gitignore`: add `*.air`, `*.metallib`, and the built binaries (the repo has no
  `.gitignore` today; `nq`, `nq_mt`, `nq_optim` etc. show up as untracked).

---

## 7. Phased execution

**Phase 0 — Prep and scaffold (no behavior change).**
1. Import nq_metal's history:
   - In nq_metal: commit the working tree first. `RESULTS.md` is modified but
     uncommitted and holds the only record of the 4-row-peel timings (including the
     37h35m N=23 run); the later `nq21_*`/`nq22_*`/`nq23_*` run logs are untracked.
     Leave `nq.air` and `nq.metallib` out.
   - Here, on a feature branch: `git subtree add --prefix=metal_import
     ~/Development/nq_metal main` (no `--squash`). This needs no modified tracked
     files; untracked files are fine.
   - One follow-up commit: `git mv metal_import/nq.metal nq_metal.metal`,
     `git mv metal_import/nq_metal.m nq_metal.m`, move `RESULTS.md` to
     `METAL_RESULTS.md`, and `git rm` the tracked binaries (`nq`, `nq_metal`), the
     2002 `nq.c` and the old `Makefile`. Run logs and the two notes files stay under
     `metal_import/` as the historical record.
2. Replace the body of the imported `nq_metal.m` `main()` path with a stub
   `solveTasksGpuMetal` returning 0 (the real port is Phase 1, done as edits to this
   file so the diff against the original stays readable). Add `Makefile`, `.gitignore`.
3. `nq.c`: `NQ_HAVE_GPU` guard, `#error` on both backends, Metal declaration,
   unified call site, the §4 comment.
4. Verify: `make cpu` and `make metal` both give correct counts for N = 4..16 (the
   stub falls through to CPU); `cc -DUSE_CUDA -fsyntax-only -Wall -Wextra nq.c` is
   clean (works on this Mac today, no nvcc needed). A real CUDA build + N=16 run on
   the NVIDIA box before merging.

**Phase 1 — Working Metal path.**
Port device/library/pipeline setup, `flush_to_gpu`, and the re-rooted `enumerate()`
per §3. Skip CUDA-style bucketing for Metal (order is irrelevant after expansion).
All counters 64-bit. Exit criterion: `NQ_SOLVER_MODE=gpu ./nq N` correct for N = 2..18.

**Phase 2 — Correctness and performance gate.**
- `make test-metal`: forced-GPU counts vs the known table (`nq.c:68-90`) for
  N = 2..18, plus forced-CPU cross-check. Runs in well under a minute.
- Edge matrix, each compared to the table: `NQ_FRONTIER_DEPTH` ∈ {1, 2, 6, N−1} for
  one odd and one even N (e.g. 13, 14); `NQ_METAL_MAX_ROWS` ∈ {8, 13} at N = 16 to
  exercise real expansion on a fast run; N ≤ 6 where depth clamps to N−1.
- Long runs: N=20 → 39,029,188,884 and N=21 → 314,666,222,712.
- Performance parity on the M1 Max against `nq_metal/RESULTS.md`: N=20 within ~10% of
  2m37s, N=21 within ~10% of 23m45s. A regression here means the expansion path or
  ARC changes cost something and must be found before going further.
- N=22 (3h50m) once, before retiring nq_metal. N=23 is optional (37h).

**Phase 3 — Parity and polish.**
- Auto-mode threshold: `NQ_GPU_MIN_TASKS` defaults to 600,000, a CUDA-tuned number
  that at depth 6 means "GPU from N=17 up". Measure the CPU/Metal crossover on the
  M1 Max (the CPU path is now orbit-weighted and faster than when nq_metal was
  benchmarked) and give Metal its own default.
- Progress line every 60 s with percent of shallow tasks consumed; print the device
  name once; honor `NQ_GPU_TIMING`.
- `readme.md` build matrix and Metal env knobs, pointing at `METAL_RESULTS.md`.
- Stop developing in `~/Development/nq_metal`; leave a README there pointing here.

**Later, only if wanted.**
- Embed the metallib (`-sectcreate` + `newLibraryWithData:`) for a single-file binary.
- Overlap CPU expansion of the next chunk with the in-flight command buffer (does not
  violate invariant 1 — it is CPU work, not a second queued CB).
- Resume file keyed on shallow-task index + running total, for multi-day N ≥ 23 runs.

---

## 8. Decisions (settled 2026-11-11)

1. **metallib packaging: load from next to the executable.** Resolved with
   `_NSGetExecutablePath`, `NQ_METALLIB` override (§6). Embedding stays in the
   "later" list.
2. **nq_metal disposition: import its commits into this repo.** Its history is
   brought in unsquashed with `git subtree add` (Phase 0 step 1); the ported files
   are then `git mv`'d into place so `git log --follow` reaches back to the original
   Metal port. After Phase 2 passes, nq_metal is no longer developed.
3. **GPU failure mid-run: exit loudly for N ≥ 20.** On a command-buffer error, retry
   that CB once on a fresh queue with one dispatch per CB. If it fails again:
   - **N ≥ 20:** print the Metal error, the shallow-task index and the running
     total to stderr, and `exit(1)`. No silent CPU restart.
   - **N < 20:** print a warning and return 0, so the CPU solves from scratch (seconds
     to minutes at these sizes).

   Either way a partial GPU total is never returned. This is the one place the
   backend calls `exit()`; setup failures (no device, metallib missing, pipeline
   error) happen before any work is done and always return 0.

---

## 9. Files

- **Imported with history, then moved:** `nq_metal.metal` (from `nq.metal`),
  `nq_metal.m`, `METAL_RESULTS.md`; remaining notes and run logs under `metal_import/`.
- **Add:** `Makefile`, `.gitignore`.
- **Edit:** `nq.c` (guards, declaration, call site, comment — roughly 25 lines),
  `readme.md`.
- **Unchanged:** `nq_types.h`, `nq_cuda.cu`, frontier generation, CPU solver.
