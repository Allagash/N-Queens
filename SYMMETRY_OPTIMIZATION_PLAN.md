# Symmetry Optimization Plan for N-Queens

## Goal
- Reduce work by exploiting more of the dihedral D4 symmetry group, beyond the single vertical mirror that the current code uses.
- Preserve exact correctness against the published `A000170` solution counts for N=8 through 20.
- Apply gains to both the CPU and the GPU paths.

## Current state

The existing code (`nq.c:484-548`) exploits only the **vertical mirror** symmetry: row 0's queen is restricted to columns `[0, N/2)`, with a separate handler for the odd-N center column, and the final count is multiplied by 2 (`nq.c:785`). This gives a factor of 2.

## The full symmetry group

The N×N board's symmetry group is **D4 = {e, r, r², r³, m_v, m_h, d_main, d_anti}**:
- 4 rotations (identity, 90°, 180°, 270°)
- 4 reflections (vertical, horizontal, main diagonal, anti-diagonal)

|D4| = 8. For typical N≥10, almost every solution orbit has stabilizer {e}, so orbit size = 8. A small minority have orbit size 4 (180°-symmetric or diagonal-involutional). Practically none have size 2 or smaller for N≥3. So `total / orbits → 8` as N grows.

The theoretical maximum reduction from full D4 exploitation is therefore close to 8×.

## Why this is harder than it looks

Of D4's eight elements, only **vertical mirror** acts on the column of row 0 in a way that factors through the row-by-row DFS. The other seven involve row N−1, the inverse permutation, or a row↔column swap. So they cannot be applied as a clean "row-0 column choice" pruning rule.

| symmetry | how it changes col_0 | DFS-tree pruning | leaf-level pruning | typical reduction |
|---------|----------------------|------------------|---------------------|-------------------|
| identity | col_0 | — | — | — |
| **vertical mirror** (current) | N−1−col_0 | yes — restrict col_0 ∈ `[0, N/2)` | — | 2× |
| horizontal mirror | depends on col_{N−1} | no | partial | small |
| 180° rotation | depends on col_{N−1} | no | partial | small |
| 90° / 270° rotation | depends on σ⁻¹(0) | no | partial | small |
| main diagonal | σ⁻¹(0) — row of queen in col 0 | weak (sees mid-search) | partial | moderate |
| anti-diagonal | N−1−σ⁻¹(N−1) | weak | partial | moderate |

The main-diagonal one is the only non-trivial extension that yields any tree-level pruning, by tracking "row of queen in column 0" during search.

## Realistic gain projection

For typical N=19/20 GPU runtime (~27s / ~210s on RTX 4060):

| approach | added symmetry | realistic speedup over current |
|----------|---------------|------------------------------|
| **A.** Per-leaf canonical-form filter only (no DFS prune) | full D4 (correctness only) | 1.0× or slightly slower (overhead) |
| **B.** Add main-diagonal DFS pruning + per-leaf orbit weighting | d_main during search | 1.3–1.8× |
| **C.** B + add 180°-rotation per-leaf filter | r² added | 1.4–2.0× |
| **D.** Burnside-style decomposition: total = canonical_search × 8 − corrections from `Fix(g)` counts | full D4 indirect | up to ~3× (substantial refactor) |
| **E.** Walker-style canonical-form enumeration (dual placement of σ and σ⁻¹) | full D4 native | up to ~4× (algorithmic rewrite) |

The 8× theoretical limit is not reachable without restructuring the DFS away from row-by-row.

## Phase 1: main-diagonal DFS pruning + per-leaf orbit weighting

This is the highest-leverage *small-ish* change.

### Idea

Diagonal reflection sends queen at `(0, c0)` to `(c0, 0)`. Pair every solution with its diagonal mirror by tracking `row_of_col0 = σ⁻¹(0)` (which row contains the queen in column 0). To break the d_main pairing, prune partial solutions where the queen in column 0 would be placed at a row `r < col_0_idx`.

This eliminates whole DFS subtrees whenever the column-0 queen is forced into a small row.

### Per-leaf orbit weighting

DFS pruning loses some orbit representatives. To recover the correct total without a static multiplier, each leaf's contribution is computed from its orbit structure:

For each leaf solution σ:
1. Compute all 8 D4 transforms of σ's column vector.
2. Let `s` = number of distinct transforms (= orbit size).
3. Let `q` = number of transforms that lie in the *pruned* domain (`col_0 ∈ [0, N/2)` AND `row_of_col0 ≥ col_0`).
4. Leaf weight = `s / q`.

Sum of leaf weights = total solutions.

Possible values: `s ∈ {1, 2, 4, 8}`, `q ∈ {1, 2, 3, 4}`, so `s/q ∈ {1, 4/3, 2, 8/3, 4}` for the cases that occur in practice. To stay in integer arithmetic, accumulate `12 × s/q` and divide by 12 at the end (LCM of plausible denominators is 12).

### Per-leaf cost

Eight transforms × O(N) work + uniqueness/in-domain analysis ≈ 200–400 ops at N=20. For ~5×10⁹ leaves at N=19 that's ~10¹² ops — order 1–2 s on the RTX 4060. The DFS-tree savings need to exceed this for a net gain.

### Implementation steps

1. `nq_types.h`: add `col_0_idx` to `NqueenTask` (column index, 0..N−1, of the row-0 queen). Possibly also a sentinel for `row_of_col0`.
2. `nq.c` frontier generation: populate `col_0_idx` for every emitted task. Easy — it's the position of the lsb in `seed.col`.
3. `nq.c` `solveTask`: 
   - Track a `positions[]` array (column placed in each row) so the leaf has the full column vector available for orbit analysis.
   - Track `row_of_col0` while descending; when placing a queen in column 0 at row `r`, prune if `r < col_0_idx`.
   - At leaf, compute the 12× weight via the orbit analysis above and accumulate.
4. `nq.c` `Nqueen`: replace the final `× 2` with `/ 12` of the accumulated 12×-weighted total.
5. `nq_cuda.cu`: mirror the same changes inside `solveTaskDevice` and `solveTasksKernel`. The `positions[]` array adds one BITFIELDTYPE per row to per-thread shared memory, so `depth_stride` rises by 1; check it still fits.
6. Validate: counts must equal the published series for N=8..16 in a quick check, and N=17..20 in a longer one.

### Risks

- **Per-leaf cost > tree savings.** If d_main pruning saves only a few percent of subtrees but the leaf analysis costs 10%, this is a net loss. The tree-savings ratio is empirical; we need to benchmark before committing. **Mitigation**: keep the symmetry behavior behind a runtime flag (`NQ_SYMMETRY=full|mirror`) so we can A/B compare.
- **Shared-memory pressure on GPU.** Adding `positions[]` to per-thread state grows shared memory by `8 × N` bytes per thread. At block size 64 and N=20, that's another ~10 KB per block, eating into the occupancy gains we just won. **Mitigation**: `positions[]` is only needed at leaf time; instead of a separate array, reconstruct it from `s_col[]` at leaf (lsb-of-difference). Costs N XORs, no memory.
- **Off-by-one in canonicalization.** Bugs here produce wrong counts that look superficially plausible. **Mitigation**: validate against `A000170` for every N=8..16 on every commit.

### Validation gate

Phase 1 must produce these exact counts before merging:

```
N=8  →             92
N=9  →            352
N=10 →            724
N=11 →          2,680
N=12 →         14,200
N=13 →         73,712
N=14 →        365,596
N=15 →      2,279,184
N=16 →     14,772,512
N=17 →     95,815,104
N=18 →    666,090,624
N=19 →  4,968,057,848
N=20 → 39,029,188,884
```

## Phase 2: 180°-rotation per-leaf filter

After Phase 1 lands.

For each leaf, the 180° rotation `r²` sends `(col_0, ..., col_{N−1})` to `(N−1−col_{N−1}, ..., N−1−col_0)`. Including `r²` in the orbit analysis (which Phase 1 already does) is automatic — Phase 1's per-leaf machinery already considers all 8 D4 transforms.

What Phase 2 *adds* is a leaf-level **early-out**: as soon as the column vector clearly cannot be the lex-smallest in its orbit (e.g., some transform's prefix already beats it), stop the orbit analysis and return weight 0. This trades a fraction of the per-leaf cost for fewer comparisons in the common asymmetric-orbit case.

Realistic additional gain over Phase 1: 5–10%.

## Phase 3 (optional, large refactor): Burnside decomposition

Rather than enumerate solutions and weight them, compute total via:

```
8 × |orbits| = |Fix(e)| + |Fix(r)| + |Fix(r²)| + |Fix(r³)|
             + |Fix(m_v)| + |Fix(m_h)| + |Fix(d_main)| + |Fix(d_anti)|
```

For N≥2:
- `|Fix(m_v)| = |Fix(m_h)| = 0` (no n-queens solution is mirror-symmetric for N≥2).
- `|Fix(r)| = |Fix(r³)|`: 90°-symmetric solutions; very few; specialized backtracker.
- `|Fix(r²)|`: 180°-symmetric solutions; specialized backtracker on N/2 rows.
- `|Fix(d_main)|`: solutions where σ is an involution; specialized backtracker on involution permutations.
- `|Fix(d_anti)|`: similar.
- `|Fix(e)| = total` (the unknown we want).

Solve for total:
```
total = 8 × |orbits| − (|Fix(r)| + |Fix(r²)| + |Fix(r³)| + |Fix(d_main)| + |Fix(d_anti)|)
```

We compute `|orbits|` (= fundamental count) by canonical-form enumeration, and the `|Fix(g)|` counts by separate small backtrackers. Each `|Fix(g)|` count is several orders of magnitude smaller than the full enumeration, so the bulk of the work shifts to the `|orbits|` count, which itself can be ~`total / 8`.

Realistic gain: up to ~3× over Phase 1, but at significant code complexity. Defer unless Phase 1 + 2 don't deliver enough.

## Validation strategy across all phases

1. A `--check` mode (or env var `NQ_SELF_TEST=1`) that runs N=8..14 internally on every binary invocation and aborts if any count is off. Adds <0.1 s.
2. Manual spot checks at N=15..20 after each phase.
3. Cross-check CPU and GPU paths against each other: counts must match for the same N.
4. Both `mirror` mode (current code path, kept as a fallback) and `full` mode (the new symmetry path) must produce identical totals. Runtime flag `NQ_SYMMETRY=mirror|full` for A/B benchmarks.

## Expected end-to-end progression

If Phase 1 lands as projected:

| N  | current GPU | after Phase 1 | speedup vs CPU baseline |
|----|-------------|---------------|------------------------|
| 19 | 27 s        | ~17 s         | ~10× |
| 20 | 212 s       | ~135 s        | ~10× |

If Phase 2 also lands:

| N  | after Phase 2 | speedup vs CPU |
|----|---------------|---------------|
| 19 | ~14 s         | ~12× |
| 20 | ~110 s        | ~13× |

Phase 3 could push these to ~7–9 s and ~70 s respectively, but with materially more risk.
