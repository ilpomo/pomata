# The pomata test policy

This document is the method of record for how pomata is tested. It exists to end a specific failure mode: an audit
finds an edge case handled or tested one way in one function and another way in its neighbour, a fix lands for that one
function, and the next audit finds the same class of difference in the function next door. That loop is not a shortage
of effort -- it is the absence of a *method*. The method is this:

> **We believe the registry and test everything uniformly. Where a function's edge-case logic genuinely differs -- and
> it will, because a rolling mean, an EMA, and a reducing ratio cannot treat a `null` identically -- we make that
> difference an explicit, documented variant, declared in one typed field. A function may depart from the baseline
> only by *declaring* it. Never by drift.**

The single source of truth is [`tests/support/registry.py`](support/registry.py): one typed `FunctionProfile` row per
public function. The shared contract suite reads each row and tests each function *by its own declared rules*, so the
whole package is exercised from one place and every genuine difference is stated, not scattered. The registry is kept
honest by [`tests/test_registry.py`](test_registry.py): a function that behaves differently from the profile it
declares -- or is added without a row -- is a red build, not a finding for the next audit.

---

## 1. The precision floor: test the math, not the hardware

pomata proves numerical correctness to a **principled, conditioning-derived tolerance**, and **not one digit tighter**.
The tolerances are named constants in [`tests/support/tolerances.py`](support/tolerances.py); that module is the single
source of truth for "how close is close enough", and no test may assert closer than the tier it belongs to.

| tier | relative | used for |
|------|----------|----------|
| `EXACT` | `1e-12` | a golden master on a fixed, deterministic series |
| `REFERENCE` | `1e-10` | agreement with the independent oracle |
| `PROPERTY` | `1e-10` | agreement under random (fuzzed) input |
| `SCALE` | `1e-6` | a rescaling test, where the rescale itself amplifies rounding |
| `STREAMING` | `1e-3` (abs) | a sequential recursion against a two-pass reference |

plus the conditioning floors `CONDITIONING_FLOOR`, `STANDARDIZED_MOMENT_FLOOR`, and `SUBNORMAL_FLOOR` for outputs that
cancel toward zero or approach the subnormal range.

**The rule.** A difference below the floor is hardware, not pomata, and we do not test it:

- **No bit-exact / sub-ULP / `== 0.0` assertion on fuzzed or cross-platform inputs.** A sub-ULP result depends on the
  CPU (x86 vs arm64), FMA contraction, the Polars version, and the order the query optimizer chooses -- none of which
  pomata owns. Asserting it is a test *designed* to flake across the CI matrix (this is the exact shape of the CDaR
  knife-edge flake that a prior audit chased for a full cycle).
- **Golden masters are rounded** (`.round(n)`) to a sane precision, so a frozen value never flakes between macOS-arm64,
  ubuntu-x86, and windows.
- **A magnitude-dependent band is sized to the data** (`input_scale ** degree * factor`), never fixed, so it is right
  at every scale; where a statistic legitimately needs a looser band (a square root that amplifies near zero, a
  difference of large terms that cancels), the departure carries a one-line reason.

The floor is not a compromise on rigor. It *is* the rigor: it draws the line exactly where "a difference means a real
coding error" ends and "a difference means IEEE-754 being IEEE-754" begins, and holds every function to the same line.

## 2. The taxonomy: the axes that decide a function's obligations

A function's edge-case obligations are fixed by *where it sits on a few axes*, not by which file it lives in. Same
coordinates → same tests, same names, same order. The axes are the fields of `FunctionProfile`:

- **macro** — `indicators` · `pnl` · `metrics`.
- **shape** — `REDUCING` (series → one scalar) · `ELEMENTWISE` (series → series) · `STRUCT` (a multi-line output).
- **domain** / **columns** — `SINGLE` · `PAIRED` (returns + benchmark) · `OHLCV` (with the column subset) · `EQUITY`.
- **windowed** — whether it takes a lookback `window`.
- **null_policy** / **nan_policy** — how an interior `null` / `NaN` flows (defined crisply in §3).
- **oracle** — the independent `*_reference` it is checked against, or `None` (component-definition / golden only).

A **scale** axis — a function's rescaling homogeneity and boundedness — belongs here too, but is intentionally left for
its own phase: it needs a richer per-input-degree model (a variance is degree-2; a VWAP is degree-1 in price and
degree-0 in volume; some oscillators are bounded only for well-formed bars) and its verification *is* the numeric scale
contract (`f(k·x)` vs `k^degree·f(x)`), not the structural flow probe the other axes use. It lands with that contract.

## 3. The null and NaN policies, defined crisply

This is the heart of the "explicit variant" idea: a `null` cannot mean the same thing to a pointwise transform, a
rolling window, a recursion, and a reduction, so each **declares** which it is. The shared contract then asserts the
matching behaviour, and the self-check proves the declaration is true of the code.

**`null_policy` — an interior `null` in the input:**

- **`SKIPPED`** *(reducing only)* — the `null` is dropped from the reduction; the scalar is exactly what it would be if
  the `null` were absent. *e.g. `sharpe_ratio`, `value_at_risk`.*
- **`PROPAGATES`** *(elementwise)* — the `null` nulls at most its own output row (and a fixed lag); the output recovers
  immediately after. A pointwise map. *e.g. `price_average`, `mom`, `returns_simple`.*
- **`IN_WINDOW_IS_NULL`** *(elementwise)* — a single `null` nulls **every rolling window that overlaps it** — about
  `window` rows — then the output recovers. *e.g. `sma`, `rsi_stochastic`, `value_at_risk_rolling`.*
- **`BRIDGED`** *(elementwise)* — a **recursion steps over** the `null`: its state carries across the gap, so later
  rows recover to the value they would have had. Identifiable because the same function **latches** a `NaN` (§ pairing
  below). *e.g. `ema`, `rma`, `atr`, `dema`.*
- **`LATCHES`** *(elementwise)* — the `null` contaminates every subsequent row and **never** recovers. *e.g. the
  Ehlers/Hilbert cycle pipeline: `mama`, `sine_wave`, `dominant_cycle_period`.*

**`nan_policy` — an interior `NaN` in the input:**

- **`POISONS`** *(reducing)* — the scalar becomes `NaN` (a non-null `NaN` is never silently skipped).
- **`PROPAGATES`** *(elementwise)* — the `NaN` nans the rows it reaches, then the output recovers.
- **`LATCHES`** *(elementwise)* — a recursion carries the `NaN` forward forever; the output never recovers.

**The recursion pairing.** A function that *bridges* a `null` is exactly one whose recurrence carries state — which
means a `NaN` in that state corrupts every later step. So **`BRIDGED` null ⇔ `LATCHES` nan**; the self-check enforces
the pair. This is why the two policies are declared together: they are two views of the same fact ("this is a
recursion"), and stating both makes the recursion explicit at the row rather than buried in the kernel.

## 4. The canonical ladder: one name, one order, one docstring per rung

Every test file lays its tests out in the same four tiers, in this order; within a tier the rungs are fixed and appear
only where the axes say they apply.

- **Contract** — `returns_expr` → `reduces_to_scalar` | `preserves_length` | `emits_struct` → `lazy_eager_parity`
  → `over_partitions_independently`.
- **Edge** — `<param>_out_of_range_raises` (per validated parameter, in signature order) → `empty` → `single_row`
  → `all_null` → `nan_<policy>` → `null_<policy>` → *(windowed:)* `warmup_null_count` → `window_exceeds_length`
  → `window_equals_length` → `window_one_*` → `constant_window_is_nan` → *(singularity guards)*.
- **Correctness** — `matches_reference` → `golden_master`.
- **Properties** — `matches_reference_for_any_input` → `matches_reference_under_missing_data`
  → `scale_homogeneity` | `scale_invariance` → `matches_reference_at_large_magnitude`.

**The naming law.** A rung has exactly **one** name across the whole suite. `single_row` is never also
`single_row_is_nan`; `window_one` is never also `window_one_is_identity`; a `nan_*` name is never used for the wrong
policy. Test-local variables follow **`{WHO}_{QUALIFIER}`** — `group_primary` / `expected_primary`, not `values` /
`case` — so two files read the same top to bottom. A contributor opening two families side by side never has to hold
"which edge is tested how, where" in their head. They read the row and the rung.

## 5. The architecture: one copy, applied to all (and the explicit variant)

Each rung is placed by how much of it is shared:

- **Universal** — identical for every member (`returns_expr`, `lazy_eager_parity`, `empty`, `all_null`,
  `over_partitions`): one parametrized module per family over `__all__`; there is no per-file copy to drift.
- **Class-parametrized** — same structure, per-member expected value driven by the registry row (its oracle, shape,
  null/NaN policy, degree): `single_row`, `nan_<policy>`, `null_<policy>`, `scale_*`, `matches_reference*`. **This is
  where a genuine difference lives as an explicit variant** — `ema` declares `BRIDGED`, `sma` declares
  `IN_WINDOW_IS_NULL`, and the one contract tests each accordingly. Not two hand-written tests that might diverge; one
  contract and two declared rows.
- **Function-specific** — a truly unique golden or a bespoke singularity of *that* function: a single, explicitly
  named, documented test in that function's own file, in the same tiered class, in the same order. Explicit, never
  implicit.

## 6. How this is enforced (so audits become natural, not heroic)

`tests/test_registry.py` is the guard. It proves, on every run:

1. **bijection** — the registry is in exact bijection with the three public `__all__` tuples: no function without a
   row, no row without a function. A new public function fails the build until it is profiled.
2. **oracle integrity** — every declared `*_reference` oracle is importable.
3. **policy is real** — each function's *actual* `null` / `NaN` flow, observed on a well-conditioned series, matches
   the `shape` and policies it declares. Only the flow is read (which rows go null/NaN, and whether the effect
   recovers), never a value, so the check is exact and platform-stable.

The consequence: a function cannot silently drift from its declared behaviour, and a new function cannot slip in
untested. The next audit does not *hunt* for parity — the suite has already asserted it.

## 7. Roadmap

- **P1 — foundation** *(this change)*: `POLICY.md`, the registry (six self-verified axes), and the self-check. No
  existing test is touched.
- **P-scale — the scale axis**: add the richer per-input-degree `scale` field to the registry, its numeric scale
  contract, and the self-check that ties the two together.
- **P2 — universal contracts**: the universal rungs move into shared modules; the per-file copies are deleted.
- **P3 / P4 / P5 — class-parametrized contracts** per family, driven by the registry.
- **P6 — singularity + floor sweep**: for each guard class, ensure every member both *handles* and *tests* it, and
  lift any below-floor assertion to its tier.

The whole program runs on an integration branch (`test-methodology`): each phase is a sub-PR merged into it one at a
time, and it reaches `main` as a single coherent change only once complete.
