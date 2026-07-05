# The pomata test policy

This document is the method of record for how pomata is tested. It exists to end a specific failure mode: an edge case
is handled or tested one way in one function and another way in its neighbour, a fix lands for that one function, and
the same class of difference persists in the function next door. That drift is not a shortage of effort -- it is the
absence of a *method*. The method is this:

> **We declare only what a test cannot observe -- a function's `null` / `NaN` policy -- and prove that declaration
> against the code on every run. Everything else (a function's family, output shape, columns, oracle) is derived from
> the public surface and its signature, never restated. A function may depart from the baseline only by *declaring* it,
> and the build proves the declaration true. Never by drift.**

The one declared fact is each function's `(null_policy, nan_policy)`, in
[`tests/support/policies.py`](support/policies.py). It is kept honest by
[`tests/test_policies.py`](test_policies.py): a function added without a policy, or whose actual `null` / `NaN`
behaviour contradicts the one it declares, is a red build. The shared contract suite iterates each family's public
`__all__` directly and observes each function's shape from a probe, so the uniform rungs sweep in every new function
automatically, with nothing to restate.

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
  knife-edge flake that once churned a full cycle before it was pinned down).
- **Golden masters are rounded** (`.round(n)`) to a sane precision, so a frozen value never flakes between macOS-arm64,
  ubuntu-x86, and windows.
- **A magnitude-dependent band is sized to the data** (`input_scale ** degree * factor`), never fixed, so it is right
  at every scale; where a statistic legitimately needs a looser band (a square root that amplifies near zero, a
  difference of large terms that cancels), the departure carries a one-line reason.

The floor is not a compromise on rigor. It *is* the rigor: it draws the line exactly where "a difference means a real
coding error" ends and "a difference means IEEE-754 being IEEE-754" begins, and holds every function to the same line.

## 2. What is declared, and what is observed

A function's obligations are fixed by *what it does*, not by which file it lives in. Almost everything a test needs is
**observed**, never declared, so it cannot drift from the code:

- **family** — which public `__all__` (indicators / pnl / metrics) holds the name.
- **shape** — read from one probe: one row → reducing; a `Struct` column → struct; else a same-length series.
- **columns** / **windowed** / arity — read from the factory signature (`pl.Expr` inputs by name, `window: int`, …).
- **oracle** — the `<name>_reference` in the family's `oracles` package (a fixed convention), or golden-only (`NO_ORACLE`).

The **only** facts a test cannot observe, and so must be *declared*, are each function's **`null_policy`** and
**`nan_policy`** (defined crisply in §3) — because the declaration encodes *intent* and the build proves the code lives
up to it. They live in [`tests/support/policies.py`](support/policies.py), one `(null_policy, nan_policy)` per name.

A function's **scale** behaviour is deliberately *not* declared either: its degree is per-input and family-specific (a
variance is degree-2; a VWAP is degree-1 in price and degree-0 in volume; a borrow cost is degree-1 in quantity; a
return is invariant), so a single typed value cannot state it without being lossy. The scale tests stay per-file; only
the rung **name** is held uniform, by the grammar guard (§6).

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
  → `scale_homogeneity` | `scale_invariance` → `matches_reference_at_large_magnitude`. The scale rung is spelled from one
  vocabulary: `scale_homogeneity` (degree ≥ 1) | `scale_invariance` (degree 0); for a per-input OHLCV function,
  `price_scale_*` and `volume_scale_*`; for a per-input pnl function, `scale_homogeneity_in_<role>` (`in_quantity` /
  `in_weight` / `in_each_input`); for an additive offset, `additive_shift_invariance`. `tests/test_grammar.py`
  makes any other spelling a red build (§6).

**The naming law.** A rung has exactly **one** name across the whole suite. `single_row` is never also
`single_row_is_nan`; `window_one` is never also `window_one_is_identity`. The null/NaN flow anchor is spelled from one
**reserved** vocabulary — `null_skipped` · `null_propagates` · `null_in_window_is_null` · `null_bridged` ·
`null_latches`, and `nan_poisons` · `nan_propagates` · `nan_latches` — each usable **only** by a function that declares
that policy (a multi-input factory may still test a per-input case under a descriptive name such as
`null_in_volume_propagates`); `tests/test_grammar.py` makes a canonical name that lies about its policy a red
build (§6). Test-local variables follow **`{WHO}_{QUALIFIER}`** — `group_primary` / `expected_primary`, not `values` /
`case` — so two files read the same top to bottom. A contributor opening two families side by side never has to hold
"which edge is tested how, where" in their head. They read the rung.

## 5. The architecture: shared where uniform, per-file where specific

Each rung is placed by how much of it is genuinely shared:

- **Universal** — identical for every member (`returns_expr`, `shape`, `lazy_eager_parity`, `empty`, `all_null`,
  `over_partitions`): one parametrized module per family over `__all__`, with shape observed from a probe; there is no
  per-file copy to drift, and a new function is swept in automatically.
- **Per-file, presence-guarded** — the rungs whose value or degeneracy is genuinely function-specific (`single_row`,
  the `null` / `NaN` value anchors, the scale rungs, singularity guards): they live in each function's own file. The
  grammar guard mandates that the `null` / `NaN` / reference anchors *exist* and that any canonical name matches the
  declared policy (§6); the self-check proves the declared policy against the code. `ema` declares `BRIDGED`, `sma`
  `IN_WINDOW_IS_NULL`, and the build holds each to it — a genuine difference is stated and proven, not scattered.
- **Function-specific** — a truly unique golden or a bespoke singularity of *that* function: a single, explicitly
  named, documented test in that function's own file, in the same tiered class, in the same order.

## 6. How this is enforced (so parity holds by construction, not by vigilance)

Two source-only checks enforce this, on every run. `tests/test_policies.py` proves three things:

1. **coverage** — the policy map is in exact bijection with the three public `__all__` tuples: no function without a
   policy, no orphan policy. A new public function fails the build until its policy is declared.
2. **oracle integrity** — unless a function is golden-only (`NO_ORACLE`), its `<name>_reference` oracle is importable.
3. **policy is real** — each function's *actual* `null` / `NaN` flow, observed on a well-conditioned series, matches
   the `(null_policy, nan_policy)` it declares. Only the flow is read (which rows go null/NaN, and whether the effect
   recovers), never a value, so the check is exact and platform-stable.

`tests/test_grammar.py` proves two more, for the rungs that stay per-file (§2, §5):

4. **presence** — every function's test file carries at least one interior-`null` test, one interior-`NaN` test, and
   one `matches_reference` test. A function shipped without an edge anchor is a red build, not the next audit's finding.
5. **canonical names do not lie** — a reserved `test_null_*` / `test_nan_*` name is used only by a function whose
   declared policy it names, and a scale rung is drawn from the one scale vocabulary (§4) and sits in a
   `Test*Properties` class. Descriptive per-input names are left free; only the canonical ones are held.

The consequence: a function cannot silently drift from its declared behaviour, cannot slip in without a policy or an
edge test, and cannot spell a canonical name a way that contradicts what it declares. Parity is not something to hunt
for — the suite has already asserted it.
