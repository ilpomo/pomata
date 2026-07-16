# Tests

`pomata` is tested by a **declarative contract suite**: every public function states its whole testing contract as
one frozen dataclass of pure data (a `Spec`, one file per function under `tests/<family>/<name>.py`), and a single
ladder of generic test functions (`tests/test_ladder.py`) checks every declared fact identically across the whole
public surface. Uniformity is not policed — it is the only thing the language can express.

## How to run

```sh
uv run pytest -n auto                          # the whole gating suite
POMATA_BENCHMARKS=1 uv run pytest -m benchmark # opt-in: performance and complexity-scaling tier
uv run pytest -m differential                  # opt-in: non-gating parity vs TA-Lib (needs the 'differential' group)
```

The property tiers draw a fixed example count through the Hypothesis profiles registered in `tests/conftest.py`
(`HYPOTHESIS_PROFILE=dev|ci`, same count). Truth is the mathematical definition, checked against naive reference
oracles (`tests/<family>/oracles/`) — the gating suite depends only on Polars and Hypothesis; TA-Lib is an
opt-in cross-check, never the arbiter.

## The shape of the framework

A per-function contract is a **frozen dataclass of pure data** — a `Spec` — and the rungs are **module-level
functions** parametrized over the specs they apply to. There is no metaprogramming: no metaclass, no
`__init_subclass__`, no runtime stamping of test functions. A declaration cannot lie by omission because the
`Spec` fields it would omit are either required by the language (no default) or made mandatory by a plain
`__post_init__`.

- **Spec per function** — one file per function, `tests/<family>/<name>.py`, data only, aggregated by explicit
  imports in `tests/all_specs.py`. A forgotten import is a red build (the bijection).
- **Rungs** — `tests/test_ladder.py`, each rung written once, `@pytest.mark.parametrize` over the applicable
  subset (a comprehension on declared fields), sub-parametrized where it reads better (per struct field, per
  validation counterexample, per scale axis).
- **The engine** — `tests/support/spec.py`: the frozen data types and the small engine the rungs delegate to
  (the deterministic probe frame, the expression builder, the lane readers, the oracle bridge, the fuzz strategies,
  the sizing helpers). It is one of the three modules exempted from `disallow_any_explicit` (with the reflective `synthesis` and the TA-Lib-bridging differential tier). The fuzz vocabulary covers the
  single-input shapes, the coherent OHLC-family bars, the strictly-positive equity-curve growth path, and the
  multi-input shapes (the pnl frames and the returns/benchmark pair, each column drawn independently from its
  role's domain); an unlisted shape raises, so the closed vocabulary can never silently under-test a new function.

## The axes are declared fields, not a class hierarchy

The public surface varies along exactly three axes (11 observed combinations across 153 functions). In the spec
ladder each axis is a **declared field** or a **derived fact**, and a rung gates its own applicability by reading it:

- **shape** — `Shape.REDUCING` (43) · `Shape.SERIES` (95) · `Shape.STRUCT` (15): what one probe row observes. A
  struct declares its ordered `fields`, and every struct-aware rung reads **all** of them (never only the first).
- **windowed** — `warmup is not None` (86): the exact leading-null count under `params` — an `int` for a series, a
  per-field mapping for a struct (always, even when every line shares the count: the form is fixed by the shape, so
  a reader never guesses which lanes a number covers). A reduction and an unwindowed transform declare
  `warmup=None`.
- **null / NaN policy** — **not declared**: derived from `pomata._policy` by function name and dispatched inside
  the shared flow rungs, so a spec cannot pair the wrong behavior with the wrong declaration.

## The declaration surface (what a spec states — nothing else)

| Field | Required | Meaning |
|---|---|---|
| `factory` | yes | the `pl.Expr` factory under test |
| `inputs` | yes | ordered input column roles (drawn from the probe-frame vocabulary) |
| `params` | yes | the canonical scalar kwargs used by probes and goldens |
| `shape` | yes | `REDUCING` / `SERIES` / `STRUCT` — the observed output shape |
| `scale` | yes | a non-empty tuple of `ScaleAxis`, or a `ScaleExempt(reason)` — never an empty tuple; an axis's `degree` is an `int` for a single-lane output and a per-field mapping for a struct (e.g. `supertrend`: `line` degree 1, `direction` degree 0) |
| `oracle` | yes | the naive reference oracle |
| `golden_input` / `golden_output` | yes | the frozen golden master (per field for a struct) |
| `warmup` | optional | exact leading-null count under `params`: an `int` for a series, a per-field mapping for a struct, `None` when nothing warms up |
| `fields` | struct | the struct's field names, in order |
| `raises` | params ⇒ yes | validation counterexamples: `(overrides, ValueError match)` |
| `golden_params` / `golden_round` | optional | the golden's own params and its rounding |
| `lands_on` | optional | landing column when it is not the first input |
| `flow_horizon` | optional | rows past a missing bar the flow must have played out by |
| `flow_deviation` | optional | a written reason the interior-missing flow is input-dependent: non-empty exempts the spec from the two flow rungs, and its flow is pinned as crafted cases instead |
| `oracle_rel_tol` / `oracle_abs_tol` | optional | a per-spec oracle-agreement band, declared per computational family (the one-pass streaming forms — rolling sums and exponential recurrences — whose accumulation rounds away from the two-pass oracle, and the standardized rolling moments' absolute floor) — always a named constant from `tests/support/tolerances.py`, never a literal |
| `cost_degree` | optional | the kernel's polynomial cost degree in the row count (`1` is the family norm; log factors ride within a degree) — the benchmark tier's scaling guard derives its per-function bound from it |
| `oracle_adapter` | optional | a frame->result callable when the oracle is not the factory's signature-mirror |
| `conditioning` | optional | a Hypothesis `assume` filter for the property tier — allowed only together with a pin that carries `covers_conditioning=True` (see below) |
| `all_null` | optional | a `Deviant(expected, reason)` when the all-null answer is not all-null |
| `pins` | optional | crafted-input cases: a tuple of `SpecPin(label, inputs, expected, reason, params_override, signed, covers_conditioning, round_to)` |
| `component_expr` | optional | a zero-argument builder of the public-function recomposition the factory must reproduce (a metamorphic identity) |

A `SpecPin` is itself pure data: `label` (its id suffix), `inputs` (the full input lanes), `expected` (the output
lanes, a per-field mapping for a struct), the required `reason` (why the case is pinned, anchored to where it came
from), an optional `params_override`, `signed` (compare the sign too, so `-0.0` is not read as `0.0`),
`covers_conditioning` (this pin is the fixed case witnessing the input regime the spec's `conditioning` filter
excludes), and `round_to` (expression-side rounding mirroring `golden_round`, declared only where the exact lanes
are platform-dependent — a transcendental pipeline on a degenerate input settles on libm-rounded fixed points that
differ across OS math libraries — with the reason naming that fact). Each pin's `inputs` keys and `expected` shape
are checked against the spec at construction, and labels must be unique.

**No exclusion without a fixed case.** A `conditioning` filter tells Hypothesis to skip an input regime, which is
exactly where a bug could hide unobserved — so a spec may declare one only if at least one pin carries
`covers_conditioning=True` and demonstrates, on a concrete input from that regime, what the function actually
returns there (checked in `__post_init__`, so an uncovered filter cannot even be imported). The reverse also
holds: a `covers_conditioning=True` pin on a spec with no filter is rejected as stale. Every filter states the
*measured* boundary it was calibrated against — in its docstring when it rides a shared cut, or in the spec-local
constant's comment when it narrows a shared threshold into one — and the shared constants themselves (`well_spread`
1e-9, `windows_well_spread` 1e-9, `CONDITIONING_FLOOR` 1e-2) are never retuned per spec.

Derived, never declared: `name` (from the factory), `family` (from `__all__`), `null_policy` / `nan_policy`
(from the registry), `spec_id` (the pytest id).

## The guarantees, all by construction

1. **Completeness of the language** — the required fields have no default, so a spec *cannot* be built without
   each one; no rung is ever silently skipped for want of a declaration.
2. **`__post_init__`** — the conditional requirements, checked loudly at construction (import time): a struct names
   its `fields`; a reduction has no `warmup`; declared `params` imply `raises` (else the validation rung is a
   no-op); `scale` is never an empty tuple (an exemption is a reasoned `ScaleExempt`); every scale axis and input
   role is real; the derived name has a declared policy and a public `__all__`; a `conditioning` filter implies a
   `covers_conditioning=True` pin, and such a pin implies the filter (no exclusion without a fixed case, no stale
   coverage claim); a struct's `warmup` and every one of its axes' `degree` are per-field mappings keyed exactly by
   its `fields`, and only a struct's — the form is fixed by the shape, so there is exactly one way to write each
   declaration and the reader is never left inferring which lanes a bare number covers.
3. **Two-way bijection with the public surface** — `tests/all_specs.py` requires each family's spec tuple to match
   that family's `__all__` exactly (a missing spec fails as loudly as a stray one) and forbids duplicate names. It
   runs at import (born red), so any collection enforces it: a function cannot be exported without a spec, and a
   spec cannot outlive its function.
4. **Shape coverage guard** — one rung observes the output shape from the probe and asserts it is exactly the
   declared `shape`. (Windowedness is *not* observed: a seed-null that is not a warm-up would make that a false
   positive, so completeness rests on the required fields, not on inference.)

## The ladder (one function per rung, canonical order)

Contract — `returns_expr`, `output_lands_on_declared_column`, `shape_matches_declaration`, `lazy_eager_parity`,
`streaming_engine_parity` (the third execution mode, at the reference band — the streaming engine's chunked
accumulation legitimately reorders a sum by an ULP), `over_partitions_independently` and
`over_interleaved_partitions` (shape-aware: a reduction broadcasts across its group's rows, an elementwise output
concatenates; `assert_matches`, never a bit-equality; the interleaved variant proves the guarantee does not depend
on contiguous groups), `bare_string_raises_type_error`.
Edge — `invalid_params_raise` (per counterexample), `all_null_input` (honoring an `all_null` `Deviant`),
`single_row`, `empty`, `interior_null_flow` / `interior_nan_flow` (the policy-dispatched flow, tail guards
included; a spec declaring a `flow_deviation` is exempt and pins its flow as crafted cases instead),
`warmup_null_count` (windowed subset, per field for a struct), `window_exceeds_length` (windowed subset,
per lane: a frame no longer than a lane's warm-up emits nothing on that lane), `no_lookahead` (non-reducing subset: a
prefix of the frame gives the prefix of the full output).
Correctness — `matches_reference`, `float32_input_parity` (a `Float32` input computes what the oracle computes on
the same rounded values — the up-cast changes the dtype, never the arithmetic), `golden_master` (rounded
expression-side), `pinned_cases` (per `SpecPin`: the
crafted frame maps to the crafted lanes, signed pins comparing the sign too).
Properties — `scale` (per `ScaleAxis`: scale only that axis's roles by a power of two, degree as declared),
`matches_reference_for_any_input` and `matches_reference_under_missing_data` (`@given(st.data())` inside
`@parametrize`, honoring a spec's `conditioning`), `matches_reference_on_bit_constant_input` (impl and oracle agree
in kind — null / NaN / ±inf / finite — on a deterministic battery of bit-constant frames, the exact-zero-pinning
regime; a spec's `conditioning` scopes it exactly as it scopes the fuzz), `matches_component_definition` (specs with
`component_expr`: the factory reproduces its recomposition from public functions, lane by lane, on the probe frame).

## Sub-parametrized ids

A struct field, a validation counterexample, a scale axis, and a pin each get their own case with a readable id:
`ichimoku-senkou_b` (per-field warm-up), `sharpe_ratio-0` (per counterexample), `ichimoku-high+low` (per scale
axis), `sharpe_ratio-zero_volatility` (per pin). To read a failure: find the rung by the name in the id, read its few
lines, then read the spec row the id names.

## The registry-derived sweeps and the source-and-docs guards

Beyond the ladder, a small set of modules holds every claim that is not per-function-contract shaped: the
registry-derived sweeps (their cases come from `ALL_SPECS`, so a new function is swept in the moment its spec
lands), the family `test_bespoke.py` modules (metrics and pnl; the Hypothesis-quantified claims the spec language cannot carry —
metamorphic identities and large-magnitude properties, as ordinary `@given` tests), the `tests/support/tests/`
unit tests of the harness itself, and the source-and-docs guards:

- `test_dtype.py` — every output lane comes back `Float64` from a `Float32` or `Int64` input.
- `test_typing.py` — every public factory declares its return type as exactly `pl.Expr`.
- `test_benchmark.py` (opt-in) — the performance and complexity-scaling tier over the whole registry.
- `test_differential.py` (opt-in) — the non-gating TA-Lib parity check.
- `test_precision_table.py` — the published precision figures in `docs/correctness.md` stay reproducible.
- `test_docstrings.py` — every public docstring conforms to the shared template (section order, argument order,
  the byte-identical `TypeError` line, the Null-before-NaN edge bullets, the reserved vocabulary).
- `test_docs_surface.py` — the README's "All N …" surface lists and headline counts match the public `__all__`.
- `test_versions.py` — the support claims (Polars floor, Python versions, OS list) in the README badges, the docs
  site, the contributing guide, and the CI matrix all match `pyproject.toml`.
- `test_expr.py` — direct unit tests of the shared validation core `pomata._expr` (canonical error messages and
  bounds), whose error branches the ladder crosses only indirectly.
- `test_package.py` — import smoke for every subpackage and the exposed `__version__`.

## Deliberate conventions (tested exactly as implemented)

A few behaviors look like inconsistencies until the domain reason is on the table; each is pinned by the suite exactly
as the function computes it, and its docstring states it:

- **Burke ratio, per-bar denominator.** `burke_ratio` sums the squared *per-bar* drawdown series, not the classic
  per-episode declines of Burke (1994): every underwater bar contributes, so deeper *and* longer drawdowns are both
  penalized. The figure is not comparable to a per-episode Burke from another library.
- **EWM dispersions, `window >= 2`.** `standard_deviation_ewma` and `variance_ewma` reject a window of one while their
  rolling twins accept `window=1, ddof=0`: the EWM form's debiasing (`bias=False`) has no defined value at a single
  observation, and a runtime flag cannot carry a `ddof`-style bound.
- **Parabolic SAR, intrabar order.** Within a bar the extreme point and acceleration factor update *before* the
  reversal check, so a bar that both makes a new extreme and crosses the stop reverses onto that bar's fresh extreme.
  Wilder's text does not fix this order; the golden masters pin the chosen one, including the seeding and reversal
  branches.
- **TA-Lib divergences.** ADXR's averaging lag, the Chande momentum oscillator's smoothing, and OBV's origin follow
  the charting authorities rather than TA-Lib, so they do not agree with it even at steady state and are held out of
  the differential tier on purpose — each justified on the function itself.

## Tolerances, conditioning, and the exact-zero closures

A tolerance is never a round number picked by feel. When the implementation and the oracle agree they still round
differently, and how far they drift is set by the statistic's *conditioning*: each band is a named constant from
`tests/support/tolerances.py`, sized to the worst residual that conditioning predicts plus a margin. The
well-conditioned kernels (recursive, windowed, and stateless means) match to a few ULP; the one-pass rolling and EWM
moments meet their two-pass oracle at the wider rolling band each spec declares; a scale-invariant output (a bounded
ratio, a cycle period, a flag) carries an *absolute* band, since sizing an `O(1)` value to the input magnitude is
meaningless; and the large-magnitude bespoke tier sizes its floor to the data (`input_scale ** degree * factor`). Any
`oracle_*_tol` departure sits under a one-line comment saying why — a bare, unexplained tolerance is treated as a
defect, and `test_spec_files.py` enforces the comment.

A separate mechanism closes the one regime where a relative band cannot hold: a window that collapses onto the
float-precision floor of a much larger value that recently slid through it. The bit-constant battery
(`matches_reference_on_bit_constant_input`) drives every column to one repeated constant — the regime where the
shipped kernels pin a zero dispersion to exactly zero while a naive two-pass mean can keep a rounding residue — and the
suite pins these float-residue closures exactly:

- the rolling skewness and kurtosis recompute any window at residue risk as a fresh two-pass mean-centered moment, so a
  departed outlier can no longer poison every later window;
- the reducing dispersions pin an exactly-constant series to exactly zero, so the first-moment ratios built on them
  (Sharpe, Sortino, the information ratio's tracking error, the M-squared that composes them) degenerate to the
  documented signed infinity rather than a residue-driven huge finite, while the higher-moment variants (the adjusted
  and probabilistic Sharpe ratios) degenerate to `NaN` — their moment corrections are `0 / 0` there;
- the rolling dispersions (the rolling volatility, the rolling variance and standard deviation, and the Bollinger
  bands built on them) pin a bit-constant window to exactly zero, so a departed outlier's residue can never be reported
  as spread.

The windowed win / loss ratios (the Chande momentum oscillator, the rolling omega) keep the documented dynamic-range
limit instead: their sliding sums are guarded at the bit-constant edge and clamped where the output is bounded.

## Glossary (plain words for the suite's terms)

- **spec** — one public function's whole testing contract, written as a single frozen dataclass of plain data in
  `tests/<family>/<name>.py`. No logic lives there: it *states* facts, the ladder *checks* them.
- **rung** — one generic test function in `tests/test_ladder.py`, parametrized over every spec it applies to.
  One rung = one guarantee (e.g. "the golden master matches"), checked identically for all functions.
- **tier** — an informal grouping of rungs by strength: contract (types and shapes), edge (fixed corner inputs),
  correctness (oracle and golden agreement), properties (Hypothesis-generated inputs).
- **oracle** — the naive, obviously-correct reimplementation of a function (plain Python, two-pass, no streaming
  tricks) that the fast Polars implementation is compared against. For the irreducibly-sequential indicators (the
  Ehlers cycle cluster, the parabolic SAR, KAMA, the Fisher transform, SuperTrend) the oracle is a structural mirror
  that confirms internal consistency; the documentation's Correctness page documents the second witness each rests on
  instead.
- **golden master** — one frozen input with its frozen, hand-verified output; a change in behavior shows up as a
  golden mismatch even if impl and oracle drift together.
- **pin (fixed case)** — one crafted input mapped to its exact expected output, with a written reason. The data
  home for a fact a random input or an oracle cannot express: a hand-computed value, a domain corner, a signed
  zero, a degenerate regime.
- **conditioning (filter)** — a per-spec predicate that tells the property tiers to skip an input regime where the
  implementation and the oracle *cannot* be expected to agree (a genuine 0/0, a branch-flip residual). Declared as
  data on the spec, never hidden in a rung.
- **`covers_conditioning`** — the flag marking the pin that witnesses a filter's excluded regime. Every filter must
  have one (checked at import): what the fuzz never sees, a fixed case must still demonstrate.
- **warm-up** — the exact count of leading `null` rows a windowed function emits before its first defined value.
- **flow** — how an interior missing value (a `null` or a `NaN` in the middle of the input) propagates through a
  function: which rows go missing, and how far downstream.
- **born red** — the rule that every new guard must first be demonstrated *failing* on a real counterexample
  (locally, never committed) before it lands green; a guard that was never red proves nothing.
- **policy** — a function's declared answer to interior `null` / `NaN`, from the `pomata._policy` registry; the
  flow rungs dispatch on it.
