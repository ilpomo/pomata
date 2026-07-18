# Tests

`pomata` is tested by a **declarative contract suite**: every public function states its whole testing contract by
calling one **per-family suite function** with a closed vocabulary of enums plus a hand-written reference oracle, and a
single ladder of generic checks derives every test from that one declaration. A contributor writes one file — a
declaration and an oracle — and the standard tests, the failure messages, and (through the generated docstring) the
prose all descend from it. Uniformity is not policed: it is the only thing the language can express.

## How to run

```sh
uv run pytest -n auto                          # the whole gating suite
uv run pytest -m differential                  # opt-in: non-gating parity vs TA-Lib (needs the 'differential' group)
```

The property tier draws a fixed example count through the Hypothesis profiles registered in `conftest.py`
(`HYPOTHESIS_PROFILE=dev|ci`, same count). Truth is the mathematical definition, checked against naive reference
oracles (`<family>/oracles/`) — the gating suite depends only on Polars and Hypothesis; TA-Lib is an opt-in
cross-check, never the arbiter.

## The shape of the framework

A per-function contract is a **frozen dataclass of pure data** — a `Declaration` (`support/declaration.py`) — built by
a **per-family suite function** (`suite_pnl` · `suite_metrics` · `suite_indicators`, each in `<family>/harness.py`).
The checks are **module-level functions** of a `Declaration` (`support/rungs.py`), parametrized over the registry in
`test_rungs.py`. There is no metaprogramming: no metaclass, no `__init_subclass__`, no runtime stamping of test
functions. A declaration cannot lie by omission — the fields it would omit are either required by the language (no
default) or made mandatory by a plain `__post_init__`.

- **Declaration per function** — one file, `<family>/<name>.py`: a call to the family suite function that both binds a
  module constant and **auto-registers** the declaration as an import side effect. The family aggregator
  (`<family>/all_<family>.py`, gathered by `all_declarations.py`) imports every declaration module, so the registry is
  populated before any collectible reads it; no explicit per-function import is maintained by hand.
- **Suite function** — `<family>/harness.py`: it fixes the family-shared facts by construction (a pnl output is always
  a same-length `Float64` series; a metric reduces unless it rolls or names a window), resolves the family's warm-up
  enum to the leading-null count the generic declaration speaks, assembles the `Declaration`, and registers it.
- **Rung** — `support/rungs.py`: one plain function of a `Declaration`, wrapped once in `test_rungs.py` and
  parametrized over `registry_all()`, so the pytest id names the function (`test_oracle_agreement[cost_borrow]`). A
  check a declaration does not activate (no golden, no pins, a scale-exempt function) skips cleanly with a reason.
- **The engine** — `support/`: the frozen data types and the small engine the rungs delegate to (`declaration.py`),
  the probe-frame builders with by-construction distinct roles (`frames.py`), the regime synthesis that *constructs* a
  degenerate input from a declared axis (`synthesis.py`), the comparison / tolerance layer (`compare.py`,
  `tolerances.py`), the failure-message format (`messages.py`), and the auto-registration and surface bijection
  (`registry.py`). `declaration.py`, `synthesis.py`, and the TA-Lib-bridging `test_differential.py` are the three
  modules exempted from `disallow-any-explicit` (they mirror each public function's own `...` signature).

## The axes are declared, in a closed family vocabulary

The public surface varies along a handful of axes; each is a **closed enum** a contributor picks from, never invents.
The three families share `shape`, the null / NaN behavior, and the scaling claim; each adds its own dialect
(`<family>/enums.py`):

- **shared** — `Shape.REDUCING` / `SERIES` / `STRUCT` (what one probe row observes; a struct declares its ordered
  `fields`, and every struct-aware rung reads **all** of them); the null / NaN behavior (below); and `scaling`, a
  non-empty tuple of `ScaleAxis(roles, degree)` or a reasoned `ScaleExempt` — never an empty tuple.
- **pnl** (the most uniform: one dialect covers sixteen of eighteen) — `BehaviorNull` {`PROPAGATES`, `BRIDGED`},
  `BehaviorNan` {`PROPAGATES`, `LATCHES`}, `SpaceCost` {`CASH`, `RETURNS`} (the cash-flow / returns-flow units,
  stated as data), `ConventionSign` {`LONG_SHORT`, `SHORT_ONLY`, `LONG_ONLY`}, `NonFinite` {`IEEE_FLOW`} (one IEEE-754
  story throughout, `inf - inf = NaN` included), `Warmup` {`NONE`, `ONE_ROW`}.
- **metrics** — `BehaviorNull` {`SKIPPED`, `IN_WINDOW_IS_NULL`, `PROPAGATES`}, `BehaviorNan` {`POISONS`,
  `PROPAGATES`}, `Annualization` {`SQRT_TIME`, `LINEAR`, `GEOMETRIC`, `NONE`}, `Degenerate` {`ZERO_DISPERSION_IS_NAN`,
  `RATIO_SIGNED_INF_OR_NAN`, `EXACT_ZERO`, `COLLAPSES`}. A **rolling twin** declares `rolling_of=<reducing twin>` and
  its `window` parameter: it inherits the twin's annualization and degenerate regime, and the twin-coherence rung
  holds its row `i` to the twin reduced over the trailing window ending at `i`.
- **indicators** (the richest) — `BehaviorNull` {`BRIDGED`, `IN_WINDOW_IS_NULL`, `PROPAGATES`, `LATCHES`, `ABSORBED`},
  `BehaviorNan` {`PROPAGATES`, `LATCHES`}, `Warmup` {`NONE`, `WINDOW_MINUS_ONE`, `WINDOW`, `EXPR`, `PER_FIELD`},
  `Seeding` (docstring metadata, not read by any rung), `RelationTalib` {`MATCHES`, `DOCUMENTED_DIVERGENCE`,
  `NO_EQUIVALENT`} — the partition the differential tier reads, each divergence / no-equivalent carrying its
  `talib_reason` on the declaration itself.

## The declaration surface (what a declaration states — nothing else)

| Field | Required | Meaning |
|---|---|---|
| `factory` | yes | the `pl.Expr` factory under test (its `__name__` is the name everything derives from) |
| `inputs` | yes | ordered input column roles, drawn from the probe-frame vocabulary |
| `params` | yes | the canonical scalar kwargs probes and goldens use (non-empty ⇒ `raises`) |
| `shape` | yes | `REDUCING` / `SERIES` / `STRUCT` |
| `behavior_null` / `behavior_nan` | yes | the family dialect's answer to an interior `null` / `NaN` |
| `oracle` | yes | the naive reference, named exactly `reference_{name}` (checked at construction) |
| `scaling` | yes | a non-empty `ScaleAxis` tuple, or a `ScaleExempt(reason)`; a struct's `degree` is a per-field mapping |
| `space` / `sign` / `nonfinite` | pnl | the cost units, the sign convention, the IEEE-flow contract |
| `annualization` / `degenerate` | metrics | the annualization convention and the degenerate-denominator regime |
| `talib` / `talib_reason` / `seeding` | indicators | the TA-Lib relation, its reason, the seeding convention |
| `warmup` | optional | exact leading-null count: an `int` for a series, a per-field mapping for a struct, `None` for a reduction / unwindowed transform |
| `fields` | struct | the struct's field names, in order |
| `raises` | params ⇒ yes | validation counterexamples: `(overrides, ValueError match)` |
| `golden` | optional | the frozen golden master (`inputs`, `output`, its own `params`, `round_to`) |
| `pins` | optional | crafted-input cases (see below) |
| `recomposition` | optional | a zero-arg builder of the public-function identity the factory must reproduce, lane by lane |
| `rolling_of` / `window` | rolling | the reducing / series twin a rolling function rolls, and its window parameter |
| `deviant` | optional | a `Deviant(expected, reason)` when the all-null answer is not all-null |
| `conditioning` | optional | a Hypothesis `assume` filter for the property tier — allowed only with a covering pin |
| `oracle_rel_tol` / `oracle_abs_tol` | optional | a per-declaration oracle band (a one-pass rolling form vs its two-pass oracle) — always a named constant from `support/tolerances.py` |
| `flow_deviation` / `flow_horizon` | optional | a reason the interior-missing flow is input-dependent (exempts the flow rungs, pinned instead); rows past a missing bar the flow must have played out by |
| `reference` / `wikipedia` | optional | the citation and URL for the definition (for the generated docstring) |

A `Pin` is itself pure data: `label` (its id suffix), `inputs` (the full input lanes), `expected` (the output lanes,
a per-field mapping for a struct), the required `reason` (why the case is pinned), an optional `params_override`,
`signed` (compare the sign too, so `-0.0` is not read as `0.0`), `covers_conditioning` (this pin witnesses the regime
the `conditioning` filter excludes), and `round_to` (expression-side rounding, declared only where the exact lanes are
platform-dependent — a transcendental pipeline on a degenerate input settles on libm-rounded fixed points that differ
across OS math libraries). Each pin's `inputs` keys and `expected` shape are checked against the declaration at
construction, and labels must be unique — snake_case, so a pin's regime is named, never inferred from its position.

**No exclusion without a fixed case.** A `conditioning` filter tells Hypothesis to skip an input regime, which is
exactly where a bug could hide — so a declaration may declare one only if at least one pin carries
`covers_conditioning=True` and demonstrates, on a concrete input from that regime, what the function actually returns
there (checked in `__post_init__`, so an uncovered filter cannot even be imported). The reverse also holds: a covering
pin on a declaration with no filter is rejected as stale. Every filter states the *measured* boundary it was
calibrated against, and the shared conditioning constants are never retuned per declaration.

Derived, never declared: `name` (from the factory), `landing` (the first input's column), and the pytest id.

## The guarantees, all by construction

1. **Completeness of the language** — the required fields have no default, so a declaration *cannot* be built without
   each one; no rung is ever silently skipped for want of a declaration.
2. **`__post_init__`** — the conditional requirements, checked loudly at construction (import time): the oracle is
   named `reference_{name}`; a struct names its `fields`; a reduction has no `warmup`; declared `params` imply
   `raises`; `scaling` is never an empty tuple; every scale axis and input role is real; a `conditioning` filter
   implies a covering pin and vice versa; a rolling twin names its window and shares its twin's inputs; a struct's
   `warmup` and every axis `degree` are per-field mappings keyed exactly by its `fields`, and only a struct's — one
   form per declaration, so a reader never infers which lanes a bare number covers.
3. **Two-way bijection with the public surface** — `test_registry.py` requires each family's registered names to
   match that family's `__all__` exactly (a missing declaration fails as loudly as a stray one) and forbids
   duplicates. A function cannot be exported without a declaration, and a declaration cannot outlive its function.

## The ladder (one function per rung)

`test_rungs.py` parametrizes each rung over the whole registry:

- **oracle_agreement** — the factory reproduces its oracle on the deterministic probe and across the fuzz domain
  (honoring `conditioning`), in kind (null / NaN / ±inf / finite) and in value, at the band the conditioning sizes.
- **golden** — the frozen golden master holds (rounded expression-side).
- **pins** — every crafted case maps to its pinned lanes (signed pins comparing the sign too).
- **recomposition** — the factory reproduces its `recomposition` identity from other public functions, lane by lane.
- **behavior_null** / **behavior_nan** — an interior missing value plays out exactly as the oracle plays it out (the
  structural flow of the family dialect); a `flow_deviation` exempts these and pins the flow as crafted cases instead.
- **nonfinite** — each input carrying ±inf flows through exactly as the oracle carries it (the pnl IEEE story; a
  family that declares no such contract skips).
- **twin_coherence** — a rolling function's row `i` equals its `rolling_of` twin reduced over the trailing window.
- **annualization** — a closed-form annualization scales the output by the declared period-count ratio.
- **scaling** — each homogeneity axis scales every lane by the declared degree; a `ScaleExempt` verifies it is not
  secretly homogeneous.
- **raises** — each validation counterexample raises its canonical `ValueError`.
- **warmup** — the output carries exactly the declared leading nulls (per field for a struct).
- **all_null** / **empty** / **single_row** — the degenerate-frame contracts (an all-null input yields all-null or the
  declared deviant; an empty frame gives the shape's empty answer; a one-row input keeps the declared shape).

The **differential** tier (`test_differential.py`, opt-in, non-gating) compares each `RelationTalib.MATCHES` indicator
against TA-Lib at the reference band; the no-twin and documented-divergence relations are accounted for but not
compared, and the partition is derived from the declarations, so it cannot drift from the suite.

## When a guard is warranted (the three-question criterion)

Beyond the ladder, a small set of source-and-docs guards holds every claim the per-function contract cannot. A guard
earns its place only if it answers **yes** to one of three questions; a check that answers **no** to all three is not a
guard but a line in the authoring canon, held by `ruff`, the type checkers, and review:

1. **Truth** — does it couple the documentation to a fact the code already states, so the two cannot drift apart? (The
   docstring's Args to the signature; the NaN vocabulary to the declared behavior; the edge-case bullets to the
   classes the declaration activates; the README counts and catalogs to `__all__` and the source modules; the TA-Lib
   split to `RelationTalib`; the cash-flow / returns-flow split to `SpaceCost`.)
2. **Distance** — does it measure mathematical correctness against an independent witness? (The oracle and golden
   agreement; the recomposition identities; the published precision figures.)
3. **Invisible** — is it unreachable by the standard tooling? (`ruff`'s pydocstyle shell checks, the type checkers,
   and the doctest gate cannot see a docstring's coupling to the declaration; these guards prove it from the source.)

The guards, by that test:

- `test_docstrings.py` — the docstring truth-couplers: Args ⇔ signature; NaN vocabulary ⇔ declared behavior; Returns
  opener ⇔ declared shape; shared parameter / Raises prose pinned; edge-case bullets ⇔ the classes the declaration
  activates (`support/edge_classes.py`); each asserted degenerate outcome ⇔ a witnessing pin; each demanded scenario ⇔
  an executed Examples block; warm-up sentence ⇔ declared warm-up; Raises section ⇔ declared counterexamples; TA-Lib
  divergence header ⇔ declared relation; `ddof` header ⇔ signature; opener variant ⇔ the mirror-oracle disclosure;
  scalar knobs keyword-only; a Note header renders as its own paragraph.
- `test_docs_surface.py` — the README and docs-site catalogs: each family's list and headline count ⇔ `__all__`; each
  indicators / metrics category ⇔ its source module; the pnl cash / returns split ⇔ `SpaceCost`; the reducer count and
  rolling-twin pairing; the TA-Lib split ⇔ `RelationTalib`; the kernel-modules note; the single-dependency claim.
- `test_oracle_docstrings.py` — every reference states its missing-data contract, and the irreducibly-sequential
  mirrors disclose that they confirm internal consistency, not independence.
- `test_precision_table.py` — the published precision figures in `docs/correctness.md` stay reproducible.
- `test_versions.py` — the support claims (Polars floor, Python versions, OS list) in the README, the docs site, the
  contributing guide, and the CI matrix all match `pyproject.toml`.
- `test_package.py` — import smoke for every subpackage and the exposed `__version__`.

## Born red, always

Every new or repaired guard — a `__post_init__` check, a rung, a source-and-docs sweep — is first demonstrated
**failing** on a real counterexample (locally, never committed) before it lands green. A guard that was never red
proves nothing. The failure messages are part of the contract: a derived test that fails prints the declaration that
generated it, the exact constructed input, expected vs observed, and a triage line — *either the declaration is wrong,
or the code has a bug* — so a red build is understood in seconds.

## The structural freeze

The suite's **architecture is frozen**: the declaration language, the family dialects, the rung ladder, and the
engine are settled. Ongoing work touches **content**, not structure. Adding a public function is a fixed checklist —
the `__all__` entry, the declaration file, its family aggregator import, the `reference_{name}` oracle, and the
docstring — and every guard fails closed at collection if a piece is missing. Extending the *language* (a new enum
member, a new field, a new rung, a new source-and-docs guard) is the deliberate exception, not the routine: prefer
extending the vocabulary over writing a bespoke test, and never write a test that polices tests.

## Deliberate conventions (tested exactly as implemented)

A few behaviors look like inconsistencies until the domain reason is on the table; each is pinned exactly as the
function computes it, and its docstring states it:

- **Burke ratio, per-bar denominator.** `burke_ratio` sums the squared *per-bar* drawdown series, not Burke's classic
  per-episode declines: every underwater bar contributes, so deeper *and* longer drawdowns are both penalized.
- **EWM dispersions, `window >= 2`.** `standard_deviation_ewma` and `variance_ewma` reject a window of one while their
  rolling twins accept `window=1, ddof=0`: the EWM debiasing (`bias=False`) has no defined value at one observation.
- **Parabolic SAR, intrabar order.** Within a bar the extreme point and acceleration factor update *before* the
  reversal check; the golden masters pin the chosen order, including the seeding and reversal branches.
- **TA-Lib divergences.** ADXR's averaging lag, the Chande momentum oscillator's smoothing, and OBV's origin follow
  the charting authorities rather than TA-Lib, so they are held out of the differential tier — each justified on the
  function itself through its `DOCUMENTED_DIVERGENCE` relation.

## Tolerances, conditioning, and the exact-zero closures

A tolerance is never a round number picked by feel. When the implementation and the oracle agree they still round
differently, and how far they drift is set by the statistic's *conditioning*: each band is a named constant from
`support/tolerances.py`, sized to the worst residual conditioning predicts plus a margin. The well-conditioned kernels
(recursive, windowed, and stateless means) match to a few ULP; the one-pass rolling and EWM moments meet their
two-pass oracle at the wider rolling band a declaration states through `oracle_*_tol`; a scale-invariant output (a
bounded ratio, a cycle period, a flag) carries an *absolute* band, since sizing an `O(1)` value to the input magnitude
is meaningless. Any band override sits under a one-line reason; a bare, unexplained tolerance is treated as a defect.

Never bit-equality on a computed float — `assert_matches` compares by kind and within a band. The one regime a
relative band cannot hold is a window that collapses onto the float-precision floor of a much larger value that
recently slid through it: the shipped kernels pin an exactly-constant dispersion to exactly zero (detected via the
rolling extremes, `max == min`), and the declarations pin these float-residue closures exactly, so a departed
outlier's residue can never be reported as spread — while the first-moment ratios built on them degenerate to the
documented signed infinity (or `NaN` where the moment correction is `0 / 0`) rather than a residue-driven huge finite.

## Glossary (plain words for the suite's terms)

- **declaration** — one public function's whole testing contract, one frozen dataclass of pure data in
  `<family>/<name>.py`. No logic lives there: it *states* facts, the rungs *check* them.
- **suite function** — `suite_pnl` / `suite_metrics` / `suite_indicators`: the one call a declaration file makes; it
  fixes the family-shared facts, builds the declaration, and registers it.
- **rung** — one generic check in `support/rungs.py`, parametrized over every declaration it applies to. One rung =
  one guarantee, checked identically across the public surface.
- **oracle** — the naive, obviously-correct reimplementation (plain Python, two-pass, no streaming tricks) the fast
  Polars implementation is compared against. For the irreducibly-sequential references (the Ehlers cycle cluster, the
  parabolic SAR, KAMA, the Fisher transform, SuperTrend) the oracle is a structural mirror that confirms internal
  consistency; the Correctness page documents the second witness each rests on.
- **golden master** — one frozen input with its frozen, hand-verified output; a behavior change shows up as a golden
  mismatch even if impl and oracle drift together.
- **pin (fixed case)** — one crafted input mapped to its exact expected output, with a written reason: the data home
  for a hand-computed value, a domain corner, a signed zero, a degenerate regime.
- **conditioning (filter)** — a per-declaration predicate that tells the property tier to skip an input regime where
  impl and oracle *cannot* agree (a genuine 0/0, a branch-flip residual). Declared as data, never hidden in a rung.
- **warm-up** — the exact count of leading `null` rows a windowed function emits before its first defined value.
- **flow** — how an interior missing value propagates through a function: which rows go missing, and how far.
- **born red** — every new guard is first demonstrated *failing* on a real counterexample before it lands green.
