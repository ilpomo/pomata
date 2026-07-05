# Tests

Tests **mirror the source layout** (`tests/<package>/test_<subject>.py`). Within each test module, categories are
separated into classes (`Test<Subject><Category>`); cross-cutting categories are selectable across the whole suite via
pytest markers.

## Categories (classes within each test module)

- **Contract** — schema / shape / dtype, lazy↔eager parity, no internal `.collect()`, per-group independence under `.over`.
- **Edge** — null vs NaN (distinct in Polars), warm-up, empty, single-row, window boundaries, zero denominators.
- **Correctness** — vs a naive closed-form reference oracle + frozen golden-master numbers.
- **Properties** — property-based (Hypothesis): bounds, scale-homogeneity, per-group independence, metamorphic
  relations.

## Precision

The value-agreement tiers (Correctness and Properties) hold every indicator to a **relative `1e-10`** — ten significant
figures — against its independent oracle: the single guarantee derived in `../CORRECTNESS.md` and pinned tier by tier in
`support/tolerances.py` (the one home for the bands, with the rationale per tier). The looser bands there are
artifact-tolerances — a rescaling that amplifies rounding, a degenerate-window absolute floor near zero, the non-gating
external differential — never a weaker value-agreement.

## Cross-cutting markers

- `differential` — non-gating checks against TA-Lib, needs the `differential` dependency group:
  `pytest -m differential`.
- `benchmark` — performance & complexity-scaling, needs `POMATA_BENCHMARKS=1`: `pytest -m benchmark`.

`oracles/` holds the naive reference implementations (also the pure-Python fallback); the golden masters are pinned
inline in the test modules.

**Oracle policy:** the source of truth is the *mathematical definition*, not another implementation. The gating
correctness tests depend only on `polars` and `hypothesis`; external libraries are an optional, non-gating lens.
