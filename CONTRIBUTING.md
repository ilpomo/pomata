# Contributing to pomata

Thanks for your interest in pomata. Contributions of all kinds are welcome — bug reports, new indicators, PnL, or
metrics functions, documentation, and fixes. This guide explains how to get set up and what a change needs to be merged.

## Philosophy

pomata's value is **verifiable correctness**. Every public function is checked against multiple independent
references — a naive closed-form implementation, property-based invariants, and frozen golden-master numbers —
under 100% branch coverage. A change ships only when that full check suite is green: a feature is not done until
it is verified.

## Development setup

pomata uses [uv](https://docs.astral.sh/uv/) for everything (Python, dependencies, and the virtual environment),
and supports Python 3.12 and newer.

```bash
git clone https://github.com/ilpomo/pomata
cd pomata
uv sync
```

## Running the checks

CI runs the gate below; run it locally before opening a pull request:

```bash
uv run ruff check                                            # lint
uv run ruff format --check                                   # formatting
uv run mypy -p pomata                                        # types (package)
uv run mypy tests                                            # types (tests)
uv run pyright                                               # types (pyright)
uv run pyright --verifytypes pomata --ignoreexternal         # public type completeness
uv run pyrefly check src/pomata                              # types (pyrefly)
uv run pytest --doctest-modules src/pomata -q                # doctests
uv run pytest --cov=pomata --cov-report=term-missing -n auto # tests + 100% branch coverage
```

CI additionally runs `uv run ty check` as an advisory (non-gating) type check.

Optionally, install the pre-commit hooks — they mirror the lint/format gate and keep `uv.lock` in sync:

```bash
uvx pre-commit install
```

## Adding a function

Every public function is a pure `pl.Expr` factory, and ships with:

- a Google-style docstring: a plain-language summary, the formula in LaTeX, `Args`/`Returns`/`Raises`, a `Note`
  on null/NaN/warm-up behavior, a `References` section, and a runnable `Examples` block;
- a naive reference implementation used as an independent oracle in the tests;
- a test module covering the contract (schema, lazy/eager parity, `.over` independence), edge cases
  (null/NaN/warm-up/single row), correctness (versus the oracle and frozen golden-master numbers), and
  properties (bounds, scale-homogeneity, behavior under missing data);
- 100% branch coverage.

## Commits and pull requests

The project follows [Conventional Commits](https://www.conventionalcommits.org). Pull requests are
**squash-merged**, so the **PR title becomes the commit message** on `main` — write it as a valid
conventional-commit subject.

- **Format** — `type(scope): subject`: imperative, lowercase, no trailing period, short (~70 chars).
  - **type**: one of `feat`, `fix`, `refactor`, `perf`, `style`, `test`, `docs`, `build`, `ci`, `chore`.
  - **scope** (optional): the area, e.g. `indicators`, `pnl`, `metrics`, or a module.
- **Branch** — name it after the PR: `type/short-kebab-subject` (e.g. `feat/keltner-channels`).
- **One logical change per pull request.** Open it, run the full gate (it must be green), then request review.
- **Description** — use the template's two sections: **Summary** (what changed and *why*) and **Details** (the
  substance, as terse bullets). Do not restate the checks — CI is the source of truth, not the body.
- **Linked issues** — link them from the PR's *Development* sidebar (or a `Closes #N` line) so they close on merge.

## Releases and labels

Releases use GitHub's native release notes: tag a SemVer version and **Generate release notes**, which group the
merged pull requests by **label** (see `.github/release.yml`) — `enhancement` → Features, `bug` → Bug Fixes,
`documentation` → Documentation, and so on. Apply the matching triage label to each pull request. Labels are for
triage and grouping; they do not classify the change *type* — that is the conventional-commit prefix's job.

## Conventions

- Code style is enforced by ruff (line length 120). Full type hints everywhere; the public API is fully typed.
- US English throughout, in code and prose.
- The runtime dependency tree is intentionally minimal (just Polars); new runtime dependencies are avoided.
