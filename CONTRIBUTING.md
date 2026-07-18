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
uv run ruff check                                                              # lint
uv run ruff format --check                                                     # formatting
uv run codespell                                                               # US spelling
uv run mypy -p pomata -p tests                                                 # types (package + tests)
uv run pyright                                                                 # types (pyright)
uv run pyright --verifytypes pomata --ignoreexternal                           # public type completeness
uv run pyrefly check src/pomata                                                # types (pyrefly)
uv run pytest --doctest-modules src/pomata -q                                  # doctests
uv run pytest --cov=pomata --cov-report=term-missing --cov-report=xml -n auto  # tests + 100% branch coverage
uv run sphinx-build -W -b html docs docs/_build/html                           # docs build (warnings are errors; gates every PR)
uv run sphinx-build -b doctest docs docs/_build/doctest                        # docs-page doctests (gates every PR)
```

CI additionally runs `uv run ty check --error-on-warning` as an advisory (non-gating) type check.

Optionally, install the pre-commit hooks — they mirror the lint/format gate and keep `uv.lock` in sync:

```bash
uvx pre-commit install
```

## Adding a function

Every public function is a pure `pl.Expr` factory, and ships with:

- a Google-style docstring: a plain-language summary, the formula in LaTeX (where a closed form exists),
  `Args`/`Returns`/`Raises`, a `Note`
  on null/NaN/warm-up behavior, a `References` section, and a runnable `Examples` block;
- a naive reference implementation used as an independent oracle in the tests;
- a declaration — one `suite_<family>(...)` call under `tests/<family>/<name>.py`, self-registering, with the
  behavior axes picked from the family's closed enum vocabulary — whose declared facts the derived rungs check
  across every tier (contract, edge, correctness versus the oracle and a golden anchor, and properties: scale
  behavior, bounds, behavior under missing data);
- 100% branch coverage;
- a line in the family catalog (`README.md`'s collapsible list and the `docs/families/*.md` page — both are
  parity-checked against `__all__` by `tests/test_docs_surface.py`).

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

Every pull request carries exactly one **type** label mirroring its conventional-commit title prefix 1:1 (`feat`,
`fix`, `perf`, `docs`, `refactor`, `style`, `test`, `build`, `ci`, `chore` — plus `breaking` whenever the title
carries `!` or the body carries a `**BREAKING**` / `BREAKING CHANGE` marker), enforced fail-closed by
`.github/workflows/labels.yml`; the **area** labels (`indicators` / `metrics` /
`pnl`) mirror the title scope and are triage-only. Releases use GitHub's native release notes: tag a SemVer version
and **Generate release notes**, which `.github/release.yml` groups by those type labels — breaking changes first,
Dependabot under `dependencies`. The tag is the single source of the version (`hatch-vcs` stamps the package from
it, and `CITATION.cff` is versionless by design, so a release needs no prep commit) — but publish deliberately:
Zenodo archives the release the moment it is published, independently of CI, and its records are immutable — a
delete-and-recreate leaves a duplicate archive behind.

## Conventions

- Code style is enforced by ruff (line length 120). Full type hints everywhere; the public API is fully typed.
- US English throughout, in code and prose.
- The runtime dependency tree is intentionally minimal (just Polars); new runtime dependencies are avoided.
