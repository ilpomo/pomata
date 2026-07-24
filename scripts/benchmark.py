"""
Measure every public function against its naive reference oracle and a native-polars anchor, at growing sizes.

Three subjects per function, all on the same seeded deterministic frame:

- ``pomata`` — the public expression, evaluated eagerly via ``select``.
- ``oracle`` — the suite's hand-written naive reference, measured at each size in ascending order until a run
  exceeds the time budget; the largest size reached is recorded, so the ceiling is a measured fact, never a guess.
- ``anchor`` — a native polars ``rolling_mean`` over the function's first input column: the hardware normalizer
  that makes ratios comparable across machines.

The panel sweep holds the row count fixed and grows the number of groups under ``.over``, measuring ``pomata``
and the anchor only (a per-group Python oracle would time the loop, not the function).

Usage::

    uv run python scripts/benchmark.py --out docs/_static/benchmarks/results.json
    uv run python scripts/benchmark.py --sizes 1000 100000 1000000 --panel-rows 200000   # a faster preview run

Timing protocol: one warm-up evaluation, then the median of up to 5 timed runs (3 past 0.5 s, 1 past 1.5 s),
garbage collector disabled inside the timed region. Machine and version metadata land in the JSON.
"""

import argparse
import datetime
import gc
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# The oracles and the declaration registry live under tests/; make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pomata
import tests.all_declarations  # noqa: F401  - registration side effects
from tests.support.declaration import build_expr
from tests.support.registry import registry_all

_SEED = 20260719
_PRICE_ROLES = frozenset({"high", "low", "close", "open"})


def _bars(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """A coherent OHLC panel from a geometric walk: low <= open, close <= high, everything positive."""
    steps = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    spread = np.abs(rng.normal(0.0, 0.006, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return {"open": open_, "high": high, "low": low, "close": close}


def _lane(rng: np.random.Generator, role: str, n: int, bars: dict[str, np.ndarray]) -> np.ndarray:
    """One realistic seeded lane for ``role`` (price roles come from the shared coherent bars)."""
    if role in _PRICE_ROLES:
        return bars[role]
    if role == "volume":
        return np.exp(rng.normal(11.0, 0.4, n))
    if role in {"returns", "benchmark", "asset_returns"}:
        return rng.normal(0.0005, 0.01, n)
    if role == "equity_curve":
        return np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
    if role == "quantity":
        return rng.integers(-20, 21, n).astype(np.float64)
    if role == "weight":
        return rng.uniform(-1.0, 1.0, n)
    if role == "dividend_per_share":
        return np.where(rng.uniform(0.0, 1.0, n) < 0.05, rng.uniform(0.1, 1.0, n), 0.0)
    if "rate" in role:
        return rng.normal(0.0001, 0.0005, n)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def _frame(declaration, n: int) -> pl.DataFrame:
    rng = np.random.default_rng([_SEED, n, abs(hash(declaration.name)) % 2**31])
    bars = _bars(rng, n)
    return pl.DataFrame({role: pl.Series(_lane(rng, role, n, bars), dtype=pl.Float64) for role in declaration.inputs})


def _timed(fn) -> float:
    gc.disable()
    try:
        start = time.perf_counter_ns()
        fn()
        return (time.perf_counter_ns() - start) / 1e9
    finally:
        gc.enable()


def _measure(fn) -> float:
    """Median of up to 5 timed runs after one warm-up (3 runs past 0.5 s, a single run past 1.5 s)."""
    fn()
    first = _timed(fn)
    if first > 1.5:
        return first
    runs = 3 if first > 0.5 else 5
    return statistics.median([first, *(_timed(fn) for _ in range(runs - 1))])


def _anchor_expr(declaration) -> pl.Expr:
    return pl.col(declaration.inputs[0]).rolling_mean(20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/_static/benchmarks/results.json")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    )
    parser.add_argument("--time-budget-min", type=float, default=30.0, help="stop climbing sizes past this wall time")
    parser.add_argument("--panel", action="store_true", help="also run the .over group-count sweep")
    parser.add_argument("--panel-rows", type=int, default=1_000_000)
    parser.add_argument("--panel-groups", type=int, nargs="+", default=[1, 10, 100, 1_000])
    parser.add_argument("--oracle-budget", type=float, default=2.0)
    parser.add_argument("--cell-budget", type=float, default=5.0, help="per-measurement budget for pomata/anchor")
    args = parser.parse_args(argv)

    declarations = sorted(registry_all(), key=lambda d: (d.family, d.name))
    exprs = {f"{d.family}.{d.name}": (build_expr(d), _anchor_expr(d)) for d in declarations}
    series: dict[str, dict] = {
        f"{d.family}.{d.name}": {"sizes": {}, "pomata_ceiling": 0, "oracle_ceiling": 0} for d in declarations
    }
    pomata_alive = dict.fromkeys(series, True)
    oracle_alive = dict.fromkeys(series, True)

    # Climb the size ladder level by level (all functions at each size), stopping before a level whose start is past
    # the wall-clock budget: the small levels fly, and the climb halts on its own before the Python kernels collapse.
    start = time.perf_counter()
    for n in sorted(args.sizes):
        elapsed = (time.perf_counter() - start) / 60.0
        if elapsed > args.time_budget_min:
            print(f"stopping before {n:,} rows: {elapsed:.1f} min spent, budget {args.time_budget_min} min", flush=True)
            break
        level_start = time.perf_counter()
        for declaration in declarations:
            key = f"{declaration.family}.{declaration.name}"
            if not pomata_alive[key] and not oracle_alive[key]:
                continue
            expr, anchor = exprs[key]
            frame = _frame(declaration, n)
            entry: dict[str, float | None] = {"pomata_s": None, "anchor_s": None, "oracle_s": None}
            if pomata_alive[key]:
                took = _measure(lambda: frame.select(expr))
                entry["pomata_s"] = took
                entry["anchor_s"] = _measure(lambda: frame.select(anchor))
                series[key]["pomata_ceiling"] = n
                pomata_alive[key] = took <= args.cell_budget
            if oracle_alive[key]:
                lanes = [frame[role].to_list() for role in declaration.inputs]
                took = _timed(lambda: declaration.oracle(*lanes, **dict(declaration.params)))
                entry["oracle_s"] = took
                series[key]["oracle_ceiling"] = n
                oracle_alive[key] = took <= args.oracle_budget
            series[key]["sizes"][str(n)] = entry
        alive = sum(pomata_alive.values())
        print(f"size {n:>12,}: {time.perf_counter() - level_start:>7.1f} s  ({alive}/{len(series)} still climbing)", flush=True)

    panel: dict[str, dict] = {}
    if args.panel:
        for declaration in declarations:
            key = f"{declaration.family}.{declaration.name}"
            expr, anchor = exprs[key]
            groups: dict[str, dict] = {}
            panel_alive = True
            for g in args.panel_groups:
                if not panel_alive:
                    break
                frame = _frame(declaration, args.panel_rows)
                tickers = np.repeat([f"T{i:04d}" for i in range(g)], math.ceil(args.panel_rows / g))[: args.panel_rows]
                frame = frame.with_columns(pl.Series("ticker", tickers))
                took = _measure(lambda: frame.select(expr.over("ticker")))
                groups[str(g)] = {"pomata_s": took, "anchor_s": _measure(lambda: frame.select(anchor.over("ticker")))}
                panel_alive = took <= args.cell_budget
            panel[key] = {"rows": args.panel_rows, "groups": groups}

    payload = {
        "metadata": {
            "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "polars": pl.__version__,
                "pomata": pomata.__version__,
                "threads": pl.thread_pool_size(),
            },
            "protocol": {
                "runs": "median of up to 5 (3 past 0.5 s, 1 past 1.5 s), one warm-up, GC off",
                "oracle_budget_seconds": args.oracle_budget,
                "cell_budget_seconds": args.cell_budget,
                "time_budget_min": args.time_budget_min,
                "sizes": sorted(args.sizes),
                "panel": args.panel,
                "seed": _SEED,
            },
        },
        "series": series,
        "panel": panel,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
