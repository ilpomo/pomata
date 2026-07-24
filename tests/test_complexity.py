"""
The complexity tier: every public function stays linear in the rows and free of the window, measured.

``pomata``'s performance contract: every function is **O(n)** in the number of rows, and the window is free
(**O(1)**) wherever Polars offers a streaming rolling primitive — which is everywhere except the window-composed
studies sanctioned by name below, built from per-offset shifts because no streaming primitive exists for their
shape (weighted means, the rolling regression family, the rolling MAD, the bars-since-extremum scan); those scale
as **O(n·w)** and the guard holds the list exact. The module verifies all of it empirically, at deliberately
small sizes: the row check fits the growth exponent between the smallest and the largest measured size and rejects
anything approaching quadratic; the window check grows the window 32x at a fixed row count and rejects a cost that
grows with it. Both bounds carry orders-of-magnitude margins, so runner jitter cannot redden them: a genuine
``O(n^2)`` lands near exponent ``2`` against a ceiling of ``1.4`` (a fixed per-call overhead can only *depress*
the measured exponent, never raise it), and a genuine ``O(w)`` grows ~32x against a ratio ceiling of ``1.5``.

The tier is timing-based, so it never gates: it activates only under ``POMATA_COMPLEXITY=1`` and must run serially
(parallel workers corrupt timings) — the nightly job does both. A deliberately super-linear function, should one
ever land, must be sanctioned by name in the fail-closed allowlist below; an entry whose function measures linear
fails as stale.
"""

import gc
import math
import os
import statistics
import time
from typing import Final

import polars as pl
import pytest

import tests.all_declarations as _registered
from tests.support.declaration import Declaration, build_expr
from tests.support.frames import probe_frame
from tests.support.registry import registry_all

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

if not os.environ.get("POMATA_COMPLEXITY"):
    pytest.skip(
        "complexity tier: set POMATA_COMPLEXITY=1 and run serially (the nightly job does)", allow_module_level=True
    )

pytestmark = pytest.mark.complexity

# The row ladder spans 64x, wide enough that linear growth clears the fixed per-select overhead of the fastest
# functions; the window sweep spans 32x at a row count where a windowed cost would be plainly visible.
_ROW_SIZES: Final = (4_000, 16_000, 64_000, 256_000)
_WINDOW_SIZES: Final = (8, 32, 128, 256)
_WINDOW_ROWS: Final = 32_000
# A linear function's endpoint exponent sits near 1.0 and overhead can only depress it; a quadratic one reaches ~2.
# The 1.4 ceiling splits the two with a wide band on both sides, so timing jitter cannot cross it.
_EXPONENT_CEILING: Final = 1.4
# Across a 32x window sweep an O(w) cost grows ~32x; a streaming O(1) one stays flat. The 1.5 ceiling tolerates
# cache and jitter effects while sitting far below any real window dependence.
_WINDOW_RATIO_CEILING: Final = 1.5

# Functions sanctioned as deliberately super-linear IN THE ROWS, by name, each with a reason — the fail-closed
# exception path. Empty: the whole surface is linear in the rows today. A stale entry fails.
_SUPERLINEAR_SANCTIONED: Final[dict[str, str]] = {}

# Functions whose cost deliberately scales with the window (O(w) work per row): each is composed from per-offset
# shifts because Polars has no streaming primitive for its shape. Fail-closed both ways: an unsanctioned function
# that grows fails, and a sanctioned one that measures flat fails as stale.
_WINDOW_SCALING_SANCTIONED: Final[dict[str, str]] = {
    "aroon": "the bars-since-extremum scan compares every offset in the window (no streaming rolling argmax)",
    "aroon_oscillator": "composes aroon's two legs; inherits the per-offset scan",
    "cci": "the mean absolute deviation is rebuilt from per-offset shifts (no streaming rolling MAD)",
    "hma": "composes wma at three windows; inherits wma's per-offset construction",
    "linear_regression": "the rolling least-squares line is built from per-offset weighted shifts",
    "linear_regression_angle": "shares linear_regression's per-offset construction",
    "linear_regression_intercept": "shares linear_regression's per-offset construction",
    "linear_regression_slope": "shares linear_regression's per-offset construction",
    "time_series_forecast": "the regression-endpoint forecast shares the per-offset construction",
    "wma": "the linear weights are applied via per-offset shifts (no streaming weighted rolling)",
}

_DECLARATIONS = sorted(registry_all(), key=lambda declaration: (declaration.family, declaration.name))
_IDS = [declaration.name for declaration in _DECLARATIONS]


def _single_window(declaration: Declaration) -> str | None:
    """The one window-like parameter to sweep, or ``None`` (no window, or several — the single-axis sweep skips)."""
    windowish = [key for key in declaration.params if "window" in key]
    return windowish[0] if len(windowish) == 1 else None


def _timed(frame: pl.DataFrame, expr: pl.Expr) -> float:
    gc.disable()
    try:
        start = time.perf_counter_ns()
        frame.select(expr)
        return (time.perf_counter_ns() - start) / 1e9
    finally:
        gc.enable()


def _measure(frame: pl.DataFrame, expr: pl.Expr) -> float:
    """Median of three timed evaluations after one warm-up."""
    frame.select(expr)
    return statistics.median(_timed(frame, expr) for _ in range(3))


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_rows_stay_linear(declaration: Declaration) -> None:
    """The growth exponent across the row ladder stays below the quadratic ceiling — the O(n) leg of the contract."""
    expr = build_expr(declaration)
    times = [_measure(probe_frame(declaration.inputs, n), expr) for n in _ROW_SIZES]
    exponent = math.log(times[-1] / times[0]) / math.log(_ROW_SIZES[-1] / _ROW_SIZES[0])
    if declaration.name in _SUPERLINEAR_SANCTIONED:
        assert exponent >= _EXPONENT_CEILING, (
            f"{declaration.name}: sanctioned as super-linear ({_SUPERLINEAR_SANCTIONED[declaration.name]}) but "
            f"measures exponent {exponent:.2f} — remove the stale allowlist entry"
        )
        return
    assert exponent < _EXPONENT_CEILING, (
        f"{declaration.name}: growth exponent {exponent:.2f} over rows {_ROW_SIZES[0]:,}..{_ROW_SIZES[-1]:,} "
        f"(times {[round(t * 1e3, 2) for t in times]} ms) breaches the linear contract — the implementation has "
        f"gone super-linear, or sanction it in _SUPERLINEAR_SANCTIONED with a reason"
    )


@pytest.mark.parametrize(
    "declaration",
    [declaration for declaration in _DECLARATIONS if _single_window(declaration) is not None],
    ids=[declaration.name for declaration in _DECLARATIONS if _single_window(declaration) is not None],
)
def test_window_stays_free(declaration: Declaration) -> None:
    """Growing the window 32x must not grow the cost — the O(1)-in-the-window leg of the contract."""
    param = _single_window(declaration)
    assert param is not None  # parametrization filter
    frame = probe_frame(declaration.inputs, _WINDOW_ROWS)
    times = [_measure(frame, build_expr(declaration, **{param: w})) for w in _WINDOW_SIZES]
    ratio = times[-1] / times[0]
    if declaration.name in _WINDOW_SCALING_SANCTIONED:
        assert ratio >= _WINDOW_RATIO_CEILING, (
            f"{declaration.name}: sanctioned as window-scaling ({_WINDOW_SCALING_SANCTIONED[declaration.name]}) "
            f"but measures flat ({ratio:.2f}x) — remove the stale allowlist entry"
        )
        return
    assert ratio < _WINDOW_RATIO_CEILING, (
        f"{declaration.name}: cost grows {ratio:.2f}x when {param} grows {_WINDOW_SIZES[0]}→{_WINDOW_SIZES[-1]} "
        f"(times {[round(t * 1e3, 2) for t in times]} ms) — the window is no longer free; fix the implementation "
        f"or sanction it in _WINDOW_SCALING_SANCTIONED with a reason"
    )
