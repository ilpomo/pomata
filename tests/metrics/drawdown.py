"""Declaration for ``pomata.metrics.drawdown`` — the running fractional decline from a prior peak, scale-invariant."""

import math

from pomata.metrics import drawdown
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_drawdown
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

DRAWDOWN = suite_metrics(
    factory=drawdown,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    annualization=Annualization.NONE,
    shape=Shape.SERIES,
    oracle=reference_drawdown,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(
        inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667)
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the drawdown is 0",
        ),
        Pin(
            label="leading_null",
            inputs={"equity_curve": (None, 1.0, 1.1, 0.99, 1.2)},
            expected=(None, 0.0, 0.0, -0.10000000000000009, 0.0),
            reason="a leading warm-up null stays null and the curve begins at the first defined equity",
        ),
        Pin(
            label="interior_null_carries_peak",
            inputs={"equity_curve": (1.0, 1.2, None, 1.1, 1.3)},
            expected=(0.0, 0.0, None, -0.08333333333333326, 0.0),
            reason="an interior null yields null at that row while the running peak carries across it",
        ),
        Pin(
            label="interior_nan_row_propagates_and_peak_carries",
            inputs={"equity_curve": (1.0, 1.1, math.nan, 0.9, 1.2)},
            expected=(0.0, 0.0, math.nan, -0.18181818181818188, 0.0),
            reason="a NaN equity yields NaN at that row while the running peak ignores it",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Drawdown_%28economics%29",
    see_also=(
        ("max_drawdown", "The deepest point of this series."),
        ("ulcer_index", "The root-mean-square of this series."),
        ("drawdown_rolling", "The trailing-window form, healed once an old peak rolls out."),
    ),
    notes=(
        (
            "Inception",
            "The running peak starts at the FIRST observation: a curve fed from "
            ":func:`~pomata.pnl.equity_curve` begins at its first post-return value, so a drawdown "
            "from the starting capital itself (an opening losing streak) is invisible by "
            "construction. Prepend a literal ``1.0`` row to count declines from inception; the "
            "convention matches quantstats (empyrical instead prepends the start).",
        ),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity makes that row ``null`` (``null`` takes precedence over ``NaN``); the "
            "running peak carries across it unchanged.",
        ),
        (
            "NaN",
            "a ``NaN`` equity yields ``NaN`` for that row; the running peak ignores it (Polars' "
            "``cum_max`` semantics), so later rows are unaffected.",
        ),
        (
            "Insufficient sample",
            "a single-row series is trivially at its own peak, so its (only) drawdown is exactly ``0``, not ``null``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The drawdown for each row, the same length as ``equity_curve`` -- ``0`` at a running "
    "peak and negative while below it. A leading input ``null`` stays ``null``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's running peak "
    "restarts independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which propagates at its row) make the missing-data "
    "handling visible:",
)
