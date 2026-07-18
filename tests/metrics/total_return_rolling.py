"""Declaration for ``pomata.metrics.total_return_rolling`` — the growth over a trailing window, scale-invariant."""

import math

from pomata.metrics import total_return_rolling
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_total_return_rolling
from tests.support.declaration import Golden, Pin, ScaleAxis

TOTAL_RETURN_ROLLING = suite_metrics(
    factory=total_return_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    window="window",
    annualization=Annualization.NONE,
    warmup=2,
    oracle=reference_total_return_rolling,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
        output=(None, None, 0.05, 0.0909, 0.0952, 0.0833, 0.087),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15)},
            expected=(None, None, None, None, 0.1499999999999999),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="endpoint_null_is_null",
            inputs={"equity_curve": (1.0, 1.1, None)},
            expected=(None, None, None),
            reason="a null at a window endpoint yields null: the result depends on both endpoints",
        ),
        Pin(
            label="interior_null_is_spanned",
            inputs={"equity_curve": (1.0, None, 1.2)},
            expected=(None, None, 0.19999999999999996),
            reason="an interior null has zero effect on a fully-defined window; only the two endpoints determine it",
        ),
        Pin(
            label="endpoint_nan_propagates",
            inputs={"equity_curve": (1.0, 1.1, math.nan)},
            expected=(None, None, math.nan),
            reason="a NaN at a window endpoint propagates to NaN",
        ),
        Pin(
            label="matches_reference_representative_curve",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
            expected=(
                None,
                None,
                None,
                0.19999999999999996,
                0.04545454545454519,
                0.23809523809523814,
                0.04166666666666674,
                0.21739130434782616,
            ),
            reason="a representative equity curve compared against the naive reference at the correctness tier",
            params_override={"window": 4},
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Total_return",
    see_also=(
        ("total_return", "The whole-series reducing form."),
        ("cagr_rolling", "The annualized (per-year) windowed counterpart."),
        ("cagr", "The whole-series, annualized growth rate."),
    ),
    opener_override="Each window matches an independent reference oracle (the endpoint ratio less one).",
    bullets=(
        (
            "Null",
            "a ``null`` equity makes that row ``null`` (``null`` takes precedence over ``NaN``) — "
            "being an endpoint quantity, an interior ``null`` does not affect the result.",
        ),
        ("NaN", "a ``NaN`` at either endpoint propagates, yielding ``NaN``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling total return for each row, the same length as the input. The first ``window "
    "- 1`` rows are ``null`` (warm-up): the window must reach back ``window`` rows before a "
    "result is emitted.",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` or ``NaN`` at a window endpoint propagates, while a ``NaN`` interior to a "
    "window is ignored:",
)
