"""Declaration for ``pomata.indicators.roc`` — the windowed rate of change in percent, propagating, scale-invariant."""

import math

from pomata.indicators import roc
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_roc
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

ROC = suite_indicators(
    factory=roc,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_roc,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 200.0, 100.0, 66.6667), params={"window": 2}
    ),
    pins=(
        Pin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            params_override={"window": 1},
            expected=(None,),
            reason="a one-element series with window=1 stays undefined",
        ),
        Pin(
            label="window_one_is_single_period_return",
            inputs={"expr": (2.0, 4.0, 6.0)},
            params_override={"window": 1},
            expected=(None, 100.0, 50.0),
            reason="the one-period simple return in percent, the documented window=1 case",
        ),
        Pin(
            label="all_nan",
            inputs={"expr": (math.nan, math.nan, math.nan)},
            params_override={"window": 1},
            expected=(None, math.nan, math.nan),
            reason="an all-NaN series: warm-up stays null, then NaN propagates once both endpoints exist",
        ),
        Pin(
            label="constant_series_is_zero",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            params_override={"window": 1},
            expected=(None, 0.0, 0.0, 0.0),
            reason="ROC of a constant non-zero series is exactly 0 once warmed up",
        ),
        Pin(
            label="zero_lagged_nonzero_change_is_signed_inf",
            inputs={"expr": (0.0, 5.0, 0.0, -5.0)},
            params_override={"window": 1},
            expected=(None, math.inf, -100.0, -math.inf),
            reason="a non-zero change over a zero lagged value is +/-inf, sign tracking the change direction",
        ),
        Pin(
            label="zero_lagged_mixed_change",
            inputs={"expr": (0.0, 0.0, 5.0)},
            params_override={"window": 1},
            expected=(None, math.nan, math.inf),
            reason="a zero change over zero is NaN (0/0), a non-zero change over zero is +inf",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Momentum_%28technical_analysis%29",
    see_also=(
        ("mom", "The absolute-difference sibling."),
        ("trix", "The one-period rate of change of a triple-smoothed EMA."),
        ("rsi", "A bounded momentum oscillator."),
    ),
    bullets=(
        ("Null", "a ``null`` value makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        ("Insufficient sample", "a look-back longer than the series never completes, so the result is ``null``."),
        (
            "Degenerate denominator",
            "when both the change and the lagged value are ``0`` the ratio is indeterminate, so the "
            "result is a ``0 / 0``, i.e. ``NaN`` — a non-zero change over a zero lagged value is "
            "``+/-inf`` (the sign tracks the change relative to the signed zero).",
        ),
        ("window == 1", "the look-back is a single bar, so ``roc`` is the one-period simple return in percent."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The ROC for each row, the same length as ``expr``. The first ``window`` values are "
    "``null`` (warm-up) -- the lagged term ``expr.shift(window)`` is undefined for the first "
    "``window`` rows, so no change can be measured there.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations to look back. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, params={"window": 2}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0)},
            intro="A ``null`` (voiding the rows that reference it) and a ``NaN`` (which propagates) make "
            "the exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (42.0,)},
            intro="**Insufficient sample** — a one-element series with a one-bar window stays undefined:",
            params={"window": 1},
        ),
        Example(
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            intro="**Degenerate denominator** — ROC of a constant non-zero series is exactly ``0`` once warmed up:",
            params={"window": 1},
        ),
        Example(
            inputs={"expr": (0.0, 5.0, 0.0, -5.0)},
            intro="**Degenerate denominator** — a non-zero change over a zero lagged value is ``+/-inf``, "
            "the sign tracking the change direction:",
            params={"window": 1},
        ),
        Example(
            inputs={"expr": (0.0, 0.0, 5.0)},
            intro="**Degenerate denominator** — a zero change over zero is ``NaN`` (``0/0``), while a "
            "non-zero change over zero is ``+inf``:",
            params={"window": 1},
        ),
    ),
)
