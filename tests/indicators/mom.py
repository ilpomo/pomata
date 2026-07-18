"""Declaration for ``pomata.indicators.mom`` — the fixed-lag momentum difference, propagating, degree-1 homogeneous."""

import math

from pomata.indicators import mom
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_mom
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

MOM = suite_indicators(
    factory=mom,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_mom,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)}, output=(None, None, None, -2.0, 4.0, 5.0, 1.0, 1.0)
    ),
    pins=(
        Pin(
            label="window_one_is_first_difference",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0)},
            expected=(None, 2.0, 2.0, 2.0),
            params_override={"window": 1},
            reason="window=1 is the first difference with a single leading null",
        ),
        Pin(
            label="all_nan_series",
            inputs={"expr": (math.nan, math.nan, math.nan, math.nan)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 1},
            reason="an all-NaN series yields null during warm-up and NaN thereafter",
        ),
        Pin(
            label="constant_series_is_zero",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, 0.0, 0.0, 0.0),
            reason="the momentum of a constant series is exactly zero once warmed up",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Momentum_%28technical_analysis%29",
    see_also=(
        ("roc", "The percentage-change sibling (scale-invariant)."),
        ("rsi", "A bounded momentum oscillator."),
        ("chande_momentum_oscillator", "A bounded net-of-gains-and-losses momentum oscillator."),
    ),
    bullets=(
        ("Null", "a ``null`` value makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there — a fixed-lag difference, "
            "not a recurrence, so a ``null`` or ``NaN`` contaminates only the (at most two) positions "
            "that reference it and never latches onto the rest of the series.",
        ),
        ("Degenerate denominator", "a flat look-back leaves ``x_t == x_{t-n}``, so the difference is exactly ``0``."),
        (
            "window == 1",
            "the look-back is a single bar, so ``mom`` is the one-step first difference ``x_t - x_{t-1}``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The momentum for each row, the same length as ``expr``. The first ``window`` values are "
    "``null`` (warm-up), clamped to the series length: unlike the moving-average family, "
    "whose warm-up is ``window - 1`` rows, the value at row ``t`` needs the observation at "
    "row ``t - window``, which first exists at ``t == window``.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations to look back. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (voiding the rows that reference it) and a ``NaN`` (which propagates) make "
    "the exact handling visible at a glance:",
)
