"""Declaration for ``pomata.indicators.trima`` — the triangular (SMA-of-SMA) rolling mean, window-nulling, degree-1."""

import math

from pomata.indicators import trima
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_trima
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

_SWEEP = (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)

TRIMA = suite_indicators(
    factory=trima,
    inputs=("expr",),
    params={"window": 5},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_trima,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0)}, output=(None, None, None, 2.3333, 2.6667), params={"window": 4}
    ),
    pins=(
        Pin(
            label="single_row_window_one_identity",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="window=1 on one row returns the value itself",
        ),
        Pin(
            label="single_row_window_exceeds_length",
            inputs={"expr": (42.0,)},
            expected=(None,),
            params_override={"window": 3},
            reason="window > length on one row yields null",
        ),
        Pin(
            label="null_in_window_recovers",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, None, 3.5),
            params_override={"window": 2},
            reason="a null inside the window yields null there, and the value returns once the window clears",
        ),
        Pin(
            label="nan_propagates_confined",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, 3.5),
            params_override={"window": 2},
            reason="a NaN inside the window yields NaN there, confined to the windows spanning it",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reduces to the identity across a full series",
        ),
        Pin(
            label="matches_reference_window_2",
            inputs={"expr": _SWEEP},
            expected=(None, 2.0, 2.5, 2.5, 3.0, 7.0, 5.5, 4.0),
            params_override={"window": 2},
            reason="the even-window branch of the odd/even weight construction at the smallest even window",
        ),
        Pin(
            label="matches_reference_window_3",
            inputs={"expr": _SWEEP},
            expected=(None, None, 2.25, 2.5, 2.75, 5.0, 6.25, 4.75),
            params_override={"window": 3},
            reason="the odd-window branch of the odd/even weight construction at the smallest non-trivial odd window",
        ),
        Pin(
            label="matches_reference_window_4",
            inputs={"expr": _SWEEP},
            expected=(
                None,
                None,
                None,
                2.3333333333333335,
                2.6666666666666665,
                4.166666666666667,
                5.166666666666667,
                5.5,
            ),
            params_override={"window": 4},
            reason="the wider even-window branch of the odd/even weight construction",
        ),
        Pin(
            label="matches_reference_window_5",
            inputs={"expr": _SWEEP},
            expected=(
                None,
                None,
                None,
                None,
                2.6666666666666665,
                3.4444444444444446,
                4.555555555555556,
                5.333333333333333,
            ),
            params_override={"window": 5},
            reason="the odd-window branch of the odd/even weight construction at the canonical window",
        ),
        Pin(
            label="matches_reference_window_6",
            inputs={"expr": _SWEEP},
            expected=(None, None, None, None, None, 3.25, 3.916666666666667, 4.833333333333334),
            params_override={"window": 6},
            reason="the widest window of the reference sweep, one past the canonical window",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Moving_average",
    see_also=(
        ("sma", "The single-pass simple moving average this double-smooths."),
        ("wma", "A single-pass linearly-weighted average, also tilting the window's weights off uniform."),
        ("hma", "Another average built by composing simpler moving averages."),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — built from two :func:`sma` passes, each holding to that same "
            "``min_samples=window`` contract.",
        ),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there — through both "
            ":func:`sma` passes it composes.",
        ),
        ("Insufficient sample", "a series shorter than ``window`` observations, so the result is ``null``."),
        ("window == 1", "both sub-windows are ``1``, so the TRIMA reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The triangular moving average for each row, the same length as the input. The first "
    "``window - 1`` values are ``null`` (warm-up), matching the uniform warm-up of the "
    "moving-average family.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (which both passes propagate) and a ``NaN`` make the missing-data handling visible:",
)
