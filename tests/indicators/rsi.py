"""Declaration for ``pomata.indicators.rsi`` — Wilder's relative strength index, gap-bridging, NaN-latching."""

import math

from pomata.indicators import rsi
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_rsi
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

RSI = suite_indicators(
    factory=rsi,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_rsi,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42)},
        output=(None, None, None, 7.0588, 59.0674, 74.1408, 80.0819, 85.8581),
    ),
    pins=(
        Pin(
            label="single_row_window_1",
            inputs={"expr": (42.0,)},
            params_override={"window": 1},
            expected=(None,),
            reason="a one-element series has no difference to seed the recursion ",
        ),
        Pin(
            label="single_row_window_2",
            inputs={"expr": (42.0,)},
            params_override={"window": 2},
            expected=(None,),
            reason="the same fact restated at window=2",
        ),
        Pin(
            label="window_one_is_move_direction",
            inputs={"expr": (1.0, 3.0, 2.0, 5.0)},
            params_override={"window": 1},
            expected=(None, 100.0, 0.0, 100.0),
            reason="window=1 collapses the Wilder smoothing to the raw move direction: 100 up, 0 down ",
        ),
        Pin(
            label="constant_series_is_nan",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            params_override={"window": 2},
            expected=(None, None, math.nan, math.nan),
            reason="zero gain and zero loss is the indeterminate 0/0 relative strength, surfaced as NaN ",
        ),
        Pin(
            label="monotone_increasing_is_hundred",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
            params_override={"window": 2},
            expected=(None, None, 100.0, 100.0, 100.0),
            reason="a strictly increasing series (no losses) saturates RSI to exactly 100 ",
        ),
        Pin(
            label="monotone_decreasing_is_zero",
            inputs={"expr": (10.0, 8.0, 6.0, 4.0, 2.0)},
            params_override={"window": 2},
            expected=(None, None, 0.0, 0.0, 0.0),
            reason="a strictly decreasing series (no gains) saturates RSI to exactly 0 ",
        ),
        Pin(
            label="leading_null_defers_first_difference",
            inputs={"expr": (None, 2.0, 4.0, 6.0, 8.0)},
            params_override={"window": 2},
            expected=(None, None, None, 100.0, 100.0),
            reason="a leading null defers the first difference; the warm-up is measured from the first non-null value ",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Relative_strength_index",
    see_also=(
        ("rma", "Wilder's moving average that smooths the gains and losses RSI is built on."),
        ("money_flow_index", "The volume-weighted analog — the same oscillator on raw money flow."),
        ("chande_momentum_oscillator", "The unsmoothed sibling that sums gains and losses over a fixed window."),
    ),
    notes=(
        (
            "Seeding",
            "The gain and loss averages use Wilder's :func:`rma`, seeded with the simple average of "
            "the first ``window`` gains and losses -- Wilder's canonical initialization, exact from "
            "the first emitted value.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "the one-bar difference reads the missing observation twice, so an interior ``null`` "
            "voids its own row and the next.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        (
            "Insufficient sample",
            "a series shorter than ``window + 1`` never completes the first difference, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "no up and no down move leaves the relative strength indeterminate, so the result is a "
            "``0 / 0``, i.e. ``NaN`` (genuinely undefined, not a conventional ``50`` or ``100``).",
        ),
        (
            "window == 1",
            "the smoothing vanishes: each row reports ``100`` on an up move, ``0`` on a down move, "
            "and ``NaN`` on no move.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The RSI for each row, the same length as ``expr``. The first ``window`` values are "
    "``null`` (warm-up) -- Wilder's RSI needs ``window + 1`` prices for its first value, "
    "since row ``0`` has no difference and the gain / loss averages count ``window`` non-null "
    "differences before emitting.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so the difference and the recursion "
    "restart per group -- note that each ticker warms up independently:",
    intro_missing="A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which "
    "propagates) make the exact handling visible at a glance:",
)
