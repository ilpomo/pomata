"""
Declaration for ``pomata.indicators.chaikin_money_flow`` — the windowed money-flow ratio, window-nulling, invariant.
"""

import math

from pomata.indicators import chaikin_money_flow
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_chaikin_money_flow
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

CHAIKIN_MONEY_FLOW = suite_indicators(
    factory=chaikin_money_flow,
    inputs=("high", "low", "close", "volume"),
    params={"window": 20},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_chaikin_money_flow,
    scaling=(
        ScaleAxis(roles=("high", "low", "close"), degree=0),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has the A/D line and Chaikin oscillator, but not the volume-normalized CMF.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 14.0),
            "low": (8.0, 9.0, 9.0, 10.0, 11.0),
            "close": (9.0, 11.0, 10.0, 12.0, 13.0),
            "volume": (100.0, 200.0, 150.0, 300.0, 250.0),
        },
        output=(None, None, 0.1481, 0.2564, 0.2619),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="zero_total_volume_is_nan",
            inputs={
                "high": (10.0, 12.0, 11.0),
                "low": (8.0, 9.0, 9.0),
                "close": (9.0, 11.0, 10.0),
                "volume": (0.0, 0.0, 0.0),
            },
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="a window whose total volume is zero divides by zero, the IEEE-754 0/0 == NaN",
        ),
        Pin(
            label="zero_volume_after_large_volume_is_nan",
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
                "close": (11.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5),
                "volume": (1e16, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
            },
            params_override={"window": 3},
            expected=(None, None, 0.0, 0.25, 0.2, 0.0, math.nan),
            reason="an all-zero-volume window still yields NaN after a large 1e16 volume has slid out: the rolling sum "
            "retains a sub-ULP residual on exit, but the exact all-zero detection pins the final window to NaN",
        ),
    ),
    reference='Chaikin, M. "Chaikin Money Flow."',
    wikipedia="https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_Money_Flow",
    see_also=(
        ("accumulation_distribution", "The cumulative (window-less) money-flow line."),
        ("money_flow_index", "A bounded windowed money-flow oscillator."),
        ("accumulation_distribution_oscillator", "Chaikin's momentum oscillator over the same line."),
    ),
    notes=(
        (
            "Zero-range bars",
            "The zero-range convention applies only to a genuine equal-range bar (``high == low``), "
            "where the multiplier is ``0`` (it adds ``0`` to the numerator while its volume still "
            "counts in the denominator) — and on such a bar it wins outright: the multiplier is ``0`` "
            "regardless of the close, so a ``null`` or ``NaN`` close on a doji is absorbed into the "
            "zero flow. On a bar with a genuine range, a ``null`` or ``NaN`` in any input leaves that "
            "bar's money-flow volume ``null`` or ``NaN``, so missing data propagates rather than "
            "being silently zeroed.",
        ),
        (
            "Clamp convention",
            "The result is clamped to its ``[-1, +1]`` bound: a malformed bar whose ``close`` prints "
            "outside its ``[low, high]`` range (pushing the multiplier past ``±1``) is pinned to the "
            "bound rather than allowed to escape it.",
        ),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window whose volume is all zero, so the result is a ``0 / 0``, i.e. ``NaN`` — detected "
            "exactly via the rolling maximum of the absolute volume, so a sub-ULP residual in the "
            "rolling-sum denominator cannot fake a finite reading (the only reachable "
            "division-by-zero case, since an all-zero volume window also zeroes the numerator).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The CMF for each row, the same length as the inputs. The first ``window - 1`` values are "
    "``null`` (warm-up) -- the value is defined only once a full window of bars is available.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 12.0, 11.0, 13.0, 14.0),
                "low": (8.0, 9.0, 9.0, 10.0, 11.0),
                "close": (9.0, 11.0, 10.0, 12.0, 13.0),
                "volume": (100.0, 200.0, 150.0, 300.0, 250.0),
            },
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0),
                "low": (10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0),
                "close": (11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
                "close": (11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0),
            },
            intro="A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which "
            "propagates) make the exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 12.0, 11.0),
                "low": (8.0, 9.0, 9.0),
                "close": (9.0, 11.0, 10.0),
                "volume": (0.0, 0.0, 0.0),
            },
            intro="**Degenerate denominator** — a window whose total volume is zero is the genuine ``0/0``, "
            "so the flow reads ``NaN``:",
            params={"window": 2},
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
                "close": (11.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5),
                "volume": (1e16, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
            },
            intro="**Degenerate denominator** — once a large volume has slid out of the window, the "
            "rolling-sum denominator can carry a sub-ULP residual, yet the exact all-zero-volume "
            "detection still resolves the final window to ``NaN`` rather than a spurious finite "
            "ratio:",
            params={"window": 3},
        ),
    ),
)
