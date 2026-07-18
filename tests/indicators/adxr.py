"""
Declaration for ``pomata.indicators.adxr`` — Wilder's average directional index rating, gap-bridging, scale-invariant.
"""

import math

from pomata.indicators import adxr
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_adxr
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

ADXR = suite_indicators(
    factory=adxr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=40,
    oracle=reference_adxr,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.DOCUMENTED_DIVERGENCE,
    talib_reason="ADXR averages ADX with its lagged self; pomata lags by `window`, TA-Lib by `window - 1`.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output=(None, None, None, None, 84.1176, 52.0588, 63.2977, 41.6489, 56.9044, 38.4522),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 11, "low": (10.0,) * 11, "close": (10.0,) * 11},
            params_override={"window": 3},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
            ),
            reason="a fully flat window makes the underlying dx the indeterminate 0/0, which poisons the Wilder "
            "smoothing recursion and the averaging of the current and one-window-back adx",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_directional_movement_index",
    see_also=(
        ("adx", "The trend-strength index this averages with its own past."),
        ("dx", "The directional index the ADX smooths."),
        ("di_plus", "A directional indicator at the base of the system."),
    ),
    notes=(
        ("Seeding", "The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster."),
        (
            "Documented TA-Lib divergence",
            "TA-Lib averages the current ADX with the ADX from ``window - 1`` bars back; pomata uses "
            "``window`` bars back, Wilder's book convention, so the two never agree even at steady "
            "state and the differential tier holds ADXR out as a documented divergence.",
        ),
    ),
    note_extension="\n\nIt is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close``.",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "inherited from :func:`adx`; a row whose ADX or whose ``window``-ago ADX is missing is "
            "itself missing.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — inherited from :func:`adx`.",
        ),
        (
            "Degenerate denominator",
            "``di+`` and ``di-`` are both zero, so the result is a ``0 / 0``, i.e. ``NaN`` — "
            "inherited from :func:`adx`, which then poisons the averaging of the current and "
            "``window``-ago values.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The ADXR for each row, the same length as the inputs, in ``[0, 100]``. Its warm-up is "
    "the :func:`adx` warm-up plus a further ``window`` rows (the look-back of the averaging).",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window, and the look-back for the averaging. "
        "Must be ``>= 1``.",
    },
    intro_basic="On a small OHLC frame with a short window:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` "
    "(which latches) make the handling visible:",
)
