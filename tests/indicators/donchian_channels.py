"""
Declaration for ``pomata.indicators.donchian_channels`` — the rolling high/low channel struct, window-nulling,
degree-1.
"""

from pomata.indicators import donchian_channels
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_donchian_channels
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

DONCHIAN_CHANNELS = suite_indicators(
    factory=donchian_channels,
    inputs=("high", "low"),
    params={"window": 20},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_donchian_channels,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Donchian channels.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        },
        output={
            "lower": (None, None, 9.0, 10.0, 11.0),
            "middle": (None, None, 11.0, 11.5, 12.5),
            "upper": (None, None, 13.0, 13.0, 14.0),
        },
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_tracks_each_bar",
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            params_override={"window": 1},
            expected={
                "lower": (9.0, 10.0, 11.0),
                "middle": (10.0, 11.0, 12.0),
                "upper": (11.0, 12.0, 13.0),
            },
            reason="window=1 makes the upper/lower channel the bar's own high/low and the middle their mean, with no "
            "warm-up",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Donchian_channel",
    see_also=(
        ("midprice", "The channel's middle band on its own."),
        ("keltner_channels", "The same band shape, an EMA midline with ATR width instead of window extremes."),
        ("bollinger_bands", "Volatility bands around a moving average rather than around the window's extremes."),
    ),
    notes=(
        (
            "Inputs",
            "``high`` and ``low`` must share a length and alignment (the same row index is one bar). "
            "The channel does not assume ``high >= low``: a malformed bar where ``high < low`` flows "
            "through unchanged (the upper band can then sit below the lower band) rather than being "
            "silently reordered.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — a ``null`` in the ``high`` window nulls ``upper`` and ``middle``, a ``null`` "
            "in the ``low`` window nulls ``lower`` and ``middle``; a fully missing bar nulls all "
            "three.",
        ),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there — per band, with ``null`` "
            "still taking precedence over ``NaN``.",
        ),
        (
            "window == 1",
            "the bands are the bar's own ``high`` and ``low``, and the middle is its :func:`price_median`.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with three ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``lower`` — the lowest ``low`` over the window. - ``middle`` — the channel midline, "
    "``(upper + lower) / 2`` (identical to :func:`midprice`). - ``upper`` — the highest "
    "``high`` over the window."
    "\n\n"
    'Read one band with ``.struct.field("upper")`` (etc.) or split all three into columns '
    "with ``.struct.unnest()``. The first ``window - 1`` rows are ``null`` (warm-up).",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window (canonically ``20``, the Donchian period). "
        "Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's bands warm up independently:",
    intro_missing="A ``null`` (nulling every band whose window reads it — here ``upper`` and ``middle``, "
    "while ``lower`` stays defined) and a ``NaN`` (which propagates) make the handling "
    "visible:",
)
