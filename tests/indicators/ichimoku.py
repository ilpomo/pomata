"""
Declaration for ``pomata.indicators.ichimoku`` — a struct of four rolling midpoints, per-field warm-ups, three
windows.
"""

from pomata.indicators import ichimoku
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_ichimoku
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

ICHIMOKU = suite_indicators(
    factory=ichimoku,
    inputs=("high", "low"),
    params={"window_tenkan": 9, "window_kijun": 26, "window_senkou": 52},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("tenkan", "kijun", "senkou_a", "senkou_b"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"tenkan": 8, "kijun": 25, "senkou_a": 25, "senkou_b": 51},
    oracle=reference_ichimoku,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"tenkan": 1, "kijun": 1, "senkou_a": 1, "senkou_b": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Ichimoku Kinko Hyo.",
    raises=(
        ({"window_tenkan": 0}, r"window_tenkan must be >= 1"),
        ({"window_kijun": 0, "window_tenkan": 1}, r"window_kijun must be >= 1"),
        ({"window_senkou": 0, "window_tenkan": 1, "window_kijun": 1}, r"window_senkou must be >= 1"),
        ({"window_kijun": 5}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
        ({"window_senkou": 10}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0),
            "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0),
        },
        output={
            "tenkan": (None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0),
            "kijun": (None, None, 10.0, 11.0, 12.0, 12.0, 12.5, 12.5),
            "senkou_a": (None, None, 10.25, 11.25, 12.25, 12.0, 12.5, 12.75),
            "senkou_b": (None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5),
        },
        params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
    ),
    reference="Hosoda, G. (1969). *Ichimoku Kinkō Hyō*.",
    wikipedia="https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D",
    see_also=(
        ("midprice", "The single rolling high-low midpoint each Ichimoku line is built from."),
        ("donchian_channels", "The same window extremes kept as separate bands rather than midpoints."),
        ("keltner_channels", "Another channel, an EMA midline with ATR bands rather than rolling midpoints."),
    ),
    notes=(
        (
            "Displacement (no lookahead)",
            "Each line is emitted aligned to the row it is computed from -- zero displacement -- so "
            "the output never reads a future bar and is safe to feed a backtest directly. The "
            "traditional chart instead plots the two leading spans ``window_kijun`` bars into the "
            "future and a *chikou* (lagging) span ``window_kijun`` bars into the past; that is a "
            "presentation choice, applied on the user's side with ``.shift(...)`` -- e.g. "
            '``...struct.field("senkou_a") .shift(window_kijun)`` to lead, '
            '``pl.col("close").shift(-window_kijun)`` to lag. The chikou span is deliberately not '
            "emitted: un-displaced it is identical to ``close``, and its backward shift reads future "
            "bars, which must never enter a backtest.",
        ),
    ),
    note_extension="\n\n"
    "Every line is homogeneous of degree ``1`` under a positive common rescaling of ``high`` "
    "and ``low`` (each is a midpoint of price extremes).",
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window_tenkan`` "
            "non-null values) — each line nulls only where its own window touches the ``null``, in "
            "either ``high`` or ``low``.",
        ),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there — per line, via the "
            "underlying rolling extremes, with ``null`` still taking precedence over ``NaN``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with four ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``tenkan`` — the conversion line; ``null`` through its ``window_tenkan - 1``-row "
    "warm-up. - ``kijun`` — the base line; ``null`` through ``window_kijun - 1`` rows. - "
    "``senkou_a`` — the cloud's first bound; ``null`` through ``window_kijun - 1`` rows (it "
    "needs both lines). - ``senkou_b`` — the cloud's second bound; ``null`` through "
    "``window_senkou - 1`` rows."
    "\n\n"
    'Read a field with ``.struct.field("tenkan")`` or split all four with '
    "``.struct.unnest()``.",
    raises_prose="ValueError: If any window is ``< 1``, or the windows are not ordered ``window_tenkan <= "
    "window_kijun <= window_senkou`` (equality is allowed and collapses the corresponding "
    "lines onto each other).",
    args_prose={
        "window_tenkan": "Conversion-line window (canonically ``9``). Must be ``>= 1``.",
        "window_kijun": "Base-line window (canonically ``26``). Must be ``>= 1`` and ``>= window_tenkan``.",
        "window_senkou": "Leading-span-B window (canonically ``52``). Must be ``>= 1`` and ``>= window_kijun``.",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0),
                "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0),
            },
            params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
            round_to=4,
            fields=("tenkan", "senkou_b"),
        ),
        Example(
            inputs={
                "high": (10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0),
                "low": (8.0, 9.0, 10.0, 11.0, 12.0, 18.0, 19.0, 20.0, 21.0, 22.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's lines warm up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
            round_to=4,
            fields=("kijun",),
        ),
        Example(
            inputs={
                "high": (10.0, 12.0, None, 13.0, float("nan"), 12.0, 15.0),
                "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0),
            },
            intro="A ``null`` (any line whose window touches it is ``null``) and a ``NaN`` (which "
            "propagates) make it visible:",
            params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
            round_to=4,
            fields=("tenkan",),
        ),
    ),
)
