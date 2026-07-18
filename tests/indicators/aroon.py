"""
Declaration for ``pomata.indicators.aroon`` — the time-since-extreme struct (up, down), window-nulling, scale-
invariant.
"""

from pomata.indicators import aroon
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_aroon
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

AROON = suite_indicators(
    factory=aroon,
    inputs=("high", "low"),
    params={"window": 25},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("up", "down"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"up": 25, "down": 25},
    oracle=reference_aroon,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"up": 0, "down": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
            "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0),
        },
        output={
            "up": (None, None, None, 66.6667, 100.0, 66.6667, 100.0, 66.6667),
            "down": (None, None, None, 0.0, 66.6667, 33.3333, 0.0, 33.3333),
        },
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="current_extreme_reads_100",
            inputs={"high": (1.0, 2.0, 3.0), "low": (3.0, 2.0, 1.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 100.0), "down": (None, None, 100.0)},
            reason="when the current bar holds the look-back high (low) the up (down) line reads 100 ",
        ),
        Pin(
            label="ties_use_most_recent_extreme",
            inputs={"high": (5.0, 5.0, 3.0), "low": (1.0, 2.0, 3.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 50.0), "down": (None, None, 0.0)},
            reason="a repeated high resolves to the most recent occurrence (one bar back, up=50) ",
        ),
    ),
    reference='Chande, T. S. (1995). "The Time Price Oscillator." *Technical Analysis of Stocks & '
    "Commodities*, 13(9), 369-374.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon",
    see_also=(
        ("aroon_oscillator", "The difference ``up - down`` as a single line."),
        ("donchian_channels", "The rolling high/low extremes Aroon locates in time."),
        ("williams_r", "Another windowed high-low range oscillator."),
    ),
    notes=(
        (
            "Tie-break and seeding",
            "When the extreme is attained more than once in the look-back, the most recent occurrence "
            "is used, so the line reads higher.",
        ),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there (it is not treated as an extreme)."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``up`` — the Aroon Up line, in ``[0, 100]``. - ``down`` — the Aroon Down line, in "
    "``[0, 100]``."
    "\n\n"
    "Both are ``null`` for the first ``window`` rows (warm-up: a full ``window + 1``-bar "
    'look-back is needed). Access the fields with ``.struct.field("up")`` / ``"down"`` or '
    "``.struct.unnest()``.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Look-back length; the extreme is sought over the last ``window + 1`` bars. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low bars:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's channel warms up independently:",
    intro_missing="A ``null`` (which nulls the affected line) and a ``NaN`` (which propagates) in ``high`` "
    "make the handling visible on the ``up`` line:",
)
