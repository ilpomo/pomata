"""
Declaration for ``pomata.indicators.midprice`` — the rolling high/low midprice of a bar series, window-nulling,
degree-1.
"""

from pomata.indicators import midprice
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_midprice
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

MIDPRICE = suite_indicators(
    factory=midprice,
    inputs=("high", "low"),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_midprice,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        },
        output=(None, None, 11.0, 11.5, 12.5),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_is_price_median",
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            params_override={"window": 1},
            expected=(10.0, 11.0, 12.0),
            reason="window=1 makes the extremes the bar's own high and low, so the midprice reduces to the per-bar "
            "price_median with no warm-up — the documented degenerate branch, mirroring midpoint's window=1 pin",
        ),
    ),
    reference="No canonical external source; the indicator is defined by the formula above.",
    see_also=(
        ("midpoint", "The same midpoint over a single series instead of a bar's high and low."),
        ("price_median", "The per-bar ``(high + low) / 2`` this collapses to at ``window == 1``."),
        ("donchian_channels", "The channel whose middle band is exactly this midprice."),
    ),
    notes=(("Inputs", "``high`` and ``low`` must share a length and alignment (the same row index is one bar)."),),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — in either ``high`` or ``low``.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "window == 1",
            "the extremes are the bar's own ``high`` and ``low``, so the midprice reduces to the "
            "per-bar :func:`price_median`.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The window midprice for each row, the same length as the inputs. The first ``window - "
    "1`` values are ``null`` (warm-up): the window must hold ``window`` non-null values "
    "before a result is emitted.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    examples=(
        Example(
            inputs={"high": (11.0, 12.0, 13.0, 12.5, 14.0), "low": (9.0, 10.0, 11.0, 11.0, 12.0)},
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"high": (11.0, 12.0, 13.0, 21.0, 22.0, 23.0), "low": (9.0, 10.0, 11.0, 19.0, 20.0, 21.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 3 + ("NVDA",) * 3,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"high": (11.0, None, 13.0, float("nan"), 15.0), "low": (9.0, 10.0, 11.0, 12.0, 13.0)},
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            intro="**window == 1** — a single-bar window makes the extremes the bar's own ``high`` and "
            "``low``, so the midprice reduces to the per-bar :func:`price_median` with no warm-up:",
            params={"window": 1},
        ),
    ),
)
