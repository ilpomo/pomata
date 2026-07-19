"""
Declaration for ``pomata.indicators.price_average`` — the OHLC mean, elementwise, propagating, degree-1 homogeneous.
"""

import math

from pomata.indicators import price_average
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_price_average
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

PRICE_AVERAGE = suite_indicators(
    factory=price_average,
    inputs=("open", "high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_average,
    scaling=(ScaleAxis(roles=("open", "high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "open": (10.0, 11.0, 12.0, 11.5, 13.0),
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.125, 12.125, 11.625, 13.125),
    ),
    pins=(
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (11.0, math.nan), "low": (9.0, 10.0), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in open and a NaN in high on the same row yields null — null wins over NaN",
        ),
    ),
    reference="No canonical external source; the indicator is defined by the formula above.",
    see_also=(
        ("price_median", "The midpoint of the bar's range, ``(high + low) / 2``."),
        ("price_typical", "The equal-weighted mean of high, low, and close."),
        ("price_weighted_close", "The OHLC summary that double-weights the close."),
    ),
    notes=(
        (
            "Inputs",
            "``open``, ``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that "
            "positional order and must share a length and alignment (the same row index is one bar).",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` price yields ``NaN`` for that row."),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The average price for each row, the same length as the inputs. There is no window and no "
    "warm-up -- every row is defined from row ``0``.",
    examples=(
        Example(
            inputs={
                "open": (10.0, 11.0, 12.0, 11.5, 13.0),
                "high": (11.0, 12.0, 13.0, 12.5, 14.0),
                "low": (9.0, 10.0, 11.0, 11.0, 12.0),
                "close": (10.0, 11.5, 12.5, 11.5, 13.5),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "open": (10.0, 11.0, 12.0, 20.0, 21.0, 22.0),
                "high": (11.0, 12.0, 13.0, 21.0, 22.0, 23.0),
                "low": (9.0, 10.0, 11.0, 19.0, 20.0, 21.0),
                "close": (10.0, 11.5, 12.5, 20.0, 21.5, 22.5),
            },
            intro="On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — "
            "for this elementwise transform ``.over`` is optional (the result is identical without "
            "it) and shown here only for consistency:",
            partition=("AAPL",) * 3 + ("NVDA",) * 3,
            round_to=4,
        ),
        Example(
            inputs={
                "open": (10.0, 11.0, 12.0, 13.0, 14.0),
                "high": (11.0, 12.0, 13.0, 14.0, 15.0),
                "low": (9.0, 10.0, 11.0, 12.0, 13.0),
                "close": (10.0, None, 12.5, float("nan"), 14.5),
            },
            intro="A ``null`` then a ``NaN`` in ``close`` (both propagate through the sum) make the "
            "missing-data handling visible at a glance:",
            round_to=4,
        ),
    ),
)
