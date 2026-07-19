"""
Declaration for ``pomata.indicators.price_typical`` — the HLC mean, elementwise, propagating, degree-1 homogeneous.
"""

import math

from pomata.indicators import price_typical
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_price_typical
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

PRICE_TYPICAL = suite_indicators(
    factory=price_typical,
    inputs=("high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_typical,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.1667, 12.1667, 11.6667, 13.1667),
    ),
    pins=(
        Pin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in high combined with a NaN in low on the same row yields null — null wins over NaN",
        ),
    ),
    reference="Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.",
    wikipedia="https://en.wikipedia.org/wiki/Typical_price",
    see_also=(
        ("cci", "The Commodity Channel Index, built on the typical price."),
        ("price_average", "The equal-weighted mean of the four OHLC prices."),
        ("price_weighted_close", "The OHLC summary that double-weights the close."),
    ),
    notes=(
        (
            "Inputs",
            "``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that "
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
    returns_body="The typical price for each row, the same length as the inputs. There is no window and no "
    "warm-up -- every row is defined from row ``0``.",
    examples=(
        Example(
            inputs={
                "high": (11.0, 12.0, 13.0, 12.5, 14.0),
                "low": (9.0, 10.0, 11.0, 11.0, 12.0),
                "close": (10.0, 11.5, 12.5, 11.5, 13.5),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "high": (11.0, 12.0, 13.0, 21.0, 22.0, 23.0),
                "low": (9.0, 10.0, 11.0, 19.0, 20.0, 21.0),
                "close": (10.0, 11.5, 12.5, 20.0, 21.5, 22.5),
            },
            intro="On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — "
            "for this elementwise transform ``.over`` is optional (the result is identical without "
            "it) and shown here only for consistency:",
            partition=("A", "A", "A", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={
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
