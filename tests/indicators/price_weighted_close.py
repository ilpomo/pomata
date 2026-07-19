"""Declaration for ``pomata.indicators.price_weighted_close`` — the close-weighted HLC mean, elementwise, degree-1."""

import math

from pomata.indicators import price_weighted_close
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_price_weighted_close
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

PRICE_WEIGHTED_CLOSE = suite_indicators(
    factory=price_weighted_close,
    inputs=("high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_weighted_close,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.25, 12.25, 11.625, 13.25),
    ),
    pins=(
        Pin(
            label="null_propagates",
            inputs={"high": (11.0, None, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, None, 12.25),
            reason="a null in exactly one input role nulls that row only ",
        ),
        Pin(
            label="nan_propagates",
            inputs={"high": (11.0, math.nan, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, math.nan, 12.25),
            reason="a NaN in exactly one input role nans that row only ",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a row carrying both a null (high) and a NaN (low) yields null — null wins over NaN ",
        ),
    ),
    reference="Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/weighted-close",
    see_also=(
        ("price_average", "The equal-weighted mean of the four OHLC prices."),
        ("price_median", "The midpoint of the bar's range, ``(high + low) / 2``."),
        ("price_typical", "The equal-weighted mean of high, low, and close."),
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
    returns_body="The weighted close price for each row, the same length as the inputs. There is no window "
    "and no warm-up -- every row is defined from row ``0``.",
    args_prose={
        "close": 'Close-price series (e.g. ``pl.col("close")``); weighted twice.',
    },
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
