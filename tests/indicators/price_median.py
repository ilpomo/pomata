"""Declaration for ``pomata.indicators.price_median`` — the high-low midpoint, elementwise, propagating, degree-1."""

import math

from pomata.indicators import price_median
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_price_median
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

PRICE_MEDIAN = suite_indicators(
    factory=price_median,
    inputs=("high", "low"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_median,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"high": (11.0, 12.0, 13.0, 12.5, 14.0), "low": (9.0, 10.0, 11.0, 11.0, 12.0)},
        output=(10.0, 11.0, 12.0, 11.75, 13.0),
    ),
    pins=(
        Pin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan)},
            expected=(10.0, None),
            reason="a null in high against a NaN in low on the same row yields null — null wins over NaN ",
        ),
    ),
    reference="Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/median-price",
    see_also=(
        ("midprice", "The rolling midpoint of the high-low range over a window."),
        ("price_average", "The equal-weighted mean of the four OHLC prices."),
        ("price_typical", "The equal-weighted mean of high, low, and close."),
    ),
    notes=(
        (
            "Inputs",
            "``high`` and ``low`` are taken as the canonical OHLC roles in that positional order and "
            "must share a length and alignment (the same row index is one bar).",
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
    returns_body="The median price for each row, the same length as the inputs. There is no window and no "
    "warm-up -- every row is defined from row ``0``.",
    intro_over="On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — "
    "for this elementwise transform ``.over`` is optional (the result is identical without "
    "it) and shown here only for consistency:",
    intro_missing="A ``null`` then a ``NaN`` in ``high`` (both propagate through the sum) make the "
    "missing-data handling visible at a glance:",
)
