"""
Declaration for ``pomata.indicators.balance_of_power`` — the bar close-open ratio, elementwise, propagating,
invariant.
"""

import math

from pomata.indicators import balance_of_power
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_balance_of_power
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

BALANCE_OF_POWER = suite_indicators(
    factory=balance_of_power,
    inputs=("open", "high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_balance_of_power,
    scaling=(ScaleAxis(roles=("open", "high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "open": (10.0, 10.0, 10.0),
            "high": (12.0, 12.0, 12.0),
            "low": (8.0, 8.0, 8.0),
            "close": (11.0, 10.0, 9.0),
        },
        output=(0.25, 0.0, -0.25),
    ),
    pins=(
        Pin(
            label="null_close_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, 12.0), "low": (8.0, 8.0), "close": (11.0, None)},
            expected=(0.25, None),
            reason="a null in one input yields null for that row on a non-flat bar; the shared flow rung nulls "
            "every role at once and cannot isolate one",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, None),
            reason="a row carrying both a null (open) and a NaN (high) yields null — null wins",
        ),
        Pin(
            label="nan_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, math.nan),
            reason="a NaN in one input propagates to NaN for that row",
        ),
        Pin(
            label="flat_bar_is_zero",
            inputs={"open": (10.0, 12.0), "high": (11.0, 12.0), "low": (9.0, 12.0), "close": (10.5, 11.0)},
            expected=(0.25, 0.0),
            reason="a flat bar (high == low, exact zero range) yields 0 by convention, over the bare 0/0",
        ),
    ),
    reference='Livshin, I. (2001). "Using the Balance of Power Indicator." *Technical Analysis of '
    "Stocks & Commodities*.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/balance-of-power-bop",
    see_also=(
        ("price_average", "Another per-bar OHLC summary, the equal-weighted mean of the four prices."),
        ("price_weighted_close", "A per-bar OHLC summary that leans on the close."),
        ("price_typical", "The per-bar high-low-close average."),
    ),
    notes=(
        (
            "Inputs",
            "``open``, ``high``, ``low``, and ``close`` are the canonical OHLC roles in that "
            "positional order and must share a length and alignment (the same row index is one bar). "
            "``balance_of_power`` is scale-invariant: multiplying all four by a common factor leaves "
            "it unchanged.",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` price yields ``NaN`` for that row."),
        (
            "Degenerate denominator",
            "when ``high == low`` the range is zero — the ``0 / 0`` degenerate — but the result is "
            "``0`` by convention (no range, no directional power); the zero-range branch fires first, "
            "so a finite flat bar reads ``0`` even when ``open`` or ``close`` is ``null``, and only a "
            "``null`` ``high`` or ``low`` (which leaves the range itself ``null``) still yields "
            "``null`` on a flat bar.",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The balance of power for each row, the same length as the inputs. There is no window and "
    "no warm-up -- every row is defined from row ``0``.",
    intro_basic="Basic usage on a small OHLC frame:",
    examples=(
        Example(
            inputs={
                "open": (10.0, 11.0, 12.0, 11.0),
                "high": (11.0, 13.0, 12.0, 13.0),
                "low": (9.0, 10.0, 11.0, 10.0),
                "close": (10.5, 12.0, 11.5, 12.0),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "open": (10.0, 11.0, 12.0, 11.0, 20.0, 21.0, 22.0, 21.0),
                "high": (11.0, 13.0, 12.0, 13.0, 21.0, 23.0, 22.0, 23.0),
                "low": (9.0, 10.0, 11.0, 10.0, 19.0, 20.0, 21.0, 20.0),
                "close": (10.5, 12.0, 11.5, 12.0, 20.5, 22.0, 21.5, 22.0),
            },
            intro="Balance of Power is elementwise, so ``.over`` is optional; each ticker yields the same "
            "per-bar reading:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={
                "open": (10.0, 12.0, 11.0, 12.0, 12.0),
                "high": (11.0, 12.0, 13.0, 14.0, 13.0),
                "low": (9.0, 12.0, 11.0, 12.0, 11.0),
                "close": (10.5, 12.0, None, 13.0, float("nan")),
            },
            intro="A flat bar (``high == low``, giving ``0``), then a ``null`` and a ``NaN`` in ``close`` "
            "make the edge handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"open": (10.0, 12.0), "high": (11.0, 12.0), "low": (9.0, 12.0), "close": (10.5, 11.0)},
            intro="**Degenerate denominator** — a flat bar with ``high`` equal to ``low`` has an exact zero "
            "range, the bare ``0/0``, but returns ``0`` by convention:",
        ),
    ),
)
