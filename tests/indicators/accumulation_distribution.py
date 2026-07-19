"""
Declaration for ``pomata.indicators.accumulation_distribution`` — the running money-flow-volume total, gap-bridging.
"""

from pomata.indicators import accumulation_distribution
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_accumulation_distribution
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

ACCUMULATION_DISTRIBUTION = suite_indicators(
    factory=accumulation_distribution,
    inputs=("high", "low", "close", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_accumulation_distribution,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 13.0, 14.0),
            "low": (8.0, 9.0, 10.0, 11.0, 12.0),
            "close": (9.0, 10.5, 10.0, 13.0, 12.5),
            "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
        },
        output=(0.0, 100.0, -200.0, 200.0, -50.0),
    ),
    reference='Chaikin, M. "Accumulation/Distribution Line."',
    wikipedia="https://en.wikipedia.org/wiki/Accumulation/distribution_index",
    see_also=(
        ("accumulation_distribution_oscillator", "The Chaikin oscillator — fast minus slow EMA of this line."),
        ("chaikin_money_flow", "The windowed money-flow ratio over the same multiplier."),
        ("obv", "Another cumulative volume-flow line."),
    ),
    notes=(
        (
            "Zero-range bars",
            "On a doji bar (``high == low``) the Money Flow Multiplier is ``0`` by convention, so the "
            "denominator never hits ``0 / 0`` and ``close`` does not enter the bar's contribution."
            "\n\n"
            "The zero-range convention applies only to a genuine equal-range bar (``high == low``), "
            "where the multiplier is ``0`` and ``close`` does not enter the contribution. A ``null`` "
            "or ``NaN`` in any input instead leaves the range ``null`` or ``NaN`` (never ``== 0``), "
            "so missing data propagates rather than being silently zeroed.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "on a genuine doji bar (``high == low``, both finite) the multiplier is ``0`` and "
            "``close`` is irrelevant, so a ``null`` in ``close`` on such a bar still yields ``0`` "
            "rather than ``null``.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — a bar whose ``high`` and ``low`` are both ``NaN`` does not take the "
            "doji branch (``NaN - NaN`` is ``NaN``, never ``== 0``), so the ``NaN`` poisons the line "
            "rather than contributing ``0``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The Accumulation/Distribution Line for each row, the same length as the inputs. There is "
    "no warm-up -- the first row already carries the first bar's Money Flow Volume, and the "
    "line is the running cumulative sum from there.",
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 13.0, 14.0),
                "low": (8.0, 9.0, 10.0, 11.0, 12.0),
                "close": (9.0, 10.5, 10.0, 13.0, 12.5),
                "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0),
                "low": (10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0),
                "close": (11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
                "close": (11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0),
            },
            intro="A ``null`` (skipped, the running total carrying across it) and a ``NaN`` (which "
            "propagates) make the exact handling visible at a glance:",
            round_to=4,
        ),
    ),
)
