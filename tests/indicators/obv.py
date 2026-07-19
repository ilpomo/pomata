"""Declaration for ``pomata.indicators.obv`` — On-Balance Volume, the signed-volume running total, gap-bridging."""

from pomata.indicators import obv
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_obv
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

OBV = suite_indicators(
    factory=obv,
    inputs=("price", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_obv,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("price",), degree=0),
    ),
    talib=RelationTalib.DOCUMENTED_DIVERGENCE,
    talib_reason="OBV is a cumulative sum with an arbitrary origin; pomata seeds OBV[0] = 0, TA-Lib uses volume[0].",
    golden=Golden(
        inputs={
            "price": (10.0, 12.0, 11.0, 11.0, 13.0, 9.0, 9.0, 14.0),
            "volume": (100.0, 200.0, 150.0, 80.0, 300.0, 250.0, 90.0, 400.0),
        },
        output=(0.0, 200.0, 50.0, 50.0, 350.0, 100.0, 100.0, 500.0),
    ),
    pins=(
        Pin(
            label="flat_price_never_moves_the_total",
            inputs={"price": (5.0, 5.0, 5.0, 5.0), "volume": (10.0, 20.0, 30.0, 40.0)},
            expected=(0.0, 0.0, 0.0, 0.0),
            reason="an unchanged price contributes no signed volume, so the running total stays at the seed 0 ",
        ),
    ),
    reference="Granville, J. E. (1963). *Granville's New Key to Stock Market Profits*. Prentice-Hall.",
    wikipedia="https://en.wikipedia.org/wiki/On-balance_volume",
    see_also=(
        ("accumulation_distribution", "Another cumulative volume line."),
        ("money_flow_index", "A bounded volume-weighted oscillator."),
        ("chaikin_money_flow", "A windowed volume-weighted money-flow ratio."),
    ),
    notes=(
        (
            "Documented TA-Lib divergence",
            "TA-Lib seeds the running total with the first bar's ``volume``; pomata seeds ``OBV[0] = "
            "0``, the first bar having no predecessor to give it a direction, so the two lines sit a "
            "constant ``volume[0]`` apart at every bar and the differential tier holds OBV out as a "
            "documented divergence.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "a ``null`` ``close`` zeroes the direction at its own row and at the following row (each "
            "``diff`` touching it is filled to ``0``), while a ``null`` ``volume`` nulls that bar's "
            "contribution directly (``0 * null`` is ``null``, even on a flat or first bar).",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — a ``NaN`` ``volume`` contaminates the total even on a flat or first "
            "bar (``0 * NaN`` is ``NaN`` under IEEE-754), and a row whose own contribution is "
            "``null`` still emits ``null`` there even after the latch.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The OBV for each row, the same length as the inputs. There is no window and no warm-up "
    "-- every row is defined, starting at ``0`` on the first row.",
    args_prose={
        "expr": 'Input series, conventionally the close (any series is accepted; e.g. ``pl.col("close")``).',
    },
    example_columns={"price": "close"},
    examples=(
        Example(
            inputs={"price": (10.0, 12.0, 11.0, 11.0, 13.0, 9.0), "volume": (100.0, 200.0, 150.0, 80.0, 300.0, 250.0)},
            round_to=4,
        ),
        Example(
            inputs={
                "price": (11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={
                "price": (10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0),
                "volume": (100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0, 105.0, 115.0),
            },
            intro="A ``null`` (skipped, the running total carrying across it) and a ``NaN`` (which "
            "propagates) make the exact handling visible at a glance:",
            round_to=4,
        ),
    ),
)
