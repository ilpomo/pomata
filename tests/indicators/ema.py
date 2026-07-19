"""Declaration for ``pomata.indicators.ema`` — the recursive exponential mean, gap-bridging, NaN-latching, degree-1."""

import math

from pomata.indicators import ema
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Seeding, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_ema
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

EMA = suite_indicators(
    factory=ema,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    seeding=Seeding.SMA_SEED,
    oracle=reference_ema,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.0, 6.0, 8.0)),
    pins=(
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, 2.0, 3.333333333333333),
            params_override={"window": 2},
            reason="an interior null yields null at that row while the recursion bridges the gap",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN latches into the recursion and poisons every subsequent value",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 (alpha=1) reproduces the input with no warm-up",
        ),
        Pin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 0.0, 0.0),
            reason="an all-zero series is the exact fixed point of the recurrence: every blend of zeros is 0.0 "
            "bit-exact, with no rounding residue after warm-up",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 4.666666666666667, 7.333333333333334, 9.666666666666668),
            reason="the gap-aware recurrence over an interior null, hand-anchored against the reference",
        ),
        Pin(
            label="interior_null_after_seed_bridged",
            inputs={"expr": (2.0, 4.0, 6.0, None, 8.0, 10.0)},
            expected=(None, None, 4.0, None, 6.666666666666667, 8.333333333333334),
            reason="a null strictly after the seed: the recursion carries its state across the gap with the documented "
            "(1 - alpha) ** k decay",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
            expected=(None, None, 4.857142857142857, 6.533333333333333, 8.32258064516129),
            params_override={"adjust": True},
            reason="the frozen golden under adjust=True (the finite-window unbiased weighting), the second EMA-mode "
            "branch a single canonical golden cannot carry",
        ),
    ),
    reference='Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving Averages." '
    "*Technometrics*, 1(3), 239-250.",
    doi="https://doi.org/10.1080/00401706.1959.10489860",
    wikipedia="https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average",
    see_also=(
        ("rma", "Wilder's variant, with smoothing factor ``1 / window``."),
        ("dema", "A lag-reduced average built from two chained EMAs."),
        ("sma", "The equal-weight simple average this is the exponential analog of."),
    ),
    notes=(
        (
            "Seeding",
            "The unadjusted recursion (the default) is seeded with the simple average of the first "
            "``window`` observations, the canonical EMA initialization; the adjusted form is exact "
            "from the first observation.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap "
            "(a leading run consumes no warm-up budget, and an interior gap decays the carried weight "
            "by ``(1 - alpha) ** k`` per Polars' ``ignore_nulls=False`` convention).",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position (a ``NaN`` still inside the warm-up shows as that warm-up's ``null`` "
            "on its own row, then latches from the first emitted row).",
        ),
        ("window == 1", "the smoothing factor is ``1``, so the EMA reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The EMA for each row, the same length as ``expr``. The first ``window - 1`` values are "
    "``null`` (warm-up), matching the uniform warm-up of the moving-average family: the value "
    "is defined only once ``window`` non-null observations have been seen.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.",
        "adjust": "When ``False`` (default) use the recursive form above. When ``True`` use the "
        "finite-window bias-corrected (adjusted) weighting that divides by the decaying sum of "
        "weights at each step. The two forms differ at every row in general (coinciding only for "
        "``window == 1`` or a constant series), the gap largest near the start of the series and "
        "decaying geometrically as the history grows.",
    },
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0)},
            intro="A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a "
            "``NaN`` (which latches) make the exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (1.0, 2.0, 3.0)},
            intro="**window == 1** — the smoothing factor ``alpha=1`` reproduces the input exactly, with no warm-up:",
            params={"window": 1},
        ),
    ),
)
