"""Declaration for ``pomata.indicators.tema`` — the triple EMA lag-correction, gap-bridging, NaN-latching, degree-1."""

import math

from pomata.indicators import tema
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_tema
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

TEMA = suite_indicators(
    factory=tema,
    inputs=("expr",),
    params={"window": 2},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=3,
    oracle=reference_tema,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)}, output=(None, None, None, 8.0, 10.0, 12.0)),
    pins=(
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0, 5.0)},
            expected=(None, None, None, None, 5.037037037037038),
            params_override={"window": 2},
            reason="the exact recovery value after an interior null bridges the cascade",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, None, None, math.nan),
            params_override={"window": 2},
            reason="a NaN latches into the cascade and poisons every value past the warm-up, mirroring the "
            "ema / dema / t3 siblings' pin",
        ),
        Pin(
            label="window_one_identity",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0)},
            expected=(1.0, 2.0, 3.0, 4.0),
            params_override={"window": 1},
            reason="window=1 collapses each of the three nested EMAs to the identity",
        ),
        Pin(
            label="all_zero_series",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero series stays exactly zero after warm-up",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (7.0, 7.0, 7.0, 7.0, 7.0, 7.0)},
            expected=(None, None, None, 7.0, 7.0, 7.0),
            params_override={"window": 2},
            reason="TEMA of a constant recovers exactly that constant after warm-up",
        ),
        Pin(
            label="window_three_golden",
            inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                3.296296296296297,
                5.399016203703705,
                5.081452546296297,
                3.234953703703704,
            ),
            params_override={"window": 3},
            reason="a second frozen golden master at window=3",
        ),
        Pin(
            label="golden_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 8.118158284023668, 10.090675959328463, 12.055955303250178),
            params_override={"window": 2, "adjust": True},
            reason="the frozen golden under adjust=True finite-window unbiased weighting",
        ),
    ),
    reference='Mulloy, P. G. (1994). "Smoothing Data with Faster Moving Averages." *Technical Analysis '
    "of Stocks & Commodities*, 12(1).",
    wikipedia="https://en.wikipedia.org/wiki/Triple_exponential_moving_average",
    see_also=(
        ("dema", "The double-EMA sibling."),
        ("t3", "The six-pass Tillson sibling."),
        ("ema", "The exponential pass this chains three times."),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        ("window == 1", "each EMA reduces to the identity, so the expression reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The TEMA for each row, the same length as ``expr``. The first ``3 * (window - 1)`` "
    "values are ``null`` (warm-up), clamped to the series length: the value is composed from "
    "three chained :func:`ema` passes of the same ``window`` (each carrying a ``window - 1`` "
    "warm-up), so the warm-up is three times that of a plain EMA. Under the default "
    "``adjust=False``, each pass is seeded with the SMA of the first ``window`` observations.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.",
        "adjust": "Whether to use the bias-corrected expanding-weights EMA. ``False`` (the default) selects "
        "the recursive technical-analysis EMA.",
    },
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)}, params={"window": 2}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
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
            inputs={"expr": (1.0, 2.0, 3.0, 4.0)},
            intro="**window == 1** — each of the three nested EMAs collapses to the identity, so the TEMA "
            "reproduces the input:",
            params={"window": 1},
        ),
    ),
)
