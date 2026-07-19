"""
Declaration for ``pomata.indicators.rma`` — Wilder's recursive mean, gap-bridging, NaN-latching, degree-1 homogeneous.
"""

from pomata.indicators import rma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Seeding, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_rma
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

RMA = suite_indicators(
    factory=rma,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    seeding=Seeding.RMA_SEED,
    oracle=reference_rma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Wilder smoothing (SMMA) as a standalone function.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.0, 5.3333, 6.8889), params={"window": 3}
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 2.0),
            params_override={"window": 3},
            reason="window equal to the series length emits exactly one defined value on the last row",
        ),
        Pin(
            label="window_one_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 (alpha=1) reproduces the input with zero warm-up",
        ),
        Pin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero recursion stays exactly at zero",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, 5.0, 5.0),
            params_override={"window": 3},
            reason="a constant input yields that same constant at every defined row",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 4.666666666666667, 6.444444444444445, 8.296296296296298),
            params_override={"window": 3},
            reason="the BRIDGED gap-decay renormalization straddling the seed row itself",
        ),
        Pin(
            label="interior_null_after_seed_bridged",
            inputs={"expr": (2.0, 4.0, 6.0, None, 8.0, 10.0)},
            expected=(None, None, 4.0, None, 5.7142857142857135, 7.142857142857142),
            params_override={"window": 3},
            reason="the post-seed gap-decay branch, pinned deterministically against the reference",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    see_also=(
        ("ema", "The same recursion with smoothing factor ``2 / (window + 1)``."),
        ("atr", "The volatility average that smooths the true range with this Wilder mean."),
        ("sma", "The equal-weight baseline."),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap "
            "(a leading run consumes no warm-up budget, and an interior gap decays the carried weight "
            "by ``(1 - alpha) ** k``, emulating ``ewm_mean(adjust=False, ignore_nulls=False)`` "
            "semantics).",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        ("window == 1", "the smoothing factor is ``1``, the warm-up vanishes, and the result reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The RMA for each row, the same length as ``expr``. The first ``window - 1`` values are "
    "``null`` (warm-up) -- the recursion emits only once ``window`` non-null observations "
    "have been counted -- seeded there with their simple average -- after which every later "
    "row is defined wherever its own input is (an interior ``null`` still voids its own row, "
    "as the Note details).",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, params={"window": 3}, round_to=4),
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
            inputs={"expr": (1.0, 2.0, 3.0)},
            intro="**window == 1** — the smoothing factor ``alpha=1`` reproduces the input exactly, with zero warm-up:",
            params={"window": 1},
        ),
    ),
)
