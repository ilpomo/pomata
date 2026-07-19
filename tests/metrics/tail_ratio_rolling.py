"""Declaration for ``pomata.metrics.tail_ratio_rolling`` — the rolling right-tail over left-tail quantile ratio."""

import math

from pomata.metrics import tail_ratio_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_tail_ratio_rolling
from tests.metrics.tail_ratio import TAIL_RATIO
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

TAIL_RATIO_ROLLING = suite_metrics(
    factory=tail_ratio_rolling,
    inputs=("returns",),
    params={"window": 5},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=TAIL_RATIO,
    window="window",
    warmup=4,
    oracle=reference_tail_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, None, None, 1.5556, 1.5556, 2.0),
    ),
    pins=(
        Pin(
            label="zero_left_tail_window_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(None, None, None, None, math.inf),
            reason="a window with a zero 5th-percentile and a non-zero 95th gives +inf ",
            params_override={"window": 5},
        ),
        Pin(
            label="all_zero_window_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None, None, math.nan),
            reason="an all-zero window gives 0/0, so the ratio is NaN ",
            params_override={"window": 3},
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Tail_risk",
    see_also=(
        ("tail_ratio", "The whole-series reducing form."),
        ("value_at_risk_rolling", "Another rolling tail-risk measure."),
        ("skewness_rolling", "The rolling moment-based companion measure of distributional asymmetry."),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`tail_ratio` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "when a window's 5th-percentile return is exactly ``0`` against a non-zero 95th the ratio "
            "is ``+inf`` (or ``NaN`` when the 95th is also ``0``) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling tail ratio for each row, the same length as the input. The first ``window - "
    "1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null values before "
    "a result is emitted.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    examples=(
        Example(inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)}, params={"window": 5}, round_to=4),
        Example(
            inputs={
                "returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.02, -0.01, 0.04, -0.03, 0.01, 0.025, -0.02)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own "
            "(the ``NVDA`` group never borrows ``AAPL``'s tail):",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            params={"window": 5},
            round_to=4,
        ),
        Example(
            inputs={"returns": (None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.005)},
            intro="A leading ``null`` and a later ``NaN`` show the per-window masking, with the result "
            "recovering once both leave the window:",
            params={"window": 5},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            intro="**Degenerate denominator** — a window with a zero 5th-percentile and a non-zero 95th "
            "gives ``+inf``:",
            params={"window": 5},
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero window gives ``0/0``, so the ratio is ``NaN``:",
            params={"window": 3},
        ),
    ),
)
