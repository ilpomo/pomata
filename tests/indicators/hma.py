"""
Declaration for ``pomata.indicators.hma`` — Hull's lag-reduced weighted mean, window-nulling, degree-1 homogeneous.
"""

from pomata.indicators import hma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_hma
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

HMA = suite_indicators(
    factory=hma,
    inputs=("expr",),
    params={"window": 4},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_hma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Hull moving average.",
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"window": 0}, r"window must be >= 2"),
    ),
    golden=Golden(
        inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
        output=(None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    ),
    pins=(
        Pin(
            label="golden_master_overshoot",
            inputs={"expr": (1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            expected=(None, None, None, None, 11.599999999999998, 11.5, 10.299999999999999, 10.0),
            reason="the lag correction 2*WMA(x,half) - WMA(x,window) over- and under-shoots the input range before the "
            "final smoothing settles",
        ),
        Pin(
            label="golden_master_round_half_up",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                5.666666666666666,
                6.666666666666665,
                7.666666666666665,
                8.666666666666664,
                9.666666666666664,
                10.666666666666664,
                11.666666666666664,
            ),
            params_override={"window": 5},
            reason="the round-half-up period reduction at window=5: half-period = floor(5/2 + 0.5) = 3, not the "
            "banker-rounded 2",
        ),
    ),
    reference='Hull, A. (2005). "Hull Moving Average."',
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma",
    see_also=(
        ("wma", "The weighted mean this composes."),
        ("sma", "The unweighted baseline."),
        ("dema", "A lag-reduced average built by the same doubling correction."),
    ),
    notes=(
        (
            "Period rounding",
            "The two period reductions use round-half-**up** (``floor(window / 2 + 0.5)`` and "
            "``floor(sqrt(window) + 0.5)``), not Python's built-in ``round`` (which rounds half to "
            "even). The two disagree on the half-period only for an odd ``window`` whose half "
            "``floor(window / 2)`` is even -- ``window`` congruent to ``1`` modulo ``4`` (``5``, "
            "``9``, ``13``, ...) -- where round-half-up takes the ``.5`` up while round-half-to-even "
            "takes it down to the even floor. For ``window`` congruent to ``3`` modulo ``4`` (``3``, "
            "``7``, ``11``, ...) the half still lands on a ``.5`` boundary but both round alike.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — propagated through every composing :func:`wma`.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The HMA for each row, the same length as ``expr``. The first ``window + s - 2`` values "
    "are ``null`` (warm-up), where :math:`s = \\lfloor \\sqrt{n} + \\tfrac{1}{2} \\rfloor`: the "
    "inner ``WMA(x, window)`` needs ``window`` observations, after which the final ``WMA(., "
    "s)`` needs ``s - 1`` more.",
    raises_prose="ValueError: If ``window < 2``. The half-period :math:`\\lfloor n / 2 + \\tfrac{1}{2} "
    "\\rfloor` collapses to ``1`` at ``window == 1`` and the HMA degenerates there, so the "
    "smallest meaningful window is ``2``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)}, params={"window": 4}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0)},
            intro="A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which "
            "propagates) make the exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
    ),
)
