"""
Declaration for ``pomata.indicators.standard_deviation_ewma`` — the EWM standard deviation, gap-bridging, degree-1.
"""

from pomata.indicators import standard_deviation_ewma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_standard_deviation_ewma
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

STANDARD_DEVIATION_EWMA = suite_indicators(
    factory=standard_deviation_ewma,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_standard_deviation_ewma,
    scaling=(ScaleAxis(roles=("price",), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib STDDEV is windowed; there is no exponentially-weighted standard deviation.",
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 1.299, 0.927, 1.2484, 0.8833, 1.1923),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="interior_null_reweights_observations",
            inputs={"price": (10.0, None, 11.0, 13.0, 12.0)},
            params_override={"window": 3},
            expected=(None, None, None, 1.2133516482134201, 0.8620067027323837),
            reason="an interior null ages the lag of 10 while contributing no term; at the last defined row the "
            "ignore_nulls=False weights reduce to 1:2:3:6, so the deviation is sqrt(107/144)",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0)},
            params_override={"window": 3, "adjust": True},
            expected=(None, None, 1.1952286093343936, 0.816496580927726, 1.1495825600777716),
            reason="the frozen golden under adjust=True (the finite-window unbiased weighting), the second EWM-mode "
            "branch a single canonical golden cannot carry — mirroring the ema family's adjusted pin",
        ),
        Pin(
            label="sample_deviation_bias_false",
            inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0)},
            params_override={"window": 3, "bias": False},
            expected=(None, None, 1.6431676725154984, 1.1443442705426587, 1.5320113653395042),
            reason="the debiased sample deviation (bias=False), the second correctness branch a single biased golden "
            "cannot carry — mirroring standard_deviation_rolling's ddof=1 pin",
        ),
    ),
    reference="J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document* (4th ed.). Cited for "
    "the concept of an exponentially-weighted variance in finance; pomata computes the "
    "mean-centered, span-parameterized form above, not RiskMetrics' zero-mean recursion with "
    "a ``lambda`` decay factor (``0.94`` daily).",
    see_also=(
        ("variance_ewma", "The square of this, of which it is the root."),
        ("standard_deviation_rolling", "The equal-weighted (rolling-window) counterpart."),
        (
            "ema",
            "The related exponential mean — note the deviations here are measured from Polars' native "
            "``ewm`` mean (seeded on the first observation), not from pomata's SMA-seeded "
            ":func:`ema`, so the two only converge past the warm-up.",
        ),
    ),
    note_extension="\n\nIt is homogeneous of degree ``1`` in ``expr`` (a spread in the input's own units).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap "
            "(a leading run consumes no warm-up, and an interior gap decays the carried weight across "
            "it, per Polars' ``ignore_nulls=False`` convention).",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The exponentially-weighted standard deviation for each row, the same length as the "
    "input. The first ``window - 1`` values are ``null`` (warm-up): the recursion emits only "
    "once ``window`` non-null observations have been seen.",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.",
        "adjust": "When ``False`` (default) use the recursive form; when ``True`` use the finite-window "
        "bias-corrected weighting (the same flag as :func:`ema`).",
        "bias": "When ``True`` (default) the population standard deviation; when ``False`` the sample "
        "one. ``True`` mirrors the ``ddof = 0`` default of :func:`standard_deviation_rolling`. "
        "See :func:`variance_ewma`.",
    },
    example_columns={"price": "x"},
    examples=(
        Example(inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"price": (10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0, 17.0)},
            intro="A ``null`` (decays across the gap) and a ``NaN`` (which propagates) make the handling visible:",
            params={"window": 3},
            round_to=4,
        ),
    ),
)
