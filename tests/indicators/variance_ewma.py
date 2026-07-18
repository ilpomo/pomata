"""Declaration for ``pomata.indicators.variance_ewma`` — the EWM variance, gap-bridging, NaN-latching, degree-2."""

from pomata.indicators import variance_ewma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_variance_ewma
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

VARIANCE_EWMA = suite_indicators(
    factory=variance_ewma,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_variance_ewma,
    scaling=(ScaleAxis(roles=("price",), degree=2),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib VAR is windowed; there is no exponentially-weighted variance.",
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 1.6875, 0.8594, 1.5586, 0.7803, 1.4216),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="interior_null_reweights_observations",
            inputs={"price": (10.0, None, 11.0, 13.0, 12.0)},
            params_override={"window": 3},
            expected=(None, None, None, 1.4722222222222232, 0.7430555555555561),
            reason="an interior null ages the lag of 10 while contributing no term; at the last defined row the "
            "ignore_nulls=False weights reduce to 1:2:3:6, giving variance 107/144",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0)},
            params_override={"window": 3, "adjust": True},
            expected=(None, None, 1.4285714285714286, 0.6666666666666666, 1.3215400624349636),
            reason="the frozen golden under adjust=True (the finite-window unbiased weighting), the second EWM-mode "
            "branch a single canonical golden cannot carry — mirroring the ema family's adjusted pin",
        ),
        Pin(
            label="sample_variance_bias_false",
            inputs={"price": (10.0, 11.0, 13.0, 12.0, 14.0)},
            params_override={"window": 3, "bias": False},
            expected=(None, None, 2.7, 1.3095238095238095, 2.347058823529412),
            reason="the debiased sample variance (bias=False), the second correctness branch a single biased golden "
            "cannot carry — mirroring variance_rolling's ddof=1 pin",
        ),
    ),
    reference="J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document* (4th ed.). Cited for "
    "the concept of an exponentially-weighted variance in finance; pomata computes the "
    "mean-centered, span-parameterized form above, not RiskMetrics' zero-mean recursion with "
    "a ``lambda`` decay factor (``0.94`` daily).",
    see_also=(
        ("standard_deviation_ewma", "Its square root, in the input's own units."),
        ("variance_rolling", "The equal-weighted (rolling-window) counterpart."),
        (
            "ema",
            "The related exponential mean — note the deviations here are measured from Polars' native "
            "``ewm`` mean (seeded on the first observation), not from pomata's SMA-seeded "
            ":func:`ema`, so the two only converge past the warm-up.",
        ),
    ),
    note_extension="\n\n"
    "``window`` must be ``>= 2``: a single observation yields a well-defined ``0`` under the "
    "default ``bias=True``, but divides by zero under the unbiased ``bias=False`` correction, "
    "so a minimum of ``2`` is enforced uniformly across both paths. It is homogeneous of "
    "degree ``2`` in ``expr`` (a variance scales with the square of the input).",
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
    returns_body="The exponentially-weighted variance for each row, the same length as the input. The "
    "first ``window - 1`` values are ``null`` (warm-up): the recursion emits only once "
    "``window`` non-null observations have been seen.",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.",
        "adjust": "When ``False`` (default) use the recursive form; when ``True`` use the finite-window "
        "bias-corrected weighting (the same flag as :func:`ema`).",
        "bias": "When ``True`` (default) the population variance (divides by the weight total); when "
        "``False`` the unbiased sample variance (the reliability correction ``1 - sum(w ** 2) / "
        "(sum w) ** 2``). ``True`` mirrors the ``ddof = 0`` default of :func:`variance_rolling`.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (decays across the gap) and a ``NaN`` (which propagates) make the handling visible:",
)
