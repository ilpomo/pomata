"""Declaration for ``pomata.indicators.variance_ewma`` — the EWM variance, gap-bridging, NaN-latching, degree-2."""

from pomata.indicators import variance_ewma
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_variance_ewma
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

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
)
