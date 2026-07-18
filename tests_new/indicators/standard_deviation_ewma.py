"""
Declaration for ``pomata.indicators.standard_deviation_ewma`` — the EWM standard deviation, gap-bridging, degree-1.
"""

from pomata.indicators import standard_deviation_ewma
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_standard_deviation_ewma
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

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
)
