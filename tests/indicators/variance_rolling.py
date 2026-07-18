"""
Declaration for ``pomata.indicators.variance_rolling`` — the rolling variance, window-nulling, degree-2 homogeneous.
"""

from pomata.indicators import variance_rolling
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_variance_rolling
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

VARIANCE_ROLLING = suite_indicators(
    factory=variance_rolling,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_variance_rolling,
    scaling=(ScaleAxis(roles=("price",), degree=2),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(inputs={"price": (2.0, 4.0, 4.0, 8.0)}, output=(None, 1.0, 0.0, 4.0), params={"window": 2}),
    pins=(
        Pin(
            label="sample_variance_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 4.0),
            reason="the sample variance (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry",
        ),
        Pin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 222222177777.78003, 0.0, 0.0),
            reason="a constant window has exactly zero dispersion even after a much larger value has left it, where "
            "an incremental rolling variance would leave a residue",
        ),
        Pin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the variance is 0 at every row with no warm-up",
        ),
    ),
)
