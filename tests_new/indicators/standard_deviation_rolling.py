"""Declaration for ``pomata.indicators.standard_deviation_rolling`` — the rolling standard deviation, window-nulling."""

from pomata.indicators import standard_deviation_rolling
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_standard_deviation_rolling
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

STANDARD_DEVIATION_ROLLING = suite_indicators(
    factory=standard_deviation_rolling,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_standard_deviation_rolling,
    scaling=(ScaleAxis(roles=("price",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(inputs={"price": (2.0, 4.0, 4.0, 8.0)}, output=(None, 1.0, 0.0, 2.0), params={"window": 2}),
    pins=(
        Pin(
            label="sample_deviation_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 2.0),
            reason="the sample deviation (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry",
        ),
        Pin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the deviation is 0 at every row with no warm-up",
        ),
        Pin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 471404.47365057963, 0.0, 0.0),
            reason="a constant window has exactly zero spread even after a much larger value has left it, where an "
            "incremental rolling standard deviation would leave a residue",
        ),
    ),
)
