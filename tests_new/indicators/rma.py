"""
Declaration for ``pomata.indicators.rma`` — Wilder's recursive mean, gap-bridging, NaN-latching, degree-1 homogeneous.
"""

from pomata.indicators import rma
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Seeding, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_rma
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

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
)
