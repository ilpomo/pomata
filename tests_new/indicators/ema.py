"""Declaration for ``pomata.indicators.ema`` — the recursive exponential mean, gap-bridging, NaN-latching, degree-1."""

import math

from pomata.indicators import ema
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Seeding, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_ema
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

EMA = suite_indicators(
    factory=ema,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    seeding=Seeding.SMA_SEED,
    oracle=reference_ema,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.0, 6.0, 8.0)),
    pins=(
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, 2.0, 3.333333333333333),
            params_override={"window": 2},
            reason="an interior null yields null at that row while the recursion bridges the gap",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN latches into the recursion and poisons every subsequent value",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 (alpha=1) reproduces the input with no warm-up",
        ),
        Pin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 0.0, 0.0),
            reason="an all-zero series is the exact fixed point of the recurrence: every blend of zeros is 0.0 "
            "bit-exact, with no rounding residue after warm-up",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 4.666666666666667, 7.333333333333334, 9.666666666666668),
            reason="the gap-aware recurrence over an interior null, hand-anchored against the reference",
        ),
        Pin(
            label="interior_null_after_seed_bridged",
            inputs={"expr": (2.0, 4.0, 6.0, None, 8.0, 10.0)},
            expected=(None, None, 4.0, None, 6.666666666666667, 8.333333333333334),
            reason="a null strictly after the seed: the recursion carries its state across the gap with the documented "
            "(1 - alpha) ** k decay",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
            expected=(None, None, 4.857142857142857, 6.533333333333333, 8.32258064516129),
            params_override={"adjust": True},
            reason="the frozen golden under adjust=True (the finite-window unbiased weighting), the second EMA-mode "
            "branch a single canonical golden cannot carry",
        ),
    ),
)
