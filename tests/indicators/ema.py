"""Spec for ``pomata.indicators.ema`` — the recursive exponential mean, gap-bridging, NaN-latching, degree-1."""

import math

from tests.indicators.oracles import ema_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import ema

EMA = Spec(
    factory=ema,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=ema_reference,
    # A linear recursive mean scales linearly with the series.
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
    golden_output=(None, None, 4.0, 6.0, 8.0),
    pins=(
        SpecPin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, 2.0, 3.333333333333333),
            params_override={"window": 2},
            reason="an interior null yields null at that row while the recursion bridges the gap",
        ),
        SpecPin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN latches into the recursion and poisons every subsequent value",
        ),
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 (alpha=1) reproduces the input with no warm-up",
        ),
        SpecPin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 0.0, 0.0),
            reason="the degenerate all-zero window stays exactly at zero; the case the subnormal-floor note keeps "
            "out of the property fuzz",
        ),
        SpecPin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 4.666666666666667, 7.333333333333334, 9.666666666666668),
            reason="the gap-aware recurrence over an interior null, hand-anchored against the reference",
        ),
        SpecPin(
            label="interior_null_after_seed_bridged",
            inputs={"expr": (2.0, 4.0, 6.0, None, 8.0, 10.0)},
            expected=(None, None, 4.0, None, 6.666666666666667, 8.333333333333334),
            reason="a null strictly after the seed: the recursion carries its state across the gap with the documented "
            "(1 - alpha) ** k decay",
        ),
        SpecPin(
            label="golden_master_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
            expected=(None, None, 4.857142857142857, 6.533333333333333, 8.32258064516129),
            params_override={"adjust": True},
            reason="the frozen golden under adjust=True (the finite-window unbiased weighting), the second EMA-mode "
            "branch a single canonical golden cannot carry",
        ),
    ),
)
