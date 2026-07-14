"""Spec for ``pomata.indicators.percentage_price_oscillator`` — the normalized EMA-difference, gap-bridging."""

import math

from tests.indicators.oracles import percentage_price_oscillator_reference
from tests.support import ABSOLUTE_TOLERANCE_ROLLING_ORACLE, RELATIVE_TOLERANCE_ROLLING_ORACLE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import percentage_price_oscillator

PERCENTAGE_PRICE_OSCILLATOR = Spec(
    factory=percentage_price_oscillator,
    inputs=("price",),
    params={"window_fast": 12, "window_slow": 26},
    shape=Shape.SERIES,
    warmup=25,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle=percentage_price_oscillator_reference,
    # A one-pass EMA ratio against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_ROLLING_ORACLE,
    # PPO normalizes the EMA difference by the slow EMA, so it is scale-INVARIANT, degree 0
    #
    scale=(ScaleAxis(roles=("price",), degree=0),),
    golden_params={"window_fast": 2, "window_slow": 3},
    golden_input={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 4.5455, 1.5152, 3.2407, 3.5613, 1.1871, 2.7484),
    pins=(
        SpecPin(
            label="equal_windows_are_zero",
            inputs={"price": (10.0, 11.0, 12.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0),
            reason="equal fast/slow windows produce identical EMAs so the oscillator cancels to exactly 0 ",
        ),
        SpecPin(
            label="zero_slow_ema_is_nan",
            inputs={"price": (0.0, 0.0, 0.0, 0.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, math.nan, math.nan),
            reason="an all-zero series drives both EMAs to exactly 0.0, so the 0/0 boundary surfaces as NaN ",
        ),
    ),
)
