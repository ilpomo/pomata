"""Spec for ``pomata.indicators.supertrend`` — the ATR-banded trend struct (line, direction), gap-bridging."""

import math

from tests.indicators.oracles import supertrend_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import supertrend

_GOLDEN_HIGH = (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0)
_GOLDEN_LOW = (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0)
_GOLDEN_CLOSE = (9.5, 10.5, 11.5, 12.5, 13.5, 12.0, 11.0, 10.2)

SUPERTREND = Spec(
    factory=supertrend,
    inputs=("high", "low", "close"),
    params={"window": 10, "multiplier": 3.0},
    shape=Shape.STRUCT,
    fields=("line", "direction"),
    warmup={"line": 9, "direction": 9},
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    oracle=supertrend_reference,
    # line is a degree-1 price level, direction a degree-0 invariant flag: the per-field degrees state the split
    # homogeneity claim the old suite tested (tests/indicators/test_supertrend.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree={"line": 1, "direction": 0}),),
    golden_params={"window": 3, "multiplier": 2.0},
    golden_input={"high": _GOLDEN_HIGH, "low": _GOLDEN_LOW, "close": _GOLDEN_CLOSE},
    golden_output={
        "line": (None, None, 8.8333, 9.7222, 10.6481, 10.6481, 10.6481, 12.9005),
        "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
    },
    pins=(
        SpecPin(
            label="flat_series_zero_atr_collapses_to_midpoint",
            inputs={
                "high": (5.0, 5.0, 5.0, 5.0, 5.0),
                "low": (5.0, 5.0, 5.0, 5.0, 5.0),
                "close": (5.0, 5.0, 5.0, 5.0, 5.0),
            },
            params_override={"window": 2},
            expected={
                "line": (None, 5.0, 5.0, 5.0, 5.0),
                "direction": (None, -1.0, -1.0, -1.0, -1.0),
            },
            reason="a constant high==low==close run has zero ATR, so both bands collapse onto the midpoint and the "
            "line tracks it with direction -1 forever (test_supertrend.py::test_flat_series)",
        ),
        SpecPin(
            label="single_row_window_one_seeds_long",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 1},
            expected={"line": (3.0,), "direction": (1.0,)},
            reason="window=1 defines the bar (ATR is the true range); close > lower seeds the trend long and the line "
            "reads the lower band (test_supertrend.py::test_single_row)",
        ),
        SpecPin(
            label="lower_band_exact_touch_stays_up",
            inputs={"high": (11.0, 13.0, 12.5), "low": (9.0, 11.0, 10.5), "close": (10.5, 12.0, 11.375)},
            params_override={"window": 1, "multiplier": 0.25},
            expected={"line": (9.5, 11.375, 11.375), "direction": (1.0, 1.0, 1.0)},
            reason="a flip requires a strict break, so a close exactly on the carried band holds the trend "
            "(test_supertrend.py::test_lower_band_exact_touch_stays_up)",
        ),
        SpecPin(
            label="downtrend_seed_nondefault_multiplier_golden",
            inputs={
                "high": (20.0, 19.0, 18.0, 17.0, 18.0, 19.0, 20.0, 21.0),
                "low": (19.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 20.0),
                "close": (19.2, 18.2, 17.2, 16.2, 17.8, 18.8, 19.8, 20.8),
            },
            params_override={"window": 2, "multiplier": 1.0},
            expected={
                "line": (None, 17.4, 18.65, 17.675, 16.0125, 17.15625, 18.228125000000002, 19.2640625),
                "direction": (None, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
            },
            reason="a second frozen reference exercising every branch (seed long, flip short, flip back long) at a "
            "non-default multiplier (test_supertrend.py::test_golden_master_downtrend_seed_nondefault_multiplier)",
        ),
        SpecPin(
            label="large_magnitude_micro_scale",
            inputs={
                "high": tuple(v * 1e-6 for v in _GOLDEN_HIGH),
                "low": tuple(v * 1e-6 for v in _GOLDEN_LOW),
                "close": tuple(v * 1e-6 for v in _GOLDEN_CLOSE),
            },
            params_override={"window": 3, "multiplier": 2.0},
            expected={
                "line": (
                    None,
                    None,
                    8.833333333333332e-06,
                    9.722222222222221e-06,
                    1.0648148148148146e-05,
                    1.0648148148148146e-05,
                    1.0648148148148146e-05,
                    1.2900548696844994e-05,
                ),
                "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
            },
            reason="the small-magnitude extreme of the old scale sweep: the golden scaled by 1e-6 stays exact "
            "(test_supertrend.py::test_matches_reference_at_large_magnitude)",
        ),
        SpecPin(
            label="large_magnitude_macro_scale",
            inputs={
                "high": tuple(v * 1e9 for v in _GOLDEN_HIGH),
                "low": tuple(v * 1e9 for v in _GOLDEN_LOW),
                "close": tuple(v * 1e9 for v in _GOLDEN_CLOSE),
            },
            params_override={"window": 3, "multiplier": 2.0},
            expected={
                "line": (
                    None,
                    None,
                    8833333333.333334,
                    9722222222.222221,
                    10648148148.148148,
                    10648148148.148148,
                    10648148148.148148,
                    12900548696.844994,
                ),
                "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
            },
            reason="the large-magnitude extreme of the old scale sweep, pinned against a regression to inf/NaN "
            "(test_supertrend.py::test_matches_reference_at_large_magnitude)",
        ),
    ),
)
