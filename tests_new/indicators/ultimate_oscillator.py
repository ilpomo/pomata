"""Spec for ``pomata.indicators.ultimate_oscillator`` — Williams' three-window oscillator, window-nulling, invariant."""

import math

from tests_new.indicators.oracles import ultimate_oscillator_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import ultimate_oscillator

_HLC_HIGH = (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5)
_HLC_LOW = (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5)
_HLC_CLOSE = (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0)

ULTIMATE_OSCILLATOR = Spec(
    factory=ultimate_oscillator,
    inputs=("high", "low", "close"),
    params={"window_short": 7, "window_medium": 14, "window_long": 28},
    shape=Shape.SERIES,
    warmup=27,
    lands_on="close",
    raises=(
        ({"window_short": 0}, r"window_short must be >= 1"),
        ({"window_medium": 0}, r"window_medium must be >= 1"),
        ({"window_long": 0}, r"window_long must be >= 1"),
        (
            {"window_short": 14, "window_medium": 7},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
        (
            {"window_medium": 28, "window_long": 14},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
    ),
    oracle=ultimate_oscillator_reference,
    # A bounded ratio in [0, 100], scale-INVARIANT, degree 0 (tests/indicators/test_ultimate_oscillator.py
    # ::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window_short": 2, "window_medium": 3, "window_long": 4},
    golden_input={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
    golden_output=(None, None, None, 60.7143, 66.6667, 65.0433, 67.619, 65.4762, 67.619, 65.4762),
    pins=(
        SpecPin(
            label="window_all_one_equal_windows_accepted",
            inputs={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 1},
            expected=(
                50.0,
                66.66666666666667,
                66.66666666666667,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
            ),
            reason="equal windows are accepted (not raised) and the minimum window=1 is fully defined from row 0 "
            "(test_ultimate_oscillator.py::test_misordered_windows_raise, the accepted equal-window branch)",
        ),
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the 0/0 degenerate on a flat well-formed series, detected via residual-free rolling maxima "
            "(test_ultimate_oscillator.py::test_flat_window_is_nan)",
        ),
        SpecPin(
            label="flat_window_is_nan_at_large_magnitude",
            inputs={"high": (1e9, 1e9, 1e9), "low": (1e9, 1e9, 1e9), "close": (1e9, 1e9, 1e9)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the exact-flat guard is residual-free at scale, yielding NaN rather than a falsely-saturated value "
            "(test_ultimate_oscillator.py::test_flat_window_is_nan_at_large_magnitude)",
        ),
    ),
)
