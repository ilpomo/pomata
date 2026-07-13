"""Spec for ``pomata.indicators.awesome_oscillator`` — the SMA-of-median difference, window-nulling, degree-1."""

from tests_new.indicators.oracles import awesome_oscillator_reference
from tests_new.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import awesome_oscillator

AWESOME_OSCILLATOR = Spec(
    factory=awesome_oscillator,
    inputs=("high", "low"),
    params={"window_fast": 5, "window_slow": 34},
    shape=Shape.SERIES,
    warmup=33,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle=awesome_oscillator_reference,
    # A difference of two SMAs of the median price, homogeneous of degree 1 (tests/indicators/test_awesome_oscillator.py
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    # A one-pass rolling-mean difference against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    golden_params={"window_fast": 2, "window_slow": 3},
    golden_input={"high": (2.0, 4.0, 6.0, 8.0, 10.0), "low": (0.0, 2.0, 4.0, 6.0, 8.0)},
    golden_output=(None, None, 1.0, 1.0, 1.0),
    pins=(
        SpecPin(
            label="single_row_equal_windows",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 1},
            expected=(0.0,),
            reason="window_fast == window_slow == 1 on one bar gives 0 (test_awesome_oscillator.py::test_single_row)",
        ),
        SpecPin(
            label="single_row_warmup",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 3},
            expected=(None,),
            reason="a slow window of 3 on one bar is still warm-up (test_awesome_oscillator.py::test_single_row)",
        ),
        SpecPin(
            label="equal_windows_is_zero",
            inputs={"high": (2.0, 4.0, 6.0, 8.0), "low": (0.0, 2.0, 4.0, 6.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0, 0.0),
            reason="equal windows give an identically-zero oscillator where defined (test_awesome_oscillator.py"
            "::test_equal_windows_is_zero)",
        ),
        SpecPin(
            label="flat_series_is_zero",
            inputs={"high": (5.0, 5.0, 5.0, 5.0, 5.0), "low": (5.0, 5.0, 5.0, 5.0, 5.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, 0.0, 0.0, 0.0),
            reason="over a constant median both averages equal it, so AO is 0 (test_awesome_oscillator.py"
            "::test_flat_series_is_zero)",
        ),
    ),
)
