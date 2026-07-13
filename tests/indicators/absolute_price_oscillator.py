"""Spec for ``pomata.indicators.absolute_price_oscillator`` — the EMA-difference oscillator, gap-bridging, degree-1."""

from tests.indicators.oracles import absolute_price_oscillator_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import absolute_price_oscillator

ABSOLUTE_PRICE_OSCILLATOR = Spec(
    factory=absolute_price_oscillator,
    inputs=("expr",),
    params={"window_fast": 2, "window_slow": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle=absolute_price_oscillator_reference,
    # A difference of two EMAs, each linear in the price (tests/indicators/test_absolute_price_oscillator.py
    # ::TestAbsolutePriceOscillatorProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    # A one-pass EMA difference against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    golden_input={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848),
    pins=(
        SpecPin(
            label="equal_windows_are_zero",
            inputs={"expr": (10.0, 11.0, 12.0)},
            expected=(None, 0.0, 0.0),
            params_override={"window_fast": 2, "window_slow": 2},
            reason="equal fast/slow windows make the two EMAs identical so the oscillator cancels to exactly 0.0 "
            "(test_absolute_price_oscillator.py::test_equal_windows_are_zero)",
        ),
    ),
)
