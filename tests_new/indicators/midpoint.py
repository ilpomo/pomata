"""Spec for ``pomata.indicators.midpoint`` — the rolling high/low midpoint of a series, window-nulling, degree-1."""

from tests.indicators.oracles import midpoint_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import midpoint

MIDPOINT = Spec(
    factory=midpoint,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=midpoint_reference,
    # The mean of the window's max and min, homogeneous of degree 1 (tests/indicators/test_midpoint.py
    # ::TestMidpointProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
    golden_output=(None, None, 2.0, 3.0, 4.0, 5.0),
    pins=(
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(1.0, 2.0, 3.0),
            reason="window=1 makes the max and min the value itself, so the midpoint reproduces the input with no "
            "warm-up (test_midpoint.py::TestMidpointEdge::test_window_one)",
        ),
    ),
)
