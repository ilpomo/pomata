"""Spec for ``pomata.indicators.mom`` — the fixed-lag momentum difference, propagating, degree-1 homogeneous."""

import math

from tests_new.indicators.oracles import mom_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import mom

MOM = Spec(
    factory=mom,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=mom_reference,
    # The fixed-lag difference x_t - x_{t-n} scales linearly with the series (tests/indicators/test_mom.py
    # ::TestMomProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)},
    golden_output=(None, None, None, -2.0, 4.0, 5.0, 1.0, 1.0),
    pins=(
        SpecPin(
            label="window_one_is_first_difference",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0)},
            expected=(None, 2.0, 2.0, 2.0),
            params_override={"window": 1},
            reason="window=1 is the first difference with a single leading null "
            "(test_mom.py::TestMomEdge::test_window_one_is_first_difference)",
        ),
        SpecPin(
            label="all_nan_series",
            inputs={"expr": (math.nan, math.nan, math.nan, math.nan)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 1},
            reason="an all-NaN series yields null during warm-up and NaN thereafter (test_mom.py::TestMomEdge"
            "::test_all_nan)",
        ),
        SpecPin(
            label="constant_series_is_zero",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, 0.0, 0.0, 0.0),
            reason="the momentum of a constant series is exactly zero once warmed up "
            "(test_mom.py::TestMomEdge::test_constant_series_is_zero)",
        ),
    ),
)
