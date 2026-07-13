"""Spec for ``pomata.indicators.trima`` — the triangular (SMA-of-SMA) rolling mean, window-nulling, degree-1."""

import math

from tests.indicators.oracles import trima_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import trima

_SWEEP = (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)

TRIMA = Spec(
    factory=trima,
    inputs=("expr",),
    params={"window": 5},
    shape=Shape.SERIES,
    warmup=4,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=trima_reference,
    # A fixed triangular-weight normalization scales linearly with the series (tests/indicators/test_trima.py
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 4},
    golden_input={"expr": (3.0, 1.0, 4.0, 1.0, 5.0)},
    golden_output=(None, None, None, 2.3333, 2.6667),
    pins=(
        SpecPin(
            label="single_row_window_one_identity",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="window=1 on one row returns the value itself (test_trima.py::TestTrimaEdge::test_single_row)",
        ),
        SpecPin(
            label="single_row_window_exceeds_length",
            inputs={"expr": (42.0,)},
            expected=(None,),
            params_override={"window": 3},
            reason="window > length on one row yields null (test_trima.py::TestTrimaEdge::test_single_row)",
        ),
        SpecPin(
            label="null_in_window_recovers",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, None, 3.5),
            params_override={"window": 2},
            reason="a null inside the window yields null there, and the value returns once the window clears "
            "(test_trima.py::TestTrimaEdge::test_null_in_window_is_null)",
        ),
        SpecPin(
            label="nan_propagates_confined",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, 3.5),
            params_override={"window": 2},
            reason="a NaN inside the window yields NaN there, confined to the windows spanning it "
            "(test_trima.py::TestTrimaEdge::test_nan_propagates)",
        ),
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reduces to the identity across a full series "
            "(test_trima.py::TestTrimaEdge::test_window_one_is_identity)",
        ),
        SpecPin(
            label="matches_reference_window_2",
            inputs={"expr": _SWEEP},
            expected=(None, 2.0, 2.5, 2.5, 3.0, 7.0, 5.5, 4.0),
            params_override={"window": 2},
            reason="the even-window branch of the old odd/even reference sweep "
            "(test_trima.py::TestTrimaCorrectness::test_matches_reference)",
        ),
        SpecPin(
            label="matches_reference_window_3",
            inputs={"expr": _SWEEP},
            expected=(None, None, 2.25, 2.5, 2.75, 5.0, 6.25, 4.75),
            params_override={"window": 3},
            reason="the odd-window branch of the old reference sweep "
            "(test_trima.py::TestTrimaCorrectness::test_matches_reference)",
        ),
        SpecPin(
            label="matches_reference_window_4",
            inputs={"expr": _SWEEP},
            expected=(
                None,
                None,
                None,
                2.3333333333333335,
                2.6666666666666665,
                4.166666666666667,
                5.166666666666667,
                5.5,
            ),
            params_override={"window": 4},
            reason="the even-window branch of the old reference sweep "
            "(test_trima.py::TestTrimaCorrectness::test_matches_reference)",
        ),
        SpecPin(
            label="matches_reference_window_5",
            inputs={"expr": _SWEEP},
            expected=(
                None,
                None,
                None,
                None,
                2.6666666666666665,
                3.4444444444444446,
                4.555555555555556,
                5.333333333333333,
            ),
            params_override={"window": 5},
            reason="the odd-window branch of the old reference sweep "
            "(test_trima.py::TestTrimaCorrectness::test_matches_reference)",
        ),
        SpecPin(
            label="matches_reference_window_6",
            inputs={"expr": _SWEEP},
            expected=(None, None, None, None, None, 3.25, 3.916666666666667, 4.833333333333334),
            params_override={"window": 6},
            reason="the widest window of the old reference sweep "
            "(test_trima.py::TestTrimaCorrectness::test_matches_reference)",
        ),
    ),
)
