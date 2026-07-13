"""Spec for ``pomata.indicators.sma`` — the simple rolling mean, window-nulling, degree-1 homogeneous."""

from tests.indicators.oracles import sma_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import sma

SMA = Spec(
    factory=sma,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=sma_reference,
    # A rolling mean scales linearly with the series (tests/indicators/test_sma.py::TestSmaProperties
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
    golden_output=(None, None, 4.0, 6.0, 8.0),
    pins=(
        SpecPin(
            label="single_row_window_one_identity",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-element series with window=1 returns the value itself "
            "(test_sma.py::TestSmaEdge::test_single_row, first assertion)",
        ),
        SpecPin(
            label="single_row_window_exceeds_length",
            inputs={"expr": (42.0,)},
            expected=(None,),
            reason="a one-element series with window=3 is entirely warm-up "
            "(test_sma.py::TestSmaEdge::test_single_row, second assertion)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 2.0),
            reason="a window equal to the series length emits exactly one defined value, the whole-series mean "
            "(test_sma.py::TestSmaEdge::test_window_equals_length)",
        ),
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reproduces the input with no warm-up "
            "(test_sma.py::TestSmaEdge::test_window_one_is_identity)",
        ),
    ),
)
