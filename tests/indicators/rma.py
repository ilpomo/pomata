"""Spec for ``pomata.indicators.rma`` — Wilder's recursive mean, gap-bridging, NaN-latching, degree-1 homogeneous."""

from tests.indicators.oracles import rma_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import rma

RMA = Spec(
    factory=rma,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=rma_reference,
    # A linear recursive mean scales linearly with the series (tests/indicators/test_rma.py
    # ::TestRmaProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
    golden_output=(None, None, 4.0, 5.3333, 6.8889),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 2.0),
            params_override={"window": 3},
            reason="window equal to the series length emits exactly one defined value on the last row "
            "(test_rma.py::TestRmaEdge::test_window_equals_length)",
        ),
        SpecPin(
            label="window_one_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 (alpha=1) reproduces the input with zero warm-up "
            "(test_rma.py::TestRmaEdge::test_window_one_is_identity)",
        ),
        SpecPin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero recursion stays exactly at zero "
            "(test_rma.py::TestRmaEdge::test_all_zero_series_is_zero)",
        ),
        SpecPin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, 5.0, 5.0),
            params_override={"window": 3},
            reason="a constant input yields that same constant at every defined row "
            "(test_rma.py::TestRmaEdge::test_constant_series)",
        ),
        SpecPin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 4.666666666666667, 6.444444444444445, 8.296296296296298),
            params_override={"window": 3},
            reason="the BRIDGED gap-decay renormalization straddling the seed row itself "
            "(test_rma.py::TestRmaEdge::test_interior_null_bridged)",
        ),
        SpecPin(
            label="interior_null_after_seed_bridged",
            inputs={"expr": (2.0, 4.0, 6.0, None, 8.0, 10.0)},
            expected=(None, None, 4.0, None, 5.7142857142857135, 7.142857142857142),
            params_override={"window": 3},
            reason="the post-seed gap-decay branch, pinned deterministically against the reference "
            "(test_rma.py::TestRmaEdge::test_interior_null_after_seed_bridged)",
        ),
    ),
)
