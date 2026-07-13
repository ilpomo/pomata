"""Spec for ``pomata.indicators.hma`` — Hull's lag-reduced weighted mean, window-nulling, degree-1 homogeneous."""

from tests.indicators.oracles import hma_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import hma

HMA = Spec(
    factory=hma,
    inputs=("expr",),
    params={"window": 4},
    shape=Shape.SERIES,
    warmup=4,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"window": 0}, r"window must be >= 2"),
    ),
    oracle=hma_reference,
    # A linear combination of WMAs scales linearly with the series (tests/indicators/test_hma.py
    # ::TestHmaProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
    golden_output=(None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    pins=(
        SpecPin(
            label="golden_master_overshoot",
            inputs={"expr": (1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            expected=(None, None, None, None, 11.599999999999998, 11.5, 10.299999999999999, 10.0),
            reason="the lag correction 2*WMA(x,half) - WMA(x,window) over- and under-shoots the input range before the "
            "final smoothing settles (test_hma.py::TestHmaCorrectness::test_golden_master_overshoot)",
        ),
        SpecPin(
            label="golden_master_round_half_up",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                5.666666666666666,
                6.666666666666665,
                7.666666666666665,
                8.666666666666664,
                9.666666666666664,
                10.666666666666664,
                11.666666666666664,
            ),
            params_override={"window": 5},
            reason="the round-half-up period reduction at window=5: half-period = floor(5/2 + 0.5) = 3, not the "
            "banker-rounded 2 (test_hma.py::TestHmaCorrectness::test_golden_master_round_half_up)",
        ),
    ),
)
