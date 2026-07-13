"""Spec for ``pomata.indicators.atr`` — Wilder's Average True Range, gap-bridging, NaN-latching, degree-1."""

from tests.indicators.oracles import atr_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import atr

ATR = Spec(
    factory=atr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=atr_reference,
    # A Wilder rma of the true range, homogeneous of degree 1 (tests/indicators/test_atr.py::TestAtrProperties
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    golden_params={"window": 3},
    golden_input={
        "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0),
        "low": (8.0, 9.0, 9.5, 10.0, 12.0, 11.0, 13.0, 15.0),
        "close": (9.0, 11.0, 10.0, 12.0, 14.0, 13.0, 15.0, 17.0),
    },
    golden_output=(None, None, 2.1667, 2.4444, 2.6296, 2.7531, 2.8354, 2.8903),
    pins=(
        SpecPin(
            label="window_one_is_true_range",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.5, 10.0), "close": (9.0, 11.0, 10.0, 12.0)},
            params_override={"window": 1},
            expected=(2.0, 3.0, 1.5, 3.0),
            reason="window=1 makes the Wilder smoothing the identity, so the ATR reproduces the true range "
            "(test_atr.py::TestAtrEdge::test_window_one_is_true_range)",
        ),
    ),
)
