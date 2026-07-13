"""Spec for ``pomata.indicators.di_minus`` — Wilder's negative directional indicator, gap-bridging, scale-invariant."""

from tests.indicators.oracles import di_minus_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import di_minus

DI_MINUS = Spec(
    factory=di_minus,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    lands_on="low",
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=di_minus_reference,
    # A percentage ratio of smoothed movements to the average true range, bounded in [0, 100] and scale-INVARIANT,
    # degree 0 (tests/indicators/test_di_minus.py::TestDiMinusProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
    },
    golden_output=(None, 0.0, 0.0, 21.0526, 7.8431, 24.0964, 9.4787),
)
