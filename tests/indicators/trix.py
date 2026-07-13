"""Spec for ``pomata.indicators.trix`` — the triple-EMA rate of change, gap-bridging, NaN-latching, degree-0."""

from tests.indicators.oracles import trix_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import trix

TRIX = Spec(
    factory=trix,
    inputs=("price",),
    params={"window": 2},
    shape=Shape.SERIES,
    warmup=4,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=trix_reference,
    # A percentage rate of change of a triple EMA, scale-INVARIANT, degree 0
    # (tests/indicators/test_trix.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("price",), degree=0),),
    # A one-pass triple-EMA ratio against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    golden_input={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, None, None, 5.4718, 7.4466, 2.989, 5.4253),
    pins=(
        SpecPin(
            label="window_one_identity_ema",
            inputs={"price": (100.0, 120.0, 90.0, 108.0)},
            expected=(None, 20.0, -25.0, 20.0),
            params_override={"window": 1},
            reason="window=1 makes every EMA pass the identity, degenerating TRIX to the one-period percentage ROC of "
            "the raw input (test_trix.py::TestTrixCorrectness::test_matches_reference, window=1)",
        ),
    ),
)
