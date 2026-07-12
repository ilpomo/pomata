"""Spec for ``pomata.pnl.equity_curve`` — the compounding cumulation, bridged nulls, latched NaNs, scale-exempt."""

from tests.pnl.oracles import equity_curve_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec

from pomata.pnl import equity_curve

EQUITY_CURVE = Spec(
    factory=equity_curve,
    inputs=("returns",),
    params={},
    shape=Shape.SERIES,
    oracle=equity_curve_reference,
    # A nonlinear compounding transform — neither scale-invariant nor homogeneous (tests/pnl/test_equity_curve.py:8);
    # the old suite stands a compounding metamorphic in place of a scale axis.
    scale=ScaleExempt(reason="nonlinear compounding: neither scale-invariant nor homogeneous"),
    golden_input={"returns": (0.1, -0.05, 0.2, 0.1)},
    golden_output=(1.1, 1.045, 1.254, 1.3794),
)
