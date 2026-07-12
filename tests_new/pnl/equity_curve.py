"""Spec for ``pomata.pnl.equity_curve`` — the compounding cumulation, bridged nulls, latched NaNs, scale-exempt."""

from tests.pnl.oracles import equity_curve_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

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
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(1.1,),
            reason="a one-element series resolves to 1 + return with no warm-up of its own (tests/pnl/"
            "test_equity_curve.py::test_single_row)",
        ),
        SpecPin(
            label="leading_null_passthrough",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 1.1, 1.32, 1.254),
            reason="a leading warm-up null stays null and the compounded curve begins at the first defined return; "
            "the function declares no warm-up window, so no generic warm-up rung exercises a leading null (tests/pnl/"
            "test_equity_curve.py::test_warmup_leading_null)",
        ),
    ),
)
