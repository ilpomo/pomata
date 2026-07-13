"""Spec for ``pomata.pnl.cumulative_pnl`` — the additive running total, bridged nulls, latched NaNs, degree-1."""

from tests_new.pnl.oracles import cumulative_pnl_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cumulative_pnl

CUMULATIVE_PNL = Spec(
    factory=cumulative_pnl,
    inputs=("returns",),
    params={},
    shape=Shape.SERIES,
    oracle=cumulative_pnl_reference,
    # The linear (additive) twin of equity_curve: degree-1 homogeneous, not scale-exempt (tests/pnl/
    # test_cumulative_pnl.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.1, -0.05, 0.2, 0.1)},
    golden_output=(0.1, 0.05, 0.25, 0.35),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(0.1,),
            reason="a one-element series resolves to that single return with no warm-up (tests/pnl/"
            "test_cumulative_pnl.py::test_single_row)",
        ),
        SpecPin(
            label="warmup_leading_null",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 0.1, 0.3, 0.25),
            reason="a leading warm-up null stays null and the running total begins at the first defined return; the "
            "function declares no warm-up window, so no generic warm-up rung exercises a leading null (tests/pnl/"
            "test_cumulative_pnl.py::test_warmup_leading_null)",
        ),
    ),
)
