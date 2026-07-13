"""Spec for ``pomata.pnl.returns_log`` — the one-bar log return, propagating, scale-invariant, additive across time."""

import math

from tests_new.pnl.oracles import returns_log_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import returns_log

RETURNS_LOG = Spec(
    factory=returns_log,
    inputs=("price",),
    params={},
    shape=Shape.SERIES,
    warmup=1,
    oracle=returns_log_reference,
    # The log of a price ratio: scaling the series cancels, leaving the return unchanged, degree 0 (tests/pnl/
    # test_returns_log.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("price",), degree=0),),
    golden_input={"price": (100.0, 105.0, 102.0, 108.0, 110.0)},
    golden_output=(None, 0.0488, -0.029, 0.0572, 0.0183),
    pins=(
        SpecPin(
            label="null_precedence",
            inputs={"price": (100.0, None, math.nan, 108.0)},
            expected=(None, None, None, math.nan),
            reason="the row where a NaN price meets the previous row's null is null (null wins), while the next return "
            "off the NaN is NaN (tests/pnl/test_returns_log.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="domain_boundaries",
            inputs={"price": (10.0, -5.0, 0.0, 5.0)},
            expected=(None, math.nan, -math.inf, math.inf),
            reason="the IEEE logarithm boundaries: a negative relative is NaN, a zero relative is -inf, a zero "
            "previous price makes the relative +inf (tests/pnl/test_returns_log.py::test_domain_boundaries)",
        ),
        SpecPin(
            label="negative_zero_positive_change",
            inputs={"price": (-0.0, 5.0)},
            expected=(None, math.nan),
            reason="over a -0.0 previous price the relative flips sign, so a positive price gives a negative relative "
            "and NaN after the log (tests/pnl/test_returns_log.py::test_negative_zero_previous_price)",
        ),
        SpecPin(
            label="negative_zero_negative_change",
            inputs={"price": (-0.0, -5.0)},
            expected=(None, math.inf),
            reason="the counterpart: over a -0.0 previous price a negative price gives a positive relative, so +inf "
            "(tests/pnl/test_returns_log.py::test_negative_zero_previous_price)",
        ),
        SpecPin(
            label="consecutive_infinities",
            inputs={"price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -math.inf, math.nan),
            reason="two consecutive equal-sign infinite prices make the second log return log(inf / inf) = NaN; the "
            "property tiers set allow_infinity=False (tests/pnl/test_returns_log.py"
            "::test_consecutive_infinities_make_nan)",
        ),
    ),
)
