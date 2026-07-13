"""Spec for ``pomata.metrics.volatility_rolling`` — the annualized rolling sample std, degree-1 homogeneous."""

import polars as pl
from tests_new.metrics.oracles import volatility_rolling_reference
from tests_new.support import windows_well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import volatility_rolling


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant window: the one-pass rolling std cannot track the two-pass oracle there. KEPT at the
    shared cut even though this spec's own onset at unit scale sits ~2.5 orders below it: the true floor here is
    quasi-ABSOLUTE, not scale-squared-relative, so at the domain's low edge (|r| ~0.01) the same shared cut is
    already within ~4x of the real onset — no single tighter constant is safe at both ends of the domain, and the
    shared cut is sized on that low edge.
    """
    return windows_well_spread(frame.to_series(0).to_list(), 4)


VOLATILITY_ROLLING = Spec(
    factory=volatility_rolling,
    inputs=("returns",),
    params={"window": 4, "periods_per_year": 252},
    shape=Shape.SERIES,
    warmup=3,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
    ),
    oracle=volatility_rolling_reference,
    conditioning=_windows_well_spread,
    # A sample standard deviation per window is degree-1 homogeneous (by analogy to the reducing volatility; the old
    # rolling suite carries no scale test).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, None, 0.352, 0.3779, 0.2898, 0.2457),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 0.32918080138428485),
            reason="when the window equals the series length only the last row is defined "
            "(test_volatility_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="constant_window_is_zero",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(None, None, 0.0, 0.0),
            reason="a constant window has zero dispersion, so the volatility is exactly 0 — the exact core of the "
            "near-constant regime the conditioning filter excludes from the property tiers "
            "(test_volatility_rolling.py::test_constant_window_is_zero)",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
    ),
)
