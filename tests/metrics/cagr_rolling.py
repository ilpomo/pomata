"""Spec for ``pomata.metrics.cagr_rolling`` — the annualized growth over a trailing window, scale-invariant."""

import math

from tests.metrics.oracles import cagr_rolling_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import cagr_rolling

CAGR_ROLLING = Spec(
    factory=cagr_rolling,
    inputs=("equity_curve",),
    params={"window": 3, "periods_per_year": 4},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
    ),
    oracle=cagr_rolling_reference,
    # The window's endpoint ratio E_t / E_{t-window+1} is unchanged by a uniform rescale of the equity, so the
    # annualized growth is scale-invariant (verified numerically; the old suite carries no scale test).
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
    golden_output=(None, None, 0.1025, 0.1901, 0.1995, 0.1736, 0.1815),
    pins=(
        SpecPin(
            label="window_crossing_zero_is_nan",
            inputs={"equity_curve": (1.0, -0.5, 0.8, 1.2)},
            expected=(None, math.nan, math.nan, 0.5),
            reason="a window whose endpoint ratio is non-positive is out of the geometric-growth domain, a loud NaN "
            "(tests/metrics/test_cagr_rolling.py::test_window_crossing_zero_is_nan)",
            params_override={"window": 2, "periods_per_year": 1},
        ),
        SpecPin(
            label="window_endpoint_exact_zero_is_nan",
            inputs={"equity_curve": (100.0, 105.0, 0.0, 110.0, 120.0)},
            expected=(None, None, math.nan, 0.0975056689342404, math.inf),
            reason="the EXACT-zero boundary of the geometric domain, distinct from the crossing pin's negative "
            "ratios: an end value of exactly 0.0 must be the loud NaN (not the plausible 0**x - 1 = -1.0), and a "
            "zero START endpoint blows the ratio to +inf, reported not clipped — found by the mutation audit "
            "(relaxing the guard from <= 0 to < 0 survived the suite, since no fixed case sat on the boundary)",
            params_override={"window": 3, "periods_per_year": 4},
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2)},
            expected=(None, None, None, 0.2751902830191333),
            reason="when the window equals the series length only the last row is defined "
            "(tests/metrics/test_cagr_rolling.py::test_window_equals_length)",
            params_override={"window": 4, "periods_per_year": 4},
        ),
        SpecPin(
            label="endpoint_null_is_null",
            inputs={"equity_curve": (None, 1.1, 1.05, 1.2, None)},
            expected=(None, None, None, 0.19008264462809898, None),
            reason="a null at either window endpoint nulls every window touching it "
            "(tests/metrics/test_cagr_rolling.py::test_endpoint_null_is_null)",
            params_override={"window": 3, "periods_per_year": 4},
        ),
        SpecPin(
            label="interior_null_is_spanned",
            inputs={"equity_curve": (1.0, None, 1.05, None, 1.15)},
            expected=(None, None, 0.10250000000000004, None, 0.19954648526077068),
            reason="an interior null (not a window endpoint) has exactly zero effect on the windows spanning it "
            "(tests/metrics/test_cagr_rolling.py::test_interior_null_is_spanned)",
            params_override={"window": 3, "periods_per_year": 4},
        ),
        SpecPin(
            label="endpoint_nan_propagates",
            inputs={"equity_curve": (1.0, 1.1, math.nan, 1.2, 1.15)},
            expected=(None, None, math.nan, 0.19008264462809898, math.nan),
            reason="a NaN at a window endpoint propagates to NaN for every window touching it "
            "(tests/metrics/test_cagr_rolling.py::test_endpoint_nan_propagates)",
            params_override={"window": 3, "periods_per_year": 4},
        ),
        SpecPin(
            label="recovers_the_constant_growth_rate",
            inputs={
                "equity_curve": (
                    1.1,
                    1.2100000000000002,
                    1.3310000000000004,
                    1.4641000000000004,
                    1.6105100000000006,
                    1.7715610000000008,
                    1.9487171000000012,
                )
            },
            expected=(
                None,
                None,
                None,
                0.10000000000000009,
                0.10000000000000009,
                0.10000000000000009,
                0.10000000000000009,
            ),
            reason="a curve compounding at a constant per-period rate has every window recover exactly that rate "
            "(tests/metrics/test_cagr_rolling.py::test_recovers_the_constant_growth_rate)",
            params_override={"window": 4, "periods_per_year": 1},
        ),
    ),
)
