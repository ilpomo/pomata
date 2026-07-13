"""Spec for ``pomata.metrics.sharpe_ratio_rolling`` — the annualized Sharpe over a trailing window, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import sharpe_ratio_rolling_reference
from tests.support import RELATIVE_TOLERANCE_SCALE, windows_well_spread
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import sharpe_ratio_rolling


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant window: the one-pass rolling std cannot track the two-pass oracle there. JUSTIFIED by
    measurement — if anything slightly PERMISSIVE: real disagreement was measured right at the cut (var_rel 1e-9,
    ~1e-4 relative deviation) and agreement only returns ~23x above it, but that sliver of admitted-and-divergent
    windows is unreachable by the property tiers' independent draws, so the shared cut is kept rather than inventing
    a spec-local widening.
    """
    return windows_well_spread(frame.to_series(0).to_list(), 3)


SHARPE_RATIO_ROLLING = Spec(
    factory=sharpe_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=sharpe_ratio_rolling_reference,
    conditioning=_windows_well_spread,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    # A ratio of a rolling mean to a rolling standard deviation at zero risk-free rate is scale-invariant, degree 0
    # (by analogy to the reducing sharpe_ratio; the old rolling suite carries no scale test).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02)},
    golden_output=(None, None, 10.1678, -1.3977, 7.2837, 1.271, 13.1689),
    pins=(
        SpecPin(
            label="zero_volatility",
            inputs={"returns": (0.5, 0.5, 0.5, 0.5)},
            expected=(None, None, math.inf, math.inf),
            reason="a constant window has zero dispersion with a positive mean, so the ratio is +inf — the exact "
            "core of the near-constant regime the conditioning filter excludes from the property tiers "
            "(test_sharpe_ratio_rolling.py::test_zero_volatility_is_inf)",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 4.593220484431882),
            reason="when the window equals the series length only the last row is defined "
            "(test_sharpe_ratio_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="flat_zero_excess_window_by_slide_is_nan",
            inputs={"returns": (-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0)},
            expected=(
                None,
                None,
                None,
                -4.522342411070932,
                -1.8213352931142792,
                7.9372539331937695,
                7.9372539331937695,
                math.nan,
            ),
            reason="a window whose excess returns are all exactly zero degenerates to 0/0 -> NaN even after larger "
            "values slid out: the exact rolling mean pins the numerator to zero, so the incremental running-sum "
            "residue cannot ride above the exactly-zero dispersion as a spurious inf — found by the mutation audit "
            "(reverting the numerator to the native rolling mean survived the suite without this pin); the residue "
            "is bit-sensitive, so the prefix must reach the kernel with exactly these bits",
            params_override={"window": 4},
        ),
    ),
)
