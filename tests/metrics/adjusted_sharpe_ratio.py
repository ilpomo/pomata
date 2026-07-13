"""Spec for ``pomata.metrics.adjusted_sharpe_ratio`` — reducing, the skew/kurtosis-adjusted Sharpe, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import adjusted_sharpe_ratio_reference
from tests.support import well_spread
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import adjusted_sharpe_ratio, sharpe_ratio


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the moments the Pezier-White correction uses are a 0/0 there. JUSTIFIED by
    measurement: this statistic embeds the skewness AND the kurtosis, and its first impl-vs-oracle breaches appear
    right at the shared cut (stdev_rel ~3.5e-5 vs the cut's 3.16e-5, zero breaches in 120 trials just above it), so
    the filter sits exactly where the family's worst member needs it and must not be narrowed.
    """
    return well_spread(frame.to_series(0).to_list())


def _adjusted_component() -> pl.Expr:
    """Recomposed from the public ``sharpe_ratio`` factory plus the skew/kurtosis correction, at the default params."""
    annualization = math.sqrt(252)
    sharpe = sharpe_ratio(pl.col("returns"), periods_per_year=252, risk_free_rate=0.0) / annualization
    correction = 1.0 + pl.col("returns").skew() / 6.0 * sharpe - pl.col("returns").kurtosis() / 24.0 * sharpe**2
    return annualization * sharpe * correction


ADJUSTED_SHARPE_RATIO = Spec(
    factory=adjusted_sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=adjusted_sharpe_ratio_reference,
    conditioning=_well_spread,
    # Scale-invariant at risk_free_rate=0, degree 0 (test_adjusted_sharpe_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.025, -0.015)},
    golden_output=(2.992,),
    component_expr=_adjusted_component,
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series yields null (the sample Sharpe ratio needs two observations) "
            "(test_adjusted_sharpe_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has undefined moments, so the result is NaN — the exact core of the "
            "near-constant regime the conditioning filter excludes from the property tiers "
            "(test_adjusted_sharpe_ratio.py::test_zero_volatility_is_nan)",
            covers_conditioning=True,
        ),
    ),
)
