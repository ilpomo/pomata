"""
Declaration for ``pomata.metrics.adjusted_sharpe_ratio`` — reducing, the skew/kurtosis-adjusted Sharpe, scale-
invariant.
"""

import math

import polars as pl

from pomata.metrics import adjusted_sharpe_ratio, sharpe_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_adjusted_sharpe_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis
from tests.support.strategies import well_spread


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


ADJUSTED_SHARPE_RATIO = suite_metrics(
    factory=adjusted_sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_adjusted_sharpe_ratio,
    recomposition=_adjusted_component,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_well_spread,
    golden=Golden(inputs={"returns": (0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.025, -0.015)}, output=(2.992,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series yields null (the sample Sharpe ratio needs two observations) ",
        ),
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has undefined moments, so the result is NaN — the exact core of the "
            "near-constant regime the conditioning filter excludes from the property tiers",
            covers_conditioning=True,
        ),
    ),
    reference='Pezier, J. & White, A. (2008). "The Relative Merits of Alternative Investments in '
    'Passive Portfolios." *Journal of Alternative Investments*, 10(4), 37-49.',
    doi="https://doi.org/10.3905/jai.2008.705531",
    wikipedia="https://en.wikipedia.org/wiki/Sharpe_ratio",
    see_also=(
        ("sharpe_ratio", "The base ratio this adjusts."),
        ("probabilistic_sharpe_ratio", "The confidence-level alternative correction for non-normality."),
        ("sortino_ratio", "The downside-deviation variant that captures the same return asymmetry differently."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from every moment); an all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "with fewer than two returns the sample Sharpe ratio is undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a constant series has an undefined Sharpe ratio and undefined moments, so the result is "
            "a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Stability",
            "a near-constant (non-bit-identical) sample sits at the float-conditioning limit: as the "
            "dispersion vanishes the skewness and excess kurtosis the Pezier-White correction embeds "
            "become an ill-conditioned ``0 / 0``, and the reference and implementation can round "
            "apart. The exactly-constant sample is pinned (the ``NaN`` above); real return samples "
            "are far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the adjusted Sharpe ratio (one value in ``select``, one per "
    "group under ``.over``). ``null`` when fewer than two returns are present (the Sharpe "
    "ratio is undefined).",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.",
    args_prose={
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
