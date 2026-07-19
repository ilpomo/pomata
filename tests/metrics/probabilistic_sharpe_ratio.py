"""
Declaration for ``pomata.metrics.probabilistic_sharpe_ratio`` — reducing, P(true Sharpe > benchmark), scale-
invariant.
"""

import math

import polars as pl

from pomata.metrics import probabilistic_sharpe_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_probabilistic_sharpe_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import well_spread


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the embedded sample Sharpe and higher moments are a 0/0 there. KEPT deliberately
    over-wide: this statistic's own divergence onset sits at var_rel ~2.5e-19, far below the shared cut of 1e-9,
    but the cut is sized on the worst family member (kurtosis) and a spec-local narrowing would buy back a
    negligible slice of draws at the price of one more magic constant — over-width here is a safe, conservative
    guard, not a hazard.
    """
    return well_spread(frame.to_series(0).to_list())


PROBABILISTIC_SHARPE_RATIO = suite_metrics(
    factory=probabilistic_sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "benchmark_sharpe": 0.0, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_probabilistic_sharpe_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"benchmark_sharpe": math.nan}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": -math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_well_spread,
    golden=Golden(
        inputs={"returns": (0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009)}, output=(0.9922,)
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the statistic is null ",
        ),
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero dispersion, so the Sharpe and higher moments are undefined, "
            "yielding NaN — the exact core of the near-constant regime the conditioning filter "
            "excludes from the property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="null_skipped_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.729973707391394,),
            reason="a null is skipped under a non-default benchmark Sharpe ",
            params_override={"benchmark_sharpe": 0.05},
        ),
        Pin(
            label="matches_reference_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.5961103866888193,),
            reason="reference agreement under a non-default benchmark Sharpe ",
            params_override={"benchmark_sharpe": 0.05},
        ),
    ),
    reference='Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." '
    "*Journal of Risk*, 15(2), 3-44.",
    doi="https://doi.org/10.21314/JOR.2012.255",
    wikipedia="https://en.wikipedia.org/wiki/Sharpe_ratio",
    see_also=(
        ("sharpe_ratio", "The point estimate this attaches a confidence level to."),
        ("adjusted_sharpe_ratio", "The point-estimate correction for the same non-normality."),
        ("sortino_ratio", "The downside-deviation Sharpe variant for the same asymmetric returns."),
    ),
    note_extension="\n\n"
    "The kurtosis term uses the non-excess (raw) kurtosis :math:`\\gamma_4`, exactly as in "
    "Bailey & López de Prado: a normal sample (:math:`\\gamma_4 = 3`) recovers the classic Lo "
    "standard error :math:`\\sqrt{(1 + \\mathrm{SR}^2 / 2) / (n - 1)}`.",
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
            "a near-constant sample sits at the float-conditioning limit: the inner variance under "
            "the square root is ill-conditioned there, so a floating-point residue can push it "
            "slightly negative (mathematically impossible by Pearson's inequality), yielding ``NaN``, "
            "while an exactly-zero inner variance (a measure-zero boundary) yields the limiting ``0`` "
            "or ``1``, reported rather than forced into range. The exactly-constant sample is pinned "
            "(the ``NaN`` above); real return samples are far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value in ``[0, 1]``: the probabilistic Sharpe ratio (one value in "
    "``select``, one per group under ``.over``). ``null`` when fewer than two returns are "
    "present (the sample Sharpe ratio is undefined).",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``benchmark_sharpe`` or "
    "``risk_free_rate`` is not finite, or if ``risk_free_rate < -1``.",
    args_prose={
        "periods_per_year": "Observations per year, used only to convert the annualized risk-free rate to a "
        "per-period rate (canonically ``252`` for daily). Must be ``>= 1``.",
        "benchmark_sharpe": "The (non-annualized) benchmark Sharpe ratio :math:`\\mathrm{SR}^{*}` to beat (default "
        "``0.0``). Must be finite.",
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
    examples=(
        Example(
            inputs={"returns": (0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009)},
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (
                    0.03,
                    -0.01,
                    0.02,
                    -0.015,
                    0.01,
                    0.005,
                    -0.02,
                    0.02,
                    -0.005,
                    0.015,
                    -0.01,
                    0.025,
                    0.0,
                    -0.012,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — a single return has no sample dispersion, so the result is ``null``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant series has zero dispersion, so the Sharpe ratio "
            "and higher moments are undefined and the result is ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
