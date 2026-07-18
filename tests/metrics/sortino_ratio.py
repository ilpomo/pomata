"""Declaration for ``pomata.metrics.sortino_ratio`` — reducing, annualized excess return over downside deviation."""

import math

from pomata.metrics import sortino_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_sortino_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

SORTINO_RATIO = suite_metrics(
    factory=sortino_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_sortino_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(4.4567,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-15.874507866387544,),
            reason="one observation has a defined population downside deviation, so the ratio is a real number",
        ),
        Pin(
            label="no_downside_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="all returns at/above the zero target: downside deviation is 0 with a positive mean, so +inf",
        ),
        Pin(
            label="null_skipped_risk_free_rate",
            inputs={"returns": (0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018)},
            expected=(7.332599938409737,),
            reason="a null is excluded jointly with a non-default risk-free rate ",
            params_override={"risk_free_rate": 0.02},
        ),
        Pin(
            label="matches_reference_risk_free_rate",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(2.4195202806468132,),
            reason="reference agreement at a non-default risk-free rate ",
            params_override={"risk_free_rate": 0.02},
        ),
        Pin(
            label="no_downside_zero_mean_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series sits at the zero target with zero mean excess, so the downside "
            "deviation and the numerator are both zero, giving a 0/0, i.e. NaN — the "
            "degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference='Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk '
    'Framework." *The Journal of Investing*, 3(3), 59-64.',
    doi="https://doi.org/10.3905/joi.3.3.59",
    wikipedia="https://en.wikipedia.org/wiki/Sortino_ratio",
    see_also=(
        ("sharpe_ratio", "The two-sided counterpart (penalizes all volatility)."),
        ("downside_deviation", "The denominator (downside-only dispersion)."),
        ("omega_ratio", "The threshold-based gain-to-loss alternative."),
        ("sortino_ratio_rolling", "The rolling (windowed) form."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the mean and the downside deviation); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "when every excess return is at or above the target the downside deviation is zero, so "
            "the ratio is ``+/-inf`` (or ``NaN`` when the mean excess is also zero) — reported, not "
            "clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized Sortino ratio (one value in ``select``, one "
    "per group under ``.over``). ``null`` when there are no returns.",
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
