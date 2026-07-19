"""
Declaration for ``pomata.metrics.value_at_risk_parametric`` — reducing, mean plus z times std, degree-1 homogeneous.
"""

from pomata.metrics import value_at_risk_parametric
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_value_at_risk_parametric
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

VALUE_AT_RISK_PARAMETRIC = suite_metrics(
    factory=value_at_risk_parametric,
    inputs=("returns",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_value_at_risk_parametric,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(-0.0732,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample standard deviation (ddof=1 needs two), so the result is null",
        ),
        Pin(
            label="constant_is_mean",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(0.01,),
            reason="a constant series collapses z*std to float noise, so the value-at-risk lands on the mean",
        ),
    ),
    reference="Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd "
    "ed.). McGraw-Hill.",
    wikipedia="https://en.wikipedia.org/wiki/Value_at_risk",
    see_also=(
        ("value_at_risk", "The historical (empirical) form."),
        ("value_at_risk_modified", "The skewness/kurtosis-corrected form (within its documented validity domain)."),
        ("conditional_value_at_risk", "The expected shortfall beyond the VaR threshold."),
    ),
    notes=(
        (
            "Gaussian assumption",
            "The estimate assumes normally distributed returns; for fat tails see "
            ":func:`value_at_risk_modified` (Cornish-Fisher) or :func:`value_at_risk` (historical).",
        ),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the mean and the standard deviation); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two returns leave the sample standard deviation undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a constant series collapses the dispersion to float noise (``~1e-17``, an additive term, never a "
            "divisor), so ``z * sigma`` vanishes and the result lands on the mean.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the parametric value-at-risk (one value in ``select``, one "
    "per group under ``.over``). ``null`` when fewer than two returns are present (the sample "
    "standard deviation is undefined).",
    raises_prose="ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.",
    args_prose={
        "confidence": "The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - "
        "confidence``. Must be in the open interval ``(0, 1)``.",
    },
    examples=(
        Example(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, round_to=4),
        Example(
            inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — one observation has no sample standard deviation (``ddof=1`` "
            "needs at least two), so the result is ``null``:",
        ),
    ),
)
