"""
Declaration for ``pomata.metrics.beta`` — reducing, the regression slope of returns on the benchmark, scale-
invariant.
"""

import math

from pomata.metrics import beta
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_beta
from tests.support.declaration import Golden, Pin, ScaleAxis

BETA = suite_metrics(
    factory=beta,
    inputs=("returns", "benchmark"),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_beta,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(1.162,),
    ),
    pins=(
        Pin(
            label="null_misalignment_drops_pair",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.01, None, -0.005, 0.018)},
            expected=(1.3157894736842104,),
            reason="a null in returns at one row and a null in benchmark at a different row each drop their "
            "pair independently",
        ),
        Pin(
            label="nan_poisons",
            inputs={"returns": (0.01, math.nan, 0.03, -0.01), "benchmark": (0.008, -0.01, 0.025, -0.005)},
            expected=(math.nan,),
            reason="a NaN in only the returns leg still poisons the whole reduction to NaN",
        ),
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="one complete pair has no regression slope (needs >= 2 observations), so the result is null",
        ),
        Pin(
            label="constant_benchmark_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant (zero-variance) benchmark gives 0/0, reported as NaN via an exact max==min "
            "guard — the exact-zero core of the near-constant regime; no conditioning filter is "
            "declared: the cov/var slope matches the oracle within one ULP even at ULP-adjacent "
            "benchmark spreads (measured down to a 1e-15 spread on base 0.1)",
        ),
        Pin(
            label="constant_benchmark_one_third",
            inputs={
                "returns": (0.01, -0.02, 0.03),
                "benchmark": (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
            },
            expected=(math.nan,),
            reason="the same guard at a constant not exactly representable in float, proving it is an exact "
            "equality check, not a rounding one",
        ),
        Pin(
            label="constant_benchmark_many_digits",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third, many-digit constant magnitude",
        ),
    ),
    reference='Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under '
    'Conditions of Risk." *The Journal of Finance*, 19(3), 425-442.',
    doi="https://doi.org/10.1111/j.1540-6261.1964.tb02865.x",
    wikipedia="https://en.wikipedia.org/wiki/Beta_%28finance%29",
    see_also=(
        ("alpha", "The benchmark-relative return that nets out beta-explained performance."),
        ("treynor_ratio", "The excess return per unit of this systematic risk."),
        ("beta_rolling", "The same slope over a trailing window."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two complete pairs leaves the regression slope undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a zero-variance benchmark leaves the slope undefined, so the result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the regression slope (one value in ``select``, one per group "
    "under ``.over``). ``null`` when fewer than two complete pairs are present.",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
