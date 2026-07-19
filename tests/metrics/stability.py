"""
Declaration for ``pomata.metrics.stability`` — reducing, the R-squared of the cumulative-log-return trend, scale-
exempt.
"""

import math

from pomata.metrics import stability
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_stability
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

STABILITY = suite_metrics(
    factory=stability,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_stability,
    scaling=ScaleExempt(
        reason="an R-squared of the cumulative log-return trend — the nonlinear log makes it neither "
        "scale-invariant nor scale-homogeneous"
    ),
    golden=Golden(inputs={"returns": (0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012)}, output=(0.9984,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.01,)},
            expected=(None,),
            reason="one observation has no dispersion; the regression needs two points",
        ),
        Pin(
            label="lone_nan_poisons",
            inputs={"returns": (math.nan,)},
            expected=(math.nan,),
            reason="a NaN poisons the result even when it is the only observation — the poison guard wins "
            "over the two-point count guard, exactly as in the cagr / total_return siblings",
        ),
        Pin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant non-zero return series has a perfectly linear cumulative log, so R-squared is 1.0",
        ),
        Pin(
            label="flat_path_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has a flat (zero-variance) cumulative log, so R-squared is NaN — the "
            "exact-flat core of the near-constant regime; impl and oracle agree on every "
            "fuzz-reachable path, so no conditioning filter is declared, and only this pin reaches "
            "the exact flat",
        ),
        Pin(
            label="out_of_domain_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01)},
            expected=(math.nan,),
            reason="a return at or below -1 makes log1p undefined, propagating to NaN",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Coefficient_of_determination",
    see_also=(
        ("cagr", "The growth rate whose steadiness this measures."),
        ("linear_regression", "The least-squares trend line whose goodness-of-fit this scores."),
        ("linear_regression_slope", "The slope of that same least-squares trend."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped; an all-null (or empty) series yields ``null`` (the time "
            "index is taken over the retained observations, so an interior gap does not leave a hole "
            "in the regression).",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Domain",
            "a return at or below ``-1`` makes the cumulative log undefined, so the result is a loud "
            "``NaN`` — never a plausible wrong number.",
        ),
        (
            "Insufficient sample",
            "fewer than two observations are present (the regression is undefined), so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "an all-zero (or otherwise perfectly flat) cumulative log has no variance to explain, so "
            "the result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value in ``[0, 1]``: the trend stability (one value in ``select``, "
    "one per group under ``.over``). ``null`` when fewer than two returns are present (the "
    "regression is undefined).",
    args_prose={
        "returns": "Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`); "
        "each must exceed ``-1`` (a return of ``-100%`` or worse makes the cumulative log "
        "undefined).",
    },
    examples=(
        Example(inputs={"returns": (0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012)}, round_to=4),
        Example(
            inputs={
                "returns": (
                    0.01,
                    0.012,
                    0.009,
                    0.011,
                    0.013,
                    0.008,
                    0.01,
                    0.012,
                    0.02,
                    -0.01,
                    0.03,
                    -0.02,
                    0.025,
                    -0.015,
                    0.018,
                    -0.012,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 8 + ("NVDA",) * 8,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, 0.012, None, 0.009, float("nan"), 0.011)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, -1.5, 0.01)},
            intro="**Domain** — a return at or below ``-1`` makes the cumulative log undefined, so the "
            "result is a loud ``NaN``:",
        ),
        Example(
            inputs={"returns": (0.01,)},
            intro="**Insufficient sample** — a single observation gives the regression no second point to "
            "fit, so the result is ``null``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series has a perfectly flat cumulative log with "
            "zero variance to explain, so the result is the ``0 / 0`` case, ``NaN``:",
        ),
    ),
)
