"""
Declaration for ``pomata.metrics.omega_ratio`` — reducing, mean gain over mean loss about a threshold, scale-
invariant.
"""

import math

from pomata.metrics import omega_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_omega_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

OMEGA_RATIO = suite_metrics(
    factory=omega_ratio,
    inputs=("returns",),
    params={"threshold": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_omega_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="one observation puts all mass on the gain side, so the ratio is +inf ",
        ),
        Pin(
            label="all_gain_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="returns all above the threshold have no downside, so the ratio is +inf ",
        ),
        Pin(
            label="all_loss_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="returns all below the threshold have no upside, so the ratio is 0 ",
        ),
        Pin(
            label="all_at_threshold_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="returns all exactly at the threshold give 0/0, so the ratio is NaN ",
        ),
        Pin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(0.6,),
            reason="agreement at a non-default threshold — the shifted-gain/loss split every other tier "
            "leaves at the 0.0 default, mirroring the rolling twin's pin",
            params_override={"threshold": 0.01},
        ),
    ),
    reference='Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *The Journal of '
    "Performance Measurement*, 6(3), 59-84.",
    wikipedia="https://en.wikipedia.org/wiki/Omega_ratio",
    see_also=(
        ("gain_to_pain_ratio", "The net-return over total-loss sibling about a zero threshold."),
        ("sortino_ratio", "The downside-deviation risk-adjusted alternative."),
        ("sharpe_ratio", "The moment-based risk-adjusted ratio."),
        ("omega_ratio_rolling", "The rolling (windowed) form."),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation above the threshold has no offsetting downside, so the result is "
            "``+inf`` — reported, not clipped.",
        ),
        (
            "Degenerate denominator",
            "when no return is below the threshold the mean loss is zero, so the ratio is ``+inf`` "
            "(or ``NaN`` when every return sits exactly at the threshold, a ``0 / 0``); with no "
            "return above it the mean gain is zero, so the ratio is exactly ``0`` — all reported, not "
            "clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the omega ratio (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no returns.",
    raises_prose="ValueError: If ``threshold`` is not finite.",
    args_prose={
        "threshold": "The **per-period** return level separating gains from losses / the minimum acceptable "
        "return (default ``0.0``); an annual target must be de-annualized by the caller before it "
        "is passed. Must be finite.",
    },
    examples=(
        Example(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, round_to=4),
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
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — a single observation above the threshold has no offsetting "
            "loss, so the ratio is ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — returns all above the threshold have no downside, so the "
            "ratio is ``+inf``:",
        ),
        Example(
            inputs={"returns": (-0.01, -0.02, -0.03)},
            intro="**Degenerate denominator** — returns all below the threshold have no upside, so the "
            "ratio is exactly ``0``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — returns all exactly at the threshold give a ``0 / 0``, so "
            "the ratio is ``NaN``:",
        ),
    ),
)
