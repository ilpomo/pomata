"""Declaration for ``pomata.metrics.omega_ratio_rolling`` — the rolling mean gain over mean loss about a threshold."""

import math

from pomata.metrics import omega_ratio_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.omega_ratio import OMEGA_RATIO
from tests.metrics.oracles import reference_omega_ratio_rolling
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

OMEGA_RATIO_ROLLING = suite_metrics(
    factory=omega_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "threshold": 0.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=OMEGA_RATIO,
    window="window",
    warmup=2,
    oracle=reference_omega_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, 2.0, 1.0, 5.0, 2.0, 1.3333),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 2.0),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, 0.6666666666666665, 0.3999999999999999, 1.5),
            reason="agreement at a non-default threshold",
            params_override={"window": 3, "threshold": 0.01},
        ),
        Pin(
            label="no_downside_window_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None, None, math.inf),
            reason="a window with no return below the threshold has zero mean loss, so +inf",
        ),
        Pin(
            label="no_activity_window_is_nan",
            inputs={"returns": (1000000000.0, 0.01, 1e-09, 0.0, 0.0)},
            expected=(None, math.inf, math.inf, math.inf, math.nan),
            reason="a large-magnitude corner guarding a spurious +inf residue against a correct 0/0=NaN",
            params_override={"window": 2},
        ),
        Pin(
            label="tiny_loss_window_matches_reference",
            inputs={"returns": (-0.01, 0.5, 0.5)},
            expected=(None, None, 99.99999999999999),
            reason="the smallest mean loss the fuzz domain can put in a window (one loss at the |r| >= 0.01 "
            "floor against gains at the 0.5 cap): even here the plain sum ratio matches the oracle "
            "exactly, so no conditioning filter is declared",
        ),
    ),
    reference='Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *The Journal of '
    "Performance Measurement*, 6(3), 59-84.",
    wikipedia="https://en.wikipedia.org/wiki/Omega_ratio",
    see_also=(
        ("omega_ratio", "The whole-series reducing form."),
        ("sortino_ratio_rolling", "The rolling downside-deviation risk-adjusted ratio."),
        ("sharpe_ratio_rolling", "The rolling total-volatility risk-adjusted ratio."),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`omega_ratio` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window with no return below the threshold has zero mean loss (forced exactly to zero, "
            "never a slid-out residue), so the ratio is ``+inf`` (or ``NaN`` when every return sits "
            "at the threshold, a ``0 / 0``) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling omega ratio for each row, the same length as the input. The first ``window - "
    "1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null values before "
    "a result is emitted.",
    raises_prose="ValueError: If ``window < 1``, or if ``threshold`` is not finite.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
        "threshold": "The **per-period** return level separating gains from losses / the minimum acceptable "
        "return (default ``0.0``); an annual target must be de-annualized by the caller before it "
        "is passed. Must be finite.",
    },
    examples=(
        Example(inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)}, params={"window": 3}, round_to=4),
        Example(
            inputs={
                "returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02, float("nan"), -0.015, 0.02, 0.01)},
            intro="A ``null`` (which voids every window that spans it) and a ``NaN`` (which propagates to "
            "its windows) make the missing-data handling visible:",
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — a window with no return below the threshold has zero mean "
            "loss, so the ratio is ``+inf``:",
            params={"window": 3},
        ),
        Example(
            inputs={"returns": (1000000000.0, 0.01, 1e-09, 0.0, 0.0)},
            intro="**Degenerate denominator** — a window of two exact zeros at the threshold, reached after "
            "a much larger value slides out, gives a ``0 / 0``, so the ratio is ``NaN``:",
            params={"window": 2},
        ),
    ),
)
