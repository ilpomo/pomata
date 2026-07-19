"""Declaration for ``pomata.metrics.volatility_rolling`` — the annualized rolling sample std, degree-1 homogeneous."""

import polars as pl

from pomata.metrics import volatility_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_volatility_rolling
from tests.metrics.volatility import VOLATILITY
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import windows_well_spread


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant window: the one-pass rolling std cannot track the two-pass oracle there. KEPT at the
    shared cut even though this spec's own onset at unit scale sits ~2.5 orders below it: the true floor here is
    quasi-ABSOLUTE, not scale-squared-relative, so at the domain's low edge (|r| ~0.01) the same shared cut is
    already within ~4x of the real onset — no single tighter constant is safe at both ends of the domain, and the
    shared cut is sized on that low edge.
    """
    return windows_well_spread(frame.to_series(0).to_list(), 4)


VOLATILITY_ROLLING = suite_metrics(
    factory=volatility_rolling,
    inputs=("returns",),
    params={"window": 4, "periods_per_year": 252},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=VOLATILITY,
    window="window",
    warmup=3,
    oracle=reference_volatility_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(({"window": 1}, r"window must be >= 2"), ({"periods_per_year": 0}, r"periods_per_year must be >= 1")),
    conditioning=_windows_well_spread,
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, None, 0.352, 0.3779, 0.2898, 0.2457),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 0.32918080138428485),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="constant_window_is_zero",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(None, None, 0.0, 0.0),
            reason="a constant window has zero dispersion, so the volatility is exactly 0 — the exact core "
            "of the near-constant regime the conditioning filter excludes from the property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Volatility_%28finance%29",
    see_also=(
        ("volatility", "The whole-series reducing form."),
        ("sharpe_ratio_rolling", "The risk-adjusted ratio whose denominator is this."),
        ("downside_deviation_rolling", "The downside-only rolling counterpart."),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`volatility` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        ("Degenerate denominator", "a window of equal returns has zero dispersion, so the result is exactly ``0``."),
        (
            "Stability",
            "the incremental one-pass rolling standard deviation carries running sums, so once a much "
            "larger value exits the window a near-constant remainder (relative spread below the "
            "conditioning floor) can diverge from a fresh two-pass computation — the excluded tail, "
            "reported as computed.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling annualized volatility for each row, the same length as the input. The first "
    "``window - 1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null "
    "values before a result is emitted.",
    raises_prose="ValueError: If ``window < 2``, or if ``periods_per_year < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    examples=(
        Example(
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.02, -0.01, 0.04, -0.03, 0.01, 0.025, -0.02)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own "
            "(the ``NVDA`` group never borrows ``AAPL``'s tail):",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (None, 0.01, -0.02, float("nan"), 0.03, -0.01, 0.02)},
            intro="A leading ``null`` and a later ``NaN`` show the per-window masking, with the result "
            "recovering once both leave the window:",
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant window has zero dispersion, so the volatility is "
            "exactly ``0``:",
            params={"window": 3, "periods_per_year": 252},
        ),
    ),
)
