"""Declaration for ``pomata.metrics.alpha_rolling`` — Jensen's alpha over a trailing window, scale-exempt."""

import math

import polars as pl

from pomata.metrics import alpha_rolling
from tests.metrics.alpha import ALPHA
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_alpha_rolling
from tests.support.declaration import Golden, Pin, ScaleExempt
from tests.support.strategies import windows_well_conditioned
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE

# Spec-local conditioning floor: the shared 1e-2 rejected ~5 orders of magnitude of well-behaved windows. On the
# benchmark-variance axis the embedded cov/var slope was probed to agree with the oracle down to var_rel 1e-6 and to
# break only at ~1e-8 (measured on treynor_ratio_rolling, the same slope over the same benchmark window; alpha's
# multiply-by-beta is strictly better conditioned than treynor's divide-by-beta), so 1e-5 keeps a 10x margin above
# the last verified-agreeing point while re-admitting three orders of the needlessly excluded band.
_CONDITIONING_FLOOR = 1e-5


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject any trailing window whose benchmark variance is too near zero for the one-pass slope to track."""
    return windows_well_conditioned(frame["benchmark"].to_list(), 4, floor=_CONDITIONING_FLOOR)


ALPHA_ROLLING = suite_metrics(
    factory=alpha_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=ALPHA,
    window="window",
    warmup=3,
    oracle=reference_alpha_rolling,
    scaling=ScaleExempt(
        reason="annualizes an excess leg against a fixed per-period risk-free constant — neither "
        "scale-homogeneous nor scale-invariant"
    ),
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_windows_well_conditioned,
    golden=Golden(
        inputs={
            "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
            "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
        },
        output=(None, None, None, -0.0864, -0.0096, -0.0227, 0.4932, 0.7998),
        params={"window": 4, "periods_per_year": 252},
    ),
    pins=(
        Pin(
            label="null_in_window_is_null",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, None, 0.018)},
            expected=(None, None, None, None, None),
            reason="a null in either leg nulls every window touching it, disjoint rows and all ",
            params_override={"window": 3},
        ),
        Pin(
            label="null_in_constant_benchmark_window_is_null",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a null in the returns leg wins over the constant-benchmark NaN branch on incomplete windows",
            params_override={"window": 3},
        ),
        Pin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a window whose benchmark is exactly constant makes the embedded slope NaN — the exact "
            "core of the near-constant regime the conditioning filter excludes from the property "
            "tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.008, -0.015, 0.025, -0.008)},
            expected=(None, None, None, -0.14222632801543245),
            reason="when the window equals the series length only the last row is defined ",
            params_override={"window": 4},
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
)
