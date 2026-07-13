"""Spec for ``pomata.metrics.treynor_ratio_rolling`` — annualized excess return per rolling beta, scale-exempt."""

import math
from collections.abc import Sequence

import polars as pl
from tests.metrics.oracles import treynor_ratio_rolling_reference
from tests.support import CONDITIONING_FLOOR, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import treynor_ratio_rolling

_BETA_FLOOR = 5e-2


def _treynor_windows_conditioned(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int
) -> bool:
    """Whether every window has a well-conditioned benchmark variance and a slope bounded away from zero."""
    for index in range(window - 1, len(benchmark)):
        pairs = [
            (value_returns, value_benchmark)
            for value_returns, value_benchmark in zip(
                returns[index - window + 1 : index + 1], benchmark[index - window + 1 : index + 1], strict=True
            )
            if value_returns is not None
            and value_benchmark is not None
            and not math.isnan(value_returns)
            and not math.isnan(value_benchmark)
        ]
        if len(pairs) < 2:
            continue
        window_returns = [pair[0] for pair in pairs]
        window_benchmark = [pair[1] for pair in pairs]
        mean_returns = sum(window_returns) / len(pairs)
        mean_benchmark = sum(window_benchmark) / len(pairs)
        variance = sum((value - mean_benchmark) ** 2 for value in window_benchmark) / len(pairs)
        scale = max(abs(value) for value in window_benchmark) or 1.0
        if variance <= scale * scale * CONDITIONING_FLOOR:
            return False
        covariance = sum(
            (value_returns - mean_returns) * (value_benchmark - mean_benchmark)
            for value_returns, value_benchmark in zip(window_returns, window_benchmark, strict=True)
        ) / len(pairs)
        if abs(covariance / variance) < _BETA_FLOOR:
            return False
    return True


def _treynor_conditioning(frame: pl.DataFrame) -> bool:
    """Every window well-conditioned for the embedded slope — the regime treynor's rolling quotient needs."""
    return _treynor_windows_conditioned(frame["returns"].to_list(), frame["benchmark"].to_list(), 4)


TREYNOR_RATIO_ROLLING = Spec(
    factory=treynor_ratio_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252},
    shape=Shape.SERIES,
    warmup=3,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=treynor_ratio_rolling_reference,
    conditioning=_treynor_conditioning,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    # An annualized excess return over a rolling slope — neither scale-homogeneous nor scale-invariant, mirroring the
    # reducing treynor_ratio twin (verified numerically).
    scale=ScaleExempt(
        reason="an annualized excess return over a rolling slope — neither scale-homogeneous nor scale-invariant"
    ),
    golden_input={
        "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
        "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
    },
    golden_params={"window": 4, "periods_per_year": 252},
    golden_output=(None, None, None, 0.9993, 0.7483, 1.4938, -0.5003, 1.8295),
    pins=(
        SpecPin(
            label="null_in_constant_benchmark_window",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a null in a window yields null (pairwise-complete gate) before the constant-benchmark NaN branch "
            "(tests/metrics/test_treynor_ratio_rolling.py::test_null_in_constant_benchmark_window_is_null)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a window whose benchmark is exactly constant makes the embedded slope NaN "
            "(tests/metrics/test_treynor_ratio_rolling.py::test_constant_benchmark_window_is_nan)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="zero_beta_window_is_inf",
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(None, None, None, math.inf),
            reason="a zero-beta window with a positive excess return gives +inf, reported not clipped "
            "(tests/metrics/test_treynor_ratio_rolling.py::test_zero_beta_window_is_inf)",
            params_override={"window": 4},
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, -0.008, 0.018)},
            expected=(None, None, None, None, 1.2350516405135523),
            reason="window equals series length, so only the last row is defined "
            "(tests/metrics/test_treynor_ratio_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="matches_reference_with_risk_free_rate",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, -0.008, 0.018)},
            expected=(None, None, 1.324869757686746, -0.015994608829144833, 2.7922196417697887),
            reason="a non-default risk-free rate shifts every window's excess, otherwise untested by any generic rung "
            "(tests/metrics/test_treynor_ratio_rolling.py::test_matches_reference_with_risk_free_rate)",
            params_override={"window": 3, "risk_free_rate": 0.02},
        ),
    ),
)
