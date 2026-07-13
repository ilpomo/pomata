"""Spec for ``pomata.metrics.omega_ratio_rolling`` — the rolling mean gain over mean loss about a threshold,
scale-invariant.
"""

import math
from collections.abc import Sequence

import polars as pl
from tests_new.metrics.oracles import omega_ratio_rolling_reference
from tests_new.support import RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import omega_ratio_rolling

_LOSS_FLOOR = 1e-3


def _windows_conditioned(values: Sequence[float | None], window: int) -> bool:
    """Whether every window's mean loss is either zero or a real fraction of the window magnitude."""
    for index in range(window - 1, len(values)):
        finite = [
            value for value in values[index - window + 1 : index + 1] if value is not None and not math.isnan(value)
        ]
        if not finite:
            continue
        mean_loss = sum(-value for value in finite if value < 0.0) / len(finite)
        scale = max(abs(value) for value in finite) or 1.0
        if 0.0 < mean_loss < scale * _LOSS_FLOOR:
            return False
    return True


def _omega_conditioning(frame: pl.DataFrame) -> bool:
    """Every window's mean loss well-conditioned — the regime the rolling ratio needs."""
    return _windows_conditioned(frame.to_series(0).to_list(), 3)


OMEGA_RATIO_ROLLING = Spec(
    factory=omega_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "threshold": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=omega_ratio_rolling_reference,
    conditioning=_omega_conditioning,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    # A ratio of a rolling mean gain to a rolling mean loss is scale-invariant (by analogy to the reducing omega).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, 2.0, 1.0, 5.0, 2.0, 1.3333),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 2.0),
            reason="when the window equals the series length only the last row is defined "
            "(test_omega_ratio_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, 0.6666666666666665, 0.3999999999999999, 1.5),
            reason="agreement at a non-default threshold "
            "(test_omega_ratio_rolling.py::test_matches_reference_with_threshold)",
            params_override={"window": 3, "threshold": 0.01},
        ),
        SpecPin(
            label="no_downside_window_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None, None, math.inf),
            reason="a window with no return below the threshold has zero mean loss, so +inf "
            "(test_omega_ratio_rolling.py::test_no_downside_window_is_inf)",
        ),
        SpecPin(
            label="no_activity_window_is_nan",
            inputs={"returns": (1000000000.0, 0.01, 1e-09, 0.0, 0.0)},
            expected=(None, math.inf, math.inf, math.inf, math.nan),
            reason="a large-magnitude corner guarding a spurious +inf residue against a correct 0/0=NaN "
            "(test_omega_ratio_rolling.py::test_no_activity_window_is_nan)",
            params_override={"window": 2},
        ),
    ),
)
