"""
Unit tests for the shared validation/normalization core in ``pomata._expr``.

These helpers are exercised indirectly by every indicator, pnl, and metric test, but they also carry the package's
input contracts (the canonical error messages and bounds), so they are pinned directly here: a drift in a bound or a
message fails this module rather than rippling through the families.
"""

import math

import polars as pl
import pytest

from pomata._expr import (
    float64_expr,
    per_period_rate,
    validate_confidence,
    validate_ddof,
    validate_finite,
    validate_periods_per_year,
    validate_positive,
    validate_window,
    validate_window_order,
)


def test_float64_expr_casts_to_float64() -> None:
    """A ``pl.Expr`` input is returned cast to ``Float64`` regardless of its source dtype."""
    frame = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int64)})
    assert frame.select(float64_expr(pl.col("x")))["x"].dtype == pl.Float64


def test_float64_expr_rejects_bare_string() -> None:
    """A bare column-name string (not a ``pl.Expr``) is rejected with the canonical ``TypeError``."""
    with pytest.raises(TypeError, match=r"expected a Polars expression \(pl\.Expr\), got str"):
        float64_expr("x")  # type: ignore[arg-type]


def test_validate_window_accepts_at_minimum() -> None:
    """A window at the minimum passes; the default minimum is ``1``."""
    validate_window(1)
    validate_window(2, minimum=2)


def test_validate_window_rejects_below_minimum() -> None:
    """A window below the minimum raises, naming the parameter and the bound."""
    with pytest.raises(ValueError, match=r"window must be >= 1, got 0"):
        validate_window(0)
    with pytest.raises(ValueError, match=r"window_fast must be >= 2, got 1"):
        validate_window(1, minimum=2, name="window_fast")


def test_validate_ddof_accepts_below_window() -> None:
    """A ``ddof`` strictly below the window passes (the divisor ``window - ddof`` stays positive)."""
    validate_ddof(0, 3)


def test_validate_ddof_rejects_at_or_above_window() -> None:
    """A ``ddof`` equal to (or above) the window raises."""
    with pytest.raises(ValueError, match=r"ddof must be < window, got ddof=2, window=2"):
        validate_ddof(2, 2)


def test_validate_ddof_rejects_negative() -> None:
    """A negative ``ddof`` (a sign slip) raises a clean ``ValueError``, not an opaque Polars ``OverflowError``."""
    with pytest.raises(ValueError, match=r"ddof must be >= 0, got -1"):
        validate_ddof(-1, 3)


def test_validate_window_order_accepts_ordered_pair() -> None:
    """An ordered (``fast <= slow``) window pair passes, including the equal case."""
    validate_window_order(3, 5)
    validate_window_order(4, 4)


def test_validate_window_order_rejects_unordered_pair() -> None:
    """A fast window larger than the slow window raises, naming both parameters."""
    with pytest.raises(
        ValueError, match=r"windows must be ordered window_fast <= window_slow, got window_fast=5, window_slow=3"
    ):
        validate_window_order(5, 3)


def test_validate_periods_per_year() -> None:
    """``periods_per_year`` must be ``>= 1``."""
    validate_periods_per_year(1)
    with pytest.raises(ValueError, match=r"periods_per_year must be >= 1, got 0"):
        validate_periods_per_year(0)


def test_per_period_rate_geometric() -> None:
    """The per-period rate is the geometric de-annualization ``(1 + r)^(1/P) - 1`` and compounds back."""
    period = per_period_rate(0.05, 252)
    assert math.isclose((1.0 + period) ** 252 - 1.0, 0.05, rel_tol=1e-12)


def test_per_period_rate_rejects_below_minus_one() -> None:
    """A rate below ``-1`` is rejected with the canonical message; ``-1`` is the accepted (total-loss) boundary."""
    with pytest.raises(ValueError, match=r"annual_rate must be >= -1"):
        per_period_rate(-1.5, 252)
    assert per_period_rate(-1.0, 252) == -1.0


def test_per_period_rate_names_the_caller_parameter() -> None:
    """``name`` puts the caller's public parameter in the domain error, not the internal ``annual_rate``."""
    with pytest.raises(ValueError, match=r"risk_free_rate must be >= -1, got -1.5"):
        per_period_rate(-1.5, 252, name="risk_free_rate")


def test_validate_finite() -> None:
    """A finite value passes; ``NaN`` and ``inf`` raise."""
    validate_finite(0.5, "threshold")
    with pytest.raises(ValueError, match=r"threshold must be a finite number, got nan"):
        validate_finite(math.nan, "threshold")
    with pytest.raises(ValueError, match=r"threshold must be a finite number, got inf"):
        validate_finite(math.inf, "threshold")


def test_validate_positive_default_requires_strictly_positive() -> None:
    """Without ``allow_zero``, the value must be finite and strictly positive; ``0`` and non-finite raise."""
    validate_positive(1.0, "rate")
    with pytest.raises(ValueError, match=r"rate must be a finite number > 0, got 0.0"):
        validate_positive(0.0, "rate")
    with pytest.raises(ValueError, match=r"rate must be a finite number > 0, got nan"):
        validate_positive(math.nan, "rate")


def test_validate_positive_allow_zero() -> None:
    """With ``allow_zero``, zero passes but a negative value still raises."""
    validate_positive(0.0, "rate", allow_zero=True)
    with pytest.raises(ValueError, match=r"rate must be a finite number >= 0, got -1.0"):
        validate_positive(-1.0, "rate", allow_zero=True)


def test_validate_confidence() -> None:
    """The confidence level must lie strictly inside the open interval ``(0, 1)``."""
    validate_confidence(0.95)
    with pytest.raises(ValueError, match=r"confidence must be in the open interval \(0, 1\), got 0.0"):
        validate_confidence(0.0)
    with pytest.raises(ValueError, match=r"confidence must be in the open interval \(0, 1\), got 1.0"):
        validate_confidence(1.0)
