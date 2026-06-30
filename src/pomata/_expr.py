"""
Shared input-normalization helper for the indicator and pnl factories.

A leaf module: it imports only ``polars`` and nothing in the package imports back into it, so it never participates in a
cycle. Every public factory routes each of its ``pl.Expr`` inputs through :func:`float64_expr` at the top of its body,
which gives the package two guarantees uniformly: a clear, early error when a caller passes a bare column name instead
of an expression, and a single output dtype (``Float64``) regardless of the input's numeric dtype.
"""

import math
from typing import cast

import polars as pl


def float64_expr(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Normalize a factory input to a ``Float64`` expression.

    Validates that ``expr`` is a Polars expression (the factories take ``pl.Expr`` only, never string column names) and
    casts it to ``Float64`` so that every indicator and pnl factory has one predictable output dtype: an ``Int64`` or
    ``Float32`` input yields a ``Float64`` result, exactly as a ``Float64`` input would, with no value drift.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        ``expr`` cast to ``Float64``.

    Raises:
        TypeError: If ``expr`` is not a ``pl.Expr`` (e.g. a bare ``str`` column name was passed).

    Note:
        The cast is a no-op when the input is already ``Float64``. A non-numeric column (e.g. a string column referenced
        by a valid ``pl.Expr``) still fails later, at collection time, where Polars reports the dtype error.
    """
    # Widen to ``object`` so the runtime guard against a caller passing a bare column-name string is a genuine narrowing
    # (not a check the type system deems unnecessary), while the public signature still promises ``pl.Expr``.
    candidate = cast("object", expr)
    if not isinstance(candidate, pl.Expr):
        raise TypeError(
            f'expected a Polars expression (pl.Expr), got {type(candidate).__name__}; pass pl.col("close"), '
            f"not a bare column name"
        )
    return candidate.cast(pl.Float64)


def validate_window(
    window: int,
    minimum: int = 1,
    *,
    name: str = "window",
) -> None:
    """
    Validate a lookback ``window`` against its minimum, raising the canonical message on failure.

    Centralizes the ``window >= minimum`` guard every factory shares, so the bound and the error message stay identical
    across the package and cannot drift as new indicators are added. Most factories accept ``minimum=1``; pass
    ``minimum=2`` where a single-observation window degenerates (e.g. :func:`hma`, the linear-regression family). Pass
    ``name`` to validate a differently-named lookback (e.g. ``window_fast``, ``window_k``) with the same message shape.

    Args:
        window: Number of observations in the moving window.
        minimum: Smallest accepted window.
        name: The parameter name, for the error message.

    Raises:
        ValueError: If ``window < minimum``.
    """
    if window < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {window}")


def validate_ddof(
    ddof: int,
    window: int,
) -> None:
    """
    Validate that the delta degrees of freedom leave a positive divisor, raising the canonical message on failure.

    Centralizes the ``ddof < window`` guard the rolling-variance factories share, so the divisor ``window - ddof``
    stays positive and the bound and its message stay identical across the package.

    Args:
        ddof: Delta degrees of freedom; the divisor is ``window - ddof``.
        window: Number of observations in the moving window.

    Raises:
        ValueError: If ``ddof >= window``.
    """
    if ddof >= window:
        raise ValueError(f"ddof must be < window, got ddof={ddof}, window={window}")


def validate_window_order(
    window_fast: int,
    window_slow: int,
    *,
    fast_name: str = "window_fast",
    slow_name: str = "window_slow",
) -> None:
    """
    Validate that a fast/slow window pair is ordered, raising the canonical message on failure.

    Centralizes the ``window_fast <= window_slow`` guard the dual-window factories share (the oscillator pairs and
    KAMA's fast/slow smoothing windows), so the bound and the message stay identical across the package and cannot
    drift. Pass ``fast_name`` / ``slow_name`` to validate a differently-named pair with the same message shape
    (mirroring :func:`validate_window`'s ``name``).

    Args:
        window_fast: The shorter (faster) window.
        window_slow: The longer (slower) window.
        fast_name: The faster parameter's name, for the error message.
        slow_name: The slower parameter's name, for the error message.

    Raises:
        ValueError: If ``window_fast > window_slow``.
    """
    if window_fast > window_slow:
        raise ValueError(
            f"windows must be ordered {fast_name} <= {slow_name}, "
            f"got {fast_name}={window_fast}, {slow_name}={window_slow}"
        )


def validate_periods_per_year(
    periods_per_year: int,
) -> None:
    """
    Validate the annualization factor, raising the canonical message on failure.

    Centralizes the ``periods_per_year >= 1`` guard shared by every annualized metric, so the bound and the message
    stay identical across the package and cannot drift.

    Args:
        periods_per_year: Observations per year used to annualize.

    Raises:
        ValueError: If ``periods_per_year < 1``.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")


def per_period_rate(
    annual_rate: float,
    periods_per_year: int,
) -> float:
    """
    Convert an annualized rate to its per-period equivalent geometrically.

    The geometric (compounding) conversion ``(1 + annual_rate)^(1 / P) - 1`` the annualized ratios share for their
    risk-free-rate handling (Sharpe, Sortino, alpha, Treynor, ...), so the convention stays identical across the
    package. The caller validates ``periods_per_year`` separately (via :func:`validate_periods_per_year`).

    Args:
        annual_rate: The annualized rate (e.g. a risk-free rate).
        periods_per_year: Observations per year used to de-annualize.

    Returns:
        The per-period rate.

    Raises:
        ValueError: If ``annual_rate < -1`` (then ``1 + annual_rate`` is negative and the fractional power is
            undefined).
    """
    if annual_rate < -1.0:
        raise ValueError(f"annual_rate must be >= -1, got {annual_rate}")
    return math.pow(1.0 + annual_rate, 1.0 / periods_per_year) - 1.0


def validate_finite(
    value: float,
    name: str,
) -> None:
    """
    Validate a scalar tuning parameter is a finite number, raising the canonical message on failure.

    Shared by the metric factories that take a finite scalar knob (``threshold``, ``risk_free_rate``), so a ``NaN`` or
    ``inf`` is rejected at the call site rather than silently poisoning the result.

    Args:
        value: The scalar parameter value.
        name: The parameter name, for the error message.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")


def validate_positive(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    """
    Validate a scalar tuning parameter is a finite, positive number, raising the canonical message on failure.

    Shared by the factories that take a finite magnitude knob (``multiplier``, ``rate``, ``fee``, ``acceleration``, …),
    so a ``NaN``, ``inf``, or out-of-sign value is rejected at the call site rather than silently poisoning the result.
    Pass ``allow_zero=True`` where zero is a valid no-op (e.g. a zero cost ``rate``).

    Args:
        value: The scalar parameter value.
        name: The parameter name, for the error message.
        allow_zero: Whether zero is accepted (``>= 0`` instead of ``> 0``).

    Raises:
        ValueError: If ``value`` is not finite, or is negative (non-positive when ``allow_zero`` is ``False``).
    """
    if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {value}")


def validate_confidence(
    confidence: float,
) -> None:
    """
    Validate a tail confidence level lies strictly inside ``(0, 1)``, raising the canonical message on failure.

    Shared by the historical tail-risk metrics (value-at-risk, conditional value-at-risk), so the open-interval bound
    and its message stay identical across the package.

    Args:
        confidence: The tail confidence level.

    Raises:
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in the open interval (0, 1), got {confidence}")
