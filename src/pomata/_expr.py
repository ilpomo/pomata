"""
Shared input-normalization, validation, and rolling-guard helpers for all three factory families (indicators, metrics,
pnl).

A leaf module: it imports only the standard library and ``polars``, and nothing in the package imports back into it,
so it never participates in a
cycle. Every public factory routes each of its ``pl.Expr`` inputs through :func:`float64_expr` — at the top of its
own body, or transitively through the factory it composes (``max_drawdown`` and friends route via ``drawdown``) —
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
        The cast is a no-op when the input is already ``Float64``. Beware that the cast is *permissive*: a temporal,
        boolean, or numeric-text string column casts silently (a datetime becomes its epoch representation, a Boolean
        ``0.0`` / ``1.0``) and the factory computes garbage without an error — only a column that cannot cast at all
        fails at collection time. Point the factories at genuinely numeric columns.
    """
    # Widen to ``object`` so the runtime guard against a caller passing a bare column-name string is a genuine narrowing
    # (not a check the type system deems unnecessary), while the public signature still promises ``pl.Expr``.
    candidate = cast("object", expr)
    if not isinstance(candidate, pl.Expr):
        raise TypeError(
            f"expected a Polars expression (pl.Expr), got {type(candidate).__name__}; pass a column expression "
            f'like pl.col("<column>"), not a bare column name'
        )
    return candidate.cast(pl.Float64)


def rolling_has_nan(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    """
    Whether the trailing ``window`` holds any ``NaN`` — the rolling counterpart of the whole-series NaN-poison guard.

    A single ``NaN`` anywhere in the window poisons that window's rolling statistic, so the consumers map it to a
    ``NaN`` result rather than a spuriously finite one; detected as the ``rolling_max`` of the ``NaN`` indicator (a
    Boolean that turns ``True`` once any value in the window is ``NaN``). Shared by the rolling dispersions and
    moments across the families so the guard cannot drift between twins.

    Args:
        expr: Input series the rolling statistic is computed over.
        window: Number of observations in the moving window.

    Returns:
        A Boolean expression: ``True`` where the trailing window holds a ``NaN``, ``null`` while it is incomplete.
    """
    return expr.is_nan().rolling_max(window, min_samples=window)


def rolling_is_constant(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    """
    Whether every value in the trailing ``window`` is identical — a zero-variance (degenerate) window.

    Compared as ``rolling_max == rolling_min`` (exact and scale-invariant, no epsilon), so it fires only on a
    bit-identical window: exactly the case where the one-pass central moments or the incremental rolling standard
    deviation leave a cancellation residue instead of an exact zero. Beware that Polars groups ``NaN == NaN`` as
    equal, so a NaN-poisoned window also compares constant — consumers check :func:`rolling_has_nan` first.

    Args:
        expr: Input series the rolling statistic is computed over.
        window: Number of observations in the moving window.

    Returns:
        A Boolean expression: ``True`` where the trailing window is bit-constant, ``null`` while it is incomplete.
    """
    return expr.rolling_max(window, min_samples=window) == expr.rolling_min(window, min_samples=window)


def rolling_mean_exact(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    """
    A rolling mean that returns the exact constant on a bit-constant window — no incremental residue.

    The native incremental rolling mean can carry a floating-point residue after a much larger value slides out of
    the window, so a window of identical values (most dangerously: all exact zeros) reports a tiny non-zero mean.
    Above a denominator pinned to exact zero by :func:`rolling_is_constant` (the rolling ratio kernels — Sharpe,
    Sortino, information ratio), that residue flips the documented ``0/0 -> NaN`` degeneracy into a spurious
    ``residue/0 -> +/-inf``. Pinning the mean of a bit-constant window to the constant itself closes the numerator
    leg of the residue class the same way the rolling dispersions already close the denominator leg.

    Args:
        expr: Input series the rolling mean is computed over.
        window: Number of observations in the moving window.

    Returns:
        The rolling mean, exact on bit-constant windows, ``null`` while the window is incomplete.
    """
    # The constancy test inlines :func:`rolling_is_constant` (``rolling_max == rolling_min``, exact, same
    # NaN-grouping caveat) so the window minimum is computed once and reused as the exact fill value.
    low = expr.rolling_min(window, min_samples=window)
    return (
        pl.when(expr.rolling_max(window, min_samples=window) == low)
        .then(low)
        .otherwise(expr.rolling_mean(window, min_samples=window))
    )


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
    Validate that the delta degrees of freedom are non-negative and leave a positive divisor, raising the canonical
    message on failure.

    Centralizes the ``0 <= ddof < window`` bound the rolling-variance factories share, so ``ddof`` is a valid count and
    the divisor ``window - ddof`` stays positive, and the bounds and their messages stay identical across the package.

    Args:
        ddof: Delta degrees of freedom; the divisor is ``window - ddof``.
        window: Number of observations in the moving window.

    Raises:
        ValueError: If ``ddof < 0`` or ``ddof >= window``.
    """
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
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
    *,
    name: str = "annual_rate",
) -> float:
    """
    Convert an annualized rate to its per-period equivalent geometrically.

    The geometric (compounding) conversion ``(1 + annual_rate)^(1 / P) - 1`` the annualized ratios share for their
    risk-free-rate handling (Sharpe, Sortino, alpha, Treynor, ...), so the convention stays identical across the
    package. The caller validates ``periods_per_year`` separately (via :func:`validate_periods_per_year`); pass ``name``
    so the domain error names the caller's public parameter (e.g. ``risk_free_rate``) rather than the internal one.

    Args:
        annual_rate: The annualized rate (e.g. a risk-free rate).
        periods_per_year: Observations per year used to de-annualize.
        name: The caller's parameter name, for the error message.

    Returns:
        The per-period rate.

    Raises:
        ValueError: If ``annual_rate < -1`` (then ``1 + annual_rate`` is negative and the fractional power is
            undefined); the message names ``name``.
    """
    if annual_rate < -1.0:
        raise ValueError(f"{name} must be >= -1, got {annual_rate}")
    return math.pow(1.0 + annual_rate, 1.0 / periods_per_year) - 1.0


def validate_finite(
    value: float,
    name: str,
) -> None:
    """
    Validate a scalar tuning parameter is a finite number, raising the canonical message on failure.

    Shared by the factories that take a finite scalar knob (the metrics' ``threshold`` and ``risk_free_rate``, the
    indicators' ``volume_factor``), so a ``NaN`` or ``inf`` is rejected at the call site rather than silently
    poisoning the result.

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

    Shared by the factories that take a finite magnitude knob (``multiplier``, ``rate``, ``fee``, …), so a ``NaN``,
    ``inf``, or out-of-sign value is rejected at the call site rather than silently poisoning the result.
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

    Shared by the tail metrics (the value-at-risk family — historical, parametric, modified, rolling — the
    conditional value-at-risk, and the conditional drawdown-at-risk), so the open-interval bound
    and its message stay identical across the package.

    Args:
        confidence: The tail confidence level.

    Raises:
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in the open interval (0, 1), got {confidence}")


def validate_unit_fraction(
    value: float,
    name: str,
) -> None:
    """
    Validate a scalar lies in the half-open unit interval ``(0, 1]``, raising the canonical message on failure.

    Shared by the factories whose smoothing / acceleration weight is a fraction of one (the parabolic SAR's
    acceleration factor and its cap, MAMA's fast / slow alpha limits), so the bound and its message stay identical
    across the package. Zero is excluded (a zero weight is a no-op) and one is included (the maximal weight); a ``NaN``
    or ``inf`` is rejected at the call site.

    Args:
        value: The scalar parameter value.
        name: The parameter name, for the error message.

    Raises:
        ValueError: If ``value`` is not in the half-open interval ``(0, 1]`` (i.e. ``<= 0``, ``> 1``, ``NaN``, or
            ``inf``).
    """
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the half-open interval (0, 1], got {value}")
