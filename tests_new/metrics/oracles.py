"""
Naive reference oracles for the metrics family — one per public function, gathered in a single module.

Each function recomputes one metric from scratch in plain Python from its canonical definition, sharing no code with
the Polars implementation it certifies, so agreement is evidence of correctness rather than coincidence. A rolling
metric's oracle is its reducing oracle applied to each trailing window (the shared :func:`rolling_reference` /
:func:`rolling_reference_pair` wrappers), and the tail-risk oracles share one type-7 empirical quantile
(:func:`type_seven_quantile`). Named ``reference_{function}`` so the declaration's binding guard can tie each to the
factory it checks.
"""

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist


def type_seven_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """
    The type-7 (linear) empirical quantile of a non-empty ascending sequence.

    Uses the virtual index ``h = probability * (n - 1)`` and linearly interpolates between the bracketing order
    statistics ``x[floor(h)]`` and ``x[ceil(h)]`` -- the Hyndman & Fan type-7 estimator shared by numpy, pandas, and
    Polars' ``"linear"`` interpolation.

    Args:
        sorted_values: The observations in ascending order; must be non-empty.
        probability: The quantile probability in ``[0, 1]``.

    Returns:
        The interpolated quantile value.
    """
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = probability * (count - 1)
    lower = math.floor(position)
    upper = min(lower + 1, count - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def rolling_reference_pair(
    reduce: Callable[[list[float | None], list[float | None]], float | None],
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Apply a two-input reducing reference ``reduce`` to each trailing window of the ``(returns, benchmark)`` pair.

    The benchmark-relative rolling policy: warm-up and any window holding a ``None`` in either leg are ``None``;
    otherwise the reducing relative oracle is recomputed over the window (it already handles ``NaN`` and the degenerate
    cases).

    Args:
        reduce: The reducing relative oracle, called as ``reduce(window_returns, window_benchmark)`` on a clean window.
        returns: The portfolio return series (may contain ``None`` and ``float('nan')``).
        benchmark: The benchmark return series, aligned with ``returns``.
        window: The trailing window length.

    Returns:
        A list the same length as the inputs: ``None`` for warm-up or any-null windows, else ``reduce`` of the window.
    """
    output: list[float | None] = []
    for index in range(len(returns)):
        if index < window - 1:
            output.append(None)
            continue
        window_returns = list(returns[index - window + 1 : index + 1])
        window_benchmark = list(benchmark[index - window + 1 : index + 1])
        if any(value is None for value in window_returns) or any(value is None for value in window_benchmark):
            output.append(None)
            continue
        output.append(reduce(window_returns, window_benchmark))
    return output


def rolling_reference(
    reduce: Callable[[list[float | None]], float | None],
    values: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Apply a reducing reference ``reduce`` to each trailing window of ``values``.

    Args:
        reduce: The reducing oracle bound to its parameters, called as ``reduce(window_slice)`` on a clean window.
        values: The input series (may contain ``None`` and ``float('nan')``).
        window: The trailing window length.

    Returns:
        A list the same length as ``values``: ``None`` for warm-up or any-null windows, else ``reduce`` of the window.
    """
    output: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            output.append(None)
            continue
        window_slice = list(values[index - window + 1 : index + 1])
        if any(value is None for value in window_slice):
            output.append(None)
            continue
        output.append(reduce(window_slice))
    return output


def reference_adjusted_sharpe_ratio(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive adjusted Sharpe ratio over a Python list.

    The Pezier & White correction ``ASR_p = SR_p * (1 + skew/6 * SR_p - excess_kurt/24 * SR_p**2)`` applied to the
    per-period excess Sharpe ratio ``SR_p`` and the population skewness / excess kurtosis, then annualized by
    ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for
    :func:`pomata.metrics.adjusted_sharpe_ratio`. ``None`` returns are skipped; with fewer than two the result is
    ``None``; a ``nan`` anywhere poisons the result to ``nan``; zero dispersion yields ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    count = len(excess)
    mean_excess = sum(excess) / count
    variance = sum((value - mean_excess) ** 2 for value in excess) / (count - 1)
    if variance == 0.0:
        return math.nan
    sharpe_per_period = mean_excess / math.sqrt(variance)
    mean_return = sum(observations) / count
    second_moment = sum((value - mean_return) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean_return) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean_return) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    excess_kurtosis = fourth_moment / (second_moment * second_moment) - 3.0
    adjusted = sharpe_per_period * (
        1.0 + skewness / 6.0 * sharpe_per_period - excess_kurtosis / 24.0 * sharpe_per_period**2
    )
    return adjusted * math.sqrt(periods_per_year)


def reference_alpha(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive annualized Jensen's alpha over two Python lists.

    The per-period mean of ``(r - rf) - beta * (b - rf)`` compounded by ``(1 + .) ** P - 1``, where ``beta`` is the
    independent :func:`reference_beta` slope and the per-period risk-free rate is
    ``(1 + risk_free_rate) ** (1 / P) - 1`` -- recomputed from scratch as the oracle for :func:`pomata.metrics.alpha`.
    The series are pairwise-complete: a pair
    contributes only where both legs are present; with fewer than two such pairs the result is ``None`` (taking
    precedence over poisoning); otherwise a ``nan`` in either leg of a retained pair poisons the result to ``nan``. An
    overflow of the annualizing power is reported as the IEEE infinity, matching the implementation's float64 result.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    slope = reference_beta(returns, benchmark)
    assert slope is not None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess_leg = [(x - rf_period) - slope * (y - rf_period) for x, y in pairs]
    base = 1.0 + sum(excess_leg) / len(excess_leg)
    try:
        growth = math.pow(base, periods_per_year)
    except OverflowError:
        growth = math.inf if (base > 0.0 or periods_per_year % 2 == 0) else -math.inf
    return growth - 1.0


def reference_alpha_rolling(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> list[float | None]:
    """
    Naive rolling annualized alpha: the reducing reference applied to each trailing returns/benchmark window
    (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: reference_alpha(
            window_returns, window_benchmark, periods_per_year, risk_free_rate
        ),
        returns,
        benchmark,
        window,
    )


def reference_beta(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> float | None:
    """
    Naive beta (regression slope) over two Python lists.

    The population covariance of the portfolio and benchmark returns over the benchmark variance --
    ``cov(r, b) / var(b)`` -- recomputed from scratch as the oracle for :func:`pomata.metrics.beta`. The series are
    pairwise-complete: a pair contributes only where both legs are present; with fewer than two such pairs the result is
    ``None`` (taking precedence over poisoning); otherwise a ``nan`` in either leg of a retained pair poisons the result
    to ``nan``. A constant (zero-variance) benchmark gives ``0 / 0`` and is reported as ``nan``, detected exactly via
    ``min == max`` rather than the two-pass variance (whose float residual is not reliably zero for every constant).
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    benchmark_values = [y for _, y in pairs]
    if max(benchmark_values) == min(benchmark_values):
        return math.nan
    count = len(pairs)
    mean_returns = sum(x for x, _ in pairs) / count
    mean_benchmark = sum(y for _, y in pairs) / count
    covariance = sum((x - mean_returns) * (y - mean_benchmark) for x, y in pairs) / count
    variance = sum((y - mean_benchmark) ** 2 for _, y in pairs) / count
    if variance == 0.0:
        return math.nan if covariance == 0.0 else math.copysign(math.inf, covariance)
    return covariance / variance


def reference_beta_rolling(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int
) -> list[float | None]:
    """
    Naive rolling beta: the reducing reference applied to each trailing returns/benchmark window (warm-up / any-null
    windows are ``None``).
    """
    return rolling_reference_pair(reference_beta, returns, benchmark, window)


def reference_burke_ratio(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive Burke ratio over a Python list.

    The excess compound annual growth rate divided by the square root of the sum of squared drawdowns --
    ``(CAGR - risk_free_rate) / sqrt(sum(D_i**2))`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.burke_ratio` by composing the independent :func:`reference_cagr` and
    :func:`reference_drawdown`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``. A drawdown-free curve gives ``+/-inf`` (or ``nan`` when the excess growth is also zero).
    """
    growth = reference_cagr(equity_curve, periods_per_year)
    if growth is None:
        return None
    if math.isnan(growth):
        return math.nan
    observations = [value for value in equity_curve if value is not None]
    declines = [value for value in reference_drawdown(observations) if value is not None]
    denominator = math.sqrt(sum(value * value for value in declines))
    excess_growth = growth - risk_free_rate
    if denominator == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / denominator


def reference_cagr(equity_curve: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive compound annual growth rate over a Python list.

    ``final ** (periods_per_year / n) - 1`` where ``final`` is the last non-null equity and ``n`` the count of non-null
    observations, recomputed from scratch as the oracle for :func:`pomata.metrics.cagr`. ``None`` equities are skipped;
    a ``nan`` anywhere poisons the result to ``nan``; with no defined observations the result is ``None``.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    defined = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in defined):
        return math.nan
    if not defined:
        return None
    return math.pow(defined[-1], periods_per_year / len(defined)) - 1.0


def reference_cagr_rolling(
    equity_curve: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    Naive rolling compound annual growth rate: the window's endpoint ratio annualized, recomputed from scratch.

    An endpoint quantity: row ``i`` is ``(E[i] / E[i - window + 1]) ** (periods_per_year / (window - 1)) - 1`` -- the
    two endpoints span ``window - 1`` periods, so the ratio is annualized over ``window - 1``. The first ``window - 1``
    rows are warm-up ``None``; a ``None`` at either endpoint yields ``None``; a ``NaN`` at either endpoint yields
    ``nan``. An interior ``None`` / ``NaN`` does not affect the result.
    """
    output: list[float | None] = []
    for index in range(len(equity_curve)):
        if index < window - 1:
            output.append(None)
            continue
        first = equity_curve[index - window + 1]
        last = equity_curve[index]
        if first is None or last is None:
            output.append(None)
        elif math.isnan(first) or math.isnan(last):
            output.append(math.nan)
        else:
            output.append(math.pow(last / first, periods_per_year / (window - 1)) - 1.0)
    return output


def reference_calmar_ratio(equity_curve: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive Calmar ratio over a Python list.

    The compound annual growth rate divided by the magnitude of the maximum drawdown -- ``CAGR / |MDD|`` -- recomputed
    from scratch as the oracle for :func:`pomata.metrics.calmar_ratio` by composing the independent
    :func:`reference_cagr` and :func:`reference_max_drawdown`. ``None`` equities are skipped; a ``nan`` anywhere poisons
    it to ``nan``; with no defined observations the result is ``None``. A drawdown-free (monotonic) curve gives
    ``+/-inf`` (or ``nan`` when the growth is zero), matching the implementation's division.
    """
    growth = reference_cagr(equity_curve, periods_per_year)
    drawdown_trough = reference_max_drawdown(equity_curve)
    if growth is None or drawdown_trough is None:
        return None
    if math.isnan(growth) or math.isnan(drawdown_trough):
        return math.nan
    denominator = abs(drawdown_trough)
    if denominator == 0.0:
        return math.nan if growth == 0.0 else math.copysign(math.inf, growth)
    return growth / denominator


def reference_capture_downside_ratio(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive downside capture ratio over two Python lists.

    The geometric annualized portfolio return over the geometric annualized benchmark return, computed over the periods
    where the benchmark return is negative -- ``(prod(1 + r) ** (P / n) - 1) / (prod(1 + b) ** (P / n) - 1)`` over those
    ``n`` periods -- recomputed from scratch as the oracle for :func:`pomata.metrics.capture_downside_ratio`. The series
    are pairwise-complete: a pair contributes only where both legs are present; with no complete pairs the result is
    ``None``; otherwise a ``nan`` in either leg of a retained pair poisons the result to ``nan`` (taking precedence over
    an empty down-market). With no down-market period the result is ``None``. A selected pair with a wiped-out leg
    (``1 + x <= 0`` on either side) is outside the geometric domain and yields ``nan``, matching the implementation's
    domain guard. A zero annualized benchmark loss gives ``+/-inf`` (or ``nan``), matching the implementation's
    division.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    selected = [(x, y) for x, y in pairs if y < 0.0]
    if not selected:
        return None
    if any(1.0 + x <= 0.0 or 1.0 + y <= 0.0 for x, y in selected):
        return math.nan
    count = len(selected)
    portfolio_growth = math.prod(1.0 + x for x, _ in selected) ** (periods_per_year / count) - 1.0
    benchmark_growth = math.prod(1.0 + y for _, y in selected) ** (periods_per_year / count) - 1.0
    if benchmark_growth == 0.0:
        return math.nan if portfolio_growth == 0.0 else math.copysign(math.inf, portfolio_growth)
    return portfolio_growth / benchmark_growth


def reference_capture_ratio(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive capture ratio over two Python lists.

    The :func:`reference_capture_upside_ratio` divided by the :func:`reference_capture_downside_ratio`, recomputed from
    scratch as the oracle for :func:`pomata.metrics.capture_ratio` by composing the two independent capture references.
    Following the implementation's IEEE division semantics: an undefined upside or downside capture (``None``)
    propagates to ``None``; a ``nan`` in either propagates to ``nan``; a (possibly signed) zero downside capture gives
    ``+/-inf`` whose sign follows the signs of both operands (or ``nan`` when the upside capture is also zero).
    """
    upside = reference_capture_upside_ratio(returns, benchmark, periods_per_year)
    downside = reference_capture_downside_ratio(returns, benchmark, periods_per_year)
    if upside is None or downside is None:
        return None
    if math.isnan(upside) or math.isnan(downside):
        return math.nan
    if downside == 0.0:
        if upside == 0.0:
            return math.nan
        return math.copysign(math.inf, math.copysign(1.0, upside) * math.copysign(1.0, downside))
    return upside / downside


def reference_capture_upside_ratio(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive upside capture ratio over two Python lists.

    The geometric annualized portfolio return over the geometric annualized benchmark return, computed over the periods
    where the benchmark return is positive -- ``(prod(1 + r) ** (P / n) - 1) / (prod(1 + b) ** (P / n) - 1)`` over those
    ``n`` periods -- recomputed from scratch as the oracle for :func:`pomata.metrics.capture_upside_ratio`. The series
    are pairwise-complete: a pair contributes only where both legs are present; with no complete pairs the result is
    ``None``; otherwise a ``nan`` in either leg of a retained pair poisons the result to ``nan`` (taking precedence over
    an empty up-market). With no up-market period the result is ``None``. A selected pair with a wiped-out leg
    (``1 + x <= 0`` on either side) is outside the geometric domain and yields ``nan``, matching the implementation's
    domain guard. A zero annualized benchmark gain gives ``+/-inf`` (or ``nan``), matching the implementation's
    division.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    selected = [(x, y) for x, y in pairs if y > 0.0]
    if not selected:
        return None
    if any(1.0 + x <= 0.0 or 1.0 + y <= 0.0 for x, y in selected):
        return math.nan
    count = len(selected)
    portfolio_growth = math.prod(1.0 + x for x, _ in selected) ** (periods_per_year / count) - 1.0
    benchmark_growth = math.prod(1.0 + y for _, y in selected) ** (periods_per_year / count) - 1.0
    if benchmark_growth == 0.0:
        return math.nan if portfolio_growth == 0.0 else math.copysign(math.inf, portfolio_growth)
    return portfolio_growth / benchmark_growth


def reference_common_sense_ratio(returns: Sequence[float | None]) -> float | None:
    """
    Naive common sense ratio over a Python list.

    The product of the profit factor and the tail ratio, recomputed from scratch as the oracle for
    :func:`pomata.metrics.common_sense_ratio` by composing the independent :func:`reference_profit_factor` and
    :func:`reference_tail_ratio`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with
    no observations the result is ``None``. It inherits the ``+inf`` / ``nan`` degeneracies of its two factors.
    """
    factor = reference_profit_factor(returns)
    tail = reference_tail_ratio(returns)
    if factor is None or tail is None:
        return None
    if math.isnan(factor) or math.isnan(tail):
        return math.nan
    return factor * tail


def reference_conditional_drawdown_at_risk(equity_curve: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive conditional drawdown at risk (Rockafellar-Uryasev tail average) over a Python list.

    The Rockafellar-Uryasev tail average of the drawdown series: with ``k = (1 - confidence) * n``, the worst
    ``floor(k)`` order statistics are summed in full and the next carries the fractional weight ``k - floor(k)``,
    divided by ``k``. Recomputed from scratch as the oracle for :func:`pomata.metrics.conditional_drawdown_at_risk`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(value for value in reference_drawdown(observations) if value is not None)
    n = len(ascending)
    k = (1.0 - confidence) * n
    floor_k = math.floor(k)
    total = sum(ascending[:floor_k])
    fraction = k - floor_k
    if floor_k < n and fraction > 0.0:
        total += fraction * ascending[floor_k]
    return total / k


def reference_conditional_value_at_risk(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive historical conditional value-at-risk (Rockafellar-Uryasev expected shortfall) over a Python list.

    The Rockafellar-Uryasev tail average of the non-null returns: with ``k = (1 - confidence) * n``, the worst
    ``floor(k)`` order statistics are summed in full and the next carries the fractional weight ``k - floor(k)``,
    divided by ``k``. Recomputed from scratch as the oracle for :func:`pomata.metrics.conditional_value_at_risk`.
    ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(observations)
    n = len(ascending)
    k = (1.0 - confidence) * n
    floor_k = math.floor(k)
    total = sum(ascending[:floor_k])
    fraction = k - floor_k
    if floor_k < n and fraction > 0.0:
        total += fraction * ascending[floor_k]
    return total / k


def reference_downside_deviation(
    returns: Sequence[float | None],
    periods_per_year: int,
    threshold: float = 0.0,
) -> float | None:
    """
    Naive annualized downside deviation over a Python list.

    The root-mean-square of the below-threshold shortfall ``min(r - threshold, 0)`` over all non-null returns,
    annualized by ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for
    :func:`pomata.metrics.downside_deviation`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no observations the result is ``None``.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold}")
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    shortfalls = [min(value - threshold, 0.0) for value in observations]
    mean_square = sum(shortfall * shortfall for shortfall in shortfalls) / len(observations)
    return math.sqrt(mean_square) * math.sqrt(periods_per_year)


def reference_downside_deviation_rolling(
    values: Sequence[float | None], window: int, periods_per_year: int, threshold: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(
        lambda window_slice: reference_downside_deviation(window_slice, periods_per_year, threshold), values, window
    )


def reference_drawdown(equity_curve: Sequence[float | None]) -> list[float | None]:
    """
    Naive running drawdown over a Python list.

    For each row, ``equity / running_peak - 1`` where the running peak is the maximum of the non-null, non-nan equities
    seen so far, recomputed from scratch as the oracle for :func:`pomata.metrics.drawdown`. A ``None`` equity yields
    ``None`` (the peak carries across it); a ``nan`` equity yields ``nan`` and is ignored by the running peak (matching
    Polars' ``cum_max``), so later rows are unaffected.
    """
    result: list[float | None] = []
    running_peak: float | None = None
    for value in equity_curve:
        if value is None:
            result.append(None)
            continue
        if math.isnan(value):
            result.append(math.nan)
            continue
        running_peak = value if running_peak is None else max(running_peak, value)
        result.append(value / running_peak - 1)
    return result


def reference_drawdown_rolling(equity_curve: Sequence[float | None], window: int) -> list[float | None]:
    """
    Naive rolling drawdown: the current equity over the trailing window's peak, less one, recomputed from scratch.

    Row ``i`` is ``E[i] / max(E[i - window + 1 : i + 1]) - 1`` over the window. The first ``window - 1`` rows are
    warm-up ``None``; a window holding any ``None`` is ``None`` (the window must hold ``window`` non-null values);
    otherwise a ``NaN`` anywhere in the window yields ``nan``.
    """
    output: list[float | None] = []
    for index in range(len(equity_curve)):
        if index < window - 1:
            output.append(None)
            continue
        window_slice = equity_curve[index - window + 1 : index + 1]
        finite = [value for value in window_slice if value is not None]
        if len(finite) < window:
            output.append(None)
        elif any(math.isnan(value) for value in finite):
            output.append(math.nan)
        else:
            output.append(finite[-1] / max(finite) - 1.0)
    return output


def reference_gain_to_pain_ratio(returns: Sequence[float | None]) -> float | None:
    """
    Naive gain to pain ratio over a Python list.

    The sum of all returns over the magnitude of the sum of the negative returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.gain_to_pain_ratio`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result
    to ``nan``; with no observations the result is ``None``; with no losses the result is ``+inf`` (or ``nan`` when the
    net return is also zero), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    net = sum(observations)
    pain = sum(-value for value in observations if value < 0.0)
    if pain == 0.0:
        return math.inf if net > 0.0 else math.nan
    return net / pain


def reference_information_ratio(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive annualized information ratio over two Python lists.

    The mean active return (portfolio minus benchmark) over its sample standard deviation (``ddof = 1``, the tracking
    error), annualized by ``sqrt(P)`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.information_ratio`. The series are pairwise-complete: a pair contributes only where both legs
    are present; with fewer than two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a
    ``nan`` in either leg of a retained pair poisons the result to ``nan``. A zero tracking error gives ``+/-inf`` (or
    ``nan`` when the mean active is also zero), matching the implementation's division; an exactly-constant active
    series is detected via ``min == max`` rather than the two-pass deviation (whose float residual is not reliably
    zero for every constant), matching the implementation's exact zero-dispersion pin.
    """
    active = [x - y for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(active) < 2:
        return None
    if any(math.isnan(value) for value in active):
        return math.nan
    if max(active) == min(active):
        return math.nan if active[0] == 0.0 else math.copysign(math.inf, active[0])
    count = len(active)
    mean_active = sum(active) / count
    tracking_error = math.sqrt(sum((value - mean_active) ** 2 for value in active) / (count - 1))
    if tracking_error == 0.0:
        return math.nan if mean_active == 0.0 else math.copysign(math.inf, mean_active)
    return mean_active / tracking_error * math.sqrt(periods_per_year)


def reference_information_ratio_rolling(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    Naive rolling information ratio: the reducing reference applied to each trailing returns/benchmark window
    (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: reference_information_ratio(
            window_returns, window_benchmark, periods_per_year
        ),
        returns,
        benchmark,
        window,
    )


def reference_kelly_criterion(returns: Sequence[float | None]) -> float | None:
    """
    Naive Kelly criterion over a Python list.

    The growth-optimal fraction ``p - (1 - p) / W`` from the win rate ``p`` and the payoff ratio ``W``, recomputed from
    scratch as the oracle for :func:`pomata.metrics.kelly_criterion` by composing the independent
    :func:`reference_win_rate` and :func:`reference_payoff_ratio`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with the win rate or payoff ratio undefined the result is ``None``.
    """
    probability = reference_win_rate(returns)
    payoff = reference_payoff_ratio(returns)
    if probability is None or payoff is None:
        return None
    if math.isnan(probability) or math.isnan(payoff):
        return math.nan
    return probability - (1.0 - probability) / payoff


def reference_kurtosis(returns: Sequence[float | None]) -> float | None:
    """
    Naive population excess (Fisher) kurtosis over a Python list.

    ``m4 / m2**2 - 3`` from the population central moments of the non-null returns, recomputed from scratch as the
    oracle for :func:`pomata.metrics.kurtosis`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no observations the result is ``None``; a zero-variance series (constant, or a single value) gives
    ``0 / 0`` and the result is ``nan``, as does a subnormal-magnitude series whose squared variance underflows to zero
    (matching the implementation, which yields ``nan`` there too).
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    count = len(observations)
    mean = sum(observations) / count
    second_moment = sum((value - mean) ** 2 for value in observations) / count
    denominator = math.pow(second_moment, 2.0)
    if denominator == 0.0:
        return math.nan
    fourth_moment = sum((value - mean) ** 4 for value in observations) / count
    return fourth_moment / denominator - 3.0


def reference_kurtosis_rolling(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(reference_kurtosis, values, window)


def reference_max_drawdown(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive maximum drawdown over a Python list: the minimum of the running drawdown.

    Built on :func:`reference_drawdown`. ``None`` drawdowns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no defined drawdown (an all-null series) the result is ``None``; otherwise the most negative
    drawdown.
    """
    declines = [value for value in reference_drawdown(equity_curve) if value is not None]
    if any(math.isnan(value) for value in declines):
        return math.nan
    if not declines:
        return None
    return min(declines)


def reference_max_drawdown_duration(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive maximum drawdown duration over a Python list.

    The longest run of consecutive observations strictly below a prior peak, recomputed from scratch as the oracle for
    :func:`pomata.metrics.max_drawdown_duration` (returned as a ``float`` count of bars over the non-null equity).
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    peak = observations[0]
    longest = 0
    current = 0
    for value in observations:
        peak = max(peak, value)
        if value < peak:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


def reference_modigliani_risk_adjusted_performance(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive Modigliani risk-adjusted performance (M-squared) over two Python lists.

    The risk-free rate plus the portfolio :func:`reference_sharpe_ratio` ratio scaled by the benchmark
    :func:`reference_volatility` -- ``rf + SR * sigma_b`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.modigliani_risk_adjusted_performance` by composing the two independent single-input
    references. The series are pairwise-complete: a pair contributes only where both legs are present; with fewer than
    two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a ``nan`` in either leg of a
    retained pair poisons the result to ``nan``. A constant portfolio gives an infinite Sharpe ratio, which propagates.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    sharpe_ratio = reference_sharpe_ratio([x for x, _ in pairs], periods_per_year, risk_free_rate)
    benchmark_volatility = reference_volatility([y for _, y in pairs], periods_per_year)
    assert sharpe_ratio is not None
    assert benchmark_volatility is not None
    return risk_free_rate + sharpe_ratio * benchmark_volatility


def reference_omega_ratio(returns: Sequence[float | None], threshold: float) -> float | None:
    """
    Naive omega ratio over a Python list.

    The mean gain above the threshold divided by the mean loss below it -- ``E[max(r - tau, 0)] / E[max(tau - r, 0)]``
    over the non-null returns, recomputed from scratch as the oracle for :func:`pomata.metrics.omega_ratio`. ``None``
    returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is ``None``.
    With no downside the ratio is ``+inf`` (or ``nan`` when there is also no upside), matching the implementation's
    division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    excess = [value - threshold for value in observations]
    count = len(excess)
    mean_gain = sum(value for value in excess if value > 0.0) / count
    mean_loss = sum(-value for value in excess if value < 0.0) / count
    if mean_loss == 0.0:
        return math.nan if mean_gain == 0.0 else math.inf
    return mean_gain / mean_loss


def reference_omega_ratio_rolling(
    values: Sequence[float | None], window: int, threshold: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: reference_omega_ratio(window_slice, threshold), values, window)


def reference_pain_index(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive pain index over a Python list.

    The mean absolute drawdown of the non-null equity, recomputed from scratch as the oracle for
    :func:`pomata.metrics.pain_index` by averaging the magnitudes of the independent :func:`reference_drawdown`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    declines = reference_drawdown(observations)
    return sum(abs(value) for value in declines if value is not None) / len(declines)


def reference_pain_ratio(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive pain ratio over a Python list.

    The excess compound annual growth rate divided by the pain index -- ``(CAGR - risk_free_rate) / PI`` -- recomputed
    from scratch as the oracle for :func:`pomata.metrics.pain_ratio` by composing the independent :func:`reference_cagr`
    and :func:`reference_pain_index`. ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``;
    with no observations the result is ``None``. A drawdown-free curve has a zero pain index and gives ``+/-inf`` (or
    ``nan`` when the excess growth is also zero).
    """
    growth = reference_cagr(equity_curve, periods_per_year)
    pain = reference_pain_index(equity_curve)
    if growth is None or pain is None:
        return None
    if math.isnan(growth) or math.isnan(pain):
        return math.nan
    excess_growth = growth - risk_free_rate
    if pain == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / pain


def reference_payoff_ratio(returns: Sequence[float | None]) -> float | None:
    """
    Naive payoff ratio over a Python list.

    The mean of the positive returns over the magnitude of the mean of the negative returns, recomputed from scratch as
    the oracle for :func:`pomata.metrics.payoff_ratio`. ``None`` returns are skipped; a ``nan`` anywhere poisons the
    result to ``nan``; with no winning returns or no losing returns the result is ``None`` (one side is undefined).
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    wins = [value for value in observations if value > 0.0]
    losses = [value for value in observations if value < 0.0]
    if not wins or not losses:
        return None
    average_win = sum(wins) / len(wins)
    average_loss = sum(losses) / len(losses)
    return average_win / -average_loss


def reference_probabilistic_sharpe_ratio(
    returns: Sequence[float | None], periods_per_year: int, benchmark_sharpe: float, risk_free_rate: float
) -> float | None:
    """
    Naive probabilistic Sharpe ratio over a Python list.

    The Bailey & López de Prado statistic ``Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR**2))``,
    recomputed from scratch as the oracle for :func:`pomata.metrics.probabilistic_sharpe_ratio`. ``SR`` is the
    non-annualized excess Sharpe ratio, ``skew`` the population skewness, and ``kurt`` the population (non-excess)
    kurtosis. ``None`` returns are skipped; with fewer than two the result is ``None``; a ``nan`` anywhere poisons the
    result to ``nan``; zero dispersion or a non-positive inner variance yields ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    count = len(excess)
    mean_excess = sum(excess) / count
    variance = sum((value - mean_excess) ** 2 for value in excess) / (count - 1)
    if variance == 0.0:
        return math.nan
    sharpe_ratio = mean_excess / math.sqrt(variance)
    mean_return = sum(observations) / count
    second_moment = sum((value - mean_return) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean_return) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean_return) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    raw_kurtosis = fourth_moment / (second_moment * second_moment)
    inner = 1.0 - skewness * sharpe_ratio + (raw_kurtosis - 1.0) / 4.0 * sharpe_ratio * sharpe_ratio
    if inner <= 0.0:
        # ``inner < 0`` is out of domain (NaN). ``inner == 0`` is the measure-zero boundary where the statistic
        # diverges, so the CDF is the limiting 0 or 1 (the shipped factory's documented behavior) -- except on an
        # exactly-equal Sharpe, where 0 / 0 stays NaN.
        diverges = inner == 0.0 and sharpe_ratio != benchmark_sharpe
        return (1.0 if sharpe_ratio > benchmark_sharpe else 0.0) if diverges else math.nan
    argument = (sharpe_ratio - benchmark_sharpe) * math.sqrt(count - 1) / math.sqrt(inner)
    return NormalDist().cdf(argument)


def reference_profit_factor(returns: Sequence[float | None]) -> float | None:
    """
    Naive profit factor over a Python list.

    The sum of the positive returns over the magnitude of the sum of the negative returns, recomputed from scratch as
    the oracle for :func:`pomata.metrics.profit_factor`. ``None`` returns are skipped; a ``nan`` anywhere poisons the
    result to ``nan``; with no observations the result is ``None``; with no losses the result is ``+inf`` (or ``nan``
    when there are also no gains), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    gain = sum(value for value in observations if value > 0.0)
    loss = sum(-value for value in observations if value < 0.0)
    if loss == 0.0:
        return math.inf if gain > 0.0 else math.nan
    return gain / loss


def reference_recovery_ratio(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive recovery factor over a Python list.

    The total return over the magnitude of the maximum drawdown, recomputed from scratch as the oracle
    for :func:`pomata.metrics.recovery_ratio` by composing the independent :func:`reference_total_return` and
    :func:`reference_max_drawdown`. Only the drawdown denominator is taken in magnitude; the total-return numerator
    keeps its sign, so a losing curve reports a negative factor. ``None`` equities are skipped; a ``nan`` anywhere
    poisons it to ``nan``; with no defined observations the result is ``None``. A drawdown-free curve gives ``+/-inf``
    with the sign of the total return (or ``nan`` when the total return is also zero), matching the implementation's
    division.
    """
    growth = reference_total_return(equity_curve)
    drawdown_trough = reference_max_drawdown(equity_curve)
    if growth is None or drawdown_trough is None:
        return None
    if math.isnan(growth) or math.isnan(drawdown_trough):
        return math.nan
    denominator = abs(drawdown_trough)
    if denominator == 0.0:
        return math.nan if growth == 0.0 else math.copysign(math.inf, growth)
    return growth / denominator


def reference_risk_of_ruin(returns: Sequence[float | None]) -> float | None:
    """
    Naive risk of ruin over a Python list.

    The symmetric gambler's-ruin probability ``min(((1 - p) / p) ** n, 1)`` from the win rate ``p`` and the count ``n``
    of non-null returns (the capital cushion in unit bets), recomputed from scratch as the oracle for
    :func:`pomata.metrics.risk_of_ruin` by composing the independent :func:`reference_win_rate`. ``None`` returns are
    skipped; a ``nan`` anywhere poisons the result to ``nan``; with no decisive returns (the win rate undefined) the
    result is ``None``. A win rate ``p <= 0.5`` gives a ratio ``>= 1``, so the probability is clamped to ``1`` (ruin is
    certain without an edge); ``p == 0`` (all losses) is ``1`` and ``p == 1`` (all wins) is ``0``.
    """
    probability = reference_win_rate(returns)
    if probability is None:
        return None
    if math.isnan(probability):
        return math.nan
    observations = sum(1 for value in returns if value is not None)
    if probability == 0.0:
        return 1.0
    return min(((1.0 - probability) / probability) ** observations, 1.0)


def reference_sharpe_ratio(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive annualized Sharpe ratio over a Python list.

    The mean excess return divided by its sample standard deviation (``ddof = 1``), annualized by ``sqrt(P)``, where the
    per-period risk-free rate is the geometric conversion ``(1 + risk_free_rate) ** (1 / P) - 1``. Recomputed from
    scratch as the oracle for :func:`pomata.metrics.sharpe_ratio`. ``None`` returns are skipped; with fewer than two
    observations the result is ``None`` (the undefined sample standard deviation takes precedence); otherwise a ``nan``
    anywhere poisons the result to ``nan``. Zero dispersion gives ``+/-inf`` (or ``nan`` when the mean excess is also
    zero), matching the implementation's division; an exactly-constant excess series is detected via ``min == max``
    rather than the two-pass deviation (whose float residual is not reliably zero for every constant), matching the
    implementation's exact zero-dispersion pin.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    if max(excess) == min(excess):
        return math.nan if excess[0] == 0.0 else math.copysign(math.inf, excess[0])
    count = len(excess)
    mean_excess = sum(excess) / count
    deviation = math.sqrt(sum((value - mean_excess) ** 2 for value in excess) / (count - 1))
    if deviation == 0.0:
        return math.nan if mean_excess == 0.0 else math.copysign(math.inf, mean_excess)
    return mean_excess / deviation * math.sqrt(periods_per_year)


def reference_sharpe_ratio_rolling(
    values: Sequence[float | None], window: int, periods_per_year: int, risk_free_rate: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(
        lambda window_slice: reference_sharpe_ratio(window_slice, periods_per_year, risk_free_rate), values, window
    )


def reference_skewness(returns: Sequence[float | None]) -> float | None:
    """
    Naive population skewness over a Python list.

    ``m3 / m2**1.5`` from the population central moments of the non-null returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.skewness`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``;
    with no observations the result is ``None``; a zero-variance series (constant, or a single value) gives ``0 / 0``
    and the result is ``nan``, as does a subnormal-magnitude series whose ``m2 ** 1.5`` underflows to zero (matching the
    implementation, which yields ``nan`` there too).
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    count = len(observations)
    mean = sum(observations) / count
    second_moment = sum((value - mean) ** 2 for value in observations) / count
    denominator = math.pow(second_moment, 1.5)
    if denominator == 0.0:
        return math.nan
    third_moment = sum((value - mean) ** 3 for value in observations) / count
    return third_moment / denominator


def reference_skewness_rolling(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(reference_skewness, values, window)


def reference_sortino_ratio(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive annualized Sortino ratio over a Python list.

    The mean excess return divided by the (population) downside deviation about the same target, annualized by
    ``sqrt(P)``, where the per-period risk-free target is ``(1 + risk_free_rate) ** (1 / P) - 1``. Recomputed from
    scratch as the oracle for :func:`pomata.metrics.sortino_ratio`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with no observations the result is ``None``. No downside gives ``+/-inf`` (or ``nan``
    when the mean excess is also zero), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    count = len(excess)
    mean_excess = sum(excess) / count
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in excess) / count)
    if downside == 0.0:
        return math.nan if mean_excess == 0.0 else math.copysign(math.inf, mean_excess)
    return mean_excess / downside * math.sqrt(periods_per_year)


def reference_sortino_ratio_rolling(
    values: Sequence[float | None], window: int, periods_per_year: int, risk_free_rate: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(
        lambda window_slice: reference_sortino_ratio(window_slice, periods_per_year, risk_free_rate), values, window
    )


def reference_stability(returns: Sequence[float | None]) -> float | None:
    """
    Naive trend stability over a Python list.

    The coefficient of determination of an ordinary-least-squares fit of the cumulative log returns on time, recomputed
    from scratch as the oracle for :func:`pomata.metrics.stability` (``R**2 = corr(t, cumulative_log)**2``). ``None``
    returns are skipped and the time index runs over the retained observations; a ``nan`` anywhere poisons the result
    to ``nan``, even as the only observation (the poison wins over the count guard, as in the cagr / total_return
    siblings); with fewer than two the result is ``None``; a return at or below ``-1`` (undefined log) poisons the
    result to ``nan``; a perfectly flat cumulative path gives ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if len(observations) < 2:
        return None
    if any(value <= -1.0 for value in observations):
        return math.nan
    cumulative: list[float] = []
    running = 0.0
    for value in observations:
        running += math.log1p(value)
        cumulative.append(running)
    count = len(cumulative)
    index = list(range(count))
    mean_index = sum(index) / count
    mean_cumulative = sum(cumulative) / count
    covariance = sum((index[i] - mean_index) * (cumulative[i] - mean_cumulative) for i in range(count))
    variance_index = sum((value - mean_index) ** 2 for value in index)
    variance_cumulative = sum((value - mean_cumulative) ** 2 for value in cumulative)
    if variance_index == 0.0 or variance_cumulative == 0.0:
        return math.nan
    correlation = covariance / math.sqrt(variance_index * variance_cumulative)
    return correlation * correlation


def reference_sterling_ratio(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float, excess: float
) -> float | None:
    """
    Naive Sterling ratio over a Python list.

    The excess compound annual growth rate divided by the average drawdown plus a cushion --
    ``(CAGR - risk_free_rate) / (PI + excess)`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.sterling_ratio` by composing the independent :func:`reference_cagr` and
    :func:`reference_pain_index`. ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with
    no observations the result is ``None``. A zero denominator gives ``+/-inf`` (or ``nan`` when the excess growth is
    also zero).
    """
    growth = reference_cagr(equity_curve, periods_per_year)
    pain = reference_pain_index(equity_curve)
    if growth is None or pain is None:
        return None
    if math.isnan(growth) or math.isnan(pain):
        return math.nan
    excess_growth = growth - risk_free_rate
    denominator = pain + excess
    if denominator == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / denominator


def reference_tail_ratio(returns: Sequence[float | None]) -> float | None:
    """
    Naive tail ratio over a Python list.

    The magnitude of the 95th-percentile return divided by the 5th-percentile return (both type-7 linear quantiles),
    recomputed from scratch as the oracle for :func:`pomata.metrics.tail_ratio`. ``None`` returns are skipped; a ``nan``
    anywhere poisons the result to ``nan``; with no observations the result is ``None``. A zero 5th-percentile gives the
    IEEE ``inf`` (or ``nan`` when the 95th percentile is also zero), matching the implementation's float division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(observations)
    right = type_seven_quantile(ascending, 0.95)
    left = type_seven_quantile(ascending, 0.05)
    if left == 0.0:
        return math.nan if right == 0.0 else math.inf
    return abs(right / left)


def reference_tail_ratio_rolling(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(reference_tail_ratio, values, window)


def reference_total_return(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive total return over a Python list: the last non-null equity minus one.

    Recomputed from scratch as the oracle for :func:`pomata.metrics.total_return`. ``None`` equities are skipped; a
    ``nan`` anywhere poisons the result to ``nan``; with no defined observations the result is ``None``.
    """
    defined = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in defined):
        return math.nan
    if not defined:
        return None
    return defined[-1] - 1


def reference_total_return_rolling(equity_curve: Sequence[float | None], window: int) -> list[float | None]:
    """
    Naive rolling total return: the window's last equity over its first, less one, recomputed from scratch.

    An endpoint quantity: row ``i`` is ``E[i] / E[i - window + 1] - 1``. The first ``window - 1`` rows are warm-up
    ``None``; a ``None`` at either endpoint yields ``None``; a ``NaN`` at either endpoint yields ``nan``. An interior
    ``None`` / ``NaN`` does not affect the result.
    """
    output: list[float | None] = []
    for index in range(len(equity_curve)):
        if index < window - 1:
            output.append(None)
            continue
        first = equity_curve[index - window + 1]
        last = equity_curve[index]
        if first is None or last is None:
            output.append(None)
        elif math.isnan(first) or math.isnan(last):
            output.append(math.nan)
        else:
            output.append(last / first - 1.0)
    return output


def reference_treynor_ratio(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive annualized Treynor ratio over two Python lists.

    The annualized arithmetic excess return ``mean(r - rf) * P`` over the :func:`reference_beta` slope, where the
    per-period risk-free rate is ``(1 + risk_free_rate) ** (1 / P) - 1`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.treynor_ratio`. The series are pairwise-complete: a pair contributes only where both legs are
    present; with fewer than two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a
    ``nan`` in either leg of a retained pair poisons the result to ``nan``. A zero beta gives ``+/-inf`` (or ``nan``
    when the excess return is also zero), matching the implementation's division.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    slope = reference_beta(returns, benchmark)
    assert slope is not None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    annualized_excess = (sum(x - rf_period for x, _ in pairs) / len(pairs)) * periods_per_year
    if slope == 0.0:
        return math.nan if annualized_excess == 0.0 else math.copysign(math.inf, annualized_excess)
    return annualized_excess / slope


def reference_treynor_ratio_rolling(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> list[float | None]:
    """
    Naive rolling Treynor ratio: the reducing reference applied to each trailing returns/benchmark window
    (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: reference_treynor_ratio(
            window_returns, window_benchmark, periods_per_year, risk_free_rate
        ),
        returns,
        benchmark,
        window,
    )


def reference_ulcer_index(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive Ulcer Index over a Python list: the root-mean-square of the running drawdown.

    Built on :func:`reference_drawdown`. ``None`` drawdowns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no defined drawdown (an all-null series) the result is ``None``; otherwise the quadratic mean of the
    drawdowns.
    """
    declines = [value for value in reference_drawdown(equity_curve) if value is not None]
    if any(math.isnan(value) for value in declines):
        return math.nan
    if not declines:
        return None
    return math.sqrt(sum(value * value for value in declines) / len(declines))


def reference_ulcer_performance_ratio(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive ulcer performance index (Martin ratio) over a Python list.

    The excess compound annual growth rate divided by the ulcer index -- ``(CAGR - risk_free_rate) / UlcerIndex`` --
    recomputed from scratch as the oracle for :func:`pomata.metrics.ulcer_performance_ratio` by composing the
    independent :func:`reference_cagr` and :func:`reference_ulcer_index`. ``None`` equities are skipped; a ``nan``
    anywhere poisons the result to ``nan``; with no defined observations the result is ``None``. A drawdown-free
    (monotonic) curve has a zero ulcer index and gives ``+/-inf`` (or ``nan`` when the excess growth is also zero),
    matching the implementation's division.
    """
    growth = reference_cagr(equity_curve, periods_per_year)
    ulcer = reference_ulcer_index(equity_curve)
    if growth is None or ulcer is None:
        return None
    if math.isnan(growth) or math.isnan(ulcer):
        return math.nan
    excess_growth = growth - risk_free_rate
    if ulcer == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / ulcer


def reference_value_at_risk(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive historical value-at-risk over a Python list.

    The ``1 - confidence`` empirical quantile (type-7 linear interpolation) of the non-null returns, recomputed from
    scratch as the oracle for :func:`pomata.metrics.value_at_risk`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with no observations the result is ``None``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    return type_seven_quantile(sorted(observations), 1.0 - confidence)


def reference_value_at_risk_modified(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive modified (Cornish-Fisher) value-at-risk over a Python list.

    The Gaussian quantile adjusted for skewness and excess kurtosis via the Cornish-Fisher expansion,
    ``mean + z_cf * std`` (sample ``std``, ``ddof = 1``), recomputed from scratch as the oracle for
    :func:`pomata.metrics.value_at_risk_modified`. ``None`` returns are skipped; with fewer than two the result is
    ``None``; a ``nan`` anywhere poisons it to ``nan``; zero dispersion (undefined skew/kurtosis) yields ``nan``.
    Out of the expansion's validity domain — a non-monotonic quantile map at ``z``, or a corrected quantile pushed
    across the median (``z_cf`` and ``z`` disagreeing in sign) — the estimate is ``nan``, mirroring the documented
    domain contract.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    count = len(observations)
    mean = sum(observations) / count
    deviation = math.sqrt(sum((value - mean) ** 2 for value in observations) / (count - 1))
    second_moment = sum((value - mean) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    excess_kurtosis = fourth_moment / (second_moment * second_moment) - 3.0
    z = NormalDist().inv_cdf(1.0 - confidence)
    z_cornish_fisher = (
        z
        + (z**2 - 1.0) / 6.0 * skewness
        + (z**3 - 3.0 * z) / 24.0 * excess_kurtosis
        - (2.0 * z**3 - 5.0 * z) / 36.0 * skewness**2
    )
    slope = 1.0 + z * skewness / 3.0 + (z**2 - 1.0) / 8.0 * excess_kurtosis - (6.0 * z**2 - 5.0) / 36.0 * skewness**2
    if slope <= 0.0 or z_cornish_fisher * z < 0.0:
        return math.nan
    return mean + z_cornish_fisher * deviation


def reference_value_at_risk_parametric(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive parametric (Gaussian) value-at-risk over a Python list.

    The normal-distribution quantile ``mean + Phi_inv(1 - confidence) * std`` (sample ``std``, ``ddof = 1``), recomputed
    from scratch as the oracle for :func:`pomata.metrics.value_at_risk_parametric`. ``None`` returns are skipped; with
    fewer than two the result is ``None``; a ``nan`` anywhere poisons the result to ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    count = len(observations)
    mean = sum(observations) / count
    deviation = math.sqrt(sum((value - mean) ** 2 for value in observations) / (count - 1))
    z = NormalDist().inv_cdf(1.0 - confidence)
    return mean + z * deviation


def reference_value_at_risk_rolling(
    values: Sequence[float | None], window: int, confidence: float = 0.95
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: reference_value_at_risk(window_slice, confidence), values, window)


def reference_volatility(returns: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive annualized sample standard deviation over a Python list.

    The two-pass sample standard deviation (``ddof = 1``) of the non-null returns, annualized by
    ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for :func:`pomata.metrics.volatility`. ``None``
    returns are skipped; with fewer than two remaining observations the standard deviation is undefined, so the result
    is ``None``; otherwise a ``nan`` propagates to the result. An exactly-constant series has zero dispersion, detected
    via ``min == max`` rather than the two-pass variance (whose float residual is not reliably zero for every
    constant), and is reported as exactly ``0.0``, matching the implementation's exact zero-dispersion pin.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    if max(observations) == min(observations):
        return 0.0
    mean = sum(observations) / len(observations)
    variance = sum((value - mean) ** 2 for value in observations) / (len(observations) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


def reference_volatility_rolling(
    values: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: reference_volatility(window_slice, periods_per_year), values, window)


def reference_win_rate(returns: Sequence[float | None]) -> float | None:
    """
    Naive win rate over a Python list.

    The count of strictly positive returns over the count of non-zero returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.win_rate`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``;
    a return of exactly ``0`` is excluded from the denominator; with no non-zero returns the result is ``None``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    decisive = [value for value in observations if value != 0.0]
    if not decisive:
        return None
    wins = sum(1 for value in decisive if value > 0.0)
    return wins / len(decisive)
