"""
Shared Hypothesis element strategies for the indicator test ladder, plus the shared ``window`` cap they draw from.

Each element strategy draws a single element; tests wrap them in ``st.lists(...)`` to build a series. They are the
canonical, one-home inputs for the property tiers — the missing-data tier, the scaling tier, and the positive-price
(OHLC) tier — so the same ``null`` / ``NaN`` / magnitude regimes are exercised identically across every indicator.
The cycle-cluster helpers (:func:`spans_even_lag_repeat`, :func:`two_segment_missing_data`) instead build a whole
series at once, since their guarantees are about the run as a whole.
"""

import math
from collections.abc import Sequence

from hypothesis import strategies as st

# The shared upper bound the property tiers draw their windows from; a tier needing a different cap passes its own.
WINDOW_MAX = 16

# Magnitude floor for finite draws: below it a squared or EWM-derived quantity underflows into the subnormal range,
# where the one-pass form and the two-pass oracle round apart (see subnormal_safe_floats). One place to tune.
SUBNORMAL_FLOOR = 1e-100

# Per-window conditioning floor: a trailing window whose variance is below ``scale ** 2 * CONDITIONING_FLOOR`` is too
# near-constant for a one-pass rolling ratio (a slope or standardized moment) to track the two-pass oracle. One place
# to tune.
CONDITIONING_FLOOR = 1e-2

# Magnitude floor for standardized-moment draws (skewness / kurtosis): these normalize by ``m2 ** p`` with ``p >= 1.5``,
# so the variance is raised above a plain square and underflows at a LARGER input magnitude than SUBNORMAL_FLOOR guards
# against (kurtosis' ``m2 ** 2`` underflows to zero around an input scale of ``1e-80``). Flooring ``|v|`` here keeps the
# standardized moment well-conditioned across the scale tier's ``2 ** +-4`` rescaling (see standardized_moment_floats).
STANDARDIZED_MOMENT_FLOOR = 1e-60


def finite_floats(bound: float = 1e6) -> st.SearchStrategy[float]:
    """
    Finite floats in ``[-bound, bound]``, the shared element strategy for the ordinary-input property tiers.

    This is the canonical "well-behaved finite value" domain: the matches-reference and bounds tiers draw a series from
    it to exercise an indicator against its naive oracle on inputs with no missing data. The ``bound`` is the
    per-indicator safe magnitude -- the largest scale at which the implementation stays well-conditioned (raise it for a
    linear indicator, lower it where a difference of large terms would cancel) -- and is declared in each test file's
    sizing section rather than hard-coded here.

    Args:
        bound: The symmetric magnitude bound; values are drawn from ``[-bound, bound]`` (default ``1e6``).

    Returns:
        A strategy producing a finite ``float`` in ``[-bound, bound]``.
    """
    return st.floats(min_value=-bound, max_value=bound, allow_nan=False, allow_infinity=False)


def missing_data_floats(min_magnitude: float = 0.0) -> st.SearchStrategy[float | None]:
    """
    A Hypothesis strategy drawing finite floats in ``[-1e6, 1e6]`` freely interleaved with ``None`` and ``NaN``.

    This is the shared element strategy for the missing-data robustness tier: drawing a list from it mixes Polars
    ``null`` (Python ``None``), ``float('nan')``, and ordinary finite values in one stream, so the indicators' interior
    ``null`` / ``NaN`` paths are exercised against their naive reference oracles rather than only on hand-picked
    literals.

    For an indicator that squares its input (variance / standard deviation / Bollinger) or whose recursive EWM mean
    collapses its abs-tol at a subnormal-magnitude ``window`` (ema / rma / dema / tema / t3), pass
    ``min_magnitude=SUBNORMAL_FLOOR`` so the finite draws are floored away from the subnormal range (see
    :func:`subnormal_safe_floats`); otherwise the one-pass streaming form and the two-pass oracle can diverge on a tiny
    value — a pure floating-point artifact.

    Args:
        min_magnitude: When ``> 0``, finite draws are restricted to ``|v| >= min_magnitude`` (default ``0.0``).

    Returns:
        A strategy producing ``None``, ``float('nan')``, or a finite ``float`` in ``[-1e6, 1e6]`` (with ``|v| >=
        min_magnitude`` when set).
    """
    finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    if min_magnitude > 0.0:
        finite = finite.filter(lambda value: abs(value) >= min_magnitude)
    return st.one_of(st.none(), st.just(math.nan), finite)


def positive_missing_data(high: float = 1e4) -> st.SearchStrategy[float | None]:
    """
    A Hypothesis strategy drawing positive finite floats in ``[1.0, high]`` freely interleaved with ``None`` and
    ``NaN``.

    The positive-price counterpart of :func:`missing_data_floats`, for indicators defined on strictly positive inputs
    (``high`` / ``low`` / ``close`` / ``volume``): it mixes Polars ``null``, ``float('nan')``, and finite values bounded
    away from zero, so the interior ``null`` / ``NaN`` paths are exercised on realistic price-like magnitudes against
    the naive oracle.

    Args:
        high: The upper bound of the drawn finite values; they are drawn from ``[1.0, high]``.

    Returns:
        A strategy producing ``None``, ``float('nan')``, or a finite ``float`` in ``[1.0, high]``.
    """
    return st.one_of(
        st.none(),
        st.just(math.nan),
        st.floats(min_value=1.0, max_value=high, allow_nan=False, allow_infinity=False),
    )


def spans_even_lag_repeat(series: Sequence[float | None]) -> bool:
    """
    Whether ``series`` contains two equal values two bars apart (``x[i] == x[i - 2]``) — the cycle-pipeline degenerate.

    The shared flat-run guard for the seven Hilbert-transform cycle indicators' agreement tiers. Ehlers' six-tap
    quadrature filter reads the four-bar smooth at EVEN lags (taps at 0, 2, 4, 6), so its in-phase component collapses
    to a pure cancellation residual whenever the smooth repeats across an even lag — which happens for a flat run
    (``a, a, a, ...``) AND for a period-two alternation (``a, b, a, b, ...``), since the smooth there is equal two bars
    apart. On that residual the implementation's explicit FIR and the oracle's compensated ``sum()`` round to opposite
    sides of zero, flipping the ``inphase != 0`` phasor branch and so the phase that fixes every downstream line; the
    two transcriptions cannot be expected to agree across that discontinuity. The naive ``earlier != later``
    guard misses it (an alternation passes that check yet hits the same branch), so the agreement tiers filter on this
    predicate instead: a series with no even-lag repeat reaches the degenerate nowhere, and only such inputs are drawn.

    Args:
        series: The candidate price series (the finite list a property tier would feed the indicator).

    Returns:
        ``True`` if some value equals the value two positions earlier (an even-lag repeat is present), else ``False``.
    """
    return any(series[index] == series[index - 2] for index in range(2, len(series)))


def spans_even_lag_run(series: Sequence[float | None], min_run: int = 6) -> bool:
    """
    Whether ``series`` contains a SUSTAINED run of at least ``min_run`` consecutive even-lag equalities
    (``x[i] == x[i - 2]`` at ``min_run`` consecutive indices) — the regime where the cycle pipeline's phasor branch
    genuinely flips.

    The narrowed twin of :func:`spans_even_lag_repeat` for the cycle indicators that read the phase branch (mama,
    sine_wave, dominant_cycle_phase). An empirical boundary probe (impl vs oracle, graduated trailing runs on the
    golden carriers) showed the single-pair predicate is ~one-to-fourteen too blunt: an ISOLATED even-lag tie — the
    overwhelming majority of what it rejects under fuzzing — never produces real disagreement (worst measured
    deviation ~1e-14 for mama, ≤2.6e-10 on unit-bounded sine_wave lanes, inside the property tiers' absolute band),
    while real branch-flip disagreement needs the four-bar smooth to repeat across an even lag for a sustained
    stretch: onset at ~9 structured bars for sine_wave (probabilistic, the norm by 11-14), ~14 for mama, and a
    whole-series flat run for dominant_cycle_phase. A run of ``k`` consecutive even-lag equalities corresponds to
    ``k + 2`` such bars, so the default ``min_run=6`` (≈ 8 bars) sits one bar below the earliest measured onset,
    rejecting every sustained flat run or period-two alternation while re-admitting the isolated coincidences.

    Args:
        series: The candidate price series (the finite list a property tier would feed the indicator).
        min_run: The minimum count of consecutive even-lag equalities that makes the series degenerate.

    Returns:
        ``True`` if some ``min_run`` consecutive indices each equal the value two positions earlier, else ``False``.
    """
    run = 0
    for index in range(2, len(series)):
        run = run + 1 if series[index] == series[index - 2] else 0
        if run >= min_run:
            return True
    return False


def subnormal_safe_floats(bound: float = 1e3) -> st.SearchStrategy[float]:
    """
    Finite floats in ``[-bound, bound]`` whose magnitude is floored at ``SUBNORMAL_FLOOR``, for any indicator whose
    property tiers compare a one-pass streaming form against a two-pass oracle and would diverge on a
    subnormal-magnitude input by a pure floating-point artifact rather than a bug.

    Two indicator families need this floor, for the same underlying reason -- a quantity derived from a tiny input
    underflows into the subnormal range, where the streaming implementation and the two-pass oracle round apart:

    - **Squaring indicators** (variance, standard deviation, Bollinger bands) compute ``v ** 2``; a draw with
      ``|v| ~ 1e-162`` gives ``v ** 2 ~ 3e-324``, below ``DBL_MIN ~ 2.2e-308``, so the square loses almost all of its
      precision and the one-pass streaming variance drifts from the two-pass oracle by far more than any tolerance.
    - **Recursive EWM means** (ema, rma, dema, tema, t3) are checked against ``input_scale(values) *
      EXACT_TOLERANCE_FACTOR``; at a subnormal-magnitude ``window`` that abs-tol collapses to exactly ``0.0`` while the
      recursion and the two-pass oracle round one ULP apart -- a deterministic failure on data that is mathematically
      fine. The degenerate all-zero ``window`` is pinned in the edge tier instead.

    Flooring ``|v|`` at ``SUBNORMAL_FLOOR`` keeps ``v ** 2`` (and the scaled ``(k * v) ** 2`` with ``|k| >= 1e-2``)
    comfortably inside the normal range and keeps ``input_scale`` above the subnormal threshold, while still spanning
    a wide magnitude. Use it in EVERY property tier of such an indicator (any-input, scale, large-magnitude), not only
    the scale tier: the underflow can surface wherever a tiny value is drawn (it was originally caught only on the
    scale tier and later re-surfaced on the any-input tier). For the missing-data tier, pass
    ``min_magnitude=SUBNORMAL_FLOOR`` to :func:`missing_data_floats` for the same reason.

    Args:
        bound: The symmetric magnitude bound; values are drawn from ``[-bound, bound]``.

    Returns:
        A strategy producing finite ``float`` values with ``SUBNORMAL_FLOOR <= |v| <= bound``.
    """
    return st.floats(min_value=-bound, max_value=bound, allow_nan=False, allow_infinity=False).filter(
        lambda value: abs(value) >= SUBNORMAL_FLOOR
    )


def standardized_moment_floats(bound: float = 1e3) -> st.SearchStrategy[float]:
    """
    Finite floats in ``[-bound, bound]`` whose magnitude is floored at ``STANDARDIZED_MOMENT_FLOOR``, for a standardized
    moment (skewness, kurtosis) whose property tiers would otherwise round apart from their two-pass oracle on a
    subnormal-magnitude input by a pure floating-point artifact rather than a bug.

    A standardized moment divides a central moment by ``m2 ** p`` with ``p >= 1.5`` (``1.5`` for skewness, ``2`` for
    kurtosis), so the variance is raised above a plain square: at an input scale near ``1e-80`` that power underflows to
    ``0`` and the value collapses to ``nan``. The plain :func:`subnormal_safe_floats` floor (``1e-100``) guards only a
    square and is too low here -- under the scale tier's ``2 ** -4`` rescaling a ``1e-100`` draw still underflows.
    Flooring ``|v|`` at ``STANDARDIZED_MOMENT_FLOOR`` keeps ``m2 ** p`` (and the rescaled ``(k * v)`` with
    ``|k| >= 2 ** -4``) comfortably inside the normal range while still spanning a wide magnitude. The subnormal
    collapse itself is pinned deterministically in the edge tier.

    Args:
        bound: The symmetric magnitude bound; values are drawn from ``[-bound, bound]``.

    Returns:
        A strategy producing finite ``float`` values with ``STANDARDIZED_MOMENT_FLOOR <= |v| <= bound``.
    """
    return st.floats(min_value=-bound, max_value=bound, allow_nan=False, allow_infinity=False).filter(
        lambda value: abs(value) >= STANDARDIZED_MOMENT_FLOOR
    )


def well_spread(values: Sequence[float | None]) -> bool:
    """
    Whether the finite values have genuine spread (variance well above rounding noise), so that a standardized moment
    (skewness, kurtosis) is well-conditioned rather than a ``0 / 0`` artifact.

    An exactly- or near-constant sample has a variance that is pure floating-point rounding noise, and the two
    transcriptions resolve it oppositely: the implementation's one-pass moment collapses to ``nan`` while the two-pass
    oracle's mean rounds the deviations to a uniform tiny offset and reports a spurious finite value (a constant series'
    "skewness" of ``1``). The paths cannot agree across that degeneracy, so the standardized-moment property tiers
    filter it out with this predicate and pin the constant case deterministically in the edge tier. Fewer than two
    finite values is left to the metric's own empty / single-value handling, where both paths already agree.

    Args:
        values: The candidate series (the finite-or-missing list a property tier would feed the metric).

    Returns:
        ``True`` if the finite values' population variance exceeds ``scale ** 2 * 1e-9`` (genuine spread), else
        ``False``.
    """
    finite = [value for value in values if value is not None and not math.isnan(value)]
    if len(finite) < 2:
        return True
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    scale = max(abs(value) for value in finite)
    return variance > scale * scale * 1e-9


def windows_well_spread(values: Sequence[float | None], window: int) -> bool:
    """
    Whether every trailing ``window`` of ``values`` is well-spread -- the per-window twin of :func:`well_spread`.

    A near-constant window drives a one-pass rolling variance / standard deviation negative (then ``NaN``), where it
    cannot track the two-pass oracle; the rolling dispersion property tiers filter such windows with this predicate.
    """
    return all(well_spread(values[index - window + 1 : index + 1]) for index in range(window - 1, len(values)))


def windows_well_conditioned(values: Sequence[float | None], window: int, floor: float = CONDITIONING_FLOOR) -> bool:
    """
    Whether every trailing ``window``'s variance is a real fraction of its magnitude (a well-conditioned slope or
    standardized moment).

    A window whose variance is below ``scale ** 2 * floor`` is too near-constant for a one-pass rolling ratio to
    track the two-pass oracle; the rolling moment-ratio and benchmark-relative property tiers filter it out and pin
    the degenerate case deterministically in the edge tier. Fewer than two finite values is skipped (both paths
    already agree there). The ``floor`` defaults to the shared conservative ``CONDITIONING_FLOOR``; a spec whose
    empirical disagreement onset was measured (see each spec's conditioning wrapper) passes its own tighter,
    spec-local floor instead — the shared constant itself is never retuned per spec.
    """
    for index in range(window - 1, len(values)):
        finite = [
            value for value in values[index - window + 1 : index + 1] if value is not None and not math.isnan(value)
        ]
        if len(finite) < 2:
            continue
        mean = sum(finite) / len(finite)
        variance = sum((value - mean) ** 2 for value in finite) / len(finite)
        scale = max(abs(value) for value in finite) or 1.0
        if variance <= scale * scale * floor:
            return False
    return True


@st.composite
def two_segment_missing_data(draw: st.DrawFn, warmup: int, high: float = 1e4, tail: int = 16) -> list[float | None]:
    """
    A guaranteed-finite prefix LONGER than ``warmup`` followed by a missing-data tail, for the cycle-cluster fuzz tier.

    The shared missing-data strategy for the seven Hilbert-transform cycle indicators. Their recurrence latches the
    entire output to ``null`` at the first ``null`` / ``NaN`` (a gap it cannot bridge), and the warm-up is long (32 or
    63 bars), so drawing every element from :func:`positive_missing_data` almost never clears the warm-up before a
    missing value appears: a defined output row needs a long run of consecutive finite draws, so the tier degenerates
    to comparing all-``null`` against all-``null`` and checks no numeric value. Splitting the draw into two segments — a
    finite prefix of ``warmup + 1 .. warmup + tail`` positive bars (so defined output is always emitted), then a tail
    of ``tail`` rows from :func:`positive_missing_data` (so the defined values actually meet ``null`` / ``NaN``) —
    restores the intended power while preserving the latch behavior the edge tier pins deterministically. The prefix is
    drawn to contain no even-lag repeat (see :func:`spans_even_lag_repeat`), the same flat-run guard the agreement tiers
    use, so the defined region is well-conditioned and a spurious branch-flip never masquerades as a missing-data bug.

    Args:
        draw: The Hypothesis draw function (injected by ``@composite``).
        warmup: The indicator's leading-``null`` run; the finite prefix is drawn strictly longer than this.
        high: The upper bound of the finite prefix values; they are drawn from ``[1.0, high]`` (default ``1e4``).
        tail: The number of missing-data rows appended after the prefix, and the prefix's defined-row span (default 16).

    Returns:
        A list whose first ``warmup + 1 .. warmup + tail`` rows are finite positive floats with no even-lag repeat,
        followed by ``tail`` rows drawn from :func:`positive_missing_data`.
    """
    defined = draw(st.integers(min_value=warmup + 1, max_value=warmup + tail))
    finite = st.floats(min_value=1.0, max_value=high, allow_nan=False, allow_infinity=False)
    prefix = draw(
        st.lists(finite, min_size=defined, max_size=defined).filter(lambda series: not spans_even_lag_repeat(series))
    )
    suffix = draw(st.lists(positive_missing_data(high), min_size=tail, max_size=tail))
    return [*prefix, *suffix]


# Coherent OHLC bars. A bar must be coherent -- ``low <= open, close <= high`` -- because any indicator that divides by
# the bar range ``high - low`` (the accumulation / distribution money-flow multiplier and everything built on it) is
# unbounded on an impossible bar (``high < low``, or ``close`` outside ``[low, high]``): a tiny or negative denominator
# the numerator escapes, so the one-pass form and the multi-pass oracle diverge by far more than any tolerance. That is
# out-of-domain input, not a bug; real bars are always coherent (a flat ``high == low`` bar IS coherent and stays
# exercised). Each bar is drawn as ONE batched ``st.tuples`` of independent primitives, then assembled in pure Python:
# ``@st.composite`` re-enters the draw machinery on every interactive ``draw()``, so a per-bar composite generates ~30x
# slower than a flat tuple of the same primitives -- and these OHLC tiers dominated the suite's wall-clock.
_FRACTION = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_MISSING = st.sampled_from((0, 1, 2))  # 0 keep, 1 -> None, 2 -> NaN; shrinks toward "keep"


def _price(max_price: float) -> st.SearchStrategy[float]:
    """
    A positive finite price in ``[1.0, max_price]``.
    """
    return st.floats(min_value=1.0, max_value=max_price, allow_nan=False, allow_infinity=False)


def _low_high(price_a: float, price_b: float) -> tuple[float, float]:
    """
    Order a price pair into ``(low, high)``.
    """
    return (price_a, price_b) if price_a <= price_b else (price_b, price_a)


def _within(low: float, high: float, fraction: float) -> float:
    """
    The point ``low + fraction * (high - low)``, clamped to ``<= high`` so a rounding step never breaks coherence.
    """
    return min(high, low + fraction * (high - low))


def _apply_missing(value: float, choice: int) -> float | None:
    """
    Keep ``value`` (``0``), drop it to ``None`` (``1``), or replace it with ``NaN`` (``2``).
    """
    if choice == 1:
        return None
    if choice == 2:
        return math.nan
    return value


def _assemble_hl(pair: tuple[float, float]) -> tuple[float, float]:
    return (max(pair), min(pair))


def coherent_hl(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float]]:
    """
    A coherent ``(high, low)`` bar (``high >= low``, both positive finite).
    """
    return st.tuples(_price(max_price), _price(max_price)).map(_assemble_hl)


def _assemble_hlc(raw: tuple[float, float, float]) -> tuple[float, float, float]:
    price_a, price_b, close_fraction = raw
    low, high = _low_high(price_a, price_b)
    return (high, low, _within(low, high, close_fraction))


def coherent_hlc(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float, float]]:
    """
    A coherent ``(high, low, close)`` bar (``low <= close <= high``, positive finite).
    """
    return st.tuples(_price(max_price), _price(max_price), _FRACTION).map(_assemble_hlc)


def _assemble_hlcv(raw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    price_a, price_b, close_fraction, volume = raw
    low, high = _low_high(price_a, price_b)
    return (high, low, _within(low, high, close_fraction), volume)


def coherent_hlcv(
    max_price: float = 1e3, max_volume: float = 1e6
) -> st.SearchStrategy[tuple[float, float, float, float]]:
    """
    A coherent ``(high, low, close, volume)`` bar (``low <= close <= high``, all positive finite).
    """
    return st.tuples(_price(max_price), _price(max_price), _FRACTION, _price(max_volume)).map(_assemble_hlcv)


def _assemble_ohlc(raw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    price_a, price_b, open_fraction, close_fraction = raw
    low, high = _low_high(price_a, price_b)
    return (_within(low, high, open_fraction), high, low, _within(low, high, close_fraction))


def coherent_ohlc(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float, float, float]]:
    """
    A coherent ``(open, high, low, close)`` bar (``low <= open, close <= high``, all positive finite).
    """
    return st.tuples(_price(max_price), _price(max_price), _FRACTION, _FRACTION).map(_assemble_ohlc)


def _assemble_hl_missing(raw: tuple[float, float, int, int]) -> tuple[float | None, float | None]:
    price_a, price_b, keep_high, keep_low = raw
    low, high = _low_high(price_a, price_b)
    return (_apply_missing(high, keep_high), _apply_missing(low, keep_low))


def coherent_hl_with_missing(max_price: float = 1e4) -> st.SearchStrategy[tuple[float | None, float | None]]:
    """
    A coherent ``(high, low)`` bar with each field independently kept, nulled, or set to ``NaN``.
    """
    return st.tuples(_price(max_price), _price(max_price), _MISSING, _MISSING).map(_assemble_hl_missing)


def _assemble_hlc_missing(
    raw: tuple[float, float, float, int, int, int],
) -> tuple[float | None, float | None, float | None]:
    price_a, price_b, close_fraction, keep_high, keep_low, keep_close = raw
    low, high = _low_high(price_a, price_b)
    close = _within(low, high, close_fraction)
    return (_apply_missing(high, keep_high), _apply_missing(low, keep_low), _apply_missing(close, keep_close))


def coherent_hlc_with_missing(
    max_price: float = 1e4,
) -> st.SearchStrategy[tuple[float | None, float | None, float | None]]:
    """
    A coherent ``(high, low, close)`` bar with each field independently kept, nulled, or set to ``NaN``.
    """
    return st.tuples(_price(max_price), _price(max_price), _FRACTION, _MISSING, _MISSING, _MISSING).map(
        _assemble_hlc_missing
    )


def _assemble_hlcv_missing(
    raw: tuple[float, float, float, float, int, int, int, int],
) -> tuple[float | None, float | None, float | None, float | None]:
    price_a, price_b, close_fraction, volume, keep_high, keep_low, keep_close, keep_volume = raw
    low, high = _low_high(price_a, price_b)
    close = _within(low, high, close_fraction)
    return (
        _apply_missing(high, keep_high),
        _apply_missing(low, keep_low),
        _apply_missing(close, keep_close),
        _apply_missing(volume, keep_volume),
    )


def coherent_hlcv_with_missing(
    max_price: float = 1e4, max_volume: float = 1e6
) -> st.SearchStrategy[tuple[float | None, float | None, float | None, float | None]]:
    """
    A coherent ``(high, low, close, volume)`` bar with each field independently kept, nulled, or set to ``NaN``.
    """
    return st.tuples(
        _price(max_price), _price(max_price), _FRACTION, _price(max_volume), _MISSING, _MISSING, _MISSING, _MISSING
    ).map(_assemble_hlcv_missing)


def _assemble_ohlc_missing(
    raw: tuple[float, float, float, float, int, int, int, int],
) -> tuple[float | None, float | None, float | None, float | None]:
    price_a, price_b, open_fraction, close_fraction, keep_open, keep_high, keep_low, keep_close = raw
    low, high = _low_high(price_a, price_b)
    open_value = _within(low, high, open_fraction)
    close = _within(low, high, close_fraction)
    return (
        _apply_missing(open_value, keep_open),
        _apply_missing(high, keep_high),
        _apply_missing(low, keep_low),
        _apply_missing(close, keep_close),
    )


def coherent_ohlc_with_missing(
    max_price: float = 1e4,
) -> st.SearchStrategy[tuple[float | None, float | None, float | None, float | None]]:
    """
    A coherent ``(open, high, low, close)`` bar with each field independently kept, nulled, or set to ``NaN``.
    """
    return st.tuples(
        _price(max_price), _price(max_price), _FRACTION, _FRACTION, _MISSING, _MISSING, _MISSING, _MISSING
    ).map(_assemble_ohlc_missing)
