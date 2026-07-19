"""
Shared Hypothesis element strategies for the suite.

Each element strategy draws a single element; callers wrap them in ``st.lists(...)`` to build a series. They are the
canonical, one-home input domains for the fuzz engine (:func:`tests.support.synthesis.fuzz_frames`), so the same
``null`` / ``NaN`` / magnitude regimes are exercised identically across the suite. The cycle-cluster predicate
(:func:`spans_even_lag_run`) instead reads a whole series at once, since its guarantee is about a sustained run; the
conditioning predicates (:func:`well_spread` and friends) read whole series or windows for the same reason.
"""

import math
from collections.abc import Sequence

from hypothesis import strategies as st

# Magnitude floor for finite draws: below it a squared or EWM-derived quantity underflows into the subnormal range,
# where the one-pass form and the two-pass oracle round apart (see subnormal_safe_floats). One place to tune.
FLOOR_SUBNORMAL = 1e-100

# Per-window conditioning floor: a trailing window whose variance is below ``scale ** 2 * FLOOR_CONDITIONING`` is too
# near-constant for a one-pass rolling ratio (a slope or standardized moment) to track the two-pass oracle. One place
# to tune.
FLOOR_CONDITIONING = 1e-2


def finite_floats(bound: float = 1e6) -> st.SearchStrategy[float]:
    """
    Finite floats in ``[-bound, bound]``, the shared element strategy for the ordinary-input property tiers.

    This is the canonical "well-behaved finite value" domain: the fuzz engine draws a series from it to exercise a
    function against its naive oracle on inputs with no missing data. The ``bound`` is the caller's safe magnitude --
    the largest scale at which the implementation stays well-conditioned -- so it is passed in, never hard-coded here.

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
    ``null`` (Python ``None``), ``float('nan')``, and ordinary finite values in one stream, so the interior ``null`` /
    ``NaN`` paths are exercised against the naive reference oracles rather than only on hand-picked literals.

    Where a squared or EWM-derived quantity would underflow into the subnormal range on a tiny draw, pass
    ``min_magnitude=FLOOR_SUBNORMAL`` so the finite draws are floored away from it; otherwise a one-pass streaming form
    and its two-pass oracle can diverge on a tiny value — a pure floating-point artifact.

    Args:
        min_magnitude: When ``> 0``, finite draws are restricted to ``|v| >= min_magnitude`` (default ``0.0``).

    Returns:
        A strategy producing ``None``, ``float('nan')``, or a finite ``float`` in ``[-1e6, 1e6]``.
    """
    finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    if min_magnitude > 0.0:
        finite = finite.filter(lambda value: abs(value) >= min_magnitude)
    return st.one_of(st.none(), st.just(math.nan), finite)


def spans_even_lag_run(series: Sequence[float | None], min_run: int = 6) -> bool:
    """
    Whether ``series`` contains a SUSTAINED run of at least ``min_run`` consecutive even-lag equalities
    (``x[i] == x[i - 2]`` at ``min_run`` consecutive indices) — the regime where a cycle pipeline's phasor branch
    genuinely flips.

    The whole-series conditioning predicate for the cycle indicators that read the phase branch. The cut targets the
    SUSTAINED run, never the isolated tie: an isolated even-lag tie never produces real disagreement (inside the
    property tiers' absolute band), while real branch-flip disagreement needs the four-bar smooth to repeat across an
    even lag for a sustained stretch. A run of ``k`` consecutive even-lag equalities corresponds to ``k + 2`` such bars,
    so the
    default ``min_run=6`` (≈ 8 bars) sits one bar below the earliest measured onset, rejecting every sustained flat run
    or period-two alternation while re-admitting the isolated coincidences.

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


def _population_variance_and_scale(finite: Sequence[float]) -> tuple[float, float]:
    """
    The population variance of ``finite`` and its magnitude scale (the largest absolute value) — the shared core of the
    spread predicates below.
    """
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return variance, max(abs(value) for value in finite)


def well_spread(values: Sequence[float | None]) -> bool:
    """
    Whether the finite values have genuine spread (variance well above rounding noise), so that a standardized moment
    (skewness, kurtosis) is well-conditioned rather than a ``0 / 0`` artifact.

    An exactly- or near-constant sample has a variance that is pure floating-point rounding noise, and the two
    transcriptions resolve it oppositely, so the standardized-moment property tiers filter it out with this predicate
    and pin the constant case deterministically in the edge tier. Fewer than two finite values is left to the metric's
    own empty / single-value handling, where both paths already agree.

    Args:
        values: The candidate series (the finite-or-missing list a property tier would feed the metric).

    Returns:
        ``True`` if the finite values' population variance exceeds ``scale ** 2 * 1e-9`` (genuine spread), else
        ``False``.
    """
    finite = [value for value in values if value is not None and not math.isnan(value)]
    if len(finite) < 2:
        return True
    variance, scale = _population_variance_and_scale(finite)
    return variance > scale * scale * 1e-9


def windows_well_spread(values: Sequence[float | None], window: int) -> bool:
    """
    Whether every trailing ``window`` of ``values`` is well-spread -- the per-window twin of :func:`well_spread`.

    A near-constant window drives a one-pass rolling variance / standard deviation negative (then ``NaN``), where it
    cannot track the two-pass oracle; the rolling dispersion property tiers filter such windows with this predicate.
    """
    return all(well_spread(values[index - window + 1 : index + 1]) for index in range(window - 1, len(values)))


def windows_well_conditioned(values: Sequence[float | None], window: int, floor: float = FLOOR_CONDITIONING) -> bool:
    """
    Whether every trailing ``window``'s variance is a real fraction of its magnitude (a well-conditioned slope or
    standardized moment).

    A window whose variance is below ``scale ** 2 * floor`` is too near-constant for a one-pass rolling ratio to track
    the two-pass oracle; the rolling moment-ratio and benchmark-relative property tiers filter it out and pin the
    degenerate case deterministically in the edge tier. Fewer than two finite values is skipped. The ``floor`` defaults
    to the shared conservative ``FLOOR_CONDITIONING``; a declaration whose empirical disagreement onset was measured
    passes its own tighter, spec-local floor instead — the shared constant itself is never retuned per declaration.
    """
    for index in range(window - 1, len(values)):
        finite = [
            value for value in values[index - window + 1 : index + 1] if value is not None and not math.isnan(value)
        ]
        if len(finite) < 2:
            continue
        variance, scale = _population_variance_and_scale(finite)
        scale = scale or 1.0
        if variance <= scale * scale * floor:
            return False
    return True


# Coherent OHLC bars. A bar must be coherent -- ``low <= open, close <= high`` -- because any function that divides by
# the bar range ``high - low`` is unbounded on an impossible bar: a tiny or negative denominator the numerator escapes,
# so the one-pass form and the multi-pass oracle diverge by far more than any tolerance. That is out-of-domain input,
# not a bug; real bars are always coherent (a flat ``high == low`` bar IS coherent and stays exercised). Each bar is
# drawn as ONE batched ``st.tuples`` of independent primitives, then assembled in pure Python: ``@st.composite``
# re-enters the draw machinery on every interactive ``draw()``, so a per-bar composite generates ~30x slower.
_FRACTION = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_MISSING = st.sampled_from((0, 1, 2))  # 0 keep, 1 -> None, 2 -> NaN; shrinks toward "keep"


def _price(max_price: float) -> st.SearchStrategy[float]:
    """A positive finite price in ``[1.0, max_price]``."""
    return st.floats(min_value=1.0, max_value=max_price, allow_nan=False, allow_infinity=False)


def _low_high(price_a: float, price_b: float) -> tuple[float, float]:
    """Order a price pair into ``(low, high)``."""
    return (price_a, price_b) if price_a <= price_b else (price_b, price_a)


def _within(low: float, high: float, fraction: float) -> float:
    """The point ``low + fraction * (high - low)``, clamped to ``<= high`` so a rounding step never breaks coherence."""
    return min(high, low + fraction * (high - low))


def apply_missing(value: float, choice: int) -> float | None:
    """Keep ``value`` (``0``), drop it to ``None`` (``1``), or replace it with ``NaN`` (``2``)."""
    if choice == 1:
        return None
    if choice == 2:
        return math.nan
    return value


def _assemble_hl(pair: tuple[float, float]) -> tuple[float, float]:
    return (max(pair), min(pair))


def coherent_hl(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float]]:
    """A coherent ``(high, low)`` bar (``high >= low``, both positive finite)."""
    return st.tuples(_price(max_price), _price(max_price)).map(_assemble_hl)


def _assemble_hlc(raw: tuple[float, float, float]) -> tuple[float, float, float]:
    price_a, price_b, close_fraction = raw
    low, high = _low_high(price_a, price_b)
    return (high, low, _within(low, high, close_fraction))


def coherent_hlc(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float, float]]:
    """A coherent ``(high, low, close)`` bar (``low <= close <= high``, positive finite)."""
    return st.tuples(_price(max_price), _price(max_price), _FRACTION).map(_assemble_hlc)


def _assemble_hlcv(raw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    price_a, price_b, close_fraction, volume = raw
    low, high = _low_high(price_a, price_b)
    return (high, low, _within(low, high, close_fraction), volume)


def coherent_hlcv(
    max_price: float = 1e3, max_volume: float = 1e6
) -> st.SearchStrategy[tuple[float, float, float, float]]:
    """A coherent ``(high, low, close, volume)`` bar (``low <= close <= high``, all positive finite)."""
    return st.tuples(_price(max_price), _price(max_price), _FRACTION, _price(max_volume)).map(_assemble_hlcv)


def _assemble_ohlc(raw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    price_a, price_b, open_fraction, close_fraction = raw
    low, high = _low_high(price_a, price_b)
    return (_within(low, high, open_fraction), high, low, _within(low, high, close_fraction))


def coherent_ohlc(max_price: float = 1e3) -> st.SearchStrategy[tuple[float, float, float, float]]:
    """A coherent ``(open, high, low, close)`` bar (``low <= open, close <= high``, all positive finite)."""
    return st.tuples(_price(max_price), _price(max_price), _FRACTION, _FRACTION).map(_assemble_ohlc)


def _assemble_hl_missing(raw: tuple[float, float, int, int]) -> tuple[float | None, float | None]:
    price_a, price_b, keep_high, keep_low = raw
    low, high = _low_high(price_a, price_b)
    return (apply_missing(high, keep_high), apply_missing(low, keep_low))


def coherent_hl_with_missing(max_price: float = 1e4) -> st.SearchStrategy[tuple[float | None, float | None]]:
    """A coherent ``(high, low)`` bar with each field independently kept, nulled, or set to ``NaN``."""
    return st.tuples(_price(max_price), _price(max_price), _MISSING, _MISSING).map(_assemble_hl_missing)


def _assemble_hlc_missing(
    raw: tuple[float, float, float, int, int, int],
) -> tuple[float | None, float | None, float | None]:
    price_a, price_b, close_fraction, keep_high, keep_low, keep_close = raw
    low, high = _low_high(price_a, price_b)
    close = _within(low, high, close_fraction)
    return (apply_missing(high, keep_high), apply_missing(low, keep_low), apply_missing(close, keep_close))


def coherent_hlc_with_missing(
    max_price: float = 1e4,
) -> st.SearchStrategy[tuple[float | None, float | None, float | None]]:
    """A coherent ``(high, low, close)`` bar with each field independently kept, nulled, or set to ``NaN``."""
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
        apply_missing(high, keep_high),
        apply_missing(low, keep_low),
        apply_missing(close, keep_close),
        apply_missing(volume, keep_volume),
    )


def coherent_hlcv_with_missing(
    max_price: float = 1e4, max_volume: float = 1e6
) -> st.SearchStrategy[tuple[float | None, float | None, float | None, float | None]]:
    """A coherent ``(high, low, close, volume)`` bar with each field independently kept, nulled, or set to ``NaN``."""
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
        apply_missing(open_value, keep_open),
        apply_missing(high, keep_high),
        apply_missing(low, keep_low),
        apply_missing(close, keep_close),
    )


def coherent_ohlc_with_missing(
    max_price: float = 1e4,
) -> st.SearchStrategy[tuple[float | None, float | None, float | None, float | None]]:
    """A coherent ``(open, high, low, close)`` bar with each field independently kept, nulled, or set to ``NaN``."""
    return st.tuples(
        _price(max_price), _price(max_price), _FRACTION, _FRACTION, _MISSING, _MISSING, _MISSING, _MISSING
    ).map(_assemble_ohlc_missing)
