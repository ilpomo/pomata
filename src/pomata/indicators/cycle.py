"""
Cycle-analysis indicators built on John Ehlers' Hilbert-transform dominant-cycle measurement.

The price is smoothed, detrended, and split into in-phase and quadrature components by a Hilbert-transform FIR filter;
a homodyne discriminator (multiplying the complex signal by its one-bar-lagged conjugate) yields the instantaneous
dominant-cycle period, and the dominant-cycle phase, sine wave, instantaneous trendline, trend/cycle flag, and the MESA
adaptive moving average all follow from it. The recurrence is sequential, so a single shared pure-Python pipeline runs
it once via ``map_batches`` (like :func:`kama`); each indicator reads its own output and masks its
warm-up. A ``null``, ``NaN``, or ``inf`` price latches ``null`` from there, since the adaptive recurrence cannot bridge
a gap.
"""

import math
from dataclasses import dataclass
from functools import partial

import polars as pl

from pomata._expr import float64_expr

__all__ = (
    "dominant_cycle_period",
    "dominant_cycle_phase",
    "hilbert_phasor",
    "hilbert_trendline",
    "mama",
    "sine_wave",
    "trend_mode",
)

# Per-indicator warm-up: the recursive smoothers settle in ~32 bars; the phase-derived outputs need a further
# dominant-cycle look-back of ~31 bars (the running transform / trendline cycle average), hence 63.
_DIRECT_WARMUP = 32
_PHASE_WARMUP = 63
_PHASE_WRAP_DEGREES = 315.0
_TREND_MODE_THRESHOLD = 0.015

_MAMA_FAST_LIMIT = 0.5
_MAMA_SLOW_LIMIT = 0.05


@dataclass(frozen=True)
class _Pipeline:
    """
    The full set of per-bar outputs of Ehlers' Hilbert-transform pipeline, each a list as long as the input prefix.
    """

    period: list[float]
    phase: list[float]
    in_phase: list[float]
    quadrature: list[float]
    sine: list[float]
    lead_sine: list[float]
    trendline: list[float]
    mama: list[float]
    fama: list[float]
    trend_mode: list[float]


def _fir(
    series: list[float],
    index: int,
    adjust: float,
) -> float:
    """
    Ehlers' six-tap Hilbert-transform quadrature filter at ``index``, amplitude-compensated by ``adjust``.
    """

    def lag(
        steps: int,
    ) -> float:
        position = index - steps
        return series[position] if position >= 0 else 0.0

    return (0.0962 * series[index] + 0.5769 * lag(2) - 0.5769 * lag(4) - 0.0962 * lag(6)) * adjust


def _clean_prefix(
    series: pl.Series,
) -> list[float]:
    """
    The leading run of finite prices, stopping at the first ``null`` or non-finite value (``NaN`` or ``inf`` -- a gap
    the recurrence cannot bridge).
    """
    prices: list[float] = []
    for value in series.to_list():
        if value is None or not math.isfinite(value):
            break
        prices.append(value)
    return prices


def _ehlers_pipeline(
    prices: list[float],
    limit_fast: float,
    limit_slow: float,
) -> _Pipeline:
    """
    Run Ehlers' shared smooth / detrend / Hilbert-quadrature / homodyne-discriminator pipeline over ``prices``.
    """
    count = len(prices)
    smooth = [0.0] * count
    detrender = [0.0] * count
    quad = [0.0] * count
    inphase = [0.0] * count
    advance_inphase = [0.0] * count
    advance_quad = [0.0] * count
    smoothed_inphase = [0.0] * count
    smoothed_quad = [0.0] * count
    real = [0.0] * count
    imag = [0.0] * count
    period = [0.0] * count
    smooth_period = [0.0] * count
    phase = [0.0] * count
    sine = [0.0] * count
    lead_sine = [0.0] * count
    raw_trend = [0.0] * count
    trendline = [0.0] * count
    phasor_phase = [0.0] * count
    mama_line = list(prices)
    fama_line = list(prices)
    mode_line = [0.0] * count
    days_in_trend = 0
    for index in range(6, count):
        previous = period[index - 1]
        adjust = 0.075 * previous + 0.54
        smooth[index] = (
            4.0 * prices[index] + 3.0 * prices[index - 1] + 2.0 * prices[index - 2] + prices[index - 3]
        ) / 10.0
        detrender[index] = _fir(smooth, index, adjust)
        quad[index] = _fir(detrender, index, adjust)
        inphase[index] = detrender[index - 3]
        advance_inphase[index] = _fir(inphase, index, adjust)
        advance_quad[index] = _fir(quad, index, adjust)
        smoothed_inphase[index] = 0.2 * (inphase[index] - advance_quad[index]) + 0.8 * smoothed_inphase[index - 1]
        smoothed_quad[index] = 0.2 * (quad[index] + advance_inphase[index]) + 0.8 * smoothed_quad[index - 1]
        real[index] = (
            0.2
            * (smoothed_inphase[index] * smoothed_inphase[index - 1] + smoothed_quad[index] * smoothed_quad[index - 1])
            + 0.8 * real[index - 1]
        )
        imag[index] = (
            0.2
            * (smoothed_inphase[index] * smoothed_quad[index - 1] - smoothed_quad[index] * smoothed_inphase[index - 1])
            + 0.8 * imag[index - 1]
        )
        estimate = previous
        if imag[index] != 0.0 and real[index] != 0.0:
            estimate = 360.0 / math.degrees(math.atan(imag[index] / real[index]))
        if previous != 0.0:
            estimate = max(0.67 * previous, min(1.5 * previous, estimate))
        estimate = max(6.0, min(50.0, estimate))
        period[index] = 0.2 * estimate + 0.8 * previous
        smooth_period[index] = 0.33 * period[index] + 0.67 * smooth_period[index - 1]
        cycle_length = max(1, int(smooth_period[index] + 0.5))
        real_part = 0.0
        imag_part = 0.0
        for step in range(min(cycle_length, index + 1)):
            radians = math.radians(360.0 * step / cycle_length)
            real_part += math.sin(radians) * smooth[index - step]
            imag_part += math.cos(radians) * smooth[index - step]
        # NOTE: guard an EXACT zero, not a fixed |imag_part| cutoff. imag_part is the cosine projection of the smoothed
        # PRICE, so it is degree-1 (scales with the level); a fixed threshold would snap a low-amplitude carrier to a
        # different phase branch under a lossless rescale -- breaking scale-invariance -- whereas atan saturates to
        # +/-90 as imag_part -> 0, so exact-zero is the continuous limit (and matches the phasor-phase guard below).
        bar_phase = math.degrees(math.atan(real_part / imag_part)) if imag_part != 0.0 else 90.0 * _sign(real_part)
        bar_phase += 90.0 + 360.0 / smooth_period[index]
        if imag_part < 0.0:
            bar_phase += 180.0
        if bar_phase > _PHASE_WRAP_DEGREES:
            bar_phase -= 360.0
        phase[index] = bar_phase
        sine[index] = math.sin(math.radians(bar_phase))
        lead_sine[index] = math.sin(math.radians(bar_phase + 45.0))
        cycle_average = sum(prices[index - step] for step in range(min(cycle_length, index + 1))) / cycle_length
        trendline[index] = (
            4.0 * cycle_average + 3.0 * raw_trend[index - 1] + 2.0 * raw_trend[index - 2] + raw_trend[index - 3]
        ) / 10.0
        raw_trend[index] = cycle_average
        phasor_phase[index] = (
            math.degrees(math.atan(quad[index] / inphase[index])) if inphase[index] != 0.0 else phasor_phase[index - 1]
        )
        delta_phase = max(1.0, phasor_phase[index - 1] - phasor_phase[index])
        alpha = max(limit_slow, limit_fast / delta_phase)
        mama_line[index] = alpha * prices[index] + (1.0 - alpha) * mama_line[index - 1]
        fama_line[index] = 0.5 * alpha * mama_line[index] + (1.0 - 0.5 * alpha) * fama_line[index - 1]
        flag = 1.0
        if (sine[index] > lead_sine[index]) != (sine[index - 1] > lead_sine[index - 1]):
            days_in_trend = 0
            flag = 0.0
        days_in_trend += 1
        if days_in_trend < 0.5 * smooth_period[index]:
            flag = 0.0
        phase_rate = phase[index] - phase[index - 1]
        if (
            smooth_period[index] != 0.0
            and 0.67 * 360.0 / smooth_period[index] < phase_rate < 1.5 * 360.0 / smooth_period[index]
        ):
            flag = 0.0
        if (
            trendline[index] != 0.0
            and abs((smooth[index] - trendline[index]) / trendline[index]) >= _TREND_MODE_THRESHOLD
        ):
            flag = 1.0
        mode_line[index] = flag
    return _Pipeline(smooth_period, phase, inphase, quad, sine, lead_sine, trendline, mama_line, fama_line, mode_line)


def _sign(
    value: float,
) -> float:
    """
    ``-1`` / ``0`` / ``1`` by the sign of ``value`` (as EasyLanguage's ``Sign``).
    """
    return (value > 0.0) - (value < 0.0)


def _emit(
    values: list[float],
    length: int,
    warmup: int,
) -> list[float | None]:
    """
    Mask the warm-up: ``values[index]`` from ``warmup`` to ``len(values)``, ``None`` before and (latched) after.
    """
    result: list[float | None] = [None] * length
    for index in range(warmup, len(values)):
        result[index] = values[index]
    return result


def _line_kernel(
    series: pl.Series,
    *,
    field: str,
    warmup: int,
) -> pl.Series:
    """
    Run the shared pipeline and emit one named single-line output, warm-up-masked and gap-latched.
    """
    prices = _clean_prefix(series)
    pipeline = _ehlers_pipeline(prices, _MAMA_FAST_LIMIT, _MAMA_SLOW_LIMIT)
    return pl.Series(_emit(getattr(pipeline, field), len(series), warmup), dtype=pl.Float64)


def _struct_kernel(
    series: pl.Series,
    *,
    fields: tuple[str, str],
    warmup: int,
    limit_fast: float = _MAMA_FAST_LIMIT,
    limit_slow: float = _MAMA_SLOW_LIMIT,
) -> pl.Series:
    """
    Run the shared pipeline and emit a two-field struct output, warm-up-masked and gap-latched.

    The pipeline attribute names in ``fields`` double as the output struct field names. ``limit_fast`` / ``limit_slow``
    default to the MAMA alpha bounds; only the MAMA/FAMA pair overrides them with its caller-supplied limits.
    """
    prices = _clean_prefix(series)
    pipeline = _ehlers_pipeline(prices, limit_fast, limit_slow)
    length = len(series)
    return pl.DataFrame(
        {
            fields[0]: _emit(getattr(pipeline, fields[0]), length, warmup),
            fields[1]: _emit(getattr(pipeline, fields[1]), length, warmup),
        }
    ).to_struct()


def dominant_cycle_period(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Dominant Cycle Period (Hilbert transform).

    John Ehlers' real-time measurement of the market's dominant cycle length, in bars. The price is smoothed and
    detrended, a Hilbert-transform FIR filter resolves it into in-phase and quadrature components, and a homodyne
    discriminator gives the instantaneous period from the angle of the resulting beat note:

    .. math::

        \mathrm{Period}_t = \frac{360}{\arctan(\mathrm{Im}_t / \mathrm{Re}_t)},

    with :math:`\arctan` returning **degrees**, which is then clamped to ``[6, 50]`` bars (and to between ``0.67`` and
    ``1.5`` times the previous bar — ``+50%`` up, ``-33%`` down) and doubly smoothed (the reported ``SmoothPeriod``).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The dominant-cycle period for each row, the same length as ``expr``, settling into ``[6, 50]`` (the raw
        estimate is clamped there before the reported double-smoothing, so the first emitted rows may sit marginally
        below ``6``). The first ``32`` rows are ``null`` (the warm-up the recursive smoothers need to settle).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``dominant_cycle_period(pl.col("close")).over("ticker")``.

    See Also:
        - :func:`dominant_cycle_phase`: The phase of the same dominant cycle.
        - :func:`hilbert_phasor`: The phasor the period is measured from.
        - :func:`hilbert_trendline`: Averages the price over one cycle of this period.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        The dominant cycle of a clean period-20 sine, read at the last bar (close to its true length of ``20`` bars):

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import dominant_cycle_period
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> round(frame.select(dominant_cycle_period(pl.col("close")).alias("p"))["p"][-1], 2)
        20.03
    """
    expr = float64_expr(expr)
    return expr.map_batches(partial(_line_kernel, field="period", warmup=_DIRECT_WARMUP), return_dtype=pl.Float64)


def dominant_cycle_phase(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Dominant Cycle Phase (Hilbert transform).

    The phase of the market's dominant cycle, in degrees: ``0`` at the upward mean-crossing, ``90`` at the cycle high,
    ``270`` at the cycle low, advancing through the cycle. It is read off a running discrete transform of the smoothed
    price over the dominant-cycle window, then lag-compensated.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The dominant-cycle phase in degrees for each row, the same length as ``expr``. The first ``63`` rows are
        ``null`` (the warm-up: the smoothers' settling plus the dominant-cycle look-back).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``dominant_cycle_phase(pl.col("close")).over("ticker")``.

        **When it breaks:**

        On a constant (flat) price the discrete transform's projections are pure cancellation residuals, so the phase
        is numerically arbitrary — there is no cycle to measure. The phase branch guards an *exact* zero of the cosine
        projection (saturating to ``±90`` as that projection vanishes), rather than the inventor's fixed ``0.001``
        absolute cutoff; this is the continuous limit and keeps the phase invariant under a lossless rescale of the
        price, whereas a fixed threshold would be scale-dependent.

    See Also:
        - :func:`dominant_cycle_period`: The length of the same dominant cycle.
        - :func:`sine_wave`: The sine of this phase.
        - :func:`mama`: The adaptive average this phase's rate drives.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        The dominant-cycle phase of a clean period-20 sine, read at the last bar (in degrees):

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import dominant_cycle_phase
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> round(frame.select(dominant_cycle_phase(pl.col("close")).alias("p"))["p"][-1], 2)
        -17.84
    """
    expr = float64_expr(expr)
    return expr.map_batches(partial(_line_kernel, field="phase", warmup=_PHASE_WARMUP), return_dtype=pl.Float64)


def hilbert_phasor(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Hilbert Transform Phasor — in-phase and quadrature components.

    The detrended price resolved into its in-phase and quadrature components by the Hilbert transform: the complex
    *phasor* whose rotation traces the dominant cycle.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        A struct ``pl.Expr`` with ``Float64`` fields ``in_phase`` / ``quadrature``, the same length as ``expr``. The
        first ``32`` rows are ``null`` (warm-up). Read one line with ``.struct.field("in_phase")`` or split both with
        ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``hilbert_phasor(pl.col("close")).over("ticker")``.

    See Also:
        - :func:`dominant_cycle_period`: Measured from this phasor by the homodyne discriminator.
        - :func:`mama`: Adapts on the rate of change of this phasor's phase.
        - :func:`dominant_cycle_phase`: The companion dominant-cycle phase.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        The in-phase and quadrature components on a clean period-20 sine, at the last bar:

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import hilbert_phasor
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> phasor = frame.select(hilbert_phasor(pl.col("close")).alias("h")).unnest("h")
        >>> round(phasor["in_phase"][-1], 2), round(phasor["quadrature"][-1], 2)
        (-0.8, 0.61)
    """
    expr = float64_expr(expr)
    return expr.map_batches(
        partial(_struct_kernel, fields=("in_phase", "quadrature"), warmup=_DIRECT_WARMUP),
        return_dtype=pl.Struct({"in_phase": pl.Float64, "quadrature": pl.Float64}),
    )


def hilbert_trendline(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Hilbert Transform Instantaneous Trendline.

    Ehlers' instantaneous trendline: the price averaged over exactly one dominant cycle (a self-adjusting moving
    average), then smoothed. Because it spans a whole cycle, the cyclic component cancels and only the trend remains.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The instantaneous trendline for each row, the same length as ``expr``, on the price scale. The first ``63``
        rows are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``hilbert_trendline(pl.col("close")).over("ticker")``.

    See Also:
        - :func:`trend_mode`: Uses the price's deviation from this trendline.
        - :func:`dominant_cycle_period`: The cycle length this averages the price over.
        - :func:`mama`: The adaptive average from the same pipeline.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        Spanning a whole cycle, the trendline cancels the swing and tracks the mean level (here ``100``):

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import hilbert_trendline
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> round(frame.select(hilbert_trendline(pl.col("close")).alias("t"))["t"][-1], 2)
        100.0
    """
    expr = float64_expr(expr)
    return expr.map_batches(partial(_line_kernel, field="trendline", warmup=_PHASE_WARMUP), return_dtype=pl.Float64)


def mama(
    expr: pl.Expr,
    *,
    limit_fast: float = 0.5,
    limit_slow: float = 0.05,
) -> pl.Expr:
    r"""
    MESA Adaptive Moving Average (MAMA), with its companion FAMA.

    John Ehlers' adaptive average: the smoothing constant tracks the rate of change of the dominant-cycle phase, so the
    average follows price closely when the cycle phase turns quickly and lags when it is slow. ``FAMA`` (the Following
    Adaptive Moving Average) is a second, slower pass used as a signal line:

    .. math::

        \alpha_t &= \max\!\Bigl(\text{limit\_slow},\ \frac{\text{limit\_fast}}{\Delta\phi_t}\Bigr), \\
        \mathrm{MAMA}_t &= \alpha_t\,\mathrm{price}_t + (1 - \alpha_t)\,\mathrm{MAMA}_{t-1}, \\
        \mathrm{FAMA}_t &= \tfrac{1}{2}\alpha_t\,\mathrm{MAMA}_t + (1 - \tfrac{1}{2}\alpha_t)\,\mathrm{FAMA}_{t-1},

    where :math:`\Delta\phi_t = \max(1,\ \phi_{t-1} - \phi_t)` is the per-bar decrease of the phasor phase, floored at
    ``1`` degree so that ``limit_fast`` is the true upper bound on :math:`\alpha_t`.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        limit_fast: Upper bound on the smoothing constant (a fast cycle). Must be in ``(0, 1]`` and ``>= limit_slow``.
        limit_slow: Lower bound on the smoothing constant (a slow cycle). Must be in ``(0, 1]``.

    Returns:
        A struct ``pl.Expr`` with ``Float64`` fields ``mama`` / ``fama``, the same length as ``expr``. The first ``32``
        rows are ``null`` (warm-up). Read one line with ``.struct.field("mama")`` or split both with
        ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``limit_fast`` or ``limit_slow`` is outside ``(0, 1]`` — the smoothing constant is a weight, so
            a limit above ``1`` makes ``1 - alpha`` negative and the recurrence diverges — or if ``limit_fast <
            limit_slow``, which would pin the adaptive smoothing constant at ``limit_slow`` and make ``limit_fast`` a
            false upper bound.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Seeding:**

        Both lines are seeded at the price prefix — ``MAMA`` and ``FAMA`` start from the price and the recurrence runs
        from there. Ehlers' original presentation instead zero-initializes both lines, so the two report different
        values across the warm-up region before the exponential weighting washes the seed out; pomata's price seed is
        the saner choice for a price-level average. Port warm-up-sensitive logic accordingly.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``mama(pl.col("close")).over("ticker")``.

    See Also:
        - :func:`hilbert_phasor`: The phasor whose phase rate sets the smoothing constant.
        - :func:`kama`: Another adaptive moving average, adapting on the efficiency ratio.
        - :func:`dominant_cycle_phase`: The dominant-cycle phase from the same pipeline.

    References:
        - Ehlers, J. F. "MAMA — The Mother of Adaptive Moving Averages." MESA Software.

    Examples:
        Both adaptive lines track the level of a clean period-20 cycle (here ``100``), at the last bar:

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import mama
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> lines = frame.select(mama(pl.col("close")).alias("m")).unnest("m")
        >>> round(lines["mama"][-1], 2), round(lines["fama"][-1], 2)
        (99.67, 99.96)
    """
    expr = float64_expr(expr)
    if not 0.0 < limit_fast <= 1.0:
        raise ValueError(f"limit_fast must be in the half-open interval (0, 1], got {limit_fast}")
    if not 0.0 < limit_slow <= 1.0:
        raise ValueError(f"limit_slow must be in the half-open interval (0, 1], got {limit_slow}")
    if limit_fast < limit_slow:
        raise ValueError(f"limit_fast must be >= limit_slow, got limit_fast={limit_fast}, limit_slow={limit_slow}")
    return expr.map_batches(
        partial(
            _struct_kernel,
            fields=("mama", "fama"),
            warmup=_DIRECT_WARMUP,
            limit_fast=limit_fast,
            limit_slow=limit_slow,
        ),
        return_dtype=pl.Struct({"mama": pl.Float64, "fama": pl.Float64}),
    )


def sine_wave(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Hilbert Transform Sine Wave.

    Ehlers' sine-wave indicator: the sine of the dominant-cycle phase, and a lead sine advanced by ``45°``. Their
    crossings mark cycle turning points and lead the price in a cycle (and diverge in a trend).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        A struct ``pl.Expr`` with ``Float64`` fields ``sine`` / ``lead_sine`` in ``[-1, 1]``, the same length as
        ``expr``. The first ``63`` rows are ``null`` (warm-up). Read one line with ``.struct.field("sine")`` or split
        both with ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``sine_wave(pl.col("close")).over("ticker")``.

        The underlying phase branch guards an *exact* zero of the cosine projection (saturating to ``±90`` as that
        projection vanishes), rather than the inventor's fixed ``0.001`` absolute cutoff; this is the continuous limit
        and keeps the sine invariant under a lossless rescale of the price, whereas a fixed threshold would be
        scale-dependent.

    See Also:
        - :func:`dominant_cycle_phase`: The phase these are the sine of.
        - :func:`trend_mode`: Combines these sine-wave crossings.
        - :func:`dominant_cycle_period`: The cycle these trace.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        The sine and lead-sine of a clean period-20 cycle, at the last bar (both in ``[-1, 1]``):

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import sine_wave
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> waves = frame.select(sine_wave(pl.col("close")).alias("s")).unnest("s")
        >>> round(waves["sine"][-1], 2), round(waves["lead_sine"][-1], 2)
        (-0.31, 0.46)
    """
    expr = float64_expr(expr)
    return expr.map_batches(
        partial(_struct_kernel, fields=("sine", "lead_sine"), warmup=_PHASE_WARMUP),
        return_dtype=pl.Struct({"sine": pl.Float64, "lead_sine": pl.Float64}),
    )


def trend_mode(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Hilbert Transform Trend vs Cycle Mode.

    Ehlers' market-mode flag: ``1.0`` when the market is trending, ``0.0`` when it is cycling. It combines the
    sine-wave crossings, the dominant-cycle phase rate, and the deviation of the smoothed price from the instantaneous
    trendline.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The mode flag (``1.0`` trend / ``0.0`` cycle) for each row, the same length as ``expr``. The first ``63`` rows
        are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- the fixed FIR smoothing and quadrature stages are computed independently, but the adaptive
        dominant-cycle period feeds back into its own measurement and the stages built on it, so the reference oracle
        replays Ehlers' pipeline and confirms its internal consistency rather than independence; the independent witness
        is the set of frozen golden masters (and, for MAMA, TA-Lib parity). Where measurable the oracle agrees to ten
        significant figures (a ``1e-10`` band) on any finite input within a sane dynamic range, except on a flat or
        period-two (even-lag) series, where the Hilbert quadrature is a pure cancellation residual and the measurement
        is ill-conditioned (there is no cycle to measure). ``CORRECTNESS.md`` gives the method and the
        float-conditioning limit beyond it.

        **Edge-case behavior:**

        - **Null / NaN / inf** — a ``null``, ``NaN``, or ``inf`` price latches ``null`` for every row from there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series and never spans series boundaries, e.g. ``trend_mode(pl.col("close")).over("ticker")``.

        The underlying phase branch guards an *exact* zero of the cosine projection (saturating to ``±90`` as that
        projection vanishes), rather than the inventor's fixed ``0.001`` absolute cutoff; this is the continuous limit
        and keeps the phase invariant under a lossless rescale of the price, whereas a fixed threshold would be
        scale-dependent.

    See Also:
        - :func:`hilbert_trendline`: The trendline the mode compares the price against.
        - :func:`sine_wave`: The sine-wave crossings the mode combines.
        - :func:`dominant_cycle_phase`: The phase rate the mode also uses.

    References:
        - Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.

    Examples:
        A small pure cycle -- its swing below ~1.5% of price -- stays in cycle mode, so the flag stays ``0`` over a
        clean low-amplitude period-20 sine:

        >>> import math
        >>> import polars as pl
        >>> from pomata.indicators import trend_mode
        >>>
        >>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())
        >>> frame.select(trend_mode(pl.col("close")).alias("t"))["t"].drop_nulls().unique().to_list()
        [0.0]
    """
    expr = float64_expr(expr)
    return expr.map_batches(partial(_line_kernel, field="trend_mode", warmup=_PHASE_WARMUP), return_dtype=pl.Float64)
