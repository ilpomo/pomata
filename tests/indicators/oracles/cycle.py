"""
Naive reference oracle for ``pomata.indicators.cycle``.

A transcription of Ehlers' Hilbert-transform pipeline (a structural mirror of the implementation, so it confirms
internal consistency rather than independence), shared by every cycle indicator's reference. The single-line indicators
return a list; the multi-line ones return a dict keyed by field.
"""

import math
from collections.abc import Sequence

_DIRECT_WARMUP = 32
_PHASE_WARMUP = 63


def _prefix(values: Sequence[float | None]) -> list[float]:
    """
    The leading run of finite prices, stopping at the first ``None`` or non-finite value (``NaN`` or ``inf`` -- a gap
    the recurrence cannot bridge).
    """
    prices: list[float] = []
    for value in values:
        if value is None or not math.isfinite(value):
            break
        prices.append(value)
    return prices


def _mask(values: list[float], length: int, warmup: int) -> list[float | None]:
    """
    ``values[index]`` from ``warmup`` to ``len(values)``, ``None`` before (warm-up) and after (latched gap).
    """
    out: list[float | None] = [None] * length
    for index in range(warmup, len(values)):
        out[index] = values[index]
    return out


def _pipeline(prices: list[float], fast_limit: float, slow_limit: float) -> dict[str, list[float]]:
    """
    Ehlers' shared smooth / detrend / Hilbert-quadrature / homodyne-discriminator pipeline over ``prices``.
    """
    n = len(prices)
    taps = (0.0962, 0.0, 0.5769, 0.0, -0.5769, 0.0, -0.0962)

    def fir(buf: list[float], at: int, gain: float) -> float:
        return gain * sum(c * buf[at - k] for k, c in enumerate(taps) if at - k >= 0)

    keys = ("smooth", "detrend", "quad", "ip", "advance_ip", "advance_q", "sip", "sq", "re", "im", "period", "sp")
    a = {key: [0.0] * n for key in keys}
    phase = [0.0] * n
    sine = [0.0] * n
    lead = [0.0] * n
    raw_trend = [0.0] * n
    trendline = [0.0] * n
    phasor = [0.0] * n
    mama = list(prices)
    fama = list(prices)
    mode = [0.0] * n
    days = 0
    for i in range(6, n):
        prior = a["period"][i - 1]
        gain = 0.075 * prior + 0.54
        a["smooth"][i] = (4.0 * prices[i] + 3.0 * prices[i - 1] + 2.0 * prices[i - 2] + prices[i - 3]) / 10.0
        a["detrend"][i] = fir(a["smooth"], i, gain)
        a["quad"][i] = fir(a["detrend"], i, gain)
        a["ip"][i] = a["detrend"][i - 3]
        a["advance_ip"][i] = fir(a["ip"], i, gain)
        a["advance_q"][i] = fir(a["quad"], i, gain)
        a["sip"][i] = 0.2 * (a["ip"][i] - a["advance_q"][i]) + 0.8 * a["sip"][i - 1]
        a["sq"][i] = 0.2 * (a["quad"][i] + a["advance_ip"][i]) + 0.8 * a["sq"][i - 1]
        a["re"][i] = 0.2 * (a["sip"][i] * a["sip"][i - 1] + a["sq"][i] * a["sq"][i - 1]) + 0.8 * a["re"][i - 1]
        a["im"][i] = 0.2 * (a["sip"][i] * a["sq"][i - 1] - a["sq"][i] * a["sip"][i - 1]) + 0.8 * a["im"][i - 1]
        value = prior
        if a["im"][i] != 0.0 and a["re"][i] != 0.0:
            value = 360.0 / math.degrees(math.atan(a["im"][i] / a["re"][i]))
        if prior != 0.0:
            value = min(1.5 * prior, max(0.67 * prior, value))
        value = min(50.0, max(6.0, value))
        a["period"][i] = 0.2 * value + 0.8 * prior
        a["sp"][i] = 0.33 * a["period"][i] + 0.67 * a["sp"][i - 1]
        window = max(1, int(a["sp"][i] + 0.5))
        rp = sum(math.sin(math.radians(360.0 * c / window)) * a["smooth"][i - c] for c in range(window) if i - c >= 0)
        ipart = sum(
            math.cos(math.radians(360.0 * c / window)) * a["smooth"][i - c] for c in range(window) if i - c >= 0
        )
        # NOTE: exact-zero guard mirrors the impl (a fixed |ipart| cutoff would break scale-invariance, cycle.py).
        ph = math.degrees(math.atan(rp / ipart)) if ipart != 0.0 else 90.0 * ((rp > 0.0) - (rp < 0.0))
        ph += 90.0 + 360.0 / a["sp"][i]
        if ipart < 0.0:
            ph += 180.0
        if ph > 315.0:
            ph -= 360.0
        phase[i] = ph
        sine[i] = math.sin(math.radians(ph))
        lead[i] = math.sin(math.radians(ph + 45.0))
        average = sum(prices[i - c] for c in range(window) if i - c >= 0) / window
        trendline[i] = (4.0 * average + 3.0 * raw_trend[i - 1] + 2.0 * raw_trend[i - 2] + raw_trend[i - 3]) / 10.0
        raw_trend[i] = average
        phasor[i] = math.degrees(math.atan(a["quad"][i] / a["ip"][i])) if a["ip"][i] != 0.0 else phasor[i - 1]
        step = max(1.0, phasor[i - 1] - phasor[i])
        alpha = max(slow_limit, fast_limit / step)
        mama[i] = alpha * prices[i] + (1.0 - alpha) * mama[i - 1]
        fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * fama[i - 1]
        flag = 1.0
        if (sine[i] > lead[i]) != (sine[i - 1] > lead[i - 1]):
            days = 0
            flag = 0.0
        days += 1
        if days < 0.5 * a["sp"][i]:
            flag = 0.0
        rate = phase[i] - phase[i - 1]
        if a["sp"][i] != 0.0 and 0.67 * 360.0 / a["sp"][i] < rate < 1.5 * 360.0 / a["sp"][i]:
            flag = 0.0
        if trendline[i] != 0.0 and abs((a["smooth"][i] - trendline[i]) / trendline[i]) >= 0.015:
            flag = 1.0
        mode[i] = flag
    return {
        "period": a["sp"],
        "phase": phase,
        "in_phase": a["ip"],
        "quadrature": a["quad"],
        "sine": sine,
        "lead_sine": lead,
        "trendline": trendline,
        "mama": mama,
        "fama": fama,
        "trend_mode": mode,
    }


def dominant_cycle_period_reference(values: Sequence[float | None]) -> list[float | None]:
    """
    Naive Ehlers dominant-cycle period, the oracle for :func:`pomata.indicators.dominant_cycle_period`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``32``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None``.
    """
    prices = _prefix(values)
    return _mask(_pipeline(prices, 0.5, 0.05)["period"], len(values), _DIRECT_WARMUP)


def dominant_cycle_phase_reference(values: Sequence[float | None]) -> list[float | None]:
    """
    Naive Ehlers dominant-cycle phase, the oracle for :func:`pomata.indicators.dominant_cycle_phase`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``63``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None``.
    """
    prices = _prefix(values)
    return _mask(_pipeline(prices, 0.5, 0.05)["phase"], len(values), _PHASE_WARMUP)


def hilbert_phasor_reference(values: Sequence[float | None]) -> dict[str, list[float | None]]:
    """
    Naive Ehlers in-phase / quadrature phasor, the oracle for :func:`pomata.indicators.hilbert_phasor`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``32``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None`` on both the ``in_phase`` and
    ``quadrature`` lanes.
    """
    prices = _prefix(values)
    pipeline = _pipeline(prices, 0.5, 0.05)
    length = len(values)
    return {field: _mask(pipeline[field], length, _DIRECT_WARMUP) for field in ("in_phase", "quadrature")}


def sine_wave_reference(values: Sequence[float | None]) -> dict[str, list[float | None]]:
    """
    Naive Ehlers sine / lead-sine wave, the oracle for :func:`pomata.indicators.sine_wave`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``63``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None`` on both the ``sine`` and
    ``lead_sine`` lanes.
    """
    prices = _prefix(values)
    pipeline = _pipeline(prices, 0.5, 0.05)
    length = len(values)
    return {field: _mask(pipeline[field], length, _PHASE_WARMUP) for field in ("sine", "lead_sine")}


def trend_mode_reference(values: Sequence[float | None]) -> list[float | None]:
    """
    Naive Ehlers trend / cycle flag, the oracle for :func:`pomata.indicators.trend_mode`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``63``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None``.
    """
    prices = _prefix(values)
    return _mask(_pipeline(prices, 0.5, 0.05)["trend_mode"], len(values), _PHASE_WARMUP)


def hilbert_trendline_reference(values: Sequence[float | None]) -> list[float | None]:
    """
    Naive Ehlers instantaneous trendline, the oracle for :func:`pomata.indicators.hilbert_trendline`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``63``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None``.
    """
    prices = _prefix(values)
    return _mask(_pipeline(prices, 0.5, 0.05)["trendline"], len(values), _PHASE_WARMUP)


def mama_reference(
    values: Sequence[float | None],
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> dict[str, list[float | None]]:
    """
    Naive Ehlers MESA adaptive moving average and its companion, the oracle for :func:`pomata.indicators.mama`.

    A structural mirror of the shipped Ehlers pipeline (the adaptive dominant-cycle feedback has no closed
    form), so its agreement confirms internal consistency rather than independence; the independent witnesses
    are the frozen golden masters and the TA-Lib differential. Its delicate points are the ``32``-row
    warm-up mask and the latching missing-data contract: the pipeline consumes the longest clean prefix, so
    every row from the first ``None``, ``NaN``, or ``inf`` onward is ``None`` on both the ``mama`` and ``fama`` lanes.
    """
    prices = _prefix(values)
    pipeline = _pipeline(prices, fast_limit, slow_limit)
    length = len(values)
    return {field: _mask(pipeline[field], length, _DIRECT_WARMUP) for field in ("mama", "fama")}
