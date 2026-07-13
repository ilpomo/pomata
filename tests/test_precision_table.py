"""
Pins the precision figures published in ``docs/trust.md`` to the suite, so they cannot silently drift or become
unreproducible.

Builds the deterministic 400-bar OHLC series the trust page quotes (closed-form, no RNG, so the figures regenerate from
a fresh clone), and for every indicator in the table checks two things: the final value still equals the published
figure, and it agrees with the independent reference oracle to the precision promise. This is the single source of truth
for the series; ``scripts/precision_table.py`` imports it to regenerate the full table, including the TA-Lib column
(which needs the optional ``differential`` dependency and is verified separately by the differential checks).
"""

import math

import polars as pl
import pytest
from tests.indicators.oracles import atr_reference, ema_reference, macd_reference, rsi_reference, sma_reference
from tests.support import RELATIVE_TOLERANCE_REFERENCE, assert_matches

from pomata.indicators import atr, ema, macd, rsi, sma

N = 400
CLOSE = [100.0 + 10.0 * math.sin(i / 9.0) + 4.0 * math.cos(i / 23.0) + 0.03 * i for i in range(N)]
HIGH = [close + 0.6 + 0.4 * abs(math.sin(i / 5.0)) for i, close in enumerate(CLOSE)]
LOW = [close - 0.6 - 0.4 * abs(math.cos(i / 7.0)) for i, close in enumerate(CLOSE)]
_FRAME = pl.DataFrame({"high": HIGH, "low": LOW, "close": CLOSE})


def _last(values: list[float | None]) -> float:
    """The final value, asserted finite (the warm-up never reaches the last bar of a 400-bar series)."""
    last = values[-1]
    assert last is not None
    return last


def _pomata_last(expr: pl.Expr) -> float:
    """The indicator's final value on the series."""
    return _last(_FRAME.select(expr.alias("y"))["y"].to_list())


_MACD_LINE = macd(pl.col("close"), window_fast=12, window_slow=26, window_signal=9).struct.field("macd")
_POMATA: dict[str, float] = {
    "sma(20)": _pomata_last(sma(pl.col("close"), 20)),
    "ema(20)": _pomata_last(ema(pl.col("close"), 20)),
    "rsi(14)": _pomata_last(rsi(pl.col("close"), 14)),
    "atr(14)": _pomata_last(atr(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
    "macd(12,26,9)": _pomata_last(_MACD_LINE),
}
_ORACLE: dict[str, float] = {
    "sma(20)": _last(sma_reference(CLOSE, 20)),
    "ema(20)": _last(ema_reference(CLOSE, 20)),
    "rsi(14)": _last(rsi_reference(CLOSE, 14)),
    "atr(14)": _last(atr_reference(HIGH, LOW, CLOSE, 14)),
    "macd(12,26,9)": _last(macd_reference(CLOSE, 12, 26, 9)["macd"]),
}
# The ``pomata`` column published in ``docs/trust.md``, frozen so any indicator (or table) change fails the suite.
_PUBLISHED: dict[str, float] = {
    "sma(20)": 105.15146076264764,
    "ema(20)": 107.7299930892346,
    "rsi(14)": 85.20908701341023,
    "atr(14)": 1.904174462198776,
    "macd(12,26,9)": 2.523444380829531,
}


class TestPrecisionTable:
    """
    The ``docs/trust.md`` precision figures, pinned and reproduced.
    """

    @pytest.mark.parametrize("name", list(_PUBLISHED))
    def test_matches_published_figure(self, name: str) -> None:
        """
        Verifies the indicator's final value still equals the figure published in the ``docs/trust.md`` precision table.
        """
        assert_matches([_POMATA[name]], [_PUBLISHED[name]], rel_tol=RELATIVE_TOLERANCE_REFERENCE)

    @pytest.mark.parametrize("name", list(_PUBLISHED))
    def test_matches_reference_oracle(self, name: str) -> None:
        """
        Verifies the indicator agrees with the independent reference oracle on the series, to the precision promise.
        """
        assert_matches([_POMATA[name]], [_ORACLE[name]], rel_tol=RELATIVE_TOLERANCE_REFERENCE)
