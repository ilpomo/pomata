"""
Regenerate the precision figures shown in ``docs/trust.md``.

Run from the repo root with the differential group (TA-Lib + NumPy) installed::

    uv run --group differential python scripts/precision_table.py

The deterministic 400-bar series and the ``pomata`` / reference figures are owned and pinned by
``tests/test_precision_table.py``; this tool reuses that series, adds the TA-Lib (C reference) column, and
prints the two blocks that appear in ``docs/trust.md`` — each delta a relative residual against its column's
reference — so the headline figures can be refreshed from a single source.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import talib

# The series and oracles live under tests/; make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pomata.indicators import atr, ema, macd, rsi, sma
from tests.indicators.oracles import (
    atr_reference,
    ema_reference,
    macd_reference,
    rsi_reference,
    sma_reference,
)
from tests.test_precision_table import CLOSE, HIGH, LOW, residual_cell

_frame = pl.DataFrame({"high": HIGH, "low": LOW, "close": CLOSE})
_close, _high, _low = np.asarray(CLOSE), np.asarray(HIGH), np.asarray(LOW)


def _last(values: list[float | None]) -> float:
    """The final value, asserted finite (the warm-up never reaches the last bar of a 400-bar series)."""
    last = values[-1]
    assert last is not None
    return last


def _pomata_last(expr: pl.Expr) -> float:
    return _last(_frame.select(expr.alias("y"))["y"].to_list())


_macd_line = macd(pl.col("close"), window_fast=12, window_slow=26, window_signal=9).struct.field("macd")
POMATA = {
    "sma(20)": _pomata_last(sma(pl.col("close"), 20)),
    "ema(20)": _pomata_last(ema(pl.col("close"), 20)),
    "rsi(14)": _pomata_last(rsi(pl.col("close"), 14)),
    "atr(14)": _pomata_last(atr(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
    "macd(12,26,9)": _pomata_last(_macd_line),
}
ORACLE = {
    "sma(20)": _last(sma_reference(CLOSE, 20)),
    "ema(20)": _last(ema_reference(CLOSE, 20)),
    "rsi(14)": _last(rsi_reference(CLOSE, 14)),
    "atr(14)": _last(atr_reference(HIGH, LOW, CLOSE, 14)),
    "macd(12,26,9)": _last(macd_reference(CLOSE, 12, 26, 9)["macd"]),
}
TALIB = {
    "sma(20)": float(talib.SMA(_close, 20)[-1]),
    "ema(20)": float(talib.EMA(_close, 20)[-1]),
    "rsi(14)": float(talib.RSI(_close, 14)[-1]),
    "atr(14)": float(talib.ATR(_high, _low, _close, 14)[-1]),
    "macd(12,26,9)": float(talib.MACD(_close, 12, 26, 9)[0][-1]),
}


def _vs_oracle(name: str) -> str:
    return residual_cell(POMATA[name], ORACLE[name])


def _vs_talib(name: str) -> str:
    return residual_cell(POMATA[name], TALIB[name])


def main() -> None:
    print("rsi(14), the final value of a 400-bar series:\n")
    print(f"    pomata      {POMATA['rsi(14)']!r}")
    print(f"    oracle      {ORACLE['rsi(14)']!r}   <- independent reimplementation")
    print(f"    TA-Lib      {TALIB['rsi(14)']!r}   <- relative delta {_vs_talib('rsi(14)')}")
    print()
    print("| indicator | pomata | vs reimplementation | vs TA-Lib |")
    print("| --- | --- | :-: | :-: |")
    for name in ("sma(20)", "ema(20)", "rsi(14)", "atr(14)", "macd(12,26,9)"):
        print(f"| `{name}` | `{POMATA[name]!r}` | {_vs_oracle(name)} | `{_vs_talib(name)}` |")


if __name__ == "__main__":
    main()
