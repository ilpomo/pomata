"""
Declaration for ``pomata.indicators.macd`` — the EMA-difference struct (line, signal, histogram), gap-bridging,
degree-1.
"""

import polars as pl

from pomata.indicators import absolute_price_oscillator, ema, macd
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_macd
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE


def _macd_component() -> pl.Expr:
    """MACD recomposed from public functions: line = the APO, signal = its EMA, histogram = their difference."""
    line = absolute_price_oscillator(pl.col("expr"), window_fast=12, window_slow=26)
    signal = ema(line, 9)
    return pl.struct(macd=line, signal=signal, histogram=line - signal).name.keep()


MACD = suite_indicators(
    factory=macd,
    inputs=("expr",),
    params={"window_fast": 12, "window_slow": 26, "window_signal": 9},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("macd", "signal", "histogram"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"macd": 25, "signal": 33, "histogram": 33},
    oracle=reference_macd,
    scaling=(ScaleAxis(roles=("expr",), degree={"macd": 1, "signal": 1, "histogram": 1}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_signal": 0}, r"window_signal must be >= 1"),
        ({"window_fast": 26, "window_slow": 12}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output={
            "macd": (None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848),
            "signal": (None, None, None, 0.3333, 0.3704, 0.4321, 0.2469, 0.3388),
            "histogram": (None, None, None, -0.1667, 0.0185, 0.0309, -0.0926, 0.046),
        },
        params={"window_fast": 2, "window_slow": 3, "window_signal": 2},
    ),
    recomposition=_macd_component,
    pins=(
        Pin(
            label="fast_equals_slow_is_zero",
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0)},
            params_override={"window_fast": 3, "window_slow": 3, "window_signal": 2},
            expected={
                "macd": (None, None, 0.0, 0.0, 0.0, 0.0),
                "signal": (None, None, None, 0.0, 0.0, 0.0),
                "histogram": (None, None, None, 0.0, 0.0, 0.0),
            },
            reason="equal fast/slow windows make the MACD line identically zero (x - x is exact +0.0), so signal and "
            "histogram are zero too",
        ),
    ),
)
