"""Spec for ``pomata.indicators.mama`` — the adaptive-alpha cycle pair, latching, golden-anchored, oracle-adapted."""

import math

import polars as pl
from tests.indicators.oracles import mama_reference
from tests.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import mama

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up.
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))


def _mama_oracle(spec: Spec, frame: pl.DataFrame) -> object:
    """Reach the oracle by position only: its kwargs (``fast_limit`` / ``slow_limit``) are not the factory's mirror."""
    return spec.oracle(frame[spec.inputs[0]].to_list())


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """
    Reject the cycle-pipeline degenerate: Ehlers' six-tap quadrature filter reads the four-bar smooth at even lags, so
    an even-lag repeat (a flat run or a period-two alternation) drives the in-phase component to a cancellation
    residual the impl's explicit FIR and the oracle's compensated sum round apart. Only the finite bars can reach it,
    so filtering them keeps the missing-data tier from rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


MAMA = Spec(
    factory=mama,
    inputs=("expr",),
    params={"limit_fast": 0.5, "limit_slow": 0.05},
    shape=Shape.STRUCT,
    fields=("mama", "fama"),
    warmup=32,
    raises=(
        ({"limit_fast": 0.0}, r"limit_fast must be in"),
        ({"limit_slow": 0.0}, r"limit_slow must be in"),
        ({"limit_fast": 3.0}, r"limit_fast must be in"),
        ({"limit_slow": 2.5}, r"limit_slow must be in"),
        ({"limit_fast": 0.05, "limit_slow": 0.5}, r"limit_fast must be >= limit_slow"),
    ),
    oracle=mama_reference,
    oracle_adapter=_mama_oracle,
    conditioning=_no_even_lag_repeat,
    # Both lines ride the price scale, so they scale linearly with the bars (tests/indicators/test_mama.py:312
    # test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "mama": (None,) * 32 + (97.9767, 97.6734, 97.3142, 96.9485, 96.6255, 96.3897, 96.2764, 96.308),
        "fama": (None,) * 32 + (99.6954, 99.6448, 99.5866, 99.5206, 99.4482, 99.3718, 99.2944, 99.2197),
    },
)
