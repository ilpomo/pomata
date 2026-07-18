"""Declaration for ``pomata.indicators.tema`` — the triple EMA lag-correction, gap-bridging, NaN-latching, degree-1."""

import math

from pomata.indicators import tema
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_tema
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

TEMA = suite_indicators(
    factory=tema,
    inputs=("expr",),
    params={"window": 2},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=3,
    oracle=reference_tema,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)}, output=(None, None, None, 8.0, 10.0, 12.0)),
    pins=(
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0, 5.0)},
            expected=(None, None, None, None, 5.037037037037038),
            params_override={"window": 2},
            reason="the exact recovery value after an interior null bridges the cascade",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, None, None, math.nan),
            params_override={"window": 2},
            reason="a NaN latches into the cascade and poisons every value past the warm-up, mirroring the "
            "ema / dema / t3 siblings' pin",
        ),
        Pin(
            label="window_one_identity",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0)},
            expected=(1.0, 2.0, 3.0, 4.0),
            params_override={"window": 1},
            reason="window=1 collapses each of the three nested EMAs to the identity",
        ),
        Pin(
            label="all_zero_series",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero series stays exactly zero after warm-up",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (7.0, 7.0, 7.0, 7.0, 7.0, 7.0)},
            expected=(None, None, None, 7.0, 7.0, 7.0),
            params_override={"window": 2},
            reason="TEMA of a constant recovers exactly that constant after warm-up",
        ),
        Pin(
            label="window_three_golden",
            inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                3.296296296296297,
                5.399016203703705,
                5.081452546296297,
                3.234953703703704,
            ),
            params_override={"window": 3},
            reason="a second frozen golden master at window=3",
        ),
        Pin(
            label="golden_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 8.118158284023668, 10.090675959328463, 12.055955303250178),
            params_override={"window": 2, "adjust": True},
            reason="the frozen golden under adjust=True finite-window unbiased weighting",
        ),
    ),
)
