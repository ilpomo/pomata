"""Spec for ``pomata.indicators.ichimoku`` — a struct of four rolling midpoints, per-field warm-ups, three windows."""

from tests.indicators.oracles import ichimoku_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import ichimoku

ICHIMOKU = Spec(
    factory=ichimoku,
    inputs=("high", "low"),
    params={"window_tenkan": 9, "window_kijun": 26, "window_senkou": 52},
    shape=Shape.STRUCT,
    fields=("tenkan", "kijun", "senkou_a", "senkou_b"),
    warmup={"tenkan": 8, "kijun": 25, "senkou_a": 25, "senkou_b": 51},
    raises=(
        ({"window_tenkan": 0}, r"window_tenkan must be >= 1"),
        ({"window_kijun": 0, "window_tenkan": 1}, r"window_kijun must be >= 1"),
        ({"window_senkou": 0, "window_tenkan": 1, "window_kijun": 1}, r"window_senkou must be >= 1"),
        ({"window_kijun": 5}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
    ),
    oracle=ichimoku_reference,
    # Every line is a windowed high-low midpoint, so all four scale linearly with the bars (tests/indicators/
    # test_ichimoku.py:363 test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    golden_params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
    golden_input={
        "high": (10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0),
        "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0),
    },
    golden_output={
        "tenkan": (None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0),
        "kijun": (None, None, 10.0, 11.0, 12.0, 12.0, 12.5, 12.5),
        "senkou_a": (None, None, 10.25, 11.25, 12.25, 12.0, 12.5, 12.75),
        "senkou_b": (None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5),
    },
)
