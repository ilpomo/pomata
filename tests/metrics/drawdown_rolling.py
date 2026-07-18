"""
Declaration for ``pomata.metrics.drawdown_rolling`` — the decline from the trailing-window peak, scale-invariant.
"""

from pomata.metrics import drawdown_rolling
from tests.metrics.drawdown import DRAWDOWN
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_drawdown_rolling
from tests.support.declaration import Golden, Pin, ScaleAxis

DRAWDOWN_ROLLING = suite_metrics(
    factory=drawdown_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=DRAWDOWN,
    window="window",
    warmup=2,
    oracle=reference_drawdown_rolling,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
        output=(None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2)},
            expected=(None, None, None, 0.0),
            reason="when the window exactly equals the series length only the last row is defined ",
            params_override={"window": 4},
        ),
        Pin(
            label="window_peak_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(None, None, 0.0, 0.0),
            reason="at a monotonically rising window's peak the drawdown is exactly 0 ",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Drawdown_%28economics%29",
    see_also=(
        ("drawdown", "The running form, measured against the all-time high to date."),
        ("max_drawdown", "The deepest all-time decline."),
        ("max_drawdown_duration", "The time dimension (longest underwater stretch)."),
    ),
    opener_override="Each window matches an independent reference oracle (the current equity over the window "
    "peak, less one).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling drawdown for each row, the same length as the input. The first ``window - "
    "1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null values before "
    "a result is emitted.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's window restarts "
    "independently and never spans the boundary:",
    intro_missing="A leading ``null`` and a later ``NaN`` make the windowed handling visible: a window "
    "covering the ``null`` is ``null``, and the ``NaN`` poisons every window it enters:",
)
