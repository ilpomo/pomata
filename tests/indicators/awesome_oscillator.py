"""Declaration for ``pomata.indicators.awesome_oscillator`` — the SMA-of-median difference, window-nulling, degree-1."""

from pomata.indicators import awesome_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_awesome_oscillator
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

AWESOME_OSCILLATOR = suite_indicators(
    factory=awesome_oscillator,
    inputs=("high", "low"),
    params={"window_fast": 5, "window_slow": 34},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=33,
    oracle=reference_awesome_oscillator,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Awesome Oscillator.",
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"high": (2.0, 4.0, 6.0, 8.0, 10.0), "low": (0.0, 2.0, 4.0, 6.0, 8.0)},
        output=(None, None, 1.0, 1.0, 1.0),
        params={"window_fast": 2, "window_slow": 3},
    ),
    pins=(
        Pin(
            label="single_row_equal_windows",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 1},
            expected=(0.0,),
            reason="window_fast == window_slow == 1 on one bar gives 0",
        ),
        Pin(
            label="single_row_warmup",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 3},
            expected=(None,),
            reason="a slow window of 3 on one bar is still warm-up",
        ),
        Pin(
            label="equal_windows_is_zero",
            inputs={"high": (2.0, 4.0, 6.0, 8.0), "low": (0.0, 2.0, 4.0, 6.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0, 0.0),
            reason="equal windows give an identically-zero oscillator where defined",
        ),
        Pin(
            label="flat_series_is_zero",
            inputs={"high": (5.0, 5.0, 5.0, 5.0, 5.0), "low": (5.0, 5.0, 5.0, 5.0, 5.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, 0.0, 0.0, 0.0),
            reason="over a constant median both averages equal it, so AO is 0",
        ),
    ),
    reference="Williams, B. (1998). *New Trading Dimensions*. Wiley.",
    see_also=(
        ("absolute_price_oscillator", "The same fast-minus-slow shape on the close, with exponential averages."),
        ("macd", "The exponential oscillator with an added signal line."),
        ("price_median", "The bar median each average is taken over."),
    ),
    notes=(("Inputs", "``high`` and ``low`` must share a length and alignment (the same row index is one bar)."),),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window_fast`` non-null values).",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        ("Insufficient sample", "a slow window longer than the series never completes, so the result is ``null``."),
        (
            "Degenerate denominator",
            "over a constant median run there is no spread between the two averages, so the "
            "oscillator is exactly ``0``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The oscillator for each row, the same length as the inputs. The first ``window_slow - "
    "1`` values are ``null`` (warm-up): both averages must be defined before their difference "
    "is.",
    raises_prose="ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > "
    "window_slow`` (the fast leg must be the shorter one; ``window_fast == window_slow`` is "
    "allowed and gives an identically-zero oscillator).",
    args_prose={
        "window_fast": "Window of the fast simple moving average (canonically ``5``). Must be ``>= 1``.",
        "window_slow": "Window of the slow simple moving average (canonically ``34``). Must be ``>= 1`` and ``>= "
        "window_fast``.",
    },
    intro_basic="Basic usage on high-low bars:",
    examples=(
        Example(
            inputs={"high": (2.0, 4.0, 6.0, 8.0, 10.0), "low": (0.0, 2.0, 4.0, 6.0, 8.0)},
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (11.0, 12.0, 13.0, 12.5, 14.0, 21.0, 22.0, 23.0, 22.5, 24.0),
                "low": (9.0, 10.0, 11.0, 11.0, 12.0, 19.0, 20.0, 21.0, 21.0, 22.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (11.0, 12.0, 13.0, 12.5, 14.0, None, 15.0, float("nan"), 16.0, 17.0),
                "low": (9.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 15.0),
            },
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make it visible:",
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"high": (2.0,), "low": (0.0,)},
            intro="**Insufficient sample** — a fast and slow window both collapsed to a single bar leave no "
            "warm-up to wait out, so a one-row series reads a well-defined ``0``:",
            params={"window_fast": 1, "window_slow": 1},
        ),
        Example(
            inputs={"high": (5.0, 5.0, 5.0, 5.0, 5.0), "low": (5.0, 5.0, 5.0, 5.0, 5.0)},
            intro="**Degenerate denominator** — a constant median price makes the fast and slow averages "
            "equal, so the oscillator is exactly ``0``:",
            params={"window_fast": 2, "window_slow": 3},
        ),
    ),
)
