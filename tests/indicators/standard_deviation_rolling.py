"""Declaration for ``pomata.indicators.standard_deviation_rolling`` — the rolling standard deviation, window-nulling."""

from pomata.indicators import standard_deviation_rolling
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_standard_deviation_rolling
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

STANDARD_DEVIATION_ROLLING = suite_indicators(
    factory=standard_deviation_rolling,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_standard_deviation_rolling,
    scaling=(ScaleAxis(roles=("price",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(inputs={"price": (2.0, 4.0, 4.0, 8.0)}, output=(None, 1.0, 0.0, 2.0), params={"window": 2}),
    pins=(
        Pin(
            label="sample_deviation_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 2.0),
            reason="the sample deviation (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry",
        ),
        Pin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the deviation is 0 at every row with no warm-up",
        ),
        Pin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 471404.47365057963, 0.0, 0.0),
            reason="a constant window has exactly zero spread even after a much larger value has left it, where an "
            "incremental rolling standard deviation would leave a residue",
        ),
    ),
    reference='Pearson, K. (1894). "Contributions to the Mathematical Theory of Evolution." '
    "*Philosophical Transactions of the Royal Society A*, 185, 71-110.",
    doi="https://doi.org/10.1098/rsta.1894.0003",
    wikipedia="https://en.wikipedia.org/wiki/Standard_deviation",
    see_also=(
        ("variance_rolling", "The square of this, of which it is the root."),
        ("sma", "The moving mean the deviations are measured from."),
        ("bollinger_bands", "Volatility bands placed a multiple of this standard deviation around the mean."),
    ),
    notes=(
        (
            "Degrees of freedom",
            "``ddof`` carries the same meaning as in :func:`variance_rolling` (population vs sample); "
            "the standard deviation is just its square root. ``ddof`` must be strictly below "
            "``window`` so the divisor stays positive.",
        ),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window of equal values has zero spread, so the result is exactly ``0`` — pinned "
            "explicitly, even where a much larger value has just left the window and the incremental "
            "rolling kernel would otherwise leave a cancellation residue.",
        ),
        ("window == 1", "a single value has no spread, so the result is ``0`` with the default ``ddof = 0``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling standard deviation for each row, the same length as the input. The first "
    "``window - 1`` values are ``null`` (warm-up): the window must hold ``window`` non-null "
    "values before a result is emitted.",
    raises_prose="ValueError: If ``window < 1``, ``ddof < 0``, or if ``ddof >= window`` (the divisor "
    "``window - ddof`` would be non-positive).",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
        "ddof": "Delta degrees of freedom — the divisor is ``window - ddof``. ``0`` (default) is the "
        "**population** standard deviation; ``1`` is the **sample** standard deviation. Must be "
        "``>= 0`` and ``< window``. See :func:`variance_rolling`.",
    },
    example_columns={"price": "x"},
    examples=(
        Example(inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 20.0, 22.0, 21.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 3 + ("NVDA",) * 3,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, None, 12.0, float("nan"), 14.0, 15.0)},
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            intro="**Degenerate denominator** — a constant window keeps exactly zero spread even after a "
            "much larger value has left it, avoiding the cancellation residue an incremental "
            "computation would leave, so the deviation is ``0``:",
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (1.0, 2.0, 3.0)},
            intro="**window == 1** — a window of one observation has no spread, so the deviation is ``0`` "
            "at every row, with no warm-up:",
            params={"window": 1},
        ),
    ),
)
