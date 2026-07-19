"""
Declaration for ``pomata.indicators.variance_rolling`` — the rolling variance, window-nulling, degree-2 homogeneous.
"""

from pomata.indicators import variance_rolling
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_variance_rolling
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

VARIANCE_ROLLING = suite_indicators(
    factory=variance_rolling,
    inputs=("price",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_variance_rolling,
    scaling=(ScaleAxis(roles=("price",), degree=2),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(inputs={"price": (2.0, 4.0, 4.0, 8.0)}, output=(None, 1.0, 0.0, 4.0), params={"window": 2}),
    pins=(
        Pin(
            label="sample_variance_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 4.0),
            reason="the sample variance (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry",
        ),
        Pin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 222222177777.78003, 0.0, 0.0),
            reason="a constant window has exactly zero dispersion even after a much larger value has left it, where "
            "an incremental rolling variance would leave a residue",
        ),
        Pin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the variance is 0 at every row with no warm-up",
        ),
    ),
    reference='Fisher, R. A. (1918). "The Correlation between Relatives on the Supposition of Mendelian '
    'Inheritance." *Transactions of the Royal Society of Edinburgh*, 52, 399-433.',
    doi="https://doi.org/10.1017/S0080456800012163",
    wikipedia="https://en.wikipedia.org/wiki/Variance",
    see_also=(
        ("standard_deviation_rolling", "Its square root, in the input's own units."),
        ("variance_ewma", "The exponentially-weighted counterpart, weighting recent observations more."),
        ("sma", "The moving mean the deviations are measured from."),
    ),
    notes=(
        (
            "Degrees of freedom",
            "``ddof`` selects the divisor ``window - ddof``: ``ddof = 0`` is the population variance "
            "(÷ ``window``), the charting convention; ``ddof = 1`` is the sample variance (÷ ``window "
            "- 1``), Bessel's unbiased estimator. The two differ by the factor ``window / (window - "
            "ddof)`` — e.g. on ``[10, 11, 12]`` the population variance is ``0.6667`` and the sample "
            "variance is ``1.0``. ``ddof`` must be strictly below ``window`` so the divisor stays "
            "positive.",
        ),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window of equal values has zero dispersion, so the result is exactly ``0`` — pinned "
            "explicitly, even where a much larger value has just left the window and the incremental "
            "rolling kernel would otherwise leave a cancellation residue.",
        ),
        ("window == 1", "a single value has no spread, so the result is ``0`` with the default ``ddof = 0``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling variance for each row, the same length as the input. The first ``window - "
    "1`` values are ``null`` (warm-up): the window must hold ``window`` non-null values "
    "before a result is emitted.",
    raises_prose="ValueError: If ``window < 1``, ``ddof < 0``, or if ``ddof >= window`` (the divisor "
    "``window - ddof`` would be non-positive).",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
        "ddof": "Delta degrees of freedom — the divisor is ``window - ddof``. ``0`` (default) divides by "
        "``window`` (the **population** variance); ``1`` divides by ``window - 1`` (the "
        "**sample** variance, the unbiased estimator used when the window is a sample of a larger "
        "population). Must be ``< window`` (the divisor ``window - ddof`` must be positive).",
    },
    example_columns={"price": "x"},
    examples=(
        Example(inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 20.0, 22.0, 21.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "B", "B", "B"),
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
            intro="**Degenerate denominator** — a constant window keeps exactly zero dispersion even after "
            "a much larger value has left it, avoiding the cancellation residue an incremental "
            "computation would leave, so the variance is ``0``:",
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (1.0, 2.0, 3.0)},
            intro="**window == 1** — a window of one observation has no spread, so the variance is ``0`` at "
            "every row, with no warm-up:",
            params={"window": 1},
        ),
    ),
)
