"""
Declaration for ``pomata.indicators.bollinger_bands`` — the SMA-and-deviation band struct, window-nulling, degree-1.
"""

import math

from pomata.indicators import bollinger_bands
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_bollinger_bands
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

BOLLINGER_BANDS = suite_indicators(
    factory=bollinger_bands,
    inputs=("price",),
    params={"window": 20, "multiplier": 2.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_bollinger_bands,
    scaling=(ScaleAxis(roles=("price",), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (2.0, 4.0, 4.0, 8.0)},
        output={
            "lower": (None, 1.0, 4.0, 2.0),
            "middle": (None, 3.0, 4.0, 6.0),
            "upper": (None, 5.0, 4.0, 10.0),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="multiplier_one_halves_the_band_width",
            inputs={"price": (2.0, 4.0, 4.0, 8.0)},
            params_override={"window": 2, "multiplier": 1.0},
            expected={
                "lower": (None, 2.0, 4.0, 4.0),
                "middle": (None, 3.0, 4.0, 6.0),
                "upper": (None, 4.0, 4.0, 8.0),
            },
            reason="the band half-width is linear in the multiplier: at multiplier=1 it is half of the default, the "
            "arithmetic a single default-multiplier golden cannot exercise",
        ),
        Pin(
            label="constant_window_collapses_bands_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected={
                "lower": (None, None, -609475.5473011592, 0.1, 0.1),
                "middle": (None, None, 333333.39999999997, 0.1, 0.1),
                "upper": (None, None, 1276142.3473011593, 0.1, 0.1),
            },
            reason="a constant window has exactly zero (pinned) deviation, so all three bands collapse onto the middle "
            "even after a much larger value has left the window, where the rolling kernel would otherwise leave a "
            "residue",
        ),
    ),
    reference="Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.",
    wikipedia="https://en.wikipedia.org/wiki/Bollinger_Bands",
    see_also=(
        ("sma", "The center band."),
        ("standard_deviation_rolling", "The band half-width, before scaling by ``multiplier``."),
        ("keltner_channels", "The same band shape with ATR width instead of a standard deviation."),
    ),
    notes=(
        (
            "Composition",
            "The bands are built from :func:`sma` (center) and the population "
            ":func:`standard_deviation_rolling` (width), so they inherit the warm-up and missing-data "
            "behavior of both — identically on every field of the struct.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — on all three fields.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there — on all three fields."),
        (
            "Degenerate denominator",
            "a window of equal values has zero standard deviation (see "
            ":func:`standard_deviation_rolling`), so all three bands collapse onto the constant — "
            "even at ``window == 1``, or just after a much larger value has left the window.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with three ``Float64`` fields, the same length as ``expr``:"
    "\n\n"
    "- ``lower`` — the lower band, ``middle - multiplier * sigma``. - ``middle`` — the center "
    "band, the :func:`sma` of ``expr``. - ``upper`` — the upper band, ``middle + multiplier * "
    "sigma``."
    "\n\n"
    'Read one band with ``.struct.field("middle")`` (etc.) or split all three into columns '
    "with ``.struct.unnest()``. For the first ``window - 1`` rows (warm-up) every field of "
    "the struct is ``null`` (the struct row itself stays a valid struct).",
    raises_prose="ValueError: If ``window < 1``, or if ``multiplier`` is not a finite number ``> 0`` (i.e. "
    "``<= 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
        "multiplier": "Number of standard deviations between the center band and each outer band (default "
        "``2.0``). Must be a finite number ``> 0`` (a non-positive width would collapse or invert "
        "the bands). The bands are symmetric; for asymmetric bands compose :func:`sma` and "
        ":func:`standard_deviation_rolling` directly.",
    },
    example_columns={"price": "close"},
    examples=(
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0)},
            params={"window": 3},
            round_to=4,
            fields=("lower", "middle", "upper"),
        ),
        Example(
            intro="Split the struct into three columns with ``.struct.unnest()``:",
            verbatim=(
                '>>> frame.select(bollinger_bands=expr).unnest("bollinger_bands").columns',
                "['lower', 'middle', 'upper']",
            ),
        ),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 20.0, 22.0, 21.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "B", "B", "B"),
            params={"window": 2},
            round_to=4,
            fields=("middle",),
        ),
        Example(
            inputs={"price": (10.0, None, 12.0, float("nan"), 14.0, 15.0)},
            intro="A ``null`` and a ``NaN`` propagate to every band; the middle band makes the handling visible:",
            params={"window": 2},
            round_to=4,
            fields=("middle",),
        ),
        Example(
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            intro="**Degenerate denominator** — a constant window has zero deviation, so all three bands "
            "collapse onto the middle even after a much larger value has left the window, where the "
            "rolling kernel would otherwise leave a residue:",
            params={"window": 3},
            round_to=4,
            fields=("lower",),
        ),
    ),
)
