"""
Shared, stateless test helpers for the indicator suite, re-exported flat for ``from tests.support import ...``.

The helpers are grouped one concern per module — :mod:`tests.support.asserts` (compare a result against an oracle),
:mod:`tests.support.bars` (transpose generated bars into columns), :mod:`tests.support.benchmarks` (time an expression
for the benchmark tier), :mod:`tests.support.columns` (canonical column-name constants), :mod:`tests.support.frames`
(materialize inputs into a frame and evaluate an expression),
:mod:`tests.support.strategies` (Hypothesis input generators and the ``window`` cap), :mod:`tests.support.synthesis`
(signature-driven call synthesis for the public factories), :mod:`tests.support.tolerances` (the named floating-point
tolerance ladder and :func:`input_scale`) — and re-exported here so every test imports
them from one place. They are plain functions rather than pytest fixtures so they compose with Hypothesis ``@given``
without leaking state across generated examples.
"""

from tests.support.asserts import assert_all_float64, assert_matches, assert_scale_homogeneous
from tests.support.bars import complete_benchmark, split_pairs, split_quads, split_triples
from tests.support.benchmarks import fastest_eval
from tests.support.columns import BENCHMARK, CLOSE, COLUMN_X, GROUP_KEY, HIGH, LOW, OPEN, RETURNS, VOLUME
from tests.support.frames import apply_expr, count_leading_nulls, materialize, materialize_struct
from tests.support.strategies import (
    CONDITIONING_FLOOR,
    STANDARDIZED_MOMENT_FLOOR,
    SUBNORMAL_FLOOR,
    WINDOW_MAX,
    coherent_hl,
    coherent_hl_with_missing,
    coherent_hlc,
    coherent_hlc_with_missing,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    coherent_ohlc,
    coherent_ohlc_with_missing,
    finite_floats,
    missing_data_floats,
    positive_missing_data,
    spans_even_lag_repeat,
    standardized_moment_floats,
    subnormal_safe_floats,
    two_segment_missing_data,
    well_spread,
    windows_well_conditioned,
    windows_well_spread,
)
from tests.support.synthesis import sample_argument, synthesize_call
from tests.support.tolerances import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    ABSOLUTE_TOLERANCE_STREAMING,
    BOUND_MARGIN,
    EXACT_TOLERANCE_FACTOR,
    RELATIVE_TOLERANCE_EXACT,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    VARIANCE_TOLERANCE_FACTOR,
    input_scale,
    streaming_abs_tol,
)

__all__ = (
    "ABSOLUTE_TOLERANCE_EXACT",
    "ABSOLUTE_TOLERANCE_PROPERTY",
    "ABSOLUTE_TOLERANCE_REFERENCE",
    "ABSOLUTE_TOLERANCE_SCALE",
    "ABSOLUTE_TOLERANCE_STREAMING",
    "BENCHMARK",
    "BOUND_MARGIN",
    "CLOSE",
    "COLUMN_X",
    "CONDITIONING_FLOOR",
    "EXACT_TOLERANCE_FACTOR",
    "GROUP_KEY",
    "HIGH",
    "LOW",
    "OPEN",
    "RELATIVE_TOLERANCE_EXACT",
    "RELATIVE_TOLERANCE_PROPERTY",
    "RELATIVE_TOLERANCE_REFERENCE",
    "RELATIVE_TOLERANCE_SCALE",
    "RETURNS",
    "STANDARDIZED_MOMENT_FLOOR",
    "STREAMING_TOLERANCE_FACTOR",
    "SUBNORMAL_FLOOR",
    "VARIANCE_TOLERANCE_FACTOR",
    "VOLUME",
    "WINDOW_MAX",
    "apply_expr",
    "assert_all_float64",
    "assert_matches",
    "assert_scale_homogeneous",
    "coherent_hl",
    "coherent_hl_with_missing",
    "coherent_hlc",
    "coherent_hlc_with_missing",
    "coherent_hlcv",
    "coherent_hlcv_with_missing",
    "coherent_ohlc",
    "coherent_ohlc_with_missing",
    "complete_benchmark",
    "count_leading_nulls",
    "fastest_eval",
    "finite_floats",
    "input_scale",
    "materialize",
    "materialize_struct",
    "missing_data_floats",
    "positive_missing_data",
    "sample_argument",
    "spans_even_lag_repeat",
    "split_pairs",
    "split_quads",
    "split_triples",
    "standardized_moment_floats",
    "streaming_abs_tol",
    "subnormal_safe_floats",
    "synthesize_call",
    "two_segment_missing_data",
    "well_spread",
    "windows_well_conditioned",
    "windows_well_spread",
)
