"""Spec for ``pomata.indicators.variance_rolling`` — the rolling variance, window-nulling, degree-2 homogeneous."""

from tests.indicators.oracles import variance_rolling_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import variance_rolling

VARIANCE_ROLLING = Spec(
    factory=variance_rolling,
    inputs=("price",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle=variance_rolling_reference,
    # A one-pass rolling second moment against a two-pass oracle: the magnitude-proportional band the old suite used
    # (input_scale ** 2 * VARIANCE_TOLERANCE_FACTOR), here the fixed streaming band over the well-conditioned domain.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    # A dispersion of the price, homogeneous of degree 2 (tests/indicators/test_variance_rolling.py
    # ::TestVarianceRollingProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("price",), degree=2),),
    golden_params={"window": 2},
    golden_input={"price": (2.0, 4.0, 4.0, 8.0)},
    golden_output=(None, 1.0, 0.0, 4.0),
    pins=(
        SpecPin(
            label="sample_variance_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 4.0),
            reason="the sample variance (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry (test_variance_rolling.py::TestVarianceRollingCorrectness"
            "::test_sample_ddof_golden)",
        ),
        SpecPin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 222222177777.78003, 0.0, 0.0),
            reason="a constant window has exactly zero dispersion even after a much larger value has left it, where an "
            "incremental rolling variance would leave a residue (test_variance_rolling.py::TestVarianceRollingEdge"
            "::test_constant_window_is_zero)",
        ),
        SpecPin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the variance is 0 at every row with no warm-up "
            "(test_variance_rolling.py::TestVarianceRollingEdge::test_window_one_is_zero)",
        ),
    ),
)
