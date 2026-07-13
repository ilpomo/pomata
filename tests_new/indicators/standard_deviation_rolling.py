"""Spec for ``pomata.indicators.standard_deviation_rolling`` — the rolling standard deviation, window-nulling."""

from tests.indicators.oracles import standard_deviation_rolling_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import standard_deviation_rolling

STANDARD_DEVIATION_ROLLING = Spec(
    factory=standard_deviation_rolling,
    inputs=("price",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"ddof": -1}, r"ddof must be >= 0"),
        ({"ddof": 14}, r"ddof must be < window"),
    ),
    oracle=standard_deviation_rolling_reference,
    # A one-pass rolling dispersion against a two-pass oracle (the square root of the rolling variance): the fixed
    # streaming band over the well-conditioned domain, matching every other one-pass moment family.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    # A dispersion in the price's units, homogeneous of degree 1 (tests/indicators/test_standard_deviation_rolling.py
    # ::TestStandardDeviationRollingProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("price",), degree=1),),
    golden_params={"window": 2},
    golden_input={"price": (2.0, 4.0, 4.0, 8.0)},
    golden_output=(None, 1.0, 0.0, 2.0),
    pins=(
        SpecPin(
            label="sample_deviation_ddof_one",
            inputs={"price": (1.0, 3.0, 5.0)},
            params_override={"window": 3, "ddof": 1},
            expected=(None, None, 2.0),
            reason="the sample deviation (ddof=1) divides by window - 1, the second correctness branch a single "
            "population golden cannot carry (test_standard_deviation_rolling.py"
            "::TestStandardDeviationRollingCorrectness::test_sample_ddof_golden)",
        ),
        SpecPin(
            label="window_one_is_zero",
            inputs={"price": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(0.0, 0.0, 0.0),
            reason="window=1 has no spread, so the deviation is 0 at every row with no warm-up "
            "(test_standard_deviation_rolling.py::TestStandardDeviationRollingEdge::test_window_one_is_zero)",
        ),
        SpecPin(
            label="constant_window_is_exactly_zero_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected=(None, None, 471404.47365057963, 0.0, 0.0),
            reason="a constant window has exactly zero spread even after a much larger value has left it, where an "
            "incremental rolling standard deviation would leave a residue (test_standard_deviation_rolling.py"
            "::TestStandardDeviationRollingEdge::test_constant_window_is_zero)",
        ),
    ),
)
