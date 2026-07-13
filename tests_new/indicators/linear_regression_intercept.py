"""Spec for ``pomata.indicators.linear_regression_intercept`` — the rolling least-squares intercept, degree-1."""

from tests_new.indicators.oracles import linear_regression_intercept_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import linear_regression_intercept

LINEAR_REGRESSION_INTERCEPT = Spec(
    factory=linear_regression_intercept,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=linear_regression_intercept_reference,
    # The window-start value of the least-squares line, homogeneous of degree 1 (tests/indicators/
    # test_linear_regression_intercept.py::TestLinearRegressionInterceptProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 9.8333, 11.5, 12.5, 12.5, 13.5),
)
