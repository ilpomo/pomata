"""Spec for ``pomata.indicators.linear_regression`` — the rolling least-squares endpoint, window-nulling, degree-1."""

from tests.indicators.oracles import linear_regression_reference
from tests.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import linear_regression

LINEAR_REGRESSION = Spec(
    factory=linear_regression,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=linear_regression_reference,
    # The fitted endpoint of the window's least-squares line, homogeneous of degree 1
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 12.8333, 12.5, 13.5, 13.5, 14.5),
)
