"""Spec for ``pomata.indicators.time_series_forecast`` — the one-step least-squares forecast, degree-1 homogeneous."""

from tests.indicators.oracles import time_series_forecast_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import time_series_forecast

TIME_SERIES_FORECAST = Spec(
    factory=time_series_forecast,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=time_series_forecast_reference,
    # The least-squares line extended one bar past the window, homogeneous of degree 1 (tests/indicators/
    # test_time_series_forecast.py::TestTimeSeriesForecastProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 14.3333, 13.0, 14.0, 14.0, 15.0),
)
