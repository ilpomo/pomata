"""Spec for ``pomata.indicators.standard_deviation_ewma`` — the EWM standard deviation, gap-bridging, degree-1."""

from tests.indicators.oracles import standard_deviation_ewma_reference
from tests.support import ABSOLUTE_TOLERANCE_ROLLING_ORACLE, RELATIVE_TOLERANCE_ROLLING_ORACLE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import standard_deviation_ewma

STANDARD_DEVIATION_EWMA = Spec(
    factory=standard_deviation_ewma,
    inputs=("price",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=standard_deviation_ewma_reference,
    # A one-pass EWM dispersion against a two-pass weighted oracle (the square root of the EWM variance): the fixed
    # streaming band over the well-conditioned domain, matching every other one-pass moment family.
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_ROLLING_ORACLE,
    # An EWM dispersion in the price's units, homogeneous of degree 1.
    scale=(ScaleAxis(roles=("price",), degree=1),),
    golden_params={"window": 3},
    golden_input={"price": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 1.299, 0.927, 1.2484, 0.8833, 1.1923),
    pins=(
        SpecPin(
            label="interior_null_reweights_observations",
            inputs={"price": (10.0, None, 11.0, 13.0, 12.0)},
            params_override={"window": 3},
            expected=(None, None, None, 1.2133516482134201, 0.8620067027323837),
            reason="an interior null ages the lag of 10 while contributing no term; at the last defined row the "
            "ignore_nulls=False weights reduce to 1:2:3:6, so the deviation is sqrt(107/144)",
        ),
    ),
)
