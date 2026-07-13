"""Spec for ``pomata.indicators.variance_ewma`` — the EWM variance, gap-bridging, NaN-latching, degree-2."""

from tests.indicators.oracles import variance_ewma_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import variance_ewma

VARIANCE_EWMA = Spec(
    factory=variance_ewma,
    inputs=("price",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=variance_ewma_reference,
    # A one-pass EWM second moment against a two-pass weighted oracle: the fixed streaming band over the
    # well-conditioned domain, matching every other one-pass moment family.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    # An EWM dispersion of the price, homogeneous of degree 2 (tests/indicators/test_variance_ewma.py
    # ::TestVarianceEwmaProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("price",), degree=2),),
    golden_params={"window": 3},
    golden_input={"price": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 1.6875, 0.8594, 1.5586, 0.7803, 1.4216),
    pins=(
        SpecPin(
            label="interior_null_reweights_observations",
            inputs={"price": (10.0, None, 11.0, 13.0, 12.0)},
            params_override={"window": 3},
            expected=(None, None, None, 1.4722222222222232, 0.7430555555555561),
            reason="an interior null ages the lag of 10 while contributing no term; at the last defined row the "
            "ignore_nulls=False weights reduce to 1:2:3:6, giving variance 107/144 (test_variance_ewma.py"
            "::TestVarianceEwmaCorrectness::test_golden_master_interior_null)",
        ),
    ),
)
