"""Spec for ``pomata.metrics.kurtosis`` — reducing, the standardized fourth moment minus three, scale-invariant."""

import polars as pl
from tests.metrics.oracles import kurtosis_reference
from tests.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.metrics import kurtosis


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart."""
    return well_spread(frame.to_series(0).to_list())


KURTOSIS = Spec(
    factory=kurtosis,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=kurtosis_reference,
    conditioning=_well_spread,
    # A standardized moment is scale-invariant, degree 0 (test_kurtosis.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)},
    golden_output=(-1.3223,),
)
