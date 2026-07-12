"""Spec for ``pomata.metrics.sharpe_ratio`` — reducing, annualized, null-skipping, NaN-poisoning, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import sharpe_ratio_reference
from tests.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.metrics import sharpe_ratio


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant return series: its dispersion is pure rounding noise, so the one-pass ratio and the
    two-pass oracle resolve the ``0 / 0`` oppositely — a degeneracy, not a bug.
    """
    return well_spread(frame.to_series(0).to_list())


SHARPE_RATIO = Spec(
    factory=sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=sharpe_ratio_reference,
    conditioning=_well_spread,
    # A ratio of a mean to a standard deviation at a zero risk-free rate: scale-invariant, degree 0 (tests/metrics/
    # test_sharpe_ratio.py:181 test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(2.4285,),
)
