"""
Declaration for ``pomata.metrics.volatility`` — reducing, the annualized sample standard deviation, degree-1
homogeneous.
"""

from pomata.metrics import volatility
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_volatility
from tests.support.declaration import Golden, Pin, ScaleAxis

VOLATILITY = suite_metrics(
    factory=volatility,
    inputs=("returns",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.EXACT_ZERO,
    oracle=reference_volatility,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -1}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -252}, r"periods_per_year must be >= 1"),
    ),
    golden=Golden(inputs={"returns": (0.1, -0.1, 0.2, -0.2)}, output=(2.8983,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the result is null",
        ),
        Pin(
            label="flat_returns_zero",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(0.0,),
            reason="a constant series has zero dispersion, so the volatility is exactly 0 — the exact-zero "
            "core of the near-constant regime; the property tiers' absolute band absorbs the "
            "rounding-noise dispersion there, so no conditioning filter is declared",
        ),
        Pin(
            label="golden_periods_per_year_1",
            inputs={"returns": (0.1, -0.1, 0.2, -0.2)},
            expected=(0.18257418583505539,),
            reason="the un-annualized golden branch: the sample std of the four values",
            params_override={"periods_per_year": 1},
        ),
    ),
)
