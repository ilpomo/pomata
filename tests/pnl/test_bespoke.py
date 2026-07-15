"""
The pnl residue the declarative spec cannot carry: the large-magnitude reference tier and the intra-function
metamorphic identities, expressed as ordinary property tests.

A :class:`~tests.support.spec.SpecPin` is a fixed crafted case and a :class:`~tests.support.spec.ScaleAxis` is
an exact homogeneity ratio at one ``k``; neither expresses a Hypothesis-quantified claim. Two such claims live here
as ordinary ``@given`` tests:

- ``test_matches_reference_at_large_magnitude`` — every degree-1-homogeneous pnl function keeps oracle agreement when
  its inputs are pushed to extreme magnitude (a numeric-stability tier, distinct from the exact-ratio ``scale`` axis),
  parametrized over the specs that carry it.
- the three cumulation metamorphics — the compounding identity of ``equity_curve``, the running-difference identity of
  ``cumulative_pnl``, and the across-time additivity of ``returns_log`` — each an intra-function recursive property,
  not a recomposition from other public functions (which is what ``component_expr`` is for).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.cost_borrow import COST_BORROW
from tests.pnl.cost_funding import COST_FUNDING
from tests.pnl.cost_notional import COST_NOTIONAL
from tests.pnl.cost_per_share import COST_PER_SHARE
from tests.pnl.cost_proportional import COST_PROPORTIONAL
from tests.pnl.cost_slippage import COST_SLIPPAGE
from tests.pnl.cumulative_pnl import CUMULATIVE_PNL
from tests.pnl.dividend import DIVIDEND
from tests.pnl.pnl_gross import PNL_GROSS
from tests.pnl.pnl_gross_inverse import PNL_GROSS_INVERSE
from tests.pnl.pnl_net import PNL_NET
from tests.pnl.returns_gross import RETURNS_GROSS
from tests.pnl.returns_net import RETURNS_NET
from tests.pnl.turnover import TURNOVER
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    finite_floats,
    input_scale,
)
from tests.support.spec import Spec, actual_lanes, fuzz_frames, reference_lanes, spec_id

from pomata.pnl import cumulative_pnl, equity_curve, returns_log

# ======================================================================================================================
# Large magnitude — the numeric-stability tier of every degree-1-homogeneous pnl function
# ======================================================================================================================

# The specs that carry the large-magnitude tier: the linear pnl and cost kernels, homogeneous of non-zero degree in
# their inputs (so the value genuinely rides the input scale), plus pnl_gross_inverse — its reciprocal legs cancel a
# joint quantity/price rescale (net degree 0), and the tier stresses its tiny-reciprocal-times-huge-quantity
# accumulation at extreme magnitudes instead. The scale-invariant siblings (cost_fixed, returns_simple, returns_log)
# and the nonlinear compounding equity_curve carry no such tier and are excluded.
LARGE_MAGNITUDE_SPECS: tuple[Spec, ...] = (
    COST_BORROW,
    COST_FUNDING,
    COST_NOTIONAL,
    COST_PER_SHARE,
    COST_PROPORTIONAL,
    COST_SLIPPAGE,
    CUMULATIVE_PNL,
    DIVIDEND,
    PNL_GROSS,
    PNL_GROSS_INVERSE,
    PNL_NET,
    RETURNS_GROSS,
    RETURNS_NET,
    TURNOVER,
)

_LARGE_MAGNITUDE_SCALES = (1e-6, 1e6, 1e9)


@pytest.mark.parametrize("spec", LARGE_MAGNITUDE_SPECS, ids=spec_id)
@given(data=st.data(), scale=st.sampled_from(_LARGE_MAGNITUDE_SCALES))
def test_matches_reference_at_large_magnitude(spec: Spec, data: st.DataObject, scale: float) -> None:
    """Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees."""
    frame = data.draw(fuzz_frames(spec, missing=False))
    scaled = frame.select([pl.col(role) * scale for role in spec.inputs])
    expected = reference_lanes(spec, scaled)
    actual = actual_lanes(spec, scaled)
    assert sorted(actual) == sorted(expected)
    for name, values in expected.items():
        assert_matches(actual[name], values, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_STREAMING)


# ======================================================================================================================
# Cumulation metamorphics — intra-function recursive identities, one per cumulation primitive
# ======================================================================================================================

# equity_curve's compounding domain: 1 + return stays in [0.1, 1.9], so the running product never underflows or flips.
_RETURNS = st.floats(min_value=-0.9, max_value=0.9, allow_nan=False, allow_infinity=False)
# returns_log's domain: strictly positive prices, so every log ratio is defined.
_POSITIVE_PRICES = st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False)


@given(returns=st.lists(_RETURNS, min_size=1, max_size=50))
def test_compounds_consecutive_returns(returns: list[float]) -> None:
    """Verifies the compounding identity: the first row is ``1 + return`` and each later ratio recovers it."""
    equity = apply_expr(returns, equity_curve(pl.col(COLUMN_X)))
    first = equity[0]
    assert first is not None
    assert math.isclose(
        first, 1.0 + returns[0], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
    )
    for index in range(1, len(returns)):
        current = equity[index]
        previous = equity[index - 1]
        assert current is not None
        assert previous is not None
        assert math.isclose(
            current / previous,
            1.0 + returns[index],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


@given(returns=st.lists(finite_floats(), min_size=1, max_size=50))
def test_running_difference_recovers_returns(returns: list[float]) -> None:
    """Verifies the running-difference identity: each row minus its predecessor recovers that bar's return."""
    cumulative = apply_expr(returns, cumulative_pnl(pl.col(COLUMN_X)))
    tolerance = input_scale(cumulative) * EXACT_TOLERANCE_FACTOR
    first = cumulative[0]
    assert first is not None
    assert math.isclose(first, returns[0], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=tolerance)
    for index in range(1, len(returns)):
        current = cumulative[index]
        previous = cumulative[index - 1]
        assert current is not None
        assert previous is not None
        assert math.isclose(current - previous, returns[index], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=tolerance)


@given(prices=st.lists(_POSITIVE_PRICES, min_size=2, max_size=50))
def test_aggregates_across_time(prices: list[float]) -> None:
    """Verifies the Meucci across-time identity: the log returns sum to the total log return ``ln(P_last / P_0)``."""
    log_returns = apply_expr(prices, returns_log(pl.col(COLUMN_X)))
    assert log_returns[0] is None
    assert all(value is not None for value in log_returns[1:])
    total = math.fsum(value for value in log_returns[1:] if value is not None)
    expected = math.log(prices[-1] / prices[0])
    assert math.isclose(total, expected, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
