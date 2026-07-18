"""
The pnl family dialect: the closed vocabularies a pnl declaration answers with.

Every axis here is a finite, closed set — a contributor picks a member, never invents one — so a declaration cannot
drift into free-form prose and the generated failure messages and (at cutover) the generated docstring read the same
label the declaration stated. The pnl family is the most uniform of the three: sixteen of the eighteen functions share
one dialect (an interior missing value propagates to the rows it reaches, then recovers), the two cumulations
(``cumulative_pnl`` and ``equity_curve``) bridge a ``null`` and latch a ``NaN``, and the ``±inf`` flow is a single
IEEE-754 story throughout.
"""

import enum


class BehaviorNull(enum.Enum):
    """What an interior ``null`` does to a pnl output (the pnl dialect of :class:`pomata._policy.NullPolicy`)."""

    PROPAGATES = "propagates"  # nulls the rows it reaches (its own, and a one-bar lag), then recovers
    BRIDGED = "bridged"  # a cumulation steps over it (the running state carries across), so later rows recover


class BehaviorNan(enum.Enum):
    """What an interior ``NaN`` does to a pnl output (the pnl dialect of :class:`pomata._policy.NanPolicy`)."""

    PROPAGATES = "propagates"  # nans the rows it reaches, then recovers (a pointwise or fixed-lag map)
    LATCHES = "latches"  # a cumulation carries it forward forever (every later row is ``NaN``)


class SpaceCost(enum.Enum):
    """The units a pnl output lives in — the README's cash-flow / returns-flow partition, stated as data."""

    CASH = "cash"  # a currency amount: a position P&L, a cash cost, a dividend, a running cash total
    RETURNS = "returns"  # a dimensionless fraction: a strategy return, a compounded curve, a weight turnover / its cost


class ConventionSign(enum.Enum):
    """Which side of the book the payoff is defined on — read off each function's own arithmetic."""

    LONG_SHORT = "long_short"  # symmetric in the position sign: longs and shorts both carry the payoff
    SHORT_ONLY = "short_only"  # charged on the short leg alone (``max(-quantity, 0)``), zero on longs and flats
    LONG_ONLY = "long_only"  # accrued on the long leg alone, zero on shorts and flats


class NonFinite(enum.Enum):
    """How a pnl function carries ``±inf`` inputs — one IEEE-754 story across the whole family."""

    IEEE_FLOW = "ieee_flow"  # ``±inf`` flows through the arithmetic unguarded, including ``inf - inf = NaN``


class Warmup(enum.Enum):
    """The warm-up a pnl function owes; the harness resolves it to a leading-null ``int`` / ``None``."""

    NONE = "none"  # no warm-up: the first row is already defined (a pointwise map, a flat-start cumulation)
    ONE_ROW = "one_row"  # exactly one leading ``null``: a one-bar-lagged transform has no previous bar to read at row 0
