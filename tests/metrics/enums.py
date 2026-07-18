"""
The metrics family dialect: the closed vocabularies a metrics declaration answers with.

Every axis here is a finite, closed set — a contributor picks a member, never invents one — so a declaration cannot
drift into free-form prose and the generated failure messages read the same label the declaration stated. The metrics
family splits along one line the pnl family does not: a *reduction* answers a whole series with one scalar (a Sharpe
ratio, a max drawdown), while a *rolling* twin answers it per trailing window. That split drives the two dialect facts
the engine reads — how an interior missing value flows (a reduction skips it; a rolling window it overlaps is null) and
how the output annualizes.
"""

import enum


class BehaviorNull(enum.Enum):
    """What an interior ``null`` does to a metrics output (the metrics dialect of ``pomata._policy.NullPolicy``)."""

    SKIPPED = "skipped"  # a reduction excludes it — the scalar is what it would be if the row were absent
    IN_WINDOW_IS_NULL = "in_window_is_null"  # every trailing window overlapping it is null, then the flow recovers
    PROPAGATES = "propagates"  # nulls the rows it reaches (its own output, and any window keyed off it), then recovers


class BehaviorNan(enum.Enum):
    """What an interior ``NaN`` does to a metrics output (the metrics dialect of ``pomata._policy.NanPolicy``)."""

    POISONS = "poisons"  # a reduction goes ``NaN`` — the contamination cannot be excluded from the scalar
    PROPAGATES = "propagates"  # nans the rows it reaches, then recovers (a windowed or fixed-lag map)


class Annualization(enum.Enum):
    """How a metric scales its per-period statistic to a yearly one — the convention its formula follows."""

    SQRT_TIME = "sqrt_time"  # multiplied by ``sqrt(periods_per_year)`` (a dispersion: volatility, the Sharpe pair)
    LINEAR = "linear"  # multiplied by ``periods_per_year`` (an arithmetic mean: the Treynor ratio)
    GEOMETRIC = "geometric"  # raised to a power of ``periods_per_year`` (a compound rate: cagr) — no closed-form ratio
    NONE = "none"  # not annualized (a drawdown, a count, a scale-free ratio) — no ``periods_per_year`` knob


class Degenerate(enum.Enum):
    """
    The degenerate-denominator regime a metric resolves, as a closed vocabulary of outcome kinds.

    The metrics divide: a dispersion, a ratio, a regression slope. When the denominator collapses on a constant or a
    trivial input, the outcome falls into one of these kinds — the census of every degenerate-denominator answer the 60
    functions actually give. It documents the regime a crafted pin witnesses; ``None`` on a declaration means the
    function has no degenerate-denominator regime to answer for (a drawdown, a running-total, a plain quantile).
    """

    ZERO_DISPERSION_IS_NAN = "zero_dispersion_is_nan"  # a standardized moment / embedded slope over a constant is 0/0
    RATIO_SIGNED_INF_OR_NAN = "ratio_signed_inf_or_nan"  # a vanishing denominator gives a signed ``inf`` or a 0/0 NaN
    EXACT_ZERO = "exact_zero"  # a dispersion over an exactly-constant series pins to exactly ``0.0`` (the min==max pin)
    COLLAPSES = "collapses"  # a trivial input collapses the statistic onto a fixed value in its bounded range
