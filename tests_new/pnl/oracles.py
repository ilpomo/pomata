"""
The naive reference oracles for the pnl family — one per public primitive, gathered in a single module.

Each function recomputes one PnL primitive from scratch in plain Python, sharing no code with the Polars implementation
it checks, so agreement between the two is evidence of correctness rather than coincidence. They target the semantics of
the project's Polars floor; each docstring states the definition, the subtle points the reimplementation must reproduce,
and its null / NaN / degeneracy contract. Named ``reference_{function}`` so the declaration's binding guard can tie each
to the factory it checks; the turnover-composed costs call :func:`reference_turnover` directly.
"""

import math
from collections.abc import Sequence


def reference_turnover(
    weight: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Turnover over a Python list.

    The absolute one-bar change ``|weight[t] - weight[t-1]|`` with the pre-series weight taken as flat (``0``),
    recomputed as the oracle for :func:`pomata.pnl.turnover`. So the first row is ``|weight[0]|`` (the entry trade
    from cash), not ``None``; its subtlety is the missing-data rule of a two-endpoint difference — a ``None`` voids its
    own row and the next, a ``nan`` propagates likewise — detailed below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``weight``: the traded fraction at each row, with the first row ``|weight[0]|``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]|`` rather than ``None``.
        - **Null** — a ``None`` makes its own row ``None`` and the next row ``None`` (the difference references the
          previous weight); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next, yielding ``nan`` there.
    """
    results: list[float | None] = []
    for index in range(len(weight)):
        current = weight[index]
        previous = weight[index - 1] if index > 0 else 0.0
        if current is None or previous is None:
            results.append(None)
        else:
            results.append(abs(current - previous))
    return results


def reference_cost_borrow(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Short-Borrow Cost over two aligned Python lists.

    The per-bar short-borrow fee ``max(-quantity, 0) * price * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_borrow`. Only the short part of the position is charged; its one subtlety is the missing-data
    rule of plain arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a
    ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``); only the short part
            (``< 0``) is charged.
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Per-bar borrow rate, as a fraction of the short notional. Must be ``>= 0``.

    Returns:
        A list the same length as the inputs: a non-negative cost on short bars and ``0`` on long or flat bars.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Long / flat** — a non-negative quantity has zero borrow cost.
        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for quantity_value, price_value in zip(quantity, price, strict=True):
        if quantity_value is None or price_value is None:
            results.append(None)
        elif math.isnan(quantity_value) or (math.isnan(price_value)):
            results.append(math.nan)
        else:
            results.append(max(-quantity_value, 0.0) * price_value * rate)
    return results


def reference_cost_fixed(
    quantity: Sequence[float | None],
    fee: float,
) -> list[float | None]:
    """
    Naive Fixed Transaction Cost over a Python list.

    A flat ``fee`` on every bar where the position changes (``turnover(quantity) > 0``) and ``0`` where it is held,
    recomputed as the oracle for :func:`pomata.pnl.cost_fixed` by composing the independent :func:`reference_turnover`.
    Its subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) —
    detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        fee: Flat charge per trade, in the account currency. Must be ``>= 0``.

    Returns:
        A list the same length as ``quantity``: ``fee`` where the quantity changes (the first row counts as a trade from
        a flat start) and ``0`` where it is held.

    Raises:
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row charges the ``fee``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError(f"fee must be a finite number >= 0, got {fee}")

    results: list[float | None] = []
    for traded in reference_turnover(quantity):
        if traded is None:
            results.append(None)
        elif math.isnan(traded):
            results.append(math.nan)
        elif traded > 0.0:
            results.append(fee)
        else:
            results.append(0.0)
    return results


def reference_cost_funding(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Funding Cost over three aligned Python lists.

    The elementwise product ``quantity * price * rate``, recomputed as the oracle for :func:`pomata.pnl.cost_funding`.
    It is a pure per-row product carrying the signed funding payment on the held notional; its one subtlety is the
    missing-data rule of plain arithmetic — ``None`` in any input propagates to ``None`` (taking precedence over
    ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument price for each bar (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Signed per-bar funding rate (may contain ``None`` and ``float('nan')``); same length as ``quantity``.

    Returns:
        A list the same length as the inputs: ``quantity * price * rate`` for each row.

    Raises:
        ValueError: If ``quantity``, ``price``, and ``rate`` do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in any input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in any input (with no ``None``) propagates to ``nan``.
    """
    if not len(quantity) == len(price) == len(rate):
        raise ValueError("quantity, price, and rate must have equal length")

    results: list[float | None] = []
    for quantity_value, price_value, rate_value in zip(quantity, price, rate, strict=True):
        if quantity_value is None or price_value is None or rate_value is None:
            results.append(None)
        else:
            results.append(quantity_value * price_value * rate_value)
    return results


def reference_cost_notional(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Notional Transaction Cost over two aligned Python lists.

    The traded notional times a flat rate, ``turnover(quantity) * price * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_notional` by composing the independent :func:`reference_turnover` with the price and rate.
    Its subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates),
    plus the missing-data rule of the multiply by price — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Proportional cost rate, the fee as a fraction of traded notional. Must be ``>= 0``.

    Returns:
        A list the same length as the inputs: the per-bar notional cost, with the first row charging on
        ``|quantity[0]| * price[0]``.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row charges on the entry trade.
        - **Null** — a ``None`` in the traded quantity (or its predecessor, via turnover) or the price yields ``None``
          (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for traded, price_value in zip(reference_turnover(quantity), price, strict=True):
        if traded is None or price_value is None:
            results.append(None)
        elif math.isnan(traded) or math.isnan(price_value):
            results.append(math.nan)
        else:
            results.append(traded * price_value * rate)
    return results


def reference_cost_per_share(
    quantity: Sequence[float | None],
    fee: float,
) -> list[float | None]:
    """
    Naive Per-Share Transaction Cost over a Python list.

    The units traded times a flat per-unit fee, ``turnover(quantity) * fee``, recomputed as the oracle for
    :func:`pomata.pnl.cost_per_share` by composing the independent :func:`reference_turnover` with the fee. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        fee: Commission per unit traded, in the account currency. Must be ``>= 0``.

    Returns:
        A list the same length as ``quantity``: the per-bar per-share cost, with the first row ``|quantity[0]| * fee``.

    Raises:
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row is ``|quantity[0]| * fee``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError(f"fee must be a finite number >= 0, got {fee}")

    return [None if traded is None else traded * fee for traded in reference_turnover(quantity)]


def reference_cost_proportional(
    weight: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Proportional Transaction Cost over a Python list.

    The traded fraction times a flat rate, ``turnover(weight) * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_proportional` by composing the independent :func:`reference_turnover` with the rate. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        rate: Proportional cost rate, the fee as a fraction of traded notional. Must be ``>= 0``.

    Returns:
        A list the same length as ``weight``: the per-bar proportional cost, with the first row
        ``|weight[0]| * rate``.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]| * rate``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")

    return [None if traded is None else traded * rate for traded in reference_turnover(weight)]


def reference_cost_slippage(
    weight: Sequence[float | None],
    half_spread: float,
) -> list[float | None]:
    """
    Naive Slippage Cost over a Python list.

    The traded fraction times a fixed half-spread, ``turnover(weight) * half_spread``, recomputed as the oracle for
    :func:`pomata.pnl.cost_slippage` by composing the independent :func:`reference_turnover` with the half-spread. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        half_spread: Fixed bid-ask half-spread crossed per trade, as a fraction. Must be ``>= 0``.

    Returns:
        A list the same length as ``weight``: the per-bar slippage cost, with the first row
        ``|weight[0]| * half_spread``.

    Raises:
        ValueError: If ``half_spread`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]| * half_spread``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(half_spread) or half_spread < 0.0:
        raise ValueError(f"half_spread must be a finite number >= 0, got {half_spread}")

    return [None if traded is None else traded * half_spread for traded in reference_turnover(weight)]


def reference_cumulative_pnl(
    returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Cumulative P&L over a Python list.

    The additive running total ``sum(returns[:t+1])``, recomputed as the oracle for
    :func:`pomata.pnl.cumulative_pnl`. Its subtlety is matching Polars' ``cum_sum`` missing-data rule: a ``None`` emits
    ``None`` while the running sum carries across it unchanged, whereas a ``nan`` enters the sum and propagates to every
    later row — detailed below.

    Args:
        returns: Input per-bar returns to cumulate (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``returns``: the running sum at each row.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` emits ``None`` at that row and the running sum carries across it unchanged.
        - **NaN** — a ``nan`` enters the running sum, so that row and every later row are ``nan``.
    """
    accumulator = 0.0
    results: list[float | None] = []
    for value in returns:
        if value is None:
            results.append(None)
        else:
            accumulator += value
            results.append(accumulator)
    return results


def reference_dividend(
    quantity: Sequence[float | None],
    dividend_per_share: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Dividend Cashflow over two aligned Python lists.

    The elementwise product ``quantity * dividend_per_share``, recomputed as the oracle for
    :func:`pomata.pnl.dividend`. It is a pure per-row product; its one subtlety is the missing-data rule of plain
    arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a ``nan``
    propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        dividend_per_share: Dividend paid per share for each bar (may contain ``None`` and ``float('nan')``); same
            length as ``quantity``.

    Returns:
        A list the same length as the inputs: ``quantity * dividend_per_share`` for each row.

    Raises:
        ValueError: If ``quantity`` and ``dividend_per_share`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if len(quantity) != len(dividend_per_share):
        raise ValueError("quantity and dividend_per_share must have equal length")

    results: list[float | None] = []
    for quantity_value, dps_value in zip(quantity, dividend_per_share, strict=True):
        if quantity_value is None or dps_value is None:
            results.append(None)
        else:
            results.append(quantity_value * dps_value)
    return results


def reference_equity_curve(
    returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Equity Curve over a Python list.

    The compounded running product ``prod(1 + returns[:t+1])``, recomputed as the oracle for
    :func:`pomata.pnl.equity_curve`. Its subtlety is matching Polars' ``cum_prod`` missing-data rule: a ``None`` emits
    ``None`` while the running product carries across it unchanged (a missing bar is a neutral factor of one), whereas a
    ``nan`` enters the product and propagates to every later row — detailed below.

    Args:
        returns: Input per-bar returns to compound (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``returns``: the compounded equity at each row, relative to a starting capital of
        ``1``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` emits ``None`` at that row and the running product carries across it unchanged; a
          leading warm-up ``None`` therefore stays ``None`` and the curve begins at the first defined return.
        - **NaN** — a ``nan`` enters the running product, so that row and every later row are ``nan``.
    """
    accumulator = 1.0
    results: list[float | None] = []
    for value in returns:
        if value is None:
            results.append(None)
        else:
            accumulator *= 1.0 + value
            results.append(accumulator)
    return results


def reference_pnl_gross(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Naive Gross Position PnL over two aligned Python lists.

    The per-bar mark-to-market PnL ``quantity * (price - price_prev) * multiplier``, recomputed as the oracle for
    :func:`pomata.pnl.pnl_gross`. Its subtleties are the warm-up first row (no previous price) and the missing-data rule
    of plain arithmetic — ``None`` in the quantity, the price, or the previous price propagates to ``None`` (taking
    precedence over ``nan``), and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        multiplier: Contract multiplier / point value. Must be ``> 0``.

    Returns:
        A list the same length as the inputs. The first entry is ``None`` (warm-up); thereafter
        ``quantity * (price - price_prev) * multiplier``.

    Raises:
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in the quantity, the price, or the previous price yields ``None`` (``None`` takes
          precedence over ``nan``).
        - **NaN** — a ``nan`` in any of them (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a finite number > 0, got {multiplier}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for index in range(len(quantity)):
        if index == 0:
            results.append(None)
            continue
        current_quantity = quantity[index]
        current_price = price[index]
        previous_price = price[index - 1]
        if current_quantity is None or current_price is None or previous_price is None:
            results.append(None)
        elif math.isnan(current_quantity) or math.isnan(current_price) or math.isnan(previous_price):
            results.append(math.nan)
        else:
            results.append(current_quantity * (current_price - previous_price) * multiplier)
    return results


def reference_pnl_gross_inverse(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Naive Gross Inverse-Contract PnL over two aligned Python lists.

    The per-bar coin-settled mark-to-market PnL ``quantity * multiplier * (1 / price_prev - 1 / price)``, recomputed as
    the oracle for :func:`pomata.pnl.pnl_gross_inverse`. Its subtleties are the warm-up first row (no previous price),
    the IEEE-754 division reproduced from Polars on the price reciprocal (a zero price is ``+/-inf``), and the
    missing-data rule of plain arithmetic — ``None`` in the quantity, the price, or the previous price propagates to
    ``None`` (taking precedence over ``nan``), and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes in contracts for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices, the quote per base unit (may contain ``None`` and ``float('nan')``); same length as
            ``quantity``.
        multiplier: Contract notional in the quote currency. Must be ``> 0``.

    Returns:
        A list the same length as the inputs. The first entry is ``None`` (warm-up); thereafter
        ``quantity * multiplier * (1 / price_prev - 1 / price)``.

    Raises:
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in the quantity, the price, or the previous price yields ``None`` (``None`` takes
          precedence over ``nan``).
        - **NaN** — a ``nan`` in any of them (with no ``None``) propagates to ``nan``.
        - **Domain** — reproducing Polars' IEEE-754 division of the reciprocal: a zero price makes ``1 / price`` carry
          the sign of the zero as ``+/-inf`` (so the bar is signed infinity), and a negative price gives a finite but
          economically meaningless value.
    """
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a finite number > 0, got {multiplier}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for index in range(len(quantity)):
        if index == 0:
            results.append(None)
            continue
        current_quantity = quantity[index]
        current_price = price[index]
        previous_price = price[index - 1]
        if current_quantity is None or current_price is None or previous_price is None:
            results.append(None)
        elif math.isnan(current_quantity) or math.isnan(current_price) or math.isnan(previous_price):
            results.append(math.nan)
        else:
            reciprocal_previous = (
                1.0 / previous_price if previous_price != 0.0 else math.copysign(math.inf, previous_price)
            )
            reciprocal_current = 1.0 / current_price if current_price != 0.0 else math.copysign(math.inf, current_price)
            results.append(current_quantity * multiplier * (reciprocal_previous - reciprocal_current))
    return results


def reference_pnl_net(
    pnl_gross: Sequence[float | None],
    cost: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Net Position PnL over two aligned Python lists.

    The elementwise difference ``pnl_gross - cost``, recomputed as the oracle for :func:`pomata.pnl.pnl_net`. It is a
    pure per-row subtraction; its one subtlety is the missing-data rule of plain arithmetic — ``None`` in either input
    propagates to ``None`` (taking precedence over ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        pnl_gross: Gross per-bar position PnL (may contain ``None`` and ``float('nan')``).
        cost: Per-bar transaction cost (may contain ``None`` and ``float('nan')``); same length as ``pnl_gross``.

    Returns:
        A list the same length as the inputs: ``pnl_gross - cost`` for each row.

    Raises:
        ValueError: If ``pnl_gross`` and ``cost`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if len(pnl_gross) != len(cost):
        raise ValueError("pnl_gross and cost must have equal length")

    results: list[float | None] = []
    for gross_value, cost_value in zip(pnl_gross, cost, strict=True):
        if gross_value is None or cost_value is None:
            results.append(None)
        else:
            results.append(gross_value - cost_value)
    return results


def reference_returns_gross(
    weight: Sequence[float | None],
    asset_returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Gross Strategy Returns over two aligned Python lists.

    The elementwise product ``weight * asset_returns``, recomputed as the oracle for
    :func:`pomata.pnl.returns_gross`. It is a pure per-row product with no lag; its one subtlety is the missing-data
    rule of plain arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a
    ``nan`` propagates to ``nan`` — detailed below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        asset_returns: Per-bar asset returns (may contain ``None`` and ``float('nan')``); same length as ``weight``.

    Returns:
        A list the same length as the inputs: ``weight * asset_returns`` for each row.

    Raises:
        ValueError: If ``weight`` and ``asset_returns`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None`` at that row) propagates to ``nan``.
    """
    if len(weight) != len(asset_returns):
        raise ValueError("weight and asset_returns must have equal length")

    results: list[float | None] = []
    for weight_value, asset_return in zip(weight, asset_returns, strict=True):
        if weight_value is None or asset_return is None:
            results.append(None)
        else:
            results.append(weight_value * asset_return)
    return results


def reference_returns_log(
    expr: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Logarithmic Returns over a Python list.

    The natural log of the price relative ``ln(P_t / P_{t-1})`` against the previous observation, recomputed as the
    oracle for :func:`pomata.pnl.returns_log`. Its subtlety is matching the IEEE-754 logarithm Polars applies to the
    relative: a zero relative is ``-inf``, a negative relative is ``nan``, and a zero previous price makes the relative
    ``+inf`` (so the log is ``+inf``); like a fixed-lag transform it is a two-endpoint operation, so missing data never
    latches — detailed below.

    Args:
        expr: Input series, the prices to difference (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``expr``. The first entry is ``None`` (warm-up); thereafter ``ln(P_t / P_{t-1})``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` at the current row or at the previous row yields ``None`` at that position.
        - **NaN** — a ``nan`` at the current row or at the previous row (and no ``None``) yields ``nan``.
        - **Domain** — reproducing Polars' IEEE-754 logarithm of the relative: a zero relative (``P_t = 0`` over a
          positive ``P_{t-1}``) is ``-inf``, a negative relative (the prices straddle zero) is ``nan``, and a zero
          previous price makes the relative ``+/-inf`` so the log is ``+inf`` (positive relative) or ``nan`` (negative).
    """
    results: list[float | None] = []
    for index in range(len(expr)):
        if index == 0:
            results.append(None)
            continue
        current = expr[index]
        past = expr[index - 1]
        if current is None or past is None:
            results.append(None)
        elif math.isnan(current) or math.isnan(past):
            results.append(math.nan)
        else:
            # ln(current / past), reproducing Polars' IEEE-754 boundary values for a zero / negative relative.
            if past == 0.0:
                relative = math.nan if current == 0.0 else math.copysign(math.inf, current) * math.copysign(1.0, past)
            else:
                relative = current / past
            if math.isnan(relative) or relative < 0.0:
                results.append(math.nan)
            elif relative == 0.0:
                results.append(-math.inf)
            elif math.isinf(relative):
                results.append(math.inf)
            else:
                results.append(math.log(relative))
    return results


def reference_returns_net(
    returns_gross: Sequence[float | None],
    cost: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Net Strategy Returns over two aligned Python lists.

    The elementwise difference ``returns_gross - cost``, recomputed as the oracle for :func:`pomata.pnl.returns_net`. It
    is a pure per-row subtraction; its one subtlety is the missing-data rule of plain arithmetic — ``None`` in either
    input propagates to ``None`` (taking precedence over ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        returns_gross: Gross per-bar strategy returns (may contain ``None`` and ``float('nan')``).
        cost: Per-bar transaction cost (may contain ``None`` and ``float('nan')``); same length as ``returns_gross``.

    Returns:
        A list the same length as the inputs: ``returns_gross - cost`` for each row.

    Raises:
        ValueError: If ``returns_gross`` and ``cost`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None`` at that row) propagates to ``nan``.
    """
    if len(returns_gross) != len(cost):
        raise ValueError("returns_gross and cost must have equal length")

    results: list[float | None] = []
    for gross, drag in zip(returns_gross, cost, strict=True):
        if gross is None or drag is None:
            results.append(None)
        else:
            results.append(gross - drag)
    return results


def reference_returns_simple(
    expr: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Simple Returns over a Python list.

    The fractional change ``P_t / P_{t-1} - 1`` against the previous observation, recomputed as the oracle for
    :func:`pomata.pnl.returns_simple`. Its subtlety is the IEEE-754 behavior when the previous value is zero (``0 / 0``
    is ``nan``, a non-zero change over zero is ``+/-inf``); like a fixed-lag difference it is a two-endpoint operation,
    so missing data never latches — detailed below.

    Args:
        expr: Input series, the prices to difference (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``expr``. The first entry is ``None`` (warm-up); thereafter the simple return
        ``P_t / P_{t-1} - 1``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` at the current row or at the previous row yields ``None`` at that position.
        - **NaN** — a ``nan`` at the current row or at the previous row (and no ``None``) yields ``nan``.
        - **Division by zero** — when the previous value is ``0`` the relative divides by zero following IEEE-754: a
          zero change (``0 / 0``) is ``nan`` and a non-zero change over zero is ``+/-inf`` (the sign tracks the change
          relative to the signed zero).
    """
    results: list[float | None] = []
    for index in range(len(expr)):
        if index == 0:
            results.append(None)
            continue
        current = expr[index]
        past = expr[index - 1]
        if current is None or past is None:
            results.append(None)
        elif math.isnan(current) or math.isnan(past):
            results.append(math.nan)
        elif past == 0.0:
            # P_t / 0 - 1 following IEEE-754: 0/0 is nan, a non-zero change over signed zero is +/-inf.
            if current == 0.0:
                results.append(math.nan)
            else:
                results.append(math.copysign(math.inf, current) * math.copysign(1.0, past))
        else:
            results.append(current / past - 1.0)
    return results
