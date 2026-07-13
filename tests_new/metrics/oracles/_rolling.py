"""
Shared windowing helper for the rolling-metric reference oracles.

A rolling metric's oracle is its reducing oracle applied to each trailing window: the first ``window - 1`` rows are
warm-up ``None``; a window holding any ``None`` is ``None`` (the rolling policy requires ``window`` non-null values, so
a ``null`` dominates -- unlike the reducing metrics, which skip nulls); otherwise the reducing oracle is recomputed over
the clean window, which already handles ``NaN`` propagation and the degenerate (``inf`` / ``NaN``) cases. This keeps
each rolling oracle a one-line wrapper over the verified reducing oracle, sharing no code with the implementation.
"""

from collections.abc import Callable, Sequence


def rolling_reference_pair(
    reduce: Callable[[list[float | None], list[float | None]], float | None],
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Apply a two-input reducing reference ``reduce`` to each trailing window of the ``(returns, benchmark)`` pair.

    The benchmark-relative rolling policy: warm-up and any window holding a ``None`` in either leg are ``None``;
    otherwise the reducing relative oracle is recomputed over the window (it already handles ``NaN`` and the degenerate
    cases).

    Args:
        reduce: The reducing relative oracle, called as ``reduce(window_returns, window_benchmark)`` on a clean window.
        returns: The portfolio return series (may contain ``None`` and ``float('nan')``).
        benchmark: The benchmark return series, aligned with ``returns``.
        window: The trailing window length.

    Returns:
        A list the same length as the inputs: ``None`` for warm-up or any-null windows, else ``reduce`` of the window.
    """
    output: list[float | None] = []
    for index in range(len(returns)):
        if index < window - 1:
            output.append(None)
            continue
        window_returns = list(returns[index - window + 1 : index + 1])
        window_benchmark = list(benchmark[index - window + 1 : index + 1])
        if any(value is None for value in window_returns) or any(value is None for value in window_benchmark):
            output.append(None)
            continue
        output.append(reduce(window_returns, window_benchmark))
    return output


def rolling_reference(
    reduce: Callable[[list[float | None]], float | None],
    values: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Apply a reducing reference ``reduce`` to each trailing window of ``values``.

    Args:
        reduce: The reducing oracle bound to its parameters, called as ``reduce(window_slice)`` on a clean window.
        values: The input series (may contain ``None`` and ``float('nan')``).
        window: The trailing window length.

    Returns:
        A list the same length as ``values``: ``None`` for warm-up or any-null windows, else ``reduce`` of the window.
    """
    output: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            output.append(None)
            continue
        window_slice = list(values[index - window + 1 : index + 1])
        if any(value is None for value in window_slice):
            output.append(None)
            continue
        output.append(reduce(window_slice))
    return output
