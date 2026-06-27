"""
pomata — a verifiably-correct, Polars-native quant toolkit.

Three families of atomic, composable building blocks:

- ``pomata.indicators`` — technical-analysis indicators as ``pl.Expr`` factories.
- ``pomata.metrics`` — performance & risk metrics.
- ``pomata.pnl`` — profit-and-loss accounting and transaction-cost models.

Every public function returns a free-standing Polars expression (or a small, explicit object); nothing forces a
dataframe shape on the caller.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pomata")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__: tuple[str, ...] = ()
