"""Contract for ``pomata.indicators.ichimoku`` — struct, per-field warm-ups, three ordered windows."""

from collections.abc import Mapping
from typing import ClassVar

from tests.indicators.oracles import ichimoku_reference
from tests_new.support import ContractProperties, ContractStruct, ContractWindowed

from pomata.indicators import ichimoku


class TestIchimoku(ContractWindowed, ContractStruct, ContractProperties):
    """Declarations only: every rung is inherited from the composed contracts."""

    factory = staticmethod(ichimoku)
    inputs: ClassVar[tuple[str, ...]] = ("high", "low")
    params: ClassVar[Mapping[str, int | float | bool]] = {
        "window_tenkan": 9,
        "window_kijun": 26,
        "window_senkou": 52,
    }
    fields: ClassVar[tuple[str, ...]] = ("tenkan", "kijun", "senkou_a", "senkou_b")
    warmup: ClassVar[int | Mapping[str, int]] = {"tenkan": 8, "kijun": 25, "senkou_a": 25, "senkou_b": 51}
    raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
        ({"window_tenkan": 0}, r"window_tenkan must be >= 1"),
        ({"window_kijun": 0, "window_tenkan": 1}, r"window_kijun must be >= 1"),
        ({"window_senkou": 0, "window_tenkan": 1, "window_kijun": 1}, r"window_senkou must be >= 1"),
        ({"window_kijun": 5}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
    )

    oracle = staticmethod(ichimoku_reference)
    golden_params: ClassVar[Mapping[str, float | bool]] = {"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4}
    golden_input: ClassVar[Mapping[str, tuple[float | None, ...]]] = {
        "high": (10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0),
        "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0),
    }
    golden_output: ClassVar[tuple[float | None, ...] | Mapping[str, tuple[float | None, ...]]] = {
        "tenkan": (None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0),
        "kijun": (None, None, 10.0, 11.0, 12.0, 12.0, 12.5, 12.5),
        "senkou_a": (None, None, 10.25, 11.25, 12.25, 12.0, 12.5, 12.75),
        "senkou_b": (None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5),
    }
