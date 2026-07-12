"""Contract for ``pomata.indicators.mama`` — the adaptive-alpha cycle pair, latching, golden-anchored."""

import math
from collections.abc import Mapping
from typing import ClassVar

import polars as pl
from tests.indicators.oracles import mama_reference
from tests_new.support import ContractProperties, ContractStruct, ContractWindowed

from pomata.indicators import mama

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up.
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))


class TestMama(ContractWindowed, ContractStruct, ContractProperties):
    """Declarations only; the oracle is the shared structural mirror, reached through ``_reference``."""

    factory = staticmethod(mama)
    inputs: ClassVar[tuple[str, ...]] = ("expr",)
    params: ClassVar[Mapping[str, int | float | bool]] = {"limit_fast": 0.5, "limit_slow": 0.05}
    fields: ClassVar[tuple[str, ...]] = ("mama", "fama")
    warmup: ClassVar[int | Mapping[str, int]] = 32
    raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
        ({"limit_fast": 0.0}, r"limit_fast must be in"),
        ({"limit_slow": 0.0}, r"limit_slow must be in"),
        ({"limit_fast": 3.0}, r"limit_fast must be in"),
        ({"limit_slow": 2.5}, r"limit_slow must be in"),
        ({"limit_fast": 0.05, "limit_slow": 0.5}, r"limit_fast must be >= limit_slow"),
    )

    oracle = staticmethod(mama_reference)
    golden_input: ClassVar[Mapping[str, tuple[float | None, ...]]] = {"expr": _SAMPLE}
    golden_output: ClassVar[tuple[float | None, ...] | Mapping[str, tuple[float | None, ...]]] = {
        "mama": (None,) * 32 + (97.9767, 97.6734, 97.3142, 96.9485, 96.6255, 96.3897, 96.2764, 96.308),
        "fama": (None,) * 32 + (99.6954, 99.6448, 99.5866, 99.5206, 99.4482, 99.3718, 99.2944, 99.2197),
    }

    def _reference(self, frame: pl.DataFrame) -> object:
        # The mirror's keyword names differ from the factory's (``fast_limit`` / ``slow_limit``), so the
        # signature-mirror default does not apply; the canonical params are the mirror's own defaults.
        return mama_reference(frame["expr"].to_list())
