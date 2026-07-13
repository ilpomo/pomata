"""
Dtype-uniformity contract for every public factory, derived from the registry.

Every factory routes each input through ``float64_expr`` (a cast to ``Float64``), so the package has one predictable
output dtype regardless of the input's numeric dtype. This tier pins that promise across the whole public surface:
``ALL_SPECS`` x {Float32, Int64}, with the probe frame cast to the input dtype and every output lane (each struct field
expanded by the engine's ``flat``) required to come back ``Float64``. Because the cases are derived from the registry,
a newly added function is swept in the moment its spec lands, and a struct that regressed one field off ``Float64`` is
caught on that field, not blurred into the whole.

The misuse path (a bare column name passed where a ``pl.Expr`` is required) is not re-pinned here: the ladder's
``test_bare_string_raises_type_error`` already covers it for every spec, so this module holds only the output-dtype
guarantee.
"""

import polars as pl
import pytest
from tests.all_specs import ALL_SPECS
from tests.support.spec import Spec, flat, probe_frame, widest_warmup

# A narrower float and a non-float, so the ``Float64`` promise is proven for a downcast and an integer input alike.
INPUT_DTYPES: tuple[type[pl.DataType], ...] = (pl.Float32, pl.Int64)


def _dtype_cases() -> tuple[list[tuple[Spec, type[pl.DataType]]], list[str]]:
    cases = [(spec, dtype) for spec in ALL_SPECS for dtype in INPUT_DTYPES]
    ids = [f"{spec.name}-{dtype}" for spec in ALL_SPECS for dtype in INPUT_DTYPES]
    return cases, ids


DTYPE_CASES, DTYPE_IDS = _dtype_cases()


@pytest.mark.parametrize(("spec", "input_dtype"), DTYPE_CASES, ids=DTYPE_IDS)
def test_output_is_float64(spec: Spec, input_dtype: type[pl.DataType]) -> None:
    """Verifies every public factory returns ``Float64`` on every lane from a Float32 or Int64 input."""
    frame = probe_frame(spec.inputs, widest_warmup(spec) + 8).with_columns(
        pl.col(role).cast(input_dtype) for role in spec.inputs
    )
    for lane in flat(spec, frame):
        assert lane.dtype == pl.Float64, f"{spec.name}: lane {lane.name!r} is {lane.dtype}, not Float64"
