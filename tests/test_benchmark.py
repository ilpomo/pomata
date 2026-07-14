"""
Performance benchmarks and complexity-scaling checks for every public function (non-gating).

One file for all three families: the per-function expression and its timing frame are *derived* from the registry —
the canonical call is :func:`build_expr`, the frame is :func:`probe_frame` at the timed size — so there is no separate
case table to keep in sync: ``ALL_SPECS`` (bijective with the public surface by the
:mod:`tests.all_specs` import-time check) *is* the coverage. A newly added function is swept in the moment its spec
lands.

Kept out of the default run: the module skips unless ``POMATA_BENCHMARKS`` is set, and a dedicated, non-gating CI job
runs it. The pytest-benchmark timings give performance visibility (and a baseline for ``--benchmark-compare``) over a
fixed 100k-row frame. The scaling check guards each function's declared polynomial cost degree at the smallest frame
its own discrimination inequality certifies (see :mod:`tests.support.benchmarks` for the derivation): the base walks
a quantized ladder until the measured cost clears the function's fixed per-call cost by
``SCALING_OVERHEAD_MULTIPLE``, so an expensive sequential kernel certifies in thousands of rows while a cheap
vectorized one walks higher — the tier's wall clock stays bounded per function whatever the kernel weighs, and grows
only linearly with the size of the registry.
"""

import os

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from tests.all_specs import ALL_SPECS
from tests.support import SCALING_OVERHEAD_MULTIPLE, fastest_eval, scaling_threshold
from tests.support.spec import Spec, build_expr, probe_frame, spec_id

if not os.environ.get("POMATA_BENCHMARKS"):
    pytest.skip("set POMATA_BENCHMARKS=1 to run the benchmark tier", allow_module_level=True)

# The quantized ladder the scaling base walks: quantization keeps the chosen base stable across runs, and the cap at
# 100k keeps the decade's large frame at 1M rows even for a kernel so cheap that no rung clears the stop rule.
_LADDER = (1_000, 10_000, 100_000)

# Frames this small sit below every window, so the evaluation is all fixed per-call cost (dispatch, column wiring,
# any map_batches round-trip) — the ``h`` of the cost model, measured per function rather than assumed.
_OVERHEAD_ROWS = 8

# time.perf_counter resolves far finer, but a Polars select never completes faster than ~tens of microseconds; the
# floor keeps one anomalous minimum from shrinking every certified base.
_OVERHEAD_FLOOR = 2e-5


@pytest.mark.benchmark
@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_throughput(benchmark: BenchmarkFixture, spec: Spec) -> None:
    """
    Records the evaluation time of each function over a fixed-size frame.
    """
    frame = probe_frame(spec.inputs, 100_000)
    expr = build_expr(spec).alias("y")
    benchmark(lambda: frame.select(expr))


@pytest.mark.benchmark
@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_scaling_matches_declared_degree(spec: Spec) -> None:
    """
    Verifies a 10x increase in rows stays under the bound the spec's declared ``cost_degree`` derives, guarding
    against a regression to a higher polynomial class.

    The base size is the smallest ladder rung whose measured cost clears ``SCALING_OVERHEAD_MULTIPLE`` times the
    function's own fixed per-call cost, so the ratio carries real scaling signal; the additive term in the bound is
    that same fixed cost — the constant of the cost model, which also keeps the rare never-certified (overhead-bound)
    kernel from flaking, where a genuine regression still explodes the absolute time.
    """
    overhead = max(fastest_eval(probe_frame(spec.inputs, _OVERHEAD_ROWS), lambda: build_expr(spec)), _OVERHEAD_FLOOR)
    base_size = _LADDER[-1]
    base_time = None
    for size in _LADDER:
        base_time = fastest_eval(probe_frame(spec.inputs, size), lambda: build_expr(spec))
        if base_time >= SCALING_OVERHEAD_MULTIPLE * overhead:
            base_size = size
            break
    assert base_time is not None  # the ladder is non-empty; the bind narrows it for the type checker
    large_time = fastest_eval(probe_frame(spec.inputs, 10 * base_size), lambda: build_expr(spec))
    bound = scaling_threshold(spec.cost_degree) * base_time + SCALING_OVERHEAD_MULTIPLE * overhead
    assert large_time < bound, (
        f"{spec.name}: 10x the rows ({base_size:,} -> {10 * base_size:,}) cost {large_time / base_time:.1f}x "
        f"(declared cost_degree {spec.cost_degree}, bound {scaling_threshold(spec.cost_degree):.0f}x) — a "
        f"regression to a higher polynomial class, or a wrong declared degree"
    )
