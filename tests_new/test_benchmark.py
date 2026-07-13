"""
Performance benchmarks and complexity-scaling checks for every public function (non-gating).

One file for all three families: the per-function expression and its timing frame are *derived* from the registry —
the canonical call is :func:`build_expr`, the frame is :func:`probe_frame` at the timed size — so there is no separate
case table to keep in sync and no ``CASES == __all__`` guard: ``ALL_SPECS`` (bijective with the public surface by the
:mod:`tests_new.all_specs` import-time check) *is* the coverage. A newly added function is swept in the moment its spec
lands.

Kept out of the default run: the module skips unless ``POMATA_BENCHMARKS`` is set, and a dedicated, non-gating CI job
runs it. The pytest-benchmark timings give performance visibility (and a baseline for ``--benchmark-compare``); the
scaling check guards against an accidental super-linear regression — an O(n) kernel slipping to O(n^2) — which the
absolute timings alone would not reveal. Every public function is covered, so a regression is caught wherever it lands;
of them only the two Python-kernel recursions (kama, parabolic_sar) are slower than vectorized (Rust under Polars) and
would gain from a future native kernel. The cycle functions (mama, the Hilbert pipeline) carry no special sizing here,
exactly as the old per-family suites timed them: the same 100k / 1M frames as every other function.
"""

import os

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from tests_new.all_specs import ALL_SPECS
from tests_new.support import fastest_eval
from tests_new.support.spec import Spec, build_expr, probe_frame, spec_id

if not os.environ.get("POMATA_BENCHMARKS"):
    pytest.skip("set POMATA_BENCHMARKS=1 to run the benchmark tier", allow_module_level=True)


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
def test_scales_sub_quadratically(spec: Spec) -> None:
    """
    Verifies that a 10x increase in rows costs less than 25x the time (plus a small additive floor), guarding
    against an O(n^2) regression.

    The bound is multiplicative with a small additive floor: the cheapest functions evaluate in well under a
    millisecond, where the time is dominated by fixed overhead rather than the row count, and the floor keeps that
    measurement noise from failing them. A genuine super-linear regression blows the absolute time up to seconds at a
    million rows, far past either term.
    """
    base = fastest_eval(probe_frame(spec.inputs, 100_000), lambda: build_expr(spec))
    large = fastest_eval(probe_frame(spec.inputs, 1_000_000), lambda: build_expr(spec))
    assert large < 25.0 * base + 0.02, f"{spec.name}: {large:.4f}s vs {base:.4f}s base for 10x the rows (super-linear?)"
