"""
Self-tests of the contract machinery: the three locks bite, the derivations are true, the toy contract runs.

Every lock is exercised with a synthetic counterexample (``register = False`` keeps the toys out of the surface
registry), so the machinery itself can never regress into a silent no-op — the failure mode the redesign exists
to eliminate.
"""

from collections.abc import Mapping
from typing import ClassVar

import polars as pl
import pytest
from tests_new.support import REGISTRY, ContractSeries, ContractWindowed, probe_frame

from pomata._policy import POLICIES, NanPolicy, NullPolicy
from pomata.indicators import sma


class TestSma(ContractWindowed, ContractSeries):
    """The reference toy: a real, fully-declared contract the machinery accepts and pytest runs."""

    register: ClassVar[bool] = False  # the sma slot belongs to the real rollout, not to this self-test toy

    factory = staticmethod(sma)
    inputs: ClassVar[tuple[str, ...]] = ("expr",)
    params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
    warmup: ClassVar[int | Mapping[str, int]] = 2
    raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
        ({"window": 0}, r"window must be >= 1"),
    )


class TestMachineryLocks:
    """The three import-time locks: completeness, naming, honesty — each proven on a counterexample."""

    def test_missing_declarations_fail_with_their_names(self) -> None:
        """Verifies a contract missing required declarations dies at class creation, naming every gap."""
        with pytest.raises(TypeError, match=r"does not declare.*inputs.*params.*warmup"):

            class TestRsi(ContractWindowed, ContractSeries):  # type: ignore[unused-ignore]
                register: ClassVar[bool] = False
                factory = staticmethod(sma)

    def test_wrong_class_name_fails(self) -> None:
        """Verifies the Test<Pascal> naming law is structural: a mismatched class name dies at creation."""
        with pytest.raises(TypeError, match=r"must be named TestSma"):

            class TestSimpleMovingAverage(ContractWindowed, ContractSeries):  # type: ignore[unused-ignore]
                register: ClassVar[bool] = False
                factory = staticmethod(sma)
                inputs: ClassVar[tuple[str, ...]] = ("expr",)
                params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
                warmup: ClassVar[int | Mapping[str, int]] = 2
                raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
                    ({"window": 0}, r"window must be >= 1"),
                )

    def test_unconsented_override_fails(self) -> None:
        """Verifies a child cannot silently shadow an inherited rung."""
        with pytest.raises(TypeError, match=r"overrides inherited rungs.*test_warmup_null_count"):

            class TestSma(ContractWindowed, ContractSeries):  # type: ignore[unused-ignore]
                register: ClassVar[bool] = False
                factory = staticmethod(sma)
                inputs: ClassVar[tuple[str, ...]] = ("expr",)
                params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
                warmup: ClassVar[int | Mapping[str, int]] = 2
                raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
                    ({"window": 0}, r"window must be >= 1"),
                )

                def test_warmup_null_count(self) -> None: ...

    def test_consented_override_is_accepted(self) -> None:
        """Verifies ``override_ok`` is the visible consent that legalizes a redefinition."""

        class TestSma(ContractWindowed, ContractSeries):
            register: ClassVar[bool] = False
            override_ok: ClassVar[frozenset[str]] = frozenset({"test_warmup_null_count"})
            factory = staticmethod(sma)
            inputs: ClassVar[tuple[str, ...]] = ("expr",)
            params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
            warmup: ClassVar[int | Mapping[str, int]] = 2
            raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
                ({"window": 0}, r"window must be >= 1"),
            )

            def test_warmup_null_count(self) -> None: ...

        assert "test_warmup_null_count" in vars(TestSma)

    def test_unknown_input_role_fails(self) -> None:
        """Verifies a declared input the probe frame cannot synthesize dies at creation."""
        with pytest.raises(TypeError, match=r"probe frame cannot build.*mystery"):

            class TestSma(ContractWindowed, ContractSeries):  # type: ignore[unused-ignore]
                register: ClassVar[bool] = False
                factory = staticmethod(sma)
                inputs: ClassVar[tuple[str, ...]] = ("mystery",)
                params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
                warmup: ClassVar[int | Mapping[str, int]] = 2
                raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
                    ({"window": 0}, r"window must be >= 1"),
                )

    def test_params_without_raises_fails(self) -> None:
        """Verifies a contract with validated params but no counterexamples dies: the rung would be a no-op."""
        with pytest.raises(TypeError, match=r"no raises counterexamples"):

            class TestSma(ContractWindowed, ContractSeries):  # type: ignore[unused-ignore]
                register: ClassVar[bool] = False
                factory = staticmethod(sma)
                inputs: ClassVar[tuple[str, ...]] = ("expr",)
                params: ClassVar[Mapping[str, int | float | bool]] = {"window": 3}
                warmup: ClassVar[int | Mapping[str, int]] = 2


class TestMachineryDerivations:
    """The derivations: name, family, and policies come from the package, never from the declaration."""

    def test_name_family_and_policies_are_derived(self) -> None:
        """Verifies the toy contract carries the derived facts the child never states."""
        assert TestSma.name == "sma"
        assert TestSma.family == "indicators"
        assert (TestSma.null_policy, TestSma.nan_policy) == POLICIES["sma"]
        assert TestSma.null_policy is NullPolicy.IN_WINDOW_IS_NULL
        assert TestSma.nan_policy is NanPolicy.PROPAGATES

    def test_unregistered_toy_stays_out_of_the_registry(self) -> None:
        """Verifies ``register = False`` keeps a self-test toy off the migrated surface."""
        assert "sma" not in REGISTRY

    def test_lands_on_defaults_to_first_input(self) -> None:
        """Verifies the landing-column default derivation."""
        assert TestSma.lands_on == "expr"


class TestProbeFrame:
    """The probe frame: distinct roles, coherent bars, ``Float64`` everywhere."""

    def test_roles_are_distinct_columns(self) -> None:
        """Verifies each declared role lands in its own distinctly-named column."""
        frame = probe_frame(("high", "low", "close", "volume"), 8)
        assert frame.columns == ["high", "low", "close", "volume"]
        assert frame.height == 8

    def test_bars_are_coherent(self) -> None:
        """Verifies high >= close >= low on every probe row, so OHLC factories see valid bars."""
        frame = probe_frame(("high", "low", "close"), 16)
        coherent = frame.select(((pl.col("high") >= pl.col("close")) & (pl.col("close") >= pl.col("low"))).all())
        assert coherent.item()
