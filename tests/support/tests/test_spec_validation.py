"""
Self-tests of the spec engine: every conditional requirement bites on a counterexample built inline.

The spec language's core guarantee is that a declaration cannot lie by omission. These tests construct deliberately
wrong specs and prove each is rejected at construction — a missing required field by the language itself (a native
``TypeError``), and every conditional rule by :meth:`Spec.__post_init__`. If any check ever regressed into a silent
no-op, one of these would go green where it must be red.
"""

import polars as pl
import pytest
from tests.support import Deviant, ScaleAxis, ScaleExempt, Shape, Spec

from pomata.indicators import sma


def _oracle(*_args: object, **_kwargs: object) -> object:
    """A stand-in oracle: ``__post_init__`` never calls it, it only checks the golden shapes around it."""
    return []


class TestNativeCompleteness:
    """The required fields have no default, so the language itself refuses an incomplete spec."""

    def test_missing_required_field_is_a_native_type_error(self) -> None:
        """Verifies omitting a required field dies at construction, naming the field."""
        with pytest.raises(TypeError, match=r"factory"):
            Spec(  # type: ignore[call-arg]
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_scale_exempt_without_reason_is_a_native_type_error(self) -> None:
        """Verifies ``ScaleExempt`` cannot be built without its mandatory reason."""
        with pytest.raises(TypeError, match=r"reason"):
            ScaleExempt()  # type: ignore[call-arg]

    def test_deviant_without_reason_is_a_native_type_error(self) -> None:
        """Verifies ``Deviant`` cannot be built without its mandatory reason."""
        with pytest.raises(TypeError, match=r"reason"):
            Deviant(expected=(1.0,))  # type: ignore[call-arg]


class TestConditionalRequirements:
    """Every conditional rule is checked in ``__post_init__`` and proven on a counterexample."""

    def test_struct_without_fields_is_rejected(self) -> None:
        """Verifies a struct that names no fields dies at construction."""
        with pytest.raises(ValueError, match=r"a struct must declare its ordered fields"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,)},
            )

    def test_reducing_with_a_warmup_is_rejected(self) -> None:
        """Verifies a reduction that declares a warm-up dies at construction."""
        with pytest.raises(ValueError, match=r"a reduction has no warm-up"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.REDUCING,
                warmup=2,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_params_without_raises_is_rejected(self) -> None:
        """Verifies declared params without validation counterexamples die at construction (the rung would no-op)."""
        with pytest.raises(ValueError, match=r"declares params but no raises"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={"window": 3},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_empty_scale_tuple_is_rejected(self) -> None:
        """Verifies an empty scale tuple dies at construction — the exemption must be a reasoned ``ScaleExempt``."""
        with pytest.raises(ValueError, match=r"empty scale tuple is never allowed"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=(),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_scale_axis_role_outside_inputs_is_rejected(self) -> None:
        """Verifies a scale axis naming a role that is not an input dies at construction."""
        with pytest.raises(ValueError, match=r"a scale axis names input roles"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=(ScaleAxis(roles=("mystery",), degree=1),),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_unknown_input_role_is_rejected(self) -> None:
        """Verifies an input the probe frame cannot build dies at construction."""
        with pytest.raises(ValueError, match=r"the probe frame can build; unknown: \['mystery'\]"):
            Spec(
                factory=sma,
                inputs=("mystery",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"mystery": (1.0,)},
                golden_output=(1.0,),
            )

    def test_name_outside_the_policy_registry_is_rejected(self) -> None:
        """Verifies a factory whose name has no declared policy dies at construction (the name is derived)."""

        def unregistered(expr: pl.Expr) -> pl.Expr:
            return expr

        with pytest.raises(ValueError, match=r"no declared policy in pomata._policy"):
            Spec(
                factory=unregistered,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )
