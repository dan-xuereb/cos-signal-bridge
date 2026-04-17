"""Coverage + structural tests for signal_bridge.operator_signatures (BRIDGE-03)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sdl.types import OperatorTag

from signal_bridge.operator_signatures import (
    OPERATOR_SIGNATURES,
    OperatorSignature,
    OperatorSignatureRegistry,
)

VALID_PARAM_TYPES = {"int", "float", "bool", "str"}
VALID_OPERAND_SHAPES = {"Scalar", "Series", "OHLC", "OHLCV"}


# ---------------------------------------------------------------------------
# Coverage — every enum value has a signature, no extras
# ---------------------------------------------------------------------------


def test_every_operator_tag_has_a_signature() -> None:
    missing = sorted(op.value for op in OperatorTag if op not in OPERATOR_SIGNATURES)
    extra = sorted(k.value for k in OPERATOR_SIGNATURES if k not in set(OperatorTag))
    assert not missing, f"Missing signatures for {len(missing)} operators: {missing}"
    assert not extra, f"Unknown operator keys in registry: {extra}"


def test_no_extra_keys_in_registry() -> None:
    """Subset guard — every registry key is a known OperatorTag."""
    assert set(OPERATOR_SIGNATURES).issubset(set(OperatorTag))


def test_total_count_is_80() -> None:
    assert len(OPERATOR_SIGNATURES) == 80


# ---------------------------------------------------------------------------
# Model-level invariants
# ---------------------------------------------------------------------------


def test_operator_signature_is_frozen() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.close]
    with pytest.raises(ValidationError):
        sig.operand_arity = 99  # type: ignore[misc]


def test_registry_alias_points_at_mapping() -> None:
    assert OperatorSignatureRegistry is OPERATOR_SIGNATURES


def test_all_param_types_are_primitive_literals() -> None:
    for tag, sig in OPERATOR_SIGNATURES.items():
        for key, ty in sig.param_types.items():
            assert (
                ty in VALID_PARAM_TYPES
            ), f"{tag.value}.param_types[{key!r}] = {ty!r} not in {VALID_PARAM_TYPES}"


def test_all_operand_shapes_are_valid() -> None:
    for tag, sig in OPERATOR_SIGNATURES.items():
        assert (
            sig.operand_shape in VALID_OPERAND_SHAPES
        ), f"{tag.value}.operand_shape = {sig.operand_shape!r} not in {VALID_OPERAND_SHAPES}"


def test_all_windows_within_spec() -> None:
    for tag, sig in OPERATOR_SIGNATURES.items():
        if sig.min_window is not None:
            assert sig.max_window is not None, f"{tag.value}: min_window set, max_window not"
            assert (
                1 <= sig.min_window <= sig.max_window <= 1024
            ), f"{tag.value}: invalid window ({sig.min_window}, {sig.max_window})"
        else:
            assert sig.max_window is None, f"{tag.value}: max_window set without min_window"


def test_all_arities_non_negative() -> None:
    for tag, sig in OPERATOR_SIGNATURES.items():
        assert sig.operand_arity >= 0, f"{tag.value}: negative arity {sig.operand_arity}"


# ---------------------------------------------------------------------------
# Per-category structural smoke tests
# ---------------------------------------------------------------------------


def test_primitive_close_shape() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.close]
    assert sig.operand_arity == 0
    assert sig.operand_shape == "Series"
    assert sig.param_types == {}
    assert sig.min_window is None
    assert sig.max_window is None


def test_lag_shape() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.lag]
    assert sig.operand_arity == 1
    assert sig.operand_shape == "Series"
    assert sig.param_types == {"n": "int"}
    assert sig.min_window == 1
    assert sig.max_window == 1024


def test_clip_shape() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.clip]
    assert sig.operand_arity == 1
    assert sig.param_types == {"lo": "float", "hi": "float"}
    assert sig.min_window is None
    assert sig.max_window is None


def test_roll_corr_shape() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.roll_corr]
    assert sig.operand_arity == 2
    assert sig.operand_shape == "Series"
    assert sig.param_types == {"n": "int"}
    assert sig.min_window == 1
    assert sig.max_window == 1024


def test_roll_autocorr_has_two_params() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.roll_autocorr]
    assert set(sig.param_types.keys()) == {"n", "lag"}
    assert sig.param_types["n"] == "int"
    assert sig.param_types["lag"] == "int"
    assert sig.operand_arity == 1


def test_ewm_accepts_span_alpha_halflife() -> None:
    sig = OPERATOR_SIGNATURES[OperatorTag.ewm]
    assert {"span", "alpha", "halflife"}.issubset(sig.param_types.keys())
    sig_std = OPERATOR_SIGNATURES[OperatorTag.ewm_std]
    assert {"span", "alpha", "halflife"}.issubset(sig_std.param_types.keys())


def test_regime_ops_have_signatures() -> None:
    gate = OPERATOR_SIGNATURES[OperatorTag.regime_gate]
    blend = OPERATOR_SIGNATURES[OperatorTag.regime_blend]
    switch = OPERATOR_SIGNATURES[OperatorTag.regime_switch]
    assert gate.operand_arity == 2
    assert blend.operand_arity == 2
    assert switch.operand_arity == 3


def test_logical_name_aliases_use_enum_member() -> None:
    # Python-name vs value caveat: OperatorTag.and_ has value "and", not "and_".
    # Registry keys must be the enum member itself, not the bare string.
    assert OperatorTag.and_ in OPERATOR_SIGNATURES
    assert OperatorTag.or_ in OPERATOR_SIGNATURES
    assert OperatorTag.not_ in OPERATOR_SIGNATURES
    # Stored keys are OperatorTag instances, not bare str (OperatorTag(str, Enum)
    # compares equal to its value via `__eq__`, so a pure `in` check doesn't
    # discriminate — verify via isinstance on the concrete key list).
    for key in OPERATOR_SIGNATURES:
        assert isinstance(key, OperatorTag), f"Key {key!r} is not an OperatorTag"


def test_composition_scale_and_combine() -> None:
    scale = OPERATOR_SIGNATURES[OperatorTag.scale]
    assert scale.operand_arity == 1
    assert scale.param_types == {"factor": "float"}
    combine = OPERATOR_SIGNATURES[OperatorTag.combine]
    assert combine.operand_arity == 2
    assert combine.param_types == {"w": "float"}


def test_all_primitives_have_arity_zero() -> None:
    """Per RESEARCH.md _PRIMITIVE_OPS: 20 primitive operators, all arity 0."""
    primitives = {
        OperatorTag.open,
        OperatorTag.high,
        OperatorTag.low,
        OperatorTag.close,
        OperatorTag.vwap,
        OperatorTag.mid,
        OperatorTag.volume,
        OperatorTag.dollar_volume,
        OperatorTag.num_trades,
        OperatorTag.nupl,
        OperatorTag.sopr,
        OperatorTag.mvrv,
        OperatorTag.nvt,
        OperatorTag.rhodl,
        OperatorTag.puell,
        OperatorTag.hash_ribbon,
        OperatorTag.funding,
        OperatorTag.oi,
        OperatorTag.oi_delta,
        OperatorTag.liquidations,
    }
    assert len(primitives) == 20
    for op in primitives:
        sig = OPERATOR_SIGNATURES[op]
        assert sig.operand_arity == 0, f"{op.value} should be arity 0, got {sig.operand_arity}"
        assert sig.param_types == {}, f"{op.value} should have empty param_types"
        assert sig.min_window is None
        assert sig.max_window is None


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_operator_signature_is_pydantic_model() -> None:
    """OperatorSignature is a usable Pydantic model with the documented fields."""
    sig = OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=1,
        max_window=1024,
    )
    assert sig.operand_arity == 1
    assert sig.param_types == {"n": "int"}
