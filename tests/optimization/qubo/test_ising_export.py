from dataclasses import FrozenInstanceError

import pytest

from aicir.optimization.qubo import Binary, IsingExport, Model, QuboBuilder


def _small_model() -> Model:
    x = Binary("ie_x")
    y = Binary("ie_y")
    z = Binary("ie_z")
    return Model(2.0 * x + 3.0 * y - 1.5 * z + 4.0 * x * y - 0.5 * y * z)


def test_model_to_ising_export_matches_named_dict_payload() -> None:
    model = _small_model()

    named = model.to_ising()
    export = model.to_ising_export()

    assert isinstance(export, IsingExport)
    assert dict(export.linear) == named["h"]
    assert dict(export.quadratic) == named["J"]
    assert export.offset == named["offset"]
    assert export.variable_names is not None
    assert {"ie_x", "ie_y", "ie_z"} <= set(export.variable_names)
    assert export.variable_metadata is not None
    assert {"ie_x", "ie_y", "ie_z"} <= {meta.name for meta in export.variable_metadata}


def test_ising_model_to_export_preserves_index_keys() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("ix_x")
    y = builder.registry.get_or_create("ix_y")
    builder.add_linear(x, 2.0)
    builder.add_quadratic(x, y, 4.0)

    ising = builder.to_ising_indices()
    export = ising.to_export()

    assert isinstance(export, IsingExport)
    assert dict(export.linear) == ising.h
    assert dict(export.quadratic) == ising.J
    assert export.offset == ising.offset
    assert export.variable_names == tuple(ising.variable_names)


def test_ising_export_is_frozen() -> None:
    export = IsingExport(linear={}, quadratic={}, offset=0.0)

    with pytest.raises(FrozenInstanceError):
        export.offset = 1.0  # type: ignore[misc]


def test_ising_export_exported_from_qubo_modeling_package() -> None:
    import aicir.optimization.qubo.modeling as modeling

    assert "IsingExport" in modeling.__all__
    assert modeling.IsingExport is IsingExport
