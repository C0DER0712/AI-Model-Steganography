import torch
from torch import nn

from utils.weights import (
    WeightTensor,
    extract_weights,
    flatten_weights,
    load_modified_weights,
    restore_weights,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(8, 4)


def test_extract_weights_preserves_state_dict_order_and_metadata() -> None:
    model = TinyModel()

    extracted = extract_weights(model)
    expected_names = [
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    ]

    assert [item.name for item in extracted] == expected_names
    assert all(item.values.dtype == torch.float32 for item in extracted)
    assert all(item.shape == tuple(model.state_dict()[item.name].shape) for item in extracted)
    assert all(item.dtype == model.state_dict()[item.name].dtype for item in extracted)
    assert "bn.num_batches_tracked" not in [item.name for item in extracted]


def test_flatten_and_restore_round_trip_values_exactly() -> None:
    model = TinyModel()
    extracted = extract_weights(model)

    flattened = flatten_weights(extracted)
    restored = restore_weights(flattened, extracted)

    assert flattened.dtype == torch.float32
    assert torch.equal(flattened, flatten_weights(restored))
    assert [item.name for item in restored] == [item.name for item in extracted]
    assert [item.shape for item in restored] == [item.shape for item in extracted]


def test_restore_rejects_wrong_flattened_size() -> None:
    model = TinyModel()
    extracted = extract_weights(model)
    flattened = flatten_weights(extracted)

    try:
        restore_weights(flattened[:-1], extracted)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("restore_weights should reject size mismatches")


def test_load_modified_weights_updates_float_state_and_preserves_integer_state() -> None:
    model = TinyModel()
    reference = extract_weights(model)
    original_integer_state = model.state_dict()["bn.num_batches_tracked"].clone()
    replacement = torch.arange(
        flatten_weights(reference).numel(),
        dtype=torch.float32,
    )

    modified = restore_weights(replacement, reference)
    load_modified_weights(model, modified)

    reloaded = extract_weights(model)
    assert torch.equal(flatten_weights(reloaded), replacement)
    assert torch.equal(
        model.state_dict()["bn.num_batches_tracked"],
        original_integer_state,
    )


def test_load_modified_weights_validates_names_and_shapes() -> None:
    model = TinyModel()
    reference = extract_weights(model)
    bad_name = WeightTensor(
        name="missing.weight",
        shape=reference[0].shape,
        dtype=reference[0].dtype,
        values=reference[0].values,
    )
    bad_shape = WeightTensor(
        name=reference[0].name,
        shape=(reference[0].values.numel(),),
        dtype=reference[0].dtype,
        values=reference[0].values.reshape(-1),
    )

    try:
        load_modified_weights(model, [bad_name])
    except KeyError as exc:
        assert "missing.weight" in str(exc)
    else:
        raise AssertionError("load_modified_weights should reject missing names")

    try:
        load_modified_weights(model, [bad_shape])
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("load_modified_weights should reject shape mismatches")
