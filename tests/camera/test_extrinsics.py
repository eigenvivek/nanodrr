"""Roundtrip tests for make_rt_inv / invert_rt_inv."""

import pytest
import torch

from nanodrr.camera.extrinsics import invert_rt_inv, make_rt_inv

ORIENTATIONS = ["AP", "PA", None]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_roundtrip(orientation):
    """invert_rt_inv should recover the inputs passed to make_rt_inv."""
    rotation = torch.tensor([[30.0, -15.0, 45.0]])
    translation = torch.tensor([[100.0, -200.0, 500.0]])

    extrinsic_inv = make_rt_inv(rotation, translation, orientation=orientation)
    recovered_rotation, recovered_translation = invert_rt_inv(extrinsic_inv, orientation=orientation)

    torch.testing.assert_close(recovered_rotation, rotation)
    torch.testing.assert_close(recovered_translation, translation)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_roundtrip_with_isocenter(orientation):
    """Roundtrip should hold when a non-zero isocenter is provided."""
    rotation = torch.tensor([[10.0, 20.0, -30.0]])
    translation = torch.tensor([[50.0, 75.0, 300.0]])
    isocenter = torch.tensor([10.0, -20.0, 5.0])

    extrinsic_inv = make_rt_inv(rotation, translation, orientation=orientation, isocenter=isocenter)
    recovered_rotation, recovered_translation = invert_rt_inv(extrinsic_inv, orientation=orientation, isocenter=isocenter)

    torch.testing.assert_close(recovered_rotation, rotation)
    torch.testing.assert_close(recovered_translation, translation)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_roundtrip_batch(orientation):
    """Roundtrip should hold across a batch of poses."""
    B = 8
    rotation = torch.rand(B, 3) * 360 - 180
    translation = torch.rand(B, 3) * 1000 - 500

    extrinsic_inv = make_rt_inv(rotation, translation, orientation=orientation)
    recovered_rotation, recovered_translation = invert_rt_inv(extrinsic_inv, orientation=orientation)

    # Compare recovered extrinsics rather than Euler angles directly, since
    # multiple angle tuples can represent the same rotation matrix.
    extrinsic_inv_recovered = make_rt_inv(recovered_rotation, recovered_translation, orientation=orientation)
    torch.testing.assert_close(extrinsic_inv_recovered, extrinsic_inv, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(recovered_translation, translation, atol=1e-3, rtol=1e-3)


def test_invalid_orientation():
    """An unrecognised orientation should raise a ValueError."""
    rotation = torch.zeros(1, 3)
    translation = torch.zeros(1, 3)

    with pytest.raises(ValueError, match="Unknown orientation"):
        make_rt_inv(rotation, translation, orientation="LAT")

    extrinsic_inv = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError, match="Unknown orientation"):
        invert_rt_inv(extrinsic_inv, orientation="LAT")
