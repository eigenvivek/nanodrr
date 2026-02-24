import torch
from jaxtyping import Float

from ..geometry import convert

_ORIENTATION_MATRICES = {
    "AP": [
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    "PA": [
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    None: [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
}


def make_rt_inv(
    rotation: Float[torch.Tensor, "B 3"],
    translation: Float[torch.Tensor, "B 3"],
    orientation: str | None = "AP",
    isocenter: Float[torch.Tensor, "3"] | None = None,
) -> Float[torch.Tensor, "B 4 4"]:
    """Construct camera-to-world (extrinsic inverse) matrices.

    Computes `extrinsic_inv = pose @ reorient`, where `pose` encodes a
    ZXY Euler rotation and translation, and `reorient` is a fixed
    orientation-dependent transform.

    Args:
        rotation:    (B, 3) ZXY Euler angles in degrees.
        translation: (B, 3) Camera position in mm, relative to `isocenter`.
        orientation: Acquisition orientation — one of `"AP"`, `"PA"`, or `None`.
        isocenter:   Optional (3,) volume center in world coordinates.

    Returns:
        (B, 4, 4) camera-to-world matrices.
    """
    pose = convert(
        rotation,
        translation,
        "euler",
        convention="ZXY",
        isocenter=_default_isocenter(isocenter, rotation),
    )
    return pose @ _get_orientation_matrix(orientation, rotation.device, rotation.dtype)


def _default_isocenter(
    isocenter: Float[torch.Tensor, "3"] | None,
    ref: torch.Tensor,
) -> Float[torch.Tensor, "3"]:
    if isocenter is not None:
        return isocenter
    return torch.zeros(3, device=ref.device, dtype=ref.dtype)


def _get_orientation_matrix(
    orientation: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "4 4"]:
    if orientation not in _ORIENTATION_MATRICES:
        raise ValueError(
            f"Unknown orientation {orientation!r}. Expected one of: 'AP', 'PA', or None."
        )
    return torch.tensor(_ORIENTATION_MATRICES[orientation], device=device, dtype=dtype)
