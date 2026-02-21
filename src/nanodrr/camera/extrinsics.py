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
    """Create 4x4 camera-to-world (extrinsic inverse) matrices.

    Composes pose and reorientation as ``extrinsic_inv = pose @ reorient``
    so that *translation* is applied in the pre-reoriented frame.

    Args:
        rotation: (B, 3) Euler angles (z, x, y) in degrees, ZXY convention.
        translation: (B, 3) camera position in mm, relative to *isocenter*
                     (or world origin when isocenter is ``None``).
        orientation: ``"AP"``, ``"PA"``, or ``None``.
        isocenter: Optional (3,) volume centre in world coordinates.

    Returns:
        (B, 4, 4) camera-to-world transformation matrices.
    """
    if orientation not in _ORIENTATION_MATRICES:
        raise ValueError(f"Unknown orientation: {orientation}. Use 'AP', 'PA', or None")

    device = rotation.device
    dtype = rotation.dtype

    if isocenter is None:
        isocenter = torch.zeros(3, device=device, dtype=dtype)

    pose = convert(rotation, translation, "euler", convention="ZXY", isocenter=isocenter)
    orientation_matrix = _get_orientation_matrix(orientation, device, dtype)

    return pose @ orientation_matrix


def _get_orientation_matrix(
    orientation: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "4 4"]:
    """Return the combined orientation + Rz(180°) matrix."""
    return torch.tensor(_ORIENTATION_MATRICES[orientation], device=device, dtype=dtype)
