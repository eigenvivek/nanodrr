import torch
from jaxtyping import Float
from roma import rotmat_to_euler

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


def invert_rt_inv(
    extrinsic_inv: Float[torch.Tensor, "B 4 4"],
    orientation: str | None = "AP",
    isocenter: Float[torch.Tensor, "3"] | None = None,
) -> tuple[Float[torch.Tensor, "B 3"], Float[torch.Tensor, "B 3"]]:
    """Recover rotation and translation from camera-to-world matrices.

    Inverts the composition performed by [`make_rt_inv`](#nanodrr.camera.extrinsics.make_rt_inv). The
    `orientation` and `isocenter` arguments must match those used during
    construction to obtain correct results.

    Args:
        extrinsic_inv: (B, 4, 4) camera-to-world matrices.
        orientation:   Acquisition orientation — one of `"AP"`, `"PA"`, or `None`.
        isocenter:     Optional (3,) volume center in world coordinates.

    Returns:
        rotation:    (B, 3) ZXY Euler angles in degrees.
        translation: (B, 3) Camera position in mm, relative to `isocenter`.
    """
    device, dtype = extrinsic_inv.device, extrinsic_inv.dtype
    iso = _default_isocenter(isocenter, extrinsic_inv).view(1, 3)

    R_orient = _get_orientation_matrix(orientation, device, dtype)[:3, :3]

    R_pose = extrinsic_inv[..., :3, :3] @ R_orient.T
    t_pose = extrinsic_inv[..., :3, 3]

    translation = torch.einsum("bij,bj->bi", R_pose.transpose(-1, -2), t_pose - iso)
    rotation = rotmat_to_euler(R_pose, order="ZXY", deg=True)

    return rotation, translation


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
