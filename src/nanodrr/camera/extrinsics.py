import torch
from jaxtyping import Float

from ..geometry import convert


def make_rt_inv(
    rotation: Float[torch.Tensor, "B 3"],
    translation: Float[torch.Tensor, "B 3"],
    orientation: str | None = "AP",
    isocenter: Float[torch.Tensor, "3"] | None = None,
) -> Float[torch.Tensor, "B 4 4"]:
    """Create 4x4 camera-to-world (extrinsic inverse) transformation matrix.

    Composes the pose and reorientation to match DiffDRR's behavior:
        extrinsic_inv = pose @ reorient

    This order means the translation is applied in the pre-reoriented frame,
    so translation=(0, 850, 0) with AP orientation places the source at Y=850
    in world coordinates (behind the patient for AP imaging).

    When isocenter is provided, the translation is interpreted as relative to
    the isocenter rather than world origin.

    Args:
        rotation: (batch, 3) Euler angles (angle_z, angle_x, angle_y) in degrees, ZXY convention
        translation: (batch, 3) camera position (mm). If isocenter is provided,
                    this is relative to isocenter; otherwise relative to world origin.
        orientation: "AP", "PA", or None for frame-of-reference
        isocenter: Optional (3,) volume isocenter in world coordinates.
                  When provided, the translation is relative to this point.

    Returns:
        (batch, 4, 4) camera-to-world transformation matrices
    """
    if orientation not in (None, "AP", "PA"):
        raise ValueError(f"Unknown orientation: {orientation}. Use 'AP', 'PA', or None")

    device = rotation.device
    dtype = rotation.dtype

    # Default isocenter to origin
    if isocenter is None:
        isocenter = torch.zeros(3, device=device, dtype=dtype)

    # Construct the C-arm pose relative to the subject's isocenter
    pose = convert(rotation, translation, "euler", convention="ZXY", isocenter=isocenter)

    # Apply orientation (pose @ combined)
    # bij,jk->bik : batched matrix times single matrix
    orientation_matrix = get_orientation_matrix(orientation, device, dtype)
    out = torch.einsum("bij,jk->bik", pose, orientation_matrix)

    return out


def euler_to_matrix(rotation: Float[torch.Tensor, "B 3"]) -> Float[torch.Tensor, "B 3 3"]:
    """Convert ZXY Euler angles (degrees) to rotation matrices.

    Args:
        rotation: Euler angles (angle_z, angle_x, angle_y) in degrees, shape (batch, 3)

    Returns:
        Rotation matrices of shape (batch, 3, 3)
    """
    angles = torch.deg2rad(rotation)
    z, x, y = angles[:, 0], angles[:, 1], angles[:, 2]

    cz, sz = torch.cos(z), torch.sin(z)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)

    # ZXY Euler rotation matrix
    R = torch.stack(
        [
            torch.stack([cy * cz - sx * sy * sz, -cx * sz, cz * sy + cy * sx * sz], dim=1),
            torch.stack([cy * sz + cz * sx * sy, cx * cz, sy * sz - cy * cz * sx], dim=1),
            torch.stack([-cx * sy, sx, cx * cy], dim=1),
        ],
        dim=1,
    )

    return R


def get_orientation_matrix(
    orientation: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "4 4"]:
    """Get the combined orientation + Rz(180°) matrix.

    Args:
        orientation: "AP", "PA", or None
        device: torch device
        dtype: torch dtype

    Returns:
        4x4 transformation matrix
    """
    if orientation == "AP":
        combined = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
    elif orientation == "PA":
        combined = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
    else:  # None - just Rz180
        combined = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

    return combined
