import torch
from jaxtyping import Float


def make_k_inv(
    sdd: float,
    delx: float,
    dely: float,
    x0: float,
    y0: float,
    height: int,
    width: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Float[torch.Tensor, "1 3 3"]:
    """Build the inverse intrinsic matrix K⁻¹ for a cone-beam projector.

    Focal lengths and principal point are derived from the physical geometry:

        fx = sdd / delx          cy = y0 / dely + height / 2
        fy = sdd / dely          cx = x0 / delx + width  / 2

    The returned matrix is the analytical inverse of:

        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]

    Args:
        sdd: Source-to-detector distance (mm).
        delx: Pixel spacing in x (mm/px).
        dely: Pixel spacing in y (mm/px).
        x0: Principal-point offset from detector centre in x (mm).
        y0: Principal-point offset from detector centre in y (mm).
        height: Detector height in pixels.
        width: Detector width in pixels.
        dtype: Optional tensor dtype.
        device: Optional tensor device.
    Returns:
        (1, 3, 3) inverse intrinsic matrix.
    """
    fx = sdd / delx
    fy = sdd / dely
    cx = x0 / delx + width / 2.0
    cy = y0 / dely + height / 2.0

    return torch.tensor(
        [
            [
                [1.0 / fx, 0.0, -cx / fx],
                [0.0, 1.0 / fy, -cy / fy],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=dtype,
        device=device,
    )
