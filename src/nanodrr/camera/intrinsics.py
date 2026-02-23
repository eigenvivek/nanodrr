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
    r"""Build the inverse intrinsic matrix $\mathbf K^{-1}$ for a cone-beam projector.

    $$
    \begin{align}
        f_x &= \frac{\mathrm{SDD}}{\Delta_x} &\quad c_x &= \frac{x_0}{\Delta_x} + \frac{W}{2} \\
        f_y &= \frac{\mathrm{SDD}}{\Delta_y} &\quad c_y &= \frac{y_0}{\Delta_y} + \frac{H}{2}
    \end{align}
    $$
    
    where `delx` $= \Delta_x$ and `dely` $= \Delta_y$.

    The returned matrix is the analytical inverse of the intrinsic matrix:

    $$
    \begin{equation}
        \mathbf K = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
        \end{bmatrix} \,,
    \end{equation}
    $$

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
