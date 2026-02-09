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
    fx = sdd / delx
    fy = sdd / dely
    cx = x0 / delx + width / 2.0
    cy = y0 / dely + height / 2.0

    fx_inv = 1.0 / fx
    fy_inv = 1.0 / fy

    return torch.tensor(
        [
            [
                [fx_inv, 0.0, -cx * fx_inv],
                [0.0, fy_inv, -cy * fy_inv],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=dtype,
        device=device,
    )
