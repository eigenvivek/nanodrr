import torch
import torch.nn.functional as F
from jaxtyping import Float


def resample(
    img: Float[torch.Tensor, "B C H W"],
    k_inv_old: Float[torch.Tensor, "B 3 3"],
    k_inv_new: Float[torch.Tensor, "B 3 3"],
) -> Float[torch.Tensor, "B C H W"]:
    r"""Resample an image from one camera's (inverse) intrinsic to another's.

    Each target pixel \(p'\) is mapped back to a source pixel via the homography

    \[ p = H p' = K_{\text{old}} K_{\text{new}}^{-1} p' \]

    and bilinearly interpolated. Out-of-bounds regions are filled with zeros.

    Args:
        img: Batch of images.
        k_inv_old: Inverse intrinsic matrices of the source images.
        k_inv_new: Inverse intrinsic matrices of the target images.

    Returns:
        Resampled images.
    """
    B, _, H, W = img.shape

    # Destination -> source homography per batch element
    H_mat = torch.linalg.inv(k_inv_old) @ k_inv_new

    # Build (H, W) grid of homogeneous destination pixel coords
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=img.dtype, device=img.device),
        torch.arange(W, dtype=img.dtype, device=img.device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1)

    # Apply per-batch homography
    coords_flat = coords.reshape(-1, 3).T.unsqueeze(0).expand(B, -1, -1)
    src = (H_mat @ coords_flat).permute(0, 2, 1).reshape(B, H, W, 3)

    # Perspective divide
    src_x = src[..., 0] / src[..., 2]
    src_y = src[..., 1] / src[..., 2]

    # Normalize to [-1, 1] for grid_sample
    grid_x = (src_x / (W - 1)) * 2 - 1
    grid_y = (src_y / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(img, grid, align_corners=True, padding_mode="zeros", mode="bilinear")
