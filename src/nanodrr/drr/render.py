import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..data import Subject
from ..geometry import transform_point


def render(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    n_samples: int = 500,
    align_corners: bool = True,
) -> Float[torch.Tensor, "B C H W"]:
    device, dtype = rt_inv.device, rt_inv.dtype
    B = rt_inv.shape[0]
    C = subject.n_classes
    N = height * width

    # Get the ray endpoints in camera coordinates
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(N, 3)
    tgt = sdd[:, None, None] * torch.einsum("bij,nj->bni", k_inv, uv1)

    # Compute step size [mm] in camera space (source is at origin)
    step_size = tgt.norm(dim=-1) / float(n_samples - 1)

    # Change coordinates: camera → world → voxel
    xform = subject.world_to_voxel @ rt_inv
    src = transform_point(xform, torch.zeros(B, 1, 3, device=device, dtype=dtype))
    tgt = transform_point(xform, tgt)

    # Linearly interpolate sample points along each ray
    t = torch.linspace(0, 1, n_samples, device=device, dtype=dtype)
    pts = src[..., None, :] + t[None, None, :, None] * (tgt - src)[..., None, :]
    pts = (2.0 * pts / subject.dims - 1.0).unsqueeze(-2)

    # Sample the volume
    img = F.grid_sample(
        subject.image.expand(B, -1, -1, -1, -1),
        pts,
        mode="bilinear",
        align_corners=align_corners,
    ).squeeze(1).squeeze(-1)  # [B, n_samples, N]
    img = img * step_size[:, None, :]

    if C == 1:  # Compute whole-volume ray marching
        return img.sum(dim=1, keepdim=True).reshape(B, C, height, width)

    # Sample the mask
    idx = F.grid_sample(
        subject.label.expand(B, -1, -1, -1, -1),
        pts,
        mode="nearest",
        align_corners=align_corners,
    ).squeeze(1).squeeze(-1).long()  # [B, n_samples, N]

    # Compute the structure-specific ray marching
    out = torch.zeros(B, C, N, device=img.device, dtype=img.dtype)
    out.scatter_add_(1, idx, img)
    return out.reshape(B, C, height, width)
