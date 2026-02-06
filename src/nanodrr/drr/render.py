import torch
import torch.nn.functional as F

from ..data import Subject
from ..geometry import transform_point


def render(
    subject: Subject,
    k_inv: torch.Tensor,
    rt_inv: torch.Tensor,
    sdd: torch.Tensor,
    height: int,
    width: int,
    n_samples: int = 500,
) -> torch.Tensor:
    device, dtype = sdd.device, sdd.dtype
    B = len(sdd)

    # Get the start and end points of each ray in camera coordinates
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    uv = uv.reshape(-1, 3)

    tgt = sdd * torch.einsum("...ij,nj->...ni", k_inv, uv)
    src = torch.zeros_like(tgt)

    # Get the length of a step in millimeters
    step_size = (tgt - src).norm(dim=-1) / float(n_samples - 1)

    # Move the ray from camera->world->voxel coordinates
    xform = subject.world_to_voxel @ rt_inv
    src = transform_point(xform, src)
    tgt = transform_point(xform, tgt)

    # Sample the volume
    t = torch.linspace(0, 1, n_samples, device=device, dtype=dtype)
    pts = src[..., None, :] + t[None, None, :, None] * (tgt - src)[..., None, :]
    pts = pts.reshape(B, 1, height * width * n_samples, 1, 3)
    pts = 2.0 * pts / subject.dims - 1.0

    sampled = F.grid_sample(
        subject.image.expand(B, -1, -1, -1, -1),
        pts,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # Integrate the samples to compute the final intensity
    sampled = sampled.reshape(B, height * width, n_samples)
    return step_size * sampled.sum(dim=-1)
