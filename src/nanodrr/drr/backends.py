import functools

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..data import Subject
from ..geometry import transform_point


def render_torch(
    subject: Subject,
    rt_inv: Float[torch.Tensor, "B 4 4"],
    src: Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"],
    tgt: Float[torch.Tensor, "B (H W) 3"],
    step_size: Float[torch.Tensor, "B (H W)"],
    n_samples: int,
    height: int,
    width: int,
) -> Float[torch.Tensor, "B C H W"]:
    """Reference `grid_sample` implementation of `render`."""
    device = rt_inv.device
    B = rt_inv.shape[0]
    C = subject.n_classes
    N = height * width

    # Change coordinates: camera → world → voxel → normalized grid
    xform = subject.world_to_grid @ rt_inv
    src = transform_point(xform, src)
    tgt = transform_point(xform, tgt)

    # Linearly interpolate sample points along each ray
    t = torch.linspace(0, 1, n_samples, device=device, dtype=src.dtype)
    pts = torch.lerp(
        src[:, None, :, None],
        tgt[:, None, :, None],
        t[None, :, None, None, None],
    )

    # Sample the volume
    img = F.grid_sample(
        subject.image.expand(B, -1, -1, -1, -1),
        pts,
        mode="bilinear",
        align_corners=False,
    )[:, 0, ..., 0]  # [B, n_samples, N]

    # step_size is constant along each ray, so scale after the reduction
    if C == 1:  # Compute whole-volume ray marching
        img = img.sum(dim=1, keepdim=True) * step_size[:, None, :]
        return img.reshape(B, C, height, width)

    # Sample the mask
    idx = F.grid_sample(
        subject.label.expand(B, -1, -1, -1, -1),
        pts,
        mode="nearest",
        align_corners=False,
    )[:, 0, ..., 0].long()  # [B, n_samples, N]

    # Compute the structure-specific ray marching
    out = torch.zeros(B, C, N, device=img.device, dtype=img.dtype)
    out.scatter_add_(1, idx, img)
    out = out * step_size[:, None, :]
    return out.reshape(B, C, height, width)


@functools.cache
def _pix_constants(shape, device, dtype):
    """Normalized-grid → pixel mapping (align_corners=False): px = u * size/2 + (size - 1)/2."""
    D, H, W = shape
    scale = torch.tensor([W / 2, H / 2, D / 2], device=device, dtype=dtype).reshape(3, 1)
    shift = torch.tensor([(W - 1) / 2, (H - 1) / 2, (D - 1) / 2], device=device, dtype=dtype)
    return scale, shift


def render_fused(
    subject: Subject,
    rt_inv: Float[torch.Tensor, "B 4 4"],
    src: Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"],
    tgt: Float[torch.Tensor, "B (H W) 3"],
    step_size: Float[torch.Tensor, "B (H W)"],
    n_samples: int,
    height: int,
    width: int,
) -> Float[torch.Tensor, "B C H W"]:
    """Fused Triton implementation of `render`.

    One kernel marches each ray in registers, so the sample grid and
    per-sample intensities of the `grid_sample` path are never materialized.
    """
    from ._fused import fused_raymarch

    if subject.image.shape[0] != 1:
        raise ValueError("backend='triton' requires an unbatched volume; use backend='torch'")

    C = subject.n_classes
    vol = subject.image[0, 0].contiguous()
    lab = subject.label[0, 0].contiguous() if C > 1 else vol

    # Fold the normalized-grid → pixel mapping into the camera → grid transform
    pix_scale, pix_shift = _pix_constants(vol.shape, rt_inv.device, rt_inv.dtype)
    M = (subject.world_to_grid @ rt_inv)[:, :3, :] * pix_scale
    M = torch.cat([M[..., :3], (M[..., 3] + pix_shift)[..., None]], dim=-1)

    # Broadcast the pose batch against shared ray geometry; the kernel
    # requires dense [B, ...] inputs
    B = max(M.shape[0], tgt.shape[0])
    if M.shape[0] != B:
        M = M.expand(B, -1, -1)
    if max(vol.numel(), B * C * height * width) >= 2**31:
        raise ValueError("tensor exceeds the kernel's int32 indexing; use backend='torch'")

    out = fused_raymarch(
        vol,
        lab,
        M,
        src.expand(B, -1, -1).contiguous(),
        tgt.expand(B, -1, -1).contiguous(),
        step_size.expand(B, -1).contiguous(),
        n_samples,
        C,
    )
    return out.reshape(B, C, height, width)
