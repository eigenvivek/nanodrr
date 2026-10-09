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


def fused_supported(subject: Subject, B: int, n_pixels: int) -> bool:
    """Hard limits of the fused kernel: one volume, int32 indexing."""
    N = B * n_pixels
    return subject.image.shape[0] == 1 and max(subject.image.numel(), subject.n_classes * N, 3 * N) < 2**31


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

    C = subject.n_classes
    vol = subject.image[0, 0].contiguous()
    lab = subject.label[0, 0].contiguous() if C > 1 else vol

    # The kernel samples at pixel coordinates, which is exactly world_to_voxel.
    # Geometry is computed in float32: the volume may be stored in half
    # precision, but half-precision sample coordinates cost ~20x accuracy
    M = (subject.world_to_voxel.float() @ rt_inv.float())[:, :3, :]

    # Broadcast the pose batch against shared ray geometry; the kernel
    # requires dense [B, ...] inputs
    B = max(M.shape[0], tgt.shape[0])
    if M.shape[0] != B:
        M = M.expand(B, -1, -1)
    if not fused_supported(subject, B, height * width):
        raise ValueError("inputs exceed the fused kernel's limits; use backend='torch'")

    out = fused_raymarch(
        vol,
        lab,
        M,
        src.expand(B, -1, -1).float().contiguous(),
        tgt.expand(B, -1, -1).float().contiguous(),
        step_size.expand(B, -1).float().contiguous(),
        n_samples,
        C,
        width,
    )
    return out.reshape(B, C, height, width)
