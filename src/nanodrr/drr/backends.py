import itertools

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
    soft_labels: bool = False,
) -> Float[torch.Tensor, "B C H W"]:
    """Reference `grid_sample` implementation of `render`."""
    device = rt_inv.device
    B = rt_inv.shape[0]
    C = subject.n_classes
    N = height * width

    if soft_labels and C > 1:
        return _render_soft(subject, rt_inv, src, tgt, step_size, n_samples, height, width)

    # Change coordinates: camera → world → voxel → normalized grid
    xform = subject.world_to_grid @ rt_inv
    src = transform_point(xform, src)
    tgt = transform_point(xform, tgt)
    D, H, W = subject.image.shape[-3:]
    hi = 1 + 1 / torch.tensor([W, H, D], device=device, dtype=torch.float32)
    src, tgt, step_size = _clip_rays(src, tgt, step_size, -hi, hi, n_samples)

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


def _clip_rays(
    src: Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"],
    tgt: Float[torch.Tensor, "B (H W) 3"],
    step_size: Float[torch.Tensor, "B (H W)"],
    lo: Float[torch.Tensor, "3"],
    hi: Float[torch.Tensor, "3"],
    n_samples: int,
) -> tuple[Float[torch.Tensor, "B (H W) 3"], Float[torch.Tensor, "B (H W) 3"], Float[torch.Tensor, "B (H W)"]]:
    r"""Clip each ray to the box `[lo, hi]` that supports the interpolated volume.

    The ray $x(t) = s + t (g - s)$ crosses the box for $t \in [t_0, t_1]$ (slab method):

    $$t_0 = \max\Big(0, \max_a \min(t_a^-, t_a^+)\Big), \qquad
    t_1 = \min\Big(1, \min_a \max(t_a^-, t_a^+)\Big)$$

    where $t_a^\pm$ are the parameters at which it crosses the two planes bounding
    axis $a$. The returned endpoints are inset by half a step, so `n_samples`
    evenly spaced points between them are the midpoints of `n_samples` equal
    parts of $[t_0, t_1]$, and the step size is scaled by $t_1 - t_0$. Rays that
    miss the box collapse to a point with zero step size.
    """
    dtype = src.dtype
    ftype = torch.promote_types(dtype, torch.float32)  # half-precision bounds are too coarse
    src, tgt = src.to(ftype), tgt.to(ftype)
    ray = tgt - src
    inv = 1.0 / torch.where(ray.abs() < 1e-12, 1e-12, ray)
    ta, tb = (lo - src) * inv, (hi - src) * inv
    t0 = torch.minimum(ta, tb).amax(dim=-1, keepdim=True).clamp(0.0, 1.0)
    t1 = torch.maximum(ta, tb).amin(dim=-1, keepdim=True).clamp(0.0, 1.0).maximum(t0)
    half = 0.5 * (t1 - t0) / n_samples
    src, tgt = src + (t0 + half) * ray, src + (t1 - half) * ray
    return src.to(dtype), tgt.to(dtype), step_size * (t1 - t0)[..., 0].to(dtype)


def _render_soft(
    subject: Subject,
    rt_inv: Float[torch.Tensor, "B 4 4"],
    src: Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"],
    tgt: Float[torch.Tensor, "B (H W) 3"],
    step_size: Float[torch.Tensor, "B (H W)"],
    n_samples: int,
    height: int,
    width: int,
) -> Float[torch.Tensor, "B C H W"]:
    r"""Per-structure ray marching with trilinear (partial-volume) labels.

    Trilinear interpolation is written out over the 8 voxels surrounding each
    sample so that every voxel's contribution is routed to that voxel's own class:

    $$I_c = \Delta s \sum_k \sum_{j=1}^{8} w_j(x_k) \, \mu_j \, [\ell_j = c]$$

    where $w_j(x_k)$ are the trilinear weights of sample $x_k$ and $\mu_j$, $\ell_j$
    are the intensity and label stored at voxel $j$. This equals rendering the
    volume masked to class $c$, for all classes in one pass and without a one-hot
    labelmap, so memory does not grow with the number of classes.
    """
    C = subject.n_classes
    N = height * width

    # Change coordinates: camera → world → voxel (integer coordinates are voxel centers)
    xform = subject.world_to_voxel @ rt_inv
    src = transform_point(xform, src)
    tgt = transform_point(xform, tgt)
    D, H, W = subject.image.shape[-3:]
    hi = torch.tensor([W, H, D], device=src.device, dtype=torch.float32)
    src, tgt, step_size = _clip_rays(src, tgt, step_size, -torch.ones_like(hi), hi, n_samples)

    # Linearly interpolate sample points along each ray
    t = torch.linspace(0, 1, n_samples, device=rt_inv.device, dtype=src.dtype)
    pts = torch.lerp(src[:, None], tgt[:, None], t[None, :, None, None])  # [B, n_samples, N, 3]
    B = pts.shape[0]

    # Per-axis strided indices and weights of the two neighbors. Out-of-bounds
    # neighbors get zero weight, matching `grid_sample`'s zero padding
    vol = subject.image.reshape(-1)
    lab = subject.label.reshape(-1)
    itype = torch.int32 if vol.numel() < 2**31 else torch.int64
    idx, wgt = [], []
    for axis, (size, stride) in enumerate(((W, 1), (H, W), (D, H * W))):
        x = pts[..., axis]
        x0 = x.floor()
        f = x - x0
        i0 = x0.to(itype)
        i1 = i0 + 1
        idx.append((i0.clamp(0, size - 1) * stride, i1.clamp(0, size - 1) * stride))
        wgt.append(((1 - f) * ((i0 >= 0) & (i0 < size)), f * ((i1 >= 0) & (i1 < size))))
    (ix, iy, iz), (wx, wy, wz) = idx, wgt

    # Route each neighbor's weighted intensity to the channel of its own label
    out = torch.zeros(B, C, N, device=vol.device, dtype=torch.promote_types(vol.dtype, pts.dtype))
    for dz, dy in itertools.product((0, 1), repeat=2):
        izy = iz[dz] + iy[dy]
        wzy = wz[dz] * wy[dy]
        for dx in (0, 1):
            j = izy + ix[dx]
            out.scatter_add_(1, lab[j].long(), wzy * wx[dx] * vol[j])

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
    soft_labels: bool = False,
) -> Float[torch.Tensor, "B C H W"]:
    """Fused Triton implementation of `render`.

    One kernel marches each ray in registers, so the sample grid and
    per-sample intensities of the `grid_sample` path are never materialized.
    """
    from ._triton import fused_raymarch

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
        soft_labels and C > 1,
    )
    return out.reshape(B, C, height, width)
