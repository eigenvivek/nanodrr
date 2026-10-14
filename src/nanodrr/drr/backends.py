import itertools
from collections.abc import Callable

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

    # Sample the volume. A reduced-precision volume is promoted to the geometry's dtype (`grid_sample`
    # needs one dtype), as in `_render_soft` and the per-sample promotion of the fused kernels
    dtype = torch.promote_types(subject.image.dtype, pts.dtype)
    img = F.grid_sample(
        subject.image.to(dtype).expand(B, -1, -1, -1, -1),
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
        subject.label.to(dtype).expand(B, -1, -1, -1, -1),
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


# Fused backend for each device type; `render(backend="auto")` picks from this table
DEVICE_BACKENDS = {"cuda": "triton", "mps": "metal"}
FUSED_BACKENDS = tuple(DEVICE_BACKENDS.values())
FUSED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def fused_supported(subject: Subject, B: int, n_pixels: int) -> bool:
    """Hard limits of the fused kernels: one volume, int32 indexing."""
    N = B * n_pixels
    return subject.image.shape[0] == 1 and max(subject.image.numel(), subject.n_classes * N, 3 * N) < 2**31


def load_backend(name: str):
    """The differentiable op of fused backend `name`, imported on first use. Explicit imports (not
    `importlib`) so that `torch.compile(fullgraph=True)` can trace through the lookup."""
    if name == "triton":
        from ._triton import fused_raymarch

        return fused_raymarch
    if name == "metal":
        from ._metal import metal_raymarch

        return metal_raymarch


def available(name: str) -> bool:
    """Whether fused backend `name` imports on this machine (e.g. triton is installed).

    Uncached on purpose: `render` calls it under `torch.compile`, where dynamo warns about (and cannot
    honor) `functools.cache` wrappers."""
    try:
        load_backend(name)
    except Exception:  # noqa: BLE001  # not installed, or a broken install
        return False
    return True


def pick_fused_backend(
    subject: Subject,
    rt_inv: Float[torch.Tensor, "B 4 4"],
    tgt: Float[torch.Tensor, "B (H W) 3"],
    height: int,
    width: int,
) -> str | None:
    """The fused backend `backend="auto"` should use for these inputs, or `None` for the torch path.

    The fused kernels need a GPU with a backend, fp32/fp16/bf16, int32-indexable sizes, and (for volume
    gradients, which use atomics) non-deterministic mode.
    """
    name = DEVICE_BACKENDS.get(rt_inv.device.type)
    deterministic_volume_grad = torch.are_deterministic_algorithms_enabled() and subject.image.requires_grad
    eligible = (
        name is not None
        and rt_inv.dtype in FUSED_DTYPES
        and subject.image.dtype in FUSED_DTYPES
        and not deterministic_volume_grad
        and fused_supported(subject, max(rt_inv.shape[0], tgt.shape[0]), height * width)
        and available(name)
    )
    return name if eligible else None


def register_fused_op(name: str, op_name: str, forward: Callable, backward: Callable):
    """Register `forward` and `backward` as the differentiable op `nanodrr::{op_name}` and return it.

    Both take dense tensors on the backend's device: `vol` and `lab` are `[D, H, W]`, `M` is `[B, 3, 4]`
    (camera → voxel pixel coordinates), `src` is `[B, N, 3]` or `[B, 1, 3]`, `tgt` is `[B, N, 3]` and
    `step` is `[B, N]` (the step size [mm] of the unclipped ray).

    - `forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels)` returns the
      float32 render `[B, C, N]`.
    - `backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam,
      soft_labels)` returns `(gvol, gM, gsrc, gtgt, gstep)`, already reduced: float32 `[D, H, W]`,
      `[B, 3, 4]`, `[B, N, 3]`, `[B, N, 3]` and `[B, N]`. Gradients that were not requested
      (`gvol` unless `need_gvol`; the last three unless `need_gcam`) are empty tensors.

    `name` is the backend's name in messages. Registered as custom ops so `torch.compile` treats the
    kernels as opaque: inductor takes shapes from the fake implementations rather than tracing the
    kernel source.
    """
    op = f"nanodrr::{op_name}"

    @torch.library.custom_op(op, mutates_args=())
    def raymarch(
        vol: torch.Tensor,
        lab: torch.Tensor,
        M: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
        step: torch.Tensor,
        n_samples: int,
        n_classes: int,
        width: int,
        soft_labels: bool = False,
    ) -> torch.Tensor:
        out = forward(vol, lab, M.contiguous(), src, tgt, step, n_samples, n_classes, width, soft_labels)
        return out.to(vol.dtype)

    @raymarch.register_fake
    def _(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels=False):
        B, N, _ = tgt.shape
        return torch.empty(B, n_classes, N, device=vol.device, dtype=vol.dtype)

    @torch.library.custom_op(f"{op}_bwd", mutates_args=())
    def raymarch_bwd(
        go: torch.Tensor,
        vol: torch.Tensor,
        lab: torch.Tensor,
        M: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
        step: torch.Tensor,
        n_samples: int,
        n_classes: int,
        width: int,
        need_gvol: bool,
        need_gcam: bool,
        soft_labels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if need_gvol and torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                f"Volume gradients on the {name} backend use nondeterministic atomics; "
                "disable deterministic algorithms or use backend='torch'"
            )
        return backward(
            go, vol, lab, M.contiguous(), src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels
        )

    @raymarch_bwd.register_fake
    def _(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam, soft_labels=False):
        B, N, _ = tgt.shape
        f32 = torch.float32
        gM = torch.empty(B, 3, 4, device=M.device, dtype=f32)
        gvol = torch.empty(vol.shape, device=M.device, dtype=f32) if need_gvol else gM.new_empty(0)
        gsrc = torch.empty(B, N, 3, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
        gtgt = torch.empty(B, N, 3, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
        gstep = torch.empty(B, N, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
        return gvol, gM, gsrc, gtgt, gstep

    def setup_context(ctx, inputs, output):
        vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels = inputs
        ctx.save_for_backward(vol, lab, M.contiguous(), src, tgt, step)
        ctx.n_samples = n_samples
        ctx.n_classes = n_classes
        ctx.width = width
        ctx.soft_labels = soft_labels

    def autograd_backward(ctx, go):
        vol, lab, M, src, tgt, step = ctx.saved_tensors
        need_gvol = ctx.needs_input_grad[0]
        need_gcam = any(ctx.needs_input_grad[i] for i in (3, 4, 5))
        gvol, gM, gsrc, gtgt, gstep = raymarch_bwd(
            go,
            vol,
            lab,
            M,
            src,
            tgt,
            step,
            ctx.n_samples,
            ctx.n_classes,
            ctx.width,
            need_gvol,
            need_gcam,
            ctx.soft_labels,
        )
        if need_gcam and src.shape[1] == 1:
            gsrc = gsrc.sum(dim=1, keepdim=True)
        return (
            gvol.to(vol.dtype) if need_gvol else None,
            None,  # lab: integer routing, not differentiable
            gM.to(M.dtype),
            gsrc.to(src.dtype) if need_gcam else None,
            gtgt.to(tgt.dtype) if need_gcam else None,
            gstep.to(step.dtype) if need_gcam else None,
            None,  # n_samples
            None,  # n_classes
            None,  # width
            None,  # soft_labels
        )

    torch.library.register_autograd(op, autograd_backward, setup_context=setup_context)
    return raymarch


def render_fused(
    backend: str,
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
    """Fused implementation of `render`, running `backend` (`"triton"` or `"metal"`).

    One kernel marches each ray in registers, so the sample grid and per-sample intensities
    of the `grid_sample` path are never materialized. `soft_labels` must already be `False` for a
    single-class subject (`render` normalizes it).
    """
    if DEVICE_BACKENDS.get(rt_inv.device.type) != backend:  # before the import, which may fail here
        device_type = next(device for device, name in DEVICE_BACKENDS.items() if name == backend)
        raise ValueError(f"backend={backend!r} requires {device_type.upper()} tensors; use backend='torch'")

    C = subject.n_classes
    B = max(rt_inv.shape[0], tgt.shape[0])
    if not fused_supported(subject, B, height * width):
        raise ValueError("inputs exceed the fused kernel's limits; use backend='torch'")

    # reshape (a view) rather than [0, 0]: its backward is a view too, whereas select's backward
    # materializes a full-volume zero tensor. Half-precision volumes are read in place and
    # promoted per sample by the kernel
    D, H, W = subject.image.shape[2:]
    vol = subject.image.reshape(D, H, W)
    if vol.dtype not in FUSED_DTYPES:
        vol = vol.float()
    vol = vol.contiguous()
    lab = subject.label.reshape(D, H, W).contiguous() if C > 1 else vol  # unused with a single class

    # The kernels sample at pixel coordinates, which is exactly world_to_voxel. Geometry is
    # computed in float32: the volume may be stored in half precision, but half-precision sample
    # coordinates cost ~20x accuracy
    M = (subject.world_to_voxel.float() @ rt_inv.float())[:, :3, :]

    # Broadcast the pose batch against shared ray geometry; the kernels require dense [B, ...] inputs
    out = load_backend(backend)(
        vol,
        lab,
        M.expand(B, -1, -1),
        src.expand(B, -1, -1).float().contiguous(),
        tgt.expand(B, -1, -1).float().contiguous(),
        step_size.expand(B, -1).float().contiguous(),
        n_samples,
        C,
        width,
        soft_labels,
    )
    return out.reshape(B, C, height, width)
