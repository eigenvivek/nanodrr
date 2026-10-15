from collections.abc import Callable

import torch
from jaxtyping import Float

from ..data import Subject

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
        from ._triton import triton_raymarch

        return triton_raymarch
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
