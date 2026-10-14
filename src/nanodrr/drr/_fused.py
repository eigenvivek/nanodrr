"""Code shared by the fused ray-marching backends (currently Triton on CUDA).

A fused backend is a pair of kernels, `forward` and `backward`, packaged as a `FusedBackend`.
Everything around them is common and lives here:

- `register_fused_op` turns a backend into a differentiable `torch.library` op (forward and
  backward ops, fake implementations for `torch.compile`, autograd wiring);
- `pick_fused_backend` is the `backend="auto"` policy (device, dtype, size and determinism limits);
  `render_fused` in `backends.py` prepares the inputs and calls the backend's op.

This module must not import `triton`, so that it loads everywhere.
Backends live in `_triton.py` and are imported lazily by name.
"""

import dataclasses
from collections.abc import Callable

import torch
from jaxtyping import Float

from ..data import Subject

# Fused backend for each device type; `render(backend="auto")` picks from this table
DEVICE_BACKENDS = {"cuda": "triton"}
FUSED_BACKENDS = tuple(DEVICE_BACKENDS.values())
FUSED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclasses.dataclass(frozen=True)
class FusedBackend:
    """A fused ray-marching kernel pair and the facts the shared code needs about it.

    Both functions take dense tensors on the backend's device: `vol` and `lab` are `[D, H, W]`,
    `M` is `[B, 3, 4]` (camera → voxel pixel coordinates), `src` is `[B, N, 3]` or `[B, 1, 3]`,
    `tgt` is `[B, N, 3]` and `step` is `[B, N]` (the step size [mm] of the unclipped ray).

    - `forward(vol, lab, M, src, tgt, step, n_samples, n_classes, width, soft_labels)` returns the
      float32 render `[B, C, N]`.
    - `backward(go, vol, lab, M, src, tgt, step, n_samples, n_classes, width, need_gvol, need_gcam,
      soft_labels)` returns `(gvol, gM, gsrc, gtgt, gstep)`, already reduced: float32 `[D, H, W]`,
      `[B, 3, 4]`, `[B, N, 3]`, `[B, N, 3]` and `[B, N]`. Gradients that were not requested
      (`gvol` unless `need_gvol`; the last three unless `need_gcam`) are empty tensors.

    `name` is used in messages (it is also the key in `DEVICE_BACKENDS`, which says which device the
    kernels run on) and `op_name` names the registered ops (`nanodrr::{op_name}` and
    `nanodrr::{op_name}_bwd`). `supports(dtype)` says whether a volume of that dtype can run.
    """

    name: str
    op_name: str
    forward: Callable
    backward: Callable
    supports: Callable[[torch.dtype], bool]


@dataclasses.dataclass(frozen=True)
class RegisteredBackend(FusedBackend):
    """A `FusedBackend` together with its registered differentiable op (see `register_fused_op`)."""

    op: Callable = dataclasses.field(kw_only=True)


def fused_supported(subject: Subject, B: int, n_pixels: int) -> bool:
    """Hard limits of the fused kernels: one volume, int32 indexing."""
    N = B * n_pixels
    return subject.image.shape[0] == 1 and max(subject.image.numel(), subject.n_classes * N, 3 * N) < 2**31


def src_strides(src: torch.Tensor) -> tuple[int, int]:
    """`(ray, batch)` strides of the ray sources `[B, N, 3]` or `[B, 1, 3]` (a broadcast source has ray stride 0)."""
    return (0 if src.shape[1] == 1 else 3), src.shape[1] * 3


def _module(name: str):
    """The module of fused backend `name`. Explicit imports (not `importlib`) so that
    `torch.compile(fullgraph=True)` can trace through the lookup."""
    if name == "triton":
        from . import _triton

        return _triton
    raise ValueError(f"Unknown fused backend {name!r}; expected one of {FUSED_BACKENDS}")


def load_backend(name: str) -> RegisteredBackend:
    """Import fused backend `name` and return its `BACKEND`."""
    return _module(name).BACKEND


def available(name: str, dtype: torch.dtype) -> bool:
    """Whether fused backend `name` can run on this machine for a volume of `dtype`.

    Cheap and uncached on purpose: `render` calls it under `torch.compile`, where dynamo warns about
    (and cannot honor) `functools.cache` wrappers. Backends answer from constants."""
    try:
        return load_backend(name).supports(dtype)
    except ImportError:  # e.g. triton is not installed
        return False


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
        and available(name, subject.image.dtype)
    )
    return name if eligible else None


def register_fused_op(backend: FusedBackend) -> RegisteredBackend:
    """Register the differentiable ops of `backend` and return it together with its `op`.

    Registered as custom ops so `torch.compile` treats the kernels as opaque: inductor takes
    shapes from the fake implementations rather than tracing the kernel source.
    """
    op = f"nanodrr::{backend.op_name}"

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
        out = backend.forward(vol, lab, M.contiguous(), src, tgt, step, n_samples, n_classes, width, soft_labels)
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
                f"Volume gradients on the {backend.name} backend use nondeterministic atomics; "
                "disable deterministic algorithms or use backend='torch'"
            )
        return backend.backward(
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

    def backward(ctx, go):
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

    torch.library.register_autograd(op, backward, setup_context=setup_context)
    return RegisteredBackend(**{f.name: getattr(backend, f.name) for f in dataclasses.fields(backend)}, op=raymarch)
