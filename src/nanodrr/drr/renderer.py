import functools

import torch
from jaxtyping import Float

from ..data import Subject
from .backends import fused_supported, render_fused, render_torch

_FUSED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@functools.cache
def _triton_available() -> bool:
    try:
        from . import _fused  # noqa: F401
    except Exception:
        return False
    return True


def render(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    n_samples: int = 500,
    orthographic: bool = False,
    src: Float[torch.Tensor, "B (H W) 3"] | None = None,
    tgt: Float[torch.Tensor, "B (H W) 3"] | None = None,
    backend: str = "auto",
) -> Float[torch.Tensor, "B C H W"]:
    """Differentiable ray marching through a volume and optional labelmap.

    Casts rays from an X-ray source through a 3D volume (`Subject.image`) and
    integrates sampled intensities along each ray to produce a synthetic
    radiograph. When the subject contains a multi-class labelmap (`Subject.label`),
    the integration is performed per-structure, yielding one channel per class.

    Args:
        subject: The volume to render. Must contain `Subject.image` (the 3D
            density volume) and optionally `Subject.label` (a multi-class
            labelmap for per-structure integration).
        k_inv: Inverse intrinsic camera matrix. Maps pixel coordinates to
            camera-space ray directions.
        rt_inv: Inverse extrinsic (world-to-camera) matrix. Transforms rays
            from camera space into world space.
        sdd: Source-to-detector distance, i.e., the distance from the X-ray
            point source to the imaging plane.
        height: Output image height in pixels.
        width: Output image width in pixels.
        n_samples: Number of samples to take along each ray. Higher values
            improve accuracy at the cost of memory and compute.
        orthographic: Render with parallel beams instead of cone beams.
        src: Pre-computed ray source positions in camera coordinates. If `None`,
            computed from `k_inv` and `rt_inv`.
        tgt: Pre-computed ray target positions (detector pixel locations) in
            camera coordinates. If `None`, computed from `k_inv` and `rt_inv`.
        backend: `"torch"` uses the reference `grid_sample` implementation;
            `"triton"` uses the fused Triton kernel; `"auto"` (default) picks
            `"triton"` on CUDA with fp32/fp16/bf16 when eligible and `"torch"`
            otherwise. The backends agree to float32 sampling precision (~1e-5);
            `"triton"` assumes an affine `rt_inv` and has no double backward.

    Returns:
        Rendered synthetic radiograph. Shape is `(B, C, H, W)` where `C` is
            the number of classes in the labelmap (or 1 if no labelmap is
            present).
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    device, dtype = rt_inv.device, rt_inv.dtype

    # Get the ray endpoints in camera coordinates
    if tgt is None:
        tgt = _make_tgt(k_inv, sdd, height, width, device, dtype)
    if src is None:
        src = _make_src(orthographic, tgt, sdd)

    # Compute step size [mm] in camera space
    step_size = (tgt - src).norm(dim=-1) / float(n_samples - 1)

    if backend == "auto":
        eligible = (
            rt_inv.is_cuda
            and dtype in _FUSED_DTYPES
            and subject.image.dtype in _FUSED_DTYPES
            and not (torch.are_deterministic_algorithms_enabled() and subject.image.requires_grad)  # gvol uses atomics
            and fused_supported(subject, max(rt_inv.shape[0], tgt.shape[0]), height * width)
            and _triton_available()
        )
        backend = "triton" if eligible else "torch"
    if backend == "triton":
        return render_fused(subject, rt_inv, src, tgt, step_size, n_samples, height, width)
    if backend == "torch":
        return render_torch(subject, rt_inv, src, tgt, step_size, n_samples, height, width)
    raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'torch', or 'triton'")


def _make_tgt(
    k_inv: Float[torch.Tensor, "B 3 3"],
    sdd: Float[torch.Tensor, "B"],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "B (H W) 3"]:
    N = height * width
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(N, 3)
    tgt = sdd[:, None, None] * torch.einsum("bij,nj->bni", k_inv, uv1)
    return tgt


def _make_src(
    orthographic: bool,
    tgt: Float[torch.Tensor, "B (H W) 3"],
    sdd: Float[torch.Tensor, "B"],
) -> Float[torch.Tensor, "B (H W) 3"] | Float[torch.Tensor, "B 1 3"]:
    if orthographic:
        src = tgt.clone()
        src[..., 2] -= sdd[:, None]
        return src
    else:
        B, _, _ = tgt.shape
        device, dtype = tgt.device, tgt.dtype
        return torch.zeros(B, 1, 3, device=device, dtype=dtype)
