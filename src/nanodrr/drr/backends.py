import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..data import Subject
from ..geometry import transform_point


def _reflect(x: torch.Tensor, twice_low: int, twice_high: int) -> torch.Tensor:
    """Reflect coordinates into `[twice_low / 2, twice_high / 2]` (as in ATen's `reflect_coordinates`)."""
    if twice_low == twice_high:
        return torch.zeros_like(x)
    low = twice_low / 2
    span = (twice_high - twice_low) / 2
    x = (x - low).abs()
    extra = torch.fmod(x, span)
    flips = torch.floor(x / span)
    return torch.where(flips % 2 == 0, extra + low, span - extra + low)


def _grid_sample_nearest(
    input: Float[torch.Tensor, "B C *S"],
    grid: Float[torch.Tensor, "B *O D"],
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Float[torch.Tensor, "B C *O"]:
    """Nearest-neighbor `F.grid_sample` for 2D (4D input) and 3D (5D input) volumes.

    Drop-in for `F.grid_sample(..., mode="nearest")`, which MPS does not
    implement for 3D inputs. Coordinates are unnormalized, padded, rounded half
    to even, and gathered from the flattened input. No gradient flows to `grid`
    (as with the native nearest mode); gradients flow to `input` via `gather`.
    """
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"Unknown padding_mode {padding_mode!r}; expected 'zeros', 'border', or 'reflection'")
    B, C, *spatial = input.shape
    ndim = len(spatial)
    if ndim not in (2, 3) or grid.shape[-1] != ndim or grid.shape[0] != B:
        raise ValueError(f"Expected a {ndim}D input of shape (B, C, *S) with a grid of shape (B, *O, {ndim})")
    out_shape = grid.shape[1:-1]

    # Grid is ordered (x, y[, z]) = reversed spatial dims
    size = torch.tensor(spatial[::-1], device=grid.device, dtype=grid.dtype)
    if align_corners:
        x = (grid + 1) / 2 * (size - 1)
    else:
        x = ((grid + 1) * size - 1) / 2

    # Apply padding to the coordinates (before rounding, as ATen does)
    if padding_mode == "border":
        x = torch.minimum(torch.clamp_min(x, 0), size - 1)
    elif padding_mode == "reflection":
        if align_corners:
            x = torch.stack([_reflect(x[..., i], 0, 2 * (n - 1)) for i, n in enumerate(spatial[::-1])], dim=-1)
        else:
            x = torch.stack([_reflect(x[..., i], -1, 2 * n - 1) for i, n in enumerate(spatial[::-1])], dim=-1)
        x = torch.minimum(torch.clamp_min(x, 0), size - 1)

    ijk = torch.round(x)
    valid = ((ijk >= 0) & (ijk < size)).all(dim=-1)  # only ever False for "zeros"
    ijk = ijk.long().clamp_min(0)

    # Flatten (x, y[, z]) to a linear index into the spatial dims
    flat = torch.zeros_like(ijk[..., 0])
    for dim, n in zip(range(ndim - 1, -1, -1), spatial):
        flat = flat * n + ijk[..., dim]
    flat = flat.masked_fill(~valid, 0).reshape(B, -1)  # [B, P]

    if input.stride(0) == 0:  # batch-expanded view: avoid materializing B copies
        out = input[0].reshape(C, -1)[:, flat].transpose(0, 1)  # [B, C, P]
    else:
        out = input.reshape(B, C, -1).gather(2, flat[:, None].expand(-1, C, -1))
    out = out * valid.reshape(B, 1, -1)
    return out.reshape(B, C, *out_shape)


def _grid_sample(
    input: Float[torch.Tensor, "B C *S"],
    grid: Float[torch.Tensor, "B *O D"],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Float[torch.Tensor, "B C *O"]:
    """`F.grid_sample`, routing around ops missing on MPS (nearest mode)."""
    if mode == "nearest" and input.device.type == "mps":
        return _grid_sample_nearest(input, grid, padding_mode, align_corners)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


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
    img = _grid_sample(
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
    idx = _grid_sample(
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
