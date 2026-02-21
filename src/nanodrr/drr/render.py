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
    src: Float[torch.Tensor, "B (H W) 3"] | None = None,
    tgt: Float[torch.Tensor, "B (H W) 3"] | None = None,
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
        align_corners: If `True`, the voxel grid corners are aligned with the
            volume boundaries (consistent with `torch.nn.functional.grid_sample`).
        src: Pre-computed ray source positions in world coordinates. If `None`,
            computed from `k_inv` and `rt_inv`.
        tgt: Pre-computed ray target positions (detector pixel locations) in
            world coordinates. If `None`, computed from `k_inv` and `rt_inv`.

    Returns:
        Rendered synthetic radiograph. Shape is `(B, C, H, W)` where `C` is
            the number of classes in the labelmap (or 1 if no labelmap is
            present).
    """
    device, dtype = rt_inv.device, rt_inv.dtype
    B = rt_inv.shape[0]
    C = subject.n_classes
    N = height * width

    # Get the ray endpoints in camera coordinates
    if src is None:
        src = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    if tgt is None:
        tgt = _make_tgt(k_inv, sdd, height, width, device, dtype)

    # Compute step size [mm] in camera space
    step_size = (tgt - src).norm(dim=-1) / float(n_samples - 1)

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
        align_corners=align_corners,
    )[:, 0, ..., 0]  # [B, n_samples, N]
    img = img * step_size[:, None, :]

    if C == 1:  # Compute whole-volume ray marching
        return img.sum(dim=1, keepdim=True).reshape(B, C, height, width)

    # Sample the mask
    idx = F.grid_sample(
        subject.label.expand(B, -1, -1, -1, -1),
        pts,
        mode="nearest",
        align_corners=align_corners,
    )[:, 0, ..., 0].long()  # [B, n_samples, N]

    # Compute the structure-specific ray marching
    out = torch.zeros(B, C, N, device=img.device, dtype=img.dtype)
    out.scatter_add_(1, idx, img)
    return out.reshape(B, C, height, width)


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
