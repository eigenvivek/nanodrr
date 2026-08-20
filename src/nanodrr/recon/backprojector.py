import functools

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..data import Subject

_FUSED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@functools.cache
def _triton_available() -> bool:
    try:
        from . import _fused  # noqa: F401
    except Exception:
        return False
    return True


def _invert_k(k_inv: Float[torch.Tensor, "B 3 3"]) -> Float[torch.Tensor, "B 3 3"]:
    """Analytic inverse of the (analytically inverted) intrinsic matrix."""
    k = torch.zeros_like(k_inv)
    k[..., 0, 0] = 1.0 / k_inv[..., 0, 0]
    k[..., 1, 1] = 1.0 / k_inv[..., 1, 1]
    k[..., 0, 2] = -k_inv[..., 0, 2] / k_inv[..., 0, 0]
    k[..., 1, 2] = -k_inv[..., 1, 2] / k_inv[..., 1, 1]
    k[..., 2, 2] = 1.0
    return k


def make_proj(
    subject: Subject,
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    orthographic: bool = False,
) -> Float[torch.Tensor, "B 3 4"]:
    r"""Fuse the voxel → detector-pixel projection into a single matrix.

    The inverse of the rendering path: voxel indices are mapped to world space
    (`Subject.voxel_to_world`), into camera space (the inverse of `rt_inv`),
    and onto the detector (`K`), giving homogeneous pixel coordinates

    $$
    (uz, vz, z)^\top = \mathbf K \, \mathbf{RT}_{[:3]} \, \mathbf V \, (i, j, k, 1)^\top \,,
    $$

    so a voxel projects to pixel $(u, v) = (uz / z, vz / z)$ at camera-space
    depth $z$. For `orthographic` cameras the intrinsic rows are rescaled so
    the third homogeneous coordinate is identically one (no perspective
    divide, unit backprojection weight).

    Matrices are inverted and composed in float64, then downcast.
    """
    rt = torch.linalg.inv(rt_inv.double())
    e = (rt @ subject.voxel_to_world.double())[..., :3, :]  # voxel indices → camera space
    if not orthographic:
        proj = _invert_k(k_inv.double()) @ e
    else:
        k_inv64, sdd64 = k_inv.double(), sdd.double()
        fx = 1.0 / k_inv64[..., 0, 0]
        fy = 1.0 / k_inv64[..., 1, 1]
        cx = -k_inv64[..., 0, 2] * fx
        cy = -k_inv64[..., 1, 2] * fy
        proj = torch.zeros_like(e)
        proj[..., 0, :] = (fx / sdd64)[..., None] * e[..., 0, :]
        proj[..., 1, :] = (fy / sdd64)[..., None] * e[..., 1, :]
        proj[..., 0, 3] += cx
        proj[..., 1, 3] += cy
        proj[..., 2, 3] = 1.0
    return proj.to(rt_inv.dtype)


def backproject(
    subject: Subject,
    proj: Float[torch.Tensor, "B C H W"],
    k_inv: Float[torch.Tensor, "B 3 3"],
    rt_inv: Float[torch.Tensor, "B 4 4"],
    sdd: Float[torch.Tensor, "B"],
    view_weights: Float[torch.Tensor, "B"] | None = None,
    orthographic: bool = False,
    backend: str = "auto",
    views_per_chunk: int = 4,
) -> Float[torch.Tensor, "1 C D H W"]:
    r"""Voxel-driven backprojection of detector images into a volume.

    The adjoint-flavored inverse of `render`: every voxel center of
    `subject.image`'s grid is projected onto each detector via `make_proj`,
    the (typically ramp-filtered) projection is sampled bilinearly there, and
    the samples are accumulated over views with the FDK distance weight,

    $$
    f(\mathbf x) \mathrel{+}= \frac{w_\beta}{z(\mathbf x)^2} \,
        q_\beta\big(u(\mathbf x), v(\mathbf x)\big) \,,
    $$

    where $z$ is the voxel's camera-space depth. With the default
    `view_weights` $w_\beta = \mathrm{SDD}^2$ this is the classical
    $(\mathrm{SDD} / z)^2$ magnification weight; for `orthographic` cameras
    $z \equiv 1$ and the default weight is one. Voxels behind the source or
    projecting outside the detector contribute zero.

    This is a gather (one read per voxel per view), unlike the scatter
    obtained by differentiating `render` with respect to the volume — the
    autograd adjoint is exact but ray-driven, so a single backprojection pass
    through it aliases; this voxel-driven version is the right operator for
    analytic reconstruction.

    Args:
        subject: Geometry template; the output is defined on the voxel grid of
            `subject.image` (its intensities are not read).
        proj: Detector images to backproject, e.g. from `fdk_filter`.
        k_inv: Inverse intrinsic camera matrix.
        rt_inv: Camera-to-world (inverse extrinsic) matrices, one per view.
        sdd: Source-to-detector distance (mm).
        view_weights: Per-view scalar weight $w_\beta$. Defaults to
            $\mathrm{SDD}^2$ (perspective) or one (orthographic).
        orthographic: Backproject along parallel beams instead of cone beams.
        backend: `"torch"` uses the reference `grid_sample` implementation;
            `"triton"` uses the fused kernel (single-channel, no autograd);
            `"auto"` picks `"triton"` on CUDA when eligible and no gradients
            are required.
        views_per_chunk: Number of views the torch backend processes per
            `grid_sample` call; lower values reduce peak memory.

    Returns:
        Accumulated volume on `subject`'s voxel grid, shape `(1, C, D, H, W)`.
    """
    B, C, _, _ = proj.shape
    depth, height, width = subject.image.shape[-3:]

    mat = make_proj(subject, k_inv, rt_inv, sdd, orthographic)
    if mat.shape[0] != B:
        mat = mat.expand(B, -1, -1)
    if view_weights is None:
        if orthographic:
            view_weights = torch.ones(B, device=proj.device, dtype=mat.dtype)
        else:
            view_weights = (sdd.to(mat.dtype) ** 2).expand(B)

    if backend == "auto":
        needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (proj, k_inv, rt_inv, sdd, view_weights))
        eligible = (
            proj.is_cuda
            and C == 1
            and proj.dtype in _FUSED_DTYPES
            and not needs_grad
            and _fused_supported(depth * height * width, proj)
            and _triton_available()
        )
        backend = "triton" if eligible else "torch"
    if backend == "triton":
        from ._fused import fused_backproject

        if not _fused_supported(depth * height * width, proj):
            raise ValueError("inputs exceed the fused kernel's limits; use backend='torch'")
        out = fused_backproject(proj.reshape(B, *proj.shape[-2:]), mat, view_weights, depth, height, width)
        return out.to(proj.dtype).reshape(1, C, depth, height, width)
    if backend == "torch":
        return _backproject_torch((depth, height, width), proj, mat, view_weights, views_per_chunk)
    raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'torch', or 'triton'")


def _fused_supported(n_voxels: int, proj: torch.Tensor) -> bool:
    """Hard limits of the fused kernel: single channel, int32 indexing."""
    return proj.shape[1] == 1 and max(n_voxels, proj.numel()) < 2**31


def _backproject_torch(
    vol_shape: tuple[int, int, int],
    proj: Float[torch.Tensor, "B C H W"],
    mat: Float[torch.Tensor, "B 3 4"],
    view_weights: Float[torch.Tensor, "B"],
    views_per_chunk: int,
) -> Float[torch.Tensor, "1 C D H W"]:
    """Reference `grid_sample` implementation of `backproject`."""
    B, C, det_h, det_w = proj.shape
    depth, height, width = vol_shape
    work_dtype = torch.promote_types(proj.dtype, torch.float32)
    device = proj.device

    # Voxel-center indices in (x, y, z) order, matching `Subject.voxel_to_world`
    iz, iy, ix = torch.meshgrid(
        torch.arange(depth, device=device, dtype=work_dtype),
        torch.arange(height, device=device, dtype=work_dtype),
        torch.arange(width, device=device, dtype=work_dtype),
        indexing="ij",
    )
    pts = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3)  # [N, 3]

    mat = mat.to(work_dtype)
    view_weights = view_weights.to(work_dtype)
    out = torch.zeros(C, pts.shape[0], device=device, dtype=work_dtype)
    for i in range(0, B, views_per_chunk):
        m, p, w = mat[i : i + views_per_chunk], proj[i : i + views_per_chunk], view_weights[i : i + views_per_chunk]

        # Homogeneous pixel coordinates (uz, vz, z) of every voxel center
        uvz = torch.einsum("vij,nj->vni", m[..., :3], pts) + m[..., None, :, 3]
        z = uvz[..., 2]
        valid = z > 1e-6
        z = torch.where(valid, z, torch.ones_like(z))

        # Continuous pixel coords u ∈ [0, W] map to grid_sample's [-1, 1]
        grid = torch.stack(
            [2.0 * uvz[..., 0] / (z * det_w) - 1.0, 2.0 * uvz[..., 1] / (z * det_h) - 1.0],
            dim=-1,
        ).unsqueeze(-2)  # [V, N, 1, 2]
        sampled = F.grid_sample(p.to(work_dtype), grid, mode="bilinear", align_corners=False)[..., 0]  # [V, C, N]
        out = out + (sampled * (valid * w[:, None] / z.square())[:, None]).sum(dim=0)

    return out.to(proj.dtype).reshape(1, C, depth, height, width)
