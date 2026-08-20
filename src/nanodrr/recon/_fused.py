import torch
import triton
import triton.language as tl

_BLOCK = 256
_WARPS = 4


@triton.jit
def _sample2d(img_ptr, iu, iv, H, W, m):
    """Load img[iv, iu] with zero padding outside bounds."""
    inb = m & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    return tl.load(img_ptr + iv * W + iu, mask=inb, other=0.0)


@triton.jit
def _bp_kernel(
    proj_ptr,
    P_ptr,
    w_ptr,
    out_ptr,
    NV,
    N,
    VH,
    VW,
    IH,
    IW,
    BLOCK: tl.constexpr,
):
    """Fused voxel-driven backprojection.

    Each program owns BLOCK voxels and loops over all NV views, so the
    accumulator lives in registers and each output voxel is written exactly
    once — a pure gather with no atomics. `P` maps voxel indices to
    homogeneous pixel coordinates (uz, vz, z); the per-view weight is
    `w[v] / z**2` (orthographic matrices produce z = 1). Voxels behind the
    source or projecting outside the detector contribute zero.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    ix = (offs % VW).to(tl.float32)
    iy = ((offs // VW) % VH).to(tl.float32)
    iz = (offs // (VW * VH)).to(tl.float32)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for v in range(NV):
        Pb = P_ptr + v * 12
        p00 = tl.load(Pb + 0)
        p01 = tl.load(Pb + 1)
        p02 = tl.load(Pb + 2)
        p03 = tl.load(Pb + 3)
        p10 = tl.load(Pb + 4)
        p11 = tl.load(Pb + 5)
        p12 = tl.load(Pb + 6)
        p13 = tl.load(Pb + 7)
        p20 = tl.load(Pb + 8)
        p21 = tl.load(Pb + 9)
        p22 = tl.load(Pb + 10)
        p23 = tl.load(Pb + 11)
        wv = tl.load(w_ptr + v)

        z = p20 * ix + p21 * iy + p22 * iz + p23
        ok = m & (z > 1e-6)
        zs = tl.where(ok, z, 1.0)
        u = (p00 * ix + p01 * iy + p02 * iz + p03) / zs - 0.5
        t = (p10 * ix + p11 * iy + p12 * iz + p13) / zs - 0.5

        u0 = tl.floor(u)
        t0 = tl.floor(t)
        fu = u - u0
        ft = t - t0
        iu = u0.to(tl.int32)
        it = t0.to(tl.int32)

        base = proj_ptr + v * IH * IW
        s00 = _sample2d(base, iu, it, IH, IW, ok)
        s01 = _sample2d(base, iu + 1, it, IH, IW, ok)
        s10 = _sample2d(base, iu, it + 1, IH, IW, ok)
        s11 = _sample2d(base, iu + 1, it + 1, IH, IW, ok)
        val = (s00 * (1 - fu) + s01 * fu) * (1 - ft) + (s10 * (1 - fu) + s11 * fu) * ft
        acc += tl.where(ok, wv / (zs * zs) * val, 0.0)

    tl.store(out_ptr + offs, acc, mask=m)


@torch.library.custom_op("nanodrr::fused_backproject", mutates_args=())
def fused_backproject(
    proj: torch.Tensor,
    mat: torch.Tensor,
    view_weights: torch.Tensor,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Fused voxel-driven backprojection (forward only).

    Registered as a custom op so `torch.compile` treats the kernel as opaque.
    No autograd formula is registered: the operator is a one-shot analytic
    reconstruction step; use `backend="torch"` in `backproject` when gradients
    with respect to the projections or geometry are needed.
    """
    B, det_h, det_w = proj.shape
    n = depth * height * width
    out = torch.empty(depth, height, width, device=proj.device, dtype=torch.float32)
    _bp_kernel[(triton.cdiv(n, _BLOCK),)](
        proj.float().contiguous(),
        mat.float().contiguous(),
        view_weights.float().contiguous(),
        out,
        B,
        n,
        height,
        width,
        det_h,
        det_w,
        BLOCK=_BLOCK,  # ty: ignore[invalid-argument-type]
        num_warps=_WARPS,  # ty: ignore[unknown-argument]
    )
    return out


@fused_backproject.register_fake
def _(proj, mat, view_weights, depth, height, width):
    return torch.empty(depth, height, width, device=proj.device, dtype=torch.float32)
